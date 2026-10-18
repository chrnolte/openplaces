"""Auto-discovery covers a unit by admin level, and reads the recipe id.

Two defects shared by the three auto-discovery paths (`spine`, `links`,
`discover`). They tested scope with `admin_str.startswith(rid_str)`, which
is a substring test: 'US-NC-WA' is Wake County's pre-2026 id and
'US-NC-WAR' is Warren County's current one, both real entries in the admin
spine, so a Wake-scoped recipe was discovered for Warren. And two of them
rebuilt the recipe id from its parts, which drops the filename suffix that
`CO_parcel-igac-2026_rural` carries, yielding an id no file has.
"""

import pandas as pd

import openplaces.io.harmonizer.discover as discover
import openplaces.io.harmonizer.links as links
import openplaces.io.harmonizer.spine as spine_module
import openplaces.recipe as recipe_module
from openplaces.core.schema import AdminId
from openplaces.diagnostics import _writes_geometry
from openplaces.io.harmonizer import HarmonizeState
from tests.links_patching import patch_links

WAKE_STALE = 'US-NC-WA'
WARREN = 'US-NC-WAR'


def _row(admin_id, source_id, version, entity_type='parcel', suffix=''):
    recipe_id = f'{admin_id}_{entity_type}-{source_id}-{version}'
    if suffix:
        recipe_id += f'_{suffix}'
    return {
        'admin_id': admin_id,
        'source_id': source_id,
        'version': version,
        'entity_type': entity_type,
        'exclude_from_auto_discover': False,
        'recipe_id': recipe_id,
        'filename_suffix': suffix,
    }


def _state(admin_id=WARREN):
    return HarmonizeState(
        recipe={'admin_id': AdminId('US')},
        admin_id=AdminId(admin_id),
        verbose=False,
        timer=None,
        spine=None,
    )


class TestExpandAutoDiscover:
    def test_a_shorter_sibling_recipe_is_not_discovered(self, monkeypatch):
        rows = pd.DataFrame(
            [_row(WAKE_STALE, 'wakeco', '2026'), _row(WARREN, 'warrenco', '2026')]
        )
        monkeypatch.setattr(spine_module, 'find_recipes', lambda *a, **k: rows)
        monkeypatch.setattr(
            recipe_module, 'find_additional_layer_recipes', lambda *a, **k: []
        )

        discovered = spine_module._expand_auto_discover(
            [{'auto_discover': True, 'entity_type': 'parcel'}], _state()
        )

        assert [s['recipe_id'] for s in discovered] == [
            'US-NC-WAR_parcel-warrenco-2026'
        ]

    def test_a_filename_suffix_is_kept(self, monkeypatch):
        rows = pd.DataFrame([_row('CO', 'igac', '2026', suffix='rural')])
        monkeypatch.setattr(spine_module, 'find_recipes', lambda *a, **k: rows)
        monkeypatch.setattr(
            recipe_module, 'find_additional_layer_recipes', lambda *a, **k: []
        )

        discovered = spine_module._expand_auto_discover(
            [{'auto_discover': True, 'entity_type': 'parcel'}], _state('CO-AN')
        )

        assert [s['recipe_id'] for s in discovered] == ['CO_parcel-igac-2026_rural']


class TestFindReferenceRecipe:
    def test_a_shorter_sibling_recipe_is_not_selected(self, monkeypatch):
        rows = pd.DataFrame([_row(WAKE_STALE, 'wakeco', '2026', 'footprint')])
        patch_links(monkeypatch, 'find_recipes', lambda *a, **k: rows)

        assert links._find_reference_recipe('footprint', AdminId(WARREN)) is None

    def test_a_filename_suffix_is_kept(self, monkeypatch):
        rows = pd.DataFrame([_row('CO', 'igac', '2026', suffix='rural')])
        patch_links(monkeypatch, 'find_recipes', lambda *a, **k: rows)

        found = links._find_reference_recipe('parcel', AdminId('CO-AN'))
        assert found == 'CO_parcel-igac-2026_rural'


class TestReferenceChoiceReadsRecipesOnly:
    """The DAG resolves the reference before the run's ingests write it.

    It declares the link sidecar from that answer, and the harmonize job
    resolves again at run time. Both must agree whatever outputs exist:
    New Hanover's footprint geospine was declared against the state
    parcel layer, found the county's `_geo` written by the time it ran,
    wrote the county sidecar and failed with a MissingOutputException.
    """

    def test_the_county_recipe_wins_before_and_after_its_ingest(
        self, monkeypatch, tmp_path
    ):
        state_id = 'US-NC_parcel-nconemap-2025'
        county_id = 'US-NC-NHA_parcel-nhcgov-2026'
        rows = pd.DataFrame(
            [
                _row('US-NC', 'nconemap', '2025'),
                _row('US-NC-NHA', 'nhcgov', '2026'),
            ]
        )
        patch_links(monkeypatch, 'find_recipes', lambda *a, **k: rows)

        # Outputs live under tmp_path, so a resolver probing the
        # filesystem sees exactly the files this test writes. The
        # links module bound get_output_path while it probed; the
        # patch stays there (raising=False) so a returning probe
        # fails this test.
        def output_path(recipe_id, *a, **k):
            return tmp_path / f'{recipe_id}.parquet'

        monkeypatch.setattr(recipe_module, 'get_output_path', output_path)
        monkeypatch.setattr(
            links.spatial, 'get_output_path', output_path, raising=False
        )
        (tmp_path / f'{state_id}_geo.parquet').touch()

        before = links._find_reference_recipe('parcel', AdminId('US-NC-NHA'))
        (tmp_path / f'{county_id}_geo.parquet').touch()
        after = links._find_reference_recipe('parcel', AdminId('US-NC-NHA'))

        assert before == after == county_id

    def test_newest_version_then_first_id_breaks_a_tie(self, monkeypatch):
        rows = pd.DataFrame(
            [
                _row('US-FL', 'floridagio', '2025'),
                _row('US-FL', 'floridagio', '2026'),
                _row('US-DC', 'dcgovtaxlots', '2026'),
                _row('US-DC', 'dcgovrecordlots', '2026'),
            ]
        )
        patch_links(monkeypatch, 'find_recipes', lambda *a, **k: rows)

        florida = links._find_reference_recipe('parcel', AdminId('US-FL-LAK'))
        district = links._find_reference_recipe('parcel', AdminId('US-DC-WAS'))

        assert florida == 'US-FL_parcel-floridagio-2026'
        assert district == 'US-DC_parcel-dcgovrecordlots-2026'

    def test_a_recipe_that_cannot_be_a_reference_is_skipped(self, monkeypatch):
        flat = _row('US-PA-DA', 'dauphinco', '2026', suffix='characteristics')
        flat['writes_geometry'] = False
        patch = _row('US-PA-DA', 'aaa', '2027')
        patch['patches'] = 'US_parcel-geospine-2026'
        supplement = _row('US-PA-DA', 'aab', '2027')
        supplement['supplements'] = 'US-PA-DA_parcel-dauphinco-2026'
        rows = pd.DataFrame(
            [flat, patch, supplement, _row('US-PA-DA', 'dauphinco', '2026')]
        )
        patch_links(monkeypatch, 'find_recipes', lambda *a, **k: rows)

        found = links._find_reference_recipe('parcel', AdminId('US-PA-DA'))
        assert found == 'US-PA-DA_parcel-dauphinco-2026'

    def test_dauphin_resolves_to_its_parcel_layer_in_the_recipe_tree(self):
        found = links._find_reference_recipe('parcel', AdminId('US-PA-DA'))
        assert found == 'US-PA-DA_parcel-dauphinco-2026'


class TestWritesGeometry:
    def test_a_flat_source_file_writes_no_geometry(self):
        assert not _writes_geometry({'uncompressed_file_name': 'roll.CSV'})
        assert not _writes_geometry({'compressed_file_name': 'roll.accdb'})

    def test_save_to_geometry_false_writes_no_geometry(self):
        assert not _writes_geometry({'save_to': {'geometry': False}})

    def test_an_undecided_declaration_keeps_the_recipe(self):
        # .tab is read with geopandas before the flat-table branch.
        assert _writes_geometry({'uncompressed_file_name': 'parcels.tab'})
        assert _writes_geometry({'uncompressed_file_name': 'parcels.gdb'})
        assert _writes_geometry({'compressed_file_name': 'parcels.zip'})
        assert _writes_geometry({})


class TestBestRecipeFor:
    def test_a_shorter_sibling_source_is_not_chosen(self):
        sources = [
            {'recipe_id': 'US-NC-WA_parcel-wakeco-2026', 'admin_id': WAKE_STALE},
            {'recipe_id': 'US-NC_parcel-nconemap-2026', 'admin_id': 'US-NC'},
        ]
        assert discover._best_recipe_for(WARREN, sources) == (
            'US-NC_parcel-nconemap-2026'
        )
