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
from openplaces.io.harmonizer import HarmonizeState

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
        monkeypatch.setattr(links, 'find_recipes', lambda *a, **k: rows)

        assert links._find_reference_recipe('footprint', AdminId(WARREN)) is None

    def test_a_filename_suffix_is_kept(self, monkeypatch):
        rows = pd.DataFrame([_row('CO', 'igac', '2026', suffix='rural')])
        monkeypatch.setattr(links, 'find_recipes', lambda *a, **k: rows)

        found = links._find_reference_recipe('parcel', AdminId('CO-AN'))
        assert found == 'CO_parcel-igac-2026_rural'


class TestBestRecipeFor:
    def test_a_shorter_sibling_source_is_not_chosen(self):
        sources = [
            {'recipe_id': 'US-NC-WA_parcel-wakeco-2026', 'admin_id': WAKE_STALE},
            {'recipe_id': 'US-NC_parcel-nconemap-2026', 'admin_id': 'US-NC'},
        ]
        assert discover._best_recipe_for(WARREN, sources) == (
            'US-NC_parcel-nconemap-2026'
        )
