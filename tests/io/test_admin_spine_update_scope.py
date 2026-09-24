"""`update_admin_spine` replaces its own slice, not a sibling's.

A scoped admin recipe is authoritative under its own unit, so the update
deletes that slice before rebuilding it. The slice was cut with a raw
string prefix, which is not containment once leaf codes mix two and three
characters: 'US-NC-WA' is Wake County's pre-2026 id and 'US-NC-WAR' is
Warren County's current one, so a recipe still carrying the short id
deleted Warren's units from the spine as well.
"""

import pandas as pd
import pytest

from openplaces.core.schema import AdminId
from openplaces.io import admin as admin_module

WAKE_STALE = 'US-NC-WA'
WAKE_TOWN = 'US-NC-WA-CA'
WARREN_TOWN = 'US-NC-WAR-FO'
NEW_WAKE_TOWN = 'US-NC-WA-ZZ'


@pytest.fixture
def spine_update(monkeypatch, tmp_path):
    """Run update_admin_spine against a fabricated spine and recipe output."""
    spine = pd.DataFrame(
        {'name': ['Cary', 'Franklinton'], 'type': ['Town', 'Town']},
        index=pd.Index([WAKE_TOWN, WARREN_TOWN], name='admin4_id'),
    )
    local = pd.DataFrame(
        {'name': ['Cary', 'Zebulon'], 'type': ['Town', 'Town']},
        index=pd.Index([WAKE_TOWN, NEW_WAKE_TOWN], name='admin4_id'),
    )
    local_path = tmp_path / 'local.parquet'
    local.to_parquet(local_path)
    out_path = tmp_path / 'admin4_test.csv'

    monkeypatch.setattr(
        admin_module,
        'get_recipe_by_id',
        lambda *a, **k: {'admin_id': AdminId(WAKE_STALE)},
    )
    monkeypatch.setattr(admin_module, 'get_admin', lambda *a, **k: spine.copy())
    monkeypatch.setattr(admin_module, 'get_output_path', lambda *a, **k: local_path)
    monkeypatch.setattr(admin_module, 'recipe_path', lambda *a, **k: out_path)

    def _run():
        admin_module.update_admin_spine(
            level=4,
            admin_recipe_id='US-NC-WA_admin-fabricated-2026_admin4',
            test=True,
            silent=True,
        )
        return pd.read_csv(out_path, index_col=0, encoding='utf-8-sig')

    return _run


def test_a_sibling_county_is_not_deleted(spine_update):
    updated = spine_update()
    assert WARREN_TOWN in updated.index
    assert updated.loc[WARREN_TOWN, 'name'] == 'Franklinton'


def test_the_recipe_still_owns_its_own_slice(spine_update):
    updated = spine_update()
    assert WAKE_TOWN in updated.index
    assert NEW_WAKE_TOWN in updated.index


def test_the_id_column_keeps_its_name(spine_update, tmp_path):
    # `get_admin` resolves the id column by header and raises on a file
    # whose first column is unnamed, so a spine written without it
    # cannot be read back: the next update fails on its own output.
    spine_update()
    header = (tmp_path / 'admin4_test.csv').read_text().splitlines()[0]
    assert header.startswith('admin4_id,')


def test_the_spine_is_written_byte_exact(spine_update, tmp_path):
    # The same form `build.remint_spine` writes, so whichever writer ran
    # last, the committed file reads the same on every platform.
    spine_update()
    raw = (tmp_path / 'admin4_test.csv').read_bytes()
    assert not raw.startswith(b'\xef\xbb\xbf')
    assert b'\r\n' not in raw


class TestRestrictedSources:
    """A source that may not be redistributed contributes identity only."""

    FRAME = pd.DataFrame(
        {
            'name': ['Alpha'],
            'type': ['District'],
            'name_original': ['Alpha (native script)'],
            'name_alternatives': ['Alfa'],
            'admin3_id_admin1': ['101'],
        },
        index=pd.Index(['XX-AA-AL'], name='admin3_id'),
    )

    def test_its_codes_and_spellings_are_dropped(self):
        kept = admin_module._publishable_spine_columns(self.FRAME, 3, True)
        assert list(kept.columns) == ['name', 'type']

    def test_an_open_source_is_passed_through(self):
        frame = self.FRAME
        assert admin_module._publishable_spine_columns(frame, 3, False) is frame

    def test_the_flag_is_read_from_the_recipe_source(self):
        from openplaces.recipe import get_recipe_by_id

        assert admin_module._redistribution_restricted(
            get_recipe_by_id('admin-gadm-4~1_admin2')
        )
        assert not admin_module._redistribution_restricted(
            get_recipe_by_id('CO_admin-dane-2025_admin2')
        )
        assert not admin_module._redistribution_restricted({'admin_id': 'XX'})


def test_copied_codes_register_their_source(monkeypatch, tmp_path):
    # A new country's codes arrive with the row that names their source,
    # which is what the provenance test checks every code against.
    path = tmp_path / 'code-sources.csv'
    path.write_text(
        'admin1_id,level,recipe_id,scheme\nXX,3,XX_admin-old-2020_admin3,a code\n',
        encoding='utf-8',
    )
    monkeypatch.setattr(admin_module, 'recipe_path', lambda *a, **k: path)
    admin_module._register_code_source(
        3, 'XX_admin-new-2026_admin3', ['XX-AA-AL', 'YY-BB-BR']
    )
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert table.to_dict('records') == [
        {
            'admin1_id': 'XX',
            'level': '3',
            'recipe_id': 'XX_admin-new-2026_admin3',
            'scheme': 'a code',
        },
        {
            'admin1_id': 'YY',
            'level': '3',
            'recipe_id': 'XX_admin-new-2026_admin3',
            'scheme': '',
        },
    ]


class TestReplaceCountries:
    """A global recipe can replace a country's slice, not only add to it."""

    @pytest.fixture
    def global_update(self, monkeypatch, tmp_path):
        spine = pd.DataFrame(
            {'name': ['Old A', 'Old B', 'Kept'], 'type': ['Region'] * 3},
            index=pd.Index(['XX-AA', 'XX-BB', 'YY-AA'], name='admin2_id'),
        )
        local = pd.DataFrame(
            {
                'name': ['New A', 'New C'],
                'type': ['Region', 'Region'],
                'admin2_id_wikidata': ['Q1', 'Q2'],
                'name_original': ['Nueva A', ''],
            },
            index=pd.Index(['XX-AA', 'XX-CC'], name='admin2_id'),
        )
        local_path = tmp_path / 'local.parquet'
        local.to_parquet(local_path)
        out_path = tmp_path / 'admin2_test.csv'
        monkeypatch.setattr(
            admin_module, 'get_recipe_by_id', lambda *a, **k: {'admin_id': AdminId()}
        )
        monkeypatch.setattr(admin_module, 'get_admin', lambda *a, **k: spine.copy())
        monkeypatch.setattr(admin_module, 'get_output_path', lambda *a, **k: local_path)
        monkeypatch.setattr(admin_module, 'recipe_path', lambda *a, **k: out_path)
        monkeypatch.setattr(admin_module, 'find_admin_recipe_id', lambda *a, **k: None)

        def _run(**kwargs):
            admin_module.update_admin_spine(
                level=2,
                admin_recipe_id='admin-fab-2026_admin2',
                test=True,
                silent=True,
                **kwargs,
            )
            return pd.read_csv(out_path, index_col=0, dtype=str, keep_default_na=False)

        return _run

    def test_the_covered_country_is_replaced_and_others_kept(self, global_update):
        updated = global_update(replace_countries=True)
        assert sorted(updated.index) == ['XX-AA', 'XX-CC', 'YY-AA']
        assert updated.loc['XX-AA', 'name'] == 'New A'
        assert updated.loc['YY-AA', 'name'] == 'Kept'

    def test_the_source_key_and_native_name_are_carried(self, global_update):
        updated = global_update(replace_countries=True)
        assert updated.loc['XX-CC', 'admin2_id_wikidata'] == 'Q2'
        assert updated.loc['XX-AA', 'name_original'] == 'Nueva A'
        assert updated.loc['YY-AA', 'admin2_id_wikidata'] == ''

    def test_without_the_flag_only_new_units_are_added(self, global_update):
        updated = global_update()
        assert sorted(updated.index) == ['XX-AA', 'XX-BB', 'XX-CC', 'YY-AA']
        assert updated.loc['XX-AA', 'name'] == 'Old A'


class TestDroppingAUnitsLevel:
    """Emptying a country's level, which `replace_countries` cannot do."""

    @pytest.fixture
    def spine(self, monkeypatch, tmp_path):
        frame = pd.DataFrame(
            {'name': ['A', 'B', 'C', 'Kept'], 'type': ['Region'] * 4},
            index=pd.Index(['XX-AA', 'XX-BB', 'YY-AA', 'ZZ-AA'], name='admin2_id'),
        )
        out_path = tmp_path / 'admin2_test.csv'
        monkeypatch.setattr(admin_module, 'get_admin', lambda *a, **k: frame.copy())
        monkeypatch.setattr(admin_module, 'recipe_path', lambda *a, **k: out_path)
        return out_path

    def test_only_the_named_countries_go(self, spine):
        dropped = admin_module.drop_admin_units(2, ['XX', 'YY'], silent=True)
        assert sorted(dropped.index) == ['XX-AA', 'XX-BB', 'YY-AA']
        written = pd.read_csv(spine, dtype=str, index_col=0)
        assert list(written.index) == ['ZZ-AA']

    def test_a_country_with_no_rows_is_not_an_error(self, spine):
        dropped = admin_module.drop_admin_units(2, ['QQ'], silent=True)
        assert dropped.empty
        written = pd.read_csv(spine, dtype=str, index_col=0)
        assert len(written) == 4

    def test_the_id_column_keeps_its_name(self, spine):
        admin_module.drop_admin_units(2, ['XX'], silent=True)
        assert spine.read_text().splitlines()[0].startswith('admin2_id,')

    def test_a_prefix_is_not_a_country(self, spine):
        # 'X' must not take 'XX'; the split is on the separator, not a
        # string prefix, which is the bug that once deleted Warren with
        # Wake.
        dropped = admin_module.drop_admin_units(2, ['X'], silent=True)
        assert dropped.empty


class TestOneCountryAtATime:
    """A global recipe that writes a file per country is read that way.

    There is no world-wide output path for such a recipe: a null admin
    id is level 0 and the recipe saves at level 1, so asking for one
    raises. Reading the countries is also what lets the world move off
    its old source one country at a time.
    """

    @pytest.fixture
    def per_country(self, monkeypatch, tmp_path):
        frames = {
            'XX': pd.DataFrame(
                {'name': ['New A'], 'type': ['Region']},
                index=pd.Index(['XX-AA'], name='admin2_id'),
            ),
            'YY': pd.DataFrame(
                {'name': ['New B'], 'type': ['Region']},
                index=pd.Index(['YY-BB'], name='admin2_id'),
            ),
        }
        paths = {}
        for country, frame in frames.items():
            path = tmp_path / f'{country}.parquet'
            frame.to_parquet(path)
            paths[country] = path

        def _output_path(recipe, admin_id=None, **kwargs):
            if not str(admin_id or ''):
                raise ValueError('admin_id is at level 0')
            return paths[str(admin_id)]

        monkeypatch.setattr(
            admin_module, 'get_recipe_by_id', lambda *a, **k: {'admin_id': AdminId()}
        )
        monkeypatch.setattr(admin_module, 'get_output_path', _output_path)
        monkeypatch.setattr(admin_module, 'get_admin_ids', lambda *a, **k: ['XX', 'YY'])
        return paths

    def test_every_country_is_read_when_none_is_named(self, per_country):
        recipe = {'admin_id': AdminId()}
        frame = admin_module._read_recipe_output(recipe)
        assert sorted(frame.index) == ['XX-AA', 'YY-BB']

    def test_naming_one_country_reads_only_that_one(self, per_country):
        recipe = {'admin_id': AdminId()}
        frame = admin_module._read_recipe_output(recipe, ['XX'])
        assert sorted(frame.index) == ['XX-AA']

    def test_a_country_with_no_output_is_not_an_error(self, per_country):
        recipe = {'admin_id': AdminId()}
        per_country['YY'].unlink()
        frame = admin_module._read_recipe_output(recipe, ['XX', 'YY'])
        assert sorted(frame.index) == ['XX-AA']

    def test_nothing_ingested_at_all_says_so(self, per_country):
        recipe = {'admin_id': AdminId()}
        for path in per_country.values():
            path.unlink()
        with pytest.raises(FileNotFoundError):
            admin_module._read_recipe_output(recipe, ['XX', 'YY'])
