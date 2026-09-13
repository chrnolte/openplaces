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
