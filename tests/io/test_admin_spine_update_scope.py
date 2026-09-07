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
