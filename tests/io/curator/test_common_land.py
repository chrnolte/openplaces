"""A plat's common land divided among its lots as an area, never as a value.

Every id, code and value below is fabricated.
"""

import numpy as np
import pandas as pd
import pytest

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.common_land import (
    apportion_common_land,
    apportion_common_land_area,
)
from openplaces.io.harmonizer import entity_links
from openplaces.io.harmonizer.entity_links import COMMON_AREA_ID, PLAT_KEY_METHOD


def _links(pairs):
    return pd.DataFrame(
        {
            'parcel_id': [lot for lot, _ in pairs],
            COMMON_AREA_ID: [common for _, common in pairs],
            'link_method': PLAT_KEY_METHOD,
            'link_source': 'exampleroll',
            'share': pd.Series([pd.NA] * len(pairs), dtype='Float64'),
        }
    )


def _curated():
    return pd.DataFrame(
        {
            'area_ha': [0.1, 0.3, 1.0, 0.5, 0.2],
            'land_value': [50_000.0, 50_000.0, np.nan, 80_000.0, np.nan],
            'land_value_imputed': [50_000.0, 50_000.0, np.nan, 80_000.0, np.nan],
            'improvement_value': [150_000.0, 140_000.0, np.nan, 1.0, np.nan],
            'is_common_area': [False, False, True, False, True],
            'subdivision_code': ['9001', '9001', '9001', '9002', '9002'],
            'plat_id_local': ['X-1', 'X-1', 'X-1', '77', '77'],
        },
        index=pd.Index(['a', 'b', 'c', 'd', 'e'], name='parcel_id'),
    )


def test_a_common_area_is_divided_in_proportion_to_lot_area():
    area = pd.Series({'a': 0.1, 'b': 0.3, 'c': 1.0})
    shares, commons = apportion_common_land_area(_links([('a', 'c'), ('b', 'c')]), area)
    assert shares['a'] == pytest.approx(0.25)
    assert shares['b'] == pytest.approx(0.75)
    assert shares.sum() == pytest.approx(1.0)
    assert commons.loc['c', 'n_lots_per_plat'] == 2
    assert commons.loc['c', 'common_land_lots_area_ha'] == pytest.approx(0.4)


def test_a_lot_linked_to_two_common_areas_takes_a_share_of_each():
    area = pd.Series({'a': 0.1, 'b': 0.3, 'c': 1.0, 'e': 0.2})
    links = _links([('a', 'c'), ('b', 'c'), ('a', 'e'), ('b', 'e')])
    shares, _ = apportion_common_land_area(links, area)
    assert shares['a'] == pytest.approx(0.25 + 0.05)
    assert shares.sum() == pytest.approx(1.2)


def _run(monkeypatch, links):
    monkeypatch.setattr(
        entity_links, 'get_plat_link_path', lambda recipe, admin_id: 'unused'
    )
    monkeypatch.setattr(entity_links, 'read_entity_link', lambda path: links)
    state = CurateState(
        recipe={'recipe_id': 'X_parcel-openplaces-2026'},
        entity_recipe={},
        admin_id=AdminId('US-XX-YY'),
        verbose=False,
        timer=None,
        curated=_curated(),
    )
    return apportion_common_land(state).curated


def test_the_step_writes_areas_counts_and_the_plat_key(monkeypatch):
    curated = _run(monkeypatch, _links([('a', 'c'), ('b', 'c')]))
    assert curated.loc['a', 'common_land_share_ha'] == pytest.approx(0.25)
    assert curated.loc['b', 'common_land_share_ha'] == pytest.approx(0.75)
    assert pd.isna(curated.loc['c', 'common_land_share_ha'])
    assert curated.loc['c', 'n_lots_per_plat'] == 2
    assert curated.loc['c', 'common_land_lots_area_ha'] == pytest.approx(0.4)
    assert pd.isna(curated.loc['a', 'n_lots_per_plat'])
    assert set(curated['common_land_plat_id'].dropna()) == {'US-XX-YY_9001/X-1'}
    assert curated['common_land_plat_id'].notna().sum() == 3
    assert curated[['common_land_share_ha']].loc[['d', 'e']].isna().all().all()


def test_the_step_writes_no_value_anywhere(monkeypatch):
    before = _curated()
    curated = _run(monkeypatch, _links([('a', 'c'), ('b', 'c')]))
    for column in before.columns:
        pd.testing.assert_series_equal(curated[column], before[column])
    # The common area keeps no land value (maintainer, 2026-10-24).
    assert pd.isna(curated.loc['c', 'land_value'])
    assert pd.isna(curated.loc['c', 'land_value_imputed'])
    new = set(curated.columns) - set(before.columns)
    assert new == {
        'common_land_share_ha',
        'n_lots_per_plat',
        'common_land_lots_area_ha',
        'common_land_plat_id',
    }


def test_a_unit_without_a_link_table_gets_empty_columns(monkeypatch):
    with pytest.warns(UserWarning, match='no valid plat link'):
        curated = _run(monkeypatch, None)
    assert curated['common_land_share_ha'].isna().all()
    assert curated['common_land_plat_id'].isna().all()


def test_links_to_rows_the_curate_dropped_are_ignored(monkeypatch):
    curated = _run(monkeypatch, _links([('a', 'c'), ('b', 'c'), ('gone', 'c')]))
    assert curated.loc['c', 'n_lots_per_plat'] == 2
    assert curated.loc['a', 'common_land_share_ha'] == pytest.approx(0.25)
