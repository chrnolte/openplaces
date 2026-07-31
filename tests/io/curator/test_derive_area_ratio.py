"""Tests for the generic `derive_area_ratio` curate step."""

from __future__ import annotations

import pandas as pd

from openplaces.io.curator import CurateState
from openplaces.io.curator.inferers import derive_area_ratio


def _state(df: pd.DataFrame) -> CurateState:
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=None,
        verbose=False,
        timer=None,
        curated=df,
    )


def test_ratio_reuses_existing_area_ha_column():
    # 1 ha = 10,000 m2; a 500 m2 footprint on a 1 ha parcel is a 5% share.
    df = pd.DataFrame({'area_ha': [1.0], 'sum_footprint_area_m2': [500.0]})
    out = derive_area_ratio(
        _state(df), value_column='sum_footprint_area_m2', output='footprint_area_share'
    ).curated
    assert out['footprint_area_share'].iloc[0] == 0.05


def test_missing_value_column_is_noop():
    df = pd.DataFrame({'area_ha': [1.0]})
    out = derive_area_ratio(
        _state(df), value_column='sum_footprint_area_m2', output='footprint_area_share'
    ).curated
    assert 'footprint_area_share' not in out.columns
