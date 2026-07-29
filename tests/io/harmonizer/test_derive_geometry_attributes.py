"""Tests for the `derive_geometry_attributes` harmonize step."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

import openplaces.io.harmonizer.spine as spine_module
from openplaces.io.harmonizer import HarmonizeState


def _state(spine):
    return HarmonizeState(
        recipe={'entity': {'entity_type': 'parcel'}},
        admin_id='US-NC-CE',
        verbose=False,
        timer=None,
        spine=spine,
    )


def test_area_and_centroid_computed_for_real_geometry():
    # 100m x 100m square in an equal-area CRS: area = 1 ha, centroid at (50, 50).
    spine = gpd.GeoDataFrame(
        {'geometry_source': ['obm']},
        geometry=[box(0, 0, 100, 100)],
        crs='epsg:6933',
    )
    state = spine_module.derive_geometry_attributes(_state(spine))
    assert state.spine['area_ha'].iloc[0] == pytest.approx(1.0, rel=1e-2)
    assert 'lat' in state.spine.columns
    assert 'long' in state.spine.columns


def test_synthetic_row_area_is_missing():
    # For a parcel spine (entity_type='parcel', excluded from the synthetic
    # match), a synthetic fallback geometry_source looks like
    # 'footprint.overture' -- a different entity's boundary standing in for
    # this parcel's own missing outline. 'parcel.X' itself would NOT count
    # as synthetic here (see synthetic_geometry_pattern's exclude behavior).
    spine = gpd.GeoDataFrame(
        {'geometry_source': ['obm', 'footprint.overture']},
        geometry=[box(0, 0, 100, 100), box(200, 0, 300, 100)],
        crs='epsg:6933',
    )
    state = spine_module.derive_geometry_attributes(_state(spine))
    assert state.spine['area_ha'].iloc[0] == pytest.approx(1.0, rel=1e-2)
    assert pd.isna(state.spine['area_ha'].iloc[1])
    # Position stays meaningful for a synthetic row even though area doesn't.
    assert not pd.isna(state.spine['lat'].iloc[1])
    assert not pd.isna(state.spine['long'].iloc[1])


def test_noop_when_spine_is_none():
    state = spine_module.derive_geometry_attributes(_state(None))
    assert state.spine is None
