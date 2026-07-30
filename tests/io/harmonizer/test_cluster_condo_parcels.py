"""Tests for `_cluster_condo_parcels` (attributes.py)."""

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from openplaces.io.harmonizer.attributes import _cluster_condo_parcels


def _parcels_with_hub():
    return gpd.GeoDataFrame(
        {
            'land_value': [0.0, 0.0, 0.0],
            'improvement_value': [300_000.0, 250_000.0, 0.0],
            'area_ha': [0.012, 0.011, 1.35],
            'geometry': [
                box(0, 0, 0.0001, 0.0001),
                box(0.0001, 0, 0.0002, 0.0001),
                box(0, -0.001, 0.0011, 0),
            ],
        },
        index=pd.Index(['U1', 'U2', 'H1'], name='parcel_id'),
        crs='epsg:4326',
    )


def test_hub_ids_identifies_the_common_area_parcel():
    result = _cluster_condo_parcels(_parcels_with_hub(), min_group_size=2)
    assert result is not None
    component, hub_ids = result
    assert set(component.index) == {'U1', 'U2', 'H1'}
    # All three resolve to the same cluster id.
    assert component.nunique() == 1
    assert hub_ids == {'H1'}


def test_no_hub_when_all_members_are_real_units():
    parcels = gpd.GeoDataFrame(
        {
            'land_value': [0.0, 0.0],
            'improvement_value': [300_000.0, 250_000.0],
            'area_ha': [0.012, 0.011],
            'geometry': [
                box(0, 0, 0.0001, 0.0001),
                box(0.0001, 0, 0.0002, 0.0001),
            ],
        },
        index=pd.Index(['U1', 'U2'], name='parcel_id'),
        crs='epsg:4326',
    )
    result = _cluster_condo_parcels(parcels, min_group_size=2)
    assert result is not None
    component, hub_ids = result
    assert set(component.index) == {'U1', 'U2'}
    assert hub_ids == set()


def test_returns_none_when_no_candidates():
    parcels = gpd.GeoDataFrame(
        {
            'land_value': [50_000.0],
            'improvement_value': [200_000.0],
            'area_ha': [0.5],
            'geometry': [box(0, 0, 0.001, 0.001)],
        },
        index=pd.Index(['P1'], name='parcel_id'),
        crs='epsg:4326',
    )
    assert _cluster_condo_parcels(parcels, min_group_size=2) is None
