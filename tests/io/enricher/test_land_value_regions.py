"""Tests for the land-value region and raster-at-centroid steps."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box

import openplaces.io.enricher.regions as regions_mod
import openplaces.io.enricher.zonal as zonal_mod
from openplaces.core.schema import AdminId
from openplaces.io.enricher import EnrichState
from openplaces.io.enricher.regions import assign_land_value_region
from openplaces.io.enricher.zonal import sample_raster_at_centroids

ALBERS = 'EPSG:5070'


def _state(spine):
    return EnrichState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US-NC-XX'),
        verbose=False,
        timer=None,
        spine=spine,
        evidence=pd.DataFrame(index=spine.index),
    )


def _spine():
    # Two fabricated footprints in two regions, one offshore.
    geometry = [
        box(100, 100, 110, 110),
        box(1100, 100, 1110, 110),
        box(5000, 0, 5010, 10),
    ]
    return gpd.GeoDataFrame(
        {'footprint_id': ['a', 'b', 'c']}, geometry=geometry, crs=ALBERS
    ).set_index('footprint_id')


def test_region_type_and_core_distance(monkeypatch):
    regions = gpd.GeoDataFrame(
        {
            'land_value_region_id': ['r1', 'r2'],
            'land_value_region_type': ['metro', 'rec'],
        },
        geometry=[box(0, 0, 1000, 1000), box(1000, 0, 2000, 1000)],
        crs=ALBERS,
    )
    cores = gpd.GeoDataFrame(
        {
            'land_value_region_id': ['r1', 'r2'],
            'land_value_region_type': ['metro', 'rec'],
        },
        geometry=[Point(105, 3105), Point(1105, 105)],
        crs=ALBERS,
    )
    datasets = {'R': regions, 'C': cores}
    monkeypatch.setattr(
        regions_mod, 'get_dataset', lambda rid, admin_id, geom=False: datasets[rid]
    )
    state = assign_land_value_region(_state(_spine()), 'R', 'C')
    ev = state.evidence
    assert ev['land_value_region_type'].tolist()[:2] == ['metro', 'rec']
    assert ev.loc['a', 'distance_to_land_value_core_km'] == pytest.approx(3.0)
    assert ev.loc['b', 'distance_to_land_value_core_km'] == pytest.approx(0.0)
    assert pd.isna(ev.loc['c', 'land_value_region_id'])


def test_raster_value_under_the_centroid(monkeypatch, tmp_path):
    path = tmp_path / 'values.tif'
    data = np.array([[1.0, 2.0], [0.0, 4.0]], dtype='float32')
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=2,
        width=2,
        count=1,
        dtype='float32',
        crs=ALBERS,
        transform=from_origin(0, 1000, 1000, 500),
        nodata=0,
    ) as dst:
        dst.write(data, 1)
    monkeypatch.setattr(zonal_mod, 'get_dataset', lambda rid, admin_id: path)
    state = sample_raster_at_centroids(_state(_spine()), 'V', 'value')
    values = state.evidence['value']
    # 'a' sits in the bottom-left cell, which holds the nodata value 0.
    assert np.isnan(values['a'])
    assert values['b'] == pytest.approx(4.0)
    assert np.isnan(values['c'])
