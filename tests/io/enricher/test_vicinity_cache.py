"""The cached vicinity raster is reused only when it covers the spine."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box

import openplaces.io.enricher.vicinity as vicinity_mod
from openplaces.core.schema import AdminId
from openplaces.io.enricher import EnrichState
from openplaces.io.enricher.vicinity import vicinity_coverage

_CRS = 'EPSG:6933'


def _write_raster(path, bounds, size=10):
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=size,
        width=size,
        count=1,
        dtype='uint8',
        crs=_CRS,
        transform=from_bounds(*bounds, size, size),
    ) as dst:
        dst.write(np.zeros((size, size), dtype='uint8'), 1)


def _spine(*boxes):
    return gpd.GeoDataFrame(
        geometry=[box(*b) for b in boxes],
        index=pd.Index([f'p{i}' for i in range(len(boxes))], name='parcel_id'),
        crs=_CRS,
    )


def _state(spine):
    return EnrichState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US-NC-BR'),
        verbose=False,
        timer=None,
        spine=spine,
        evidence=pd.DataFrame(index=spine.index),
    )


@pytest.fixture
def source_raster(monkeypatch, tmp_path):
    path = tmp_path / 'source.tif'
    _write_raster(path, (0, 0, 1000, 1000))
    monkeypatch.setattr(vicinity_mod, 'resolve_raster_path', lambda p: path)
    return path


@pytest.fixture
def computed(monkeypatch):
    """Stub the convolution; record the bounds each call was asked for."""
    calls = []

    def fake_compute(source, bounds, bounds_crs=None, px_radius=60):
        calls.append(tuple(bounds))
        size = 10
        transform = from_bounds(*bounds, size, size)
        return np.zeros((size, size), dtype='uint8'), transform, _CRS

    monkeypatch.setattr(vicinity_mod, 'compute_vicinity_coverage', fake_compute)
    monkeypatch.setattr(vicinity_mod, 'sample_raster', lambda state, *a: state)
    return calls


def test_wider_run_recomputes_a_raster_written_for_a_subset(source_raster, computed):
    towns = _spine((100, 100, 200, 200))
    county = _spine((100, 100, 200, 200), (700, 700, 900, 900))

    vicinity_coverage(_state(towns), 'k', str(source_raster))
    vicinity_coverage(_state(county), 'k', str(source_raster))

    assert computed == [(100, 100, 200, 200), (100, 100, 900, 900)]


def test_narrower_run_reuses_a_county_wide_raster(source_raster, computed):
    county = _spine((100, 100, 200, 200), (700, 700, 900, 900))
    towns = _spine((100, 100, 200, 200))

    vicinity_coverage(_state(county), 'k', str(source_raster))
    vicinity_coverage(_state(towns), 'k', str(source_raster))

    assert computed == [(100, 100, 900, 900)]
