"""Tests for `save_parquet`/`read_parquet`'s `combined` (single-file) mode.

`combined=True` writes attributes and geometry into one geoparquet file, with
no `_geo` sidecar and no synthetic `_join_id` column -- for a terminal,
externally-shared deliverable rather than the default split two-file layout
used for internal joinable-table processing.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from openplaces.io import read_parquet, save_parquet


def _gdf():
    return gpd.GeoDataFrame(
        {'value': [1, 2]},
        geometry=[box(0, 0, 1, 1), box(2, 2, 3, 3)],
        crs='EPSG:4326',
        index=pd.Index(['a', 'b'], name='footprint_id'),
    )


def test_combined_writes_single_file_no_sidecar(tmp_path):
    path = tmp_path / 'out.parquet'
    save_parquet(_gdf(), path, combined=True)

    assert path.exists()
    assert not path.with_stem(path.stem + '_geo').exists()
    assert not path.with_stem(path.stem + '_geo_simplified').exists()


def test_combined_round_trip_geom_true(tmp_path):
    path = tmp_path / 'out.parquet'
    original = _gdf()
    save_parquet(original, path, combined=True)

    out = read_parquet(path, geom=True)
    assert isinstance(out, gpd.GeoDataFrame)
    assert out.crs == original.crs
    assert out['value'].tolist() == [1, 2]
    assert '_join_id' not in out.columns
    assert 'geo_id' not in out.columns


def test_combined_round_trip_geom_false_drops_geometry(tmp_path):
    path = tmp_path / 'out.parquet'
    save_parquet(_gdf(), path, combined=True)

    out = read_parquet(path, geom=False)
    assert not isinstance(out, gpd.GeoDataFrame)
    assert 'geometry' not in out.columns
    assert out['value'].tolist() == [1, 2]


def test_combined_simplified_geom_raises(tmp_path):
    path = tmp_path / 'out.parquet'
    save_parquet(_gdf(), path, combined=True)

    with pytest.raises(ValueError, match='combined geoparquet file'):
        read_parquet(path, geom='simplified')


def test_combined_columns_kwarg_keeps_geometry_when_geom_true(tmp_path):
    path = tmp_path / 'out.parquet'
    save_parquet(_gdf(), path, combined=True)

    out = read_parquet(path, geom=True, columns=['value'])
    assert 'geometry' in out.columns
    assert list(out.columns) == ['value', 'geometry']
