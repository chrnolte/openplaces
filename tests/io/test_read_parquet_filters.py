"""Read-path guards for `read_parquet` across both parquet layouts.

The two layouts are the split one (attributes plus a `_geo` sidecar,
written by `save_parquet`) and the combined one (`combined=True`, one
geoparquet file). Each fix below is a case where a caller's arguments
were silently ignored, leaked an internal column, or crashed.
"""

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import pytest
from shapely.geometry import box

from openplaces.io import read_parquet, save_parquet


def _gdf():
    return gpd.GeoDataFrame(
        {
            'geo_id': ['a', 'b', 'c'],
            'state': ['MA', 'MA', 'RI'],
            'value': [1, 2, 3],
        },
        geometry=[box(0, 0, 1, 1), box(10, 10, 11, 11), box(20, 20, 21, 21)],
        crs='EPSG:4326',
    )


def _write_split(tmp_path):
    # geo_id stays a column, so the read has to drop it again when the
    # caller's column list did not ask for it.
    path = tmp_path / 'split.parquet'
    save_parquet(_gdf(), path)
    return path


def _write_combined(tmp_path):
    path = tmp_path / 'combined.parquet'
    save_parquet(_gdf(), path, combined=True)
    return path


def test_combined_bbox_and_filters_are_both_applied(tmp_path):
    # A bbox used to replace `filters` outright, so a caller narrowing
    # to one state and also passing a bbox got the other state back.
    path = _write_combined(tmp_path)
    out = read_parquet(
        path,
        geom=True,
        filters=[('state', '==', 'MA')],
        bbox=(-1, -1, 100, 100),
    )
    assert set(out['state']) == {'MA'}
    assert len(out) == 2


def test_combined_bbox_alone_still_filters_spatially(tmp_path):
    path = _write_combined(tmp_path)
    out = read_parquet(path, geom=True, bbox=(-1, -1, 5, 5))
    assert out['value'].tolist() == [1]


def test_combined_without_geometry_hides_the_covering_bbox(tmp_path):
    # write_covering_bbox=True adds an internal struct column that no
    # attribute registry entry knows; it must not reach a caller.
    path = _write_combined(tmp_path)
    assert 'bbox' in pq.ParquetFile(path).schema_arrow.names

    out = read_parquet(path, geom=False)
    assert 'bbox' not in out.columns
    assert 'geometry' not in out.columns
    assert out['value'].tolist() == [1, 2, 3]


def test_split_geometry_read_with_zero_matching_rows(tmp_path):
    # An empty predicate result built an empty `in` list, which pyarrow
    # rejected against the sidecar's string column.
    path = _write_split(tmp_path)
    out = read_parquet(path, geom=True, filters=[('state', '==', 'ZZ')])
    assert isinstance(out, gpd.GeoDataFrame)
    assert len(out) == 0
    assert 'geometry' in out.columns
    assert out.crs is not None


def test_split_geometry_read_with_columns_omitting_the_join_id(tmp_path):
    # The join id has to be read even when the caller did not ask for
    # it, and dropped again afterwards.
    path = _write_split(tmp_path)
    out = read_parquet(path, geom=True, columns=['value'])
    assert isinstance(out, gpd.GeoDataFrame)
    assert sorted(out.columns) == ['geometry', 'value']
    assert out['value'].tolist() == [1, 2, 3]


def test_split_geometry_read_keeps_a_requested_join_id(tmp_path):
    path = _write_split(tmp_path)
    out = read_parquet(path, geom=True, columns=['geo_id', 'value'])
    assert sorted(out.columns) == ['geo_id', 'geometry', 'value']


def test_split_geometry_read_when_the_join_id_is_the_index(tmp_path):
    # save_parquet also accepts geo_id as the index, in which case the
    # join runs against the index rather than a column.
    path = tmp_path / 'indexed.parquet'
    save_parquet(_gdf().set_index('geo_id'), path)
    out = read_parquet(path, geom=True, columns=['value'])
    assert sorted(out.columns) == ['geometry', 'value']
    assert out.index.name == 'geo_id'
    assert out.geometry.notna().all()


def test_split_read_without_geometry_is_unchanged(tmp_path):
    path = _write_split(tmp_path)
    out = read_parquet(path, columns=['value'])
    assert isinstance(out, pd.DataFrame)
    assert out.columns.tolist() == ['value']


def test_split_geometry_read_without_a_join_column_raises(tmp_path):
    path = tmp_path / 'plain.parquet'
    pd.DataFrame({'value': [1]}).to_parquet(path, index=False)
    with pytest.raises(ValueError, match='Could not identify column'):
        read_parquet(path, geom=True)
