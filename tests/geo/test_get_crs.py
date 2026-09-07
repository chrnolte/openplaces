"""`get_crs` on GeoParquet whose column metadata has no usable CRS.

Two cases the GeoParquet spec separates: an explicit null `crs` (what
geopandas writes for a frame with no CRS) means the data is not
georeferenced, while an absent `crs` key means OGC:CRS84.
"""

from __future__ import annotations

import json

import geopandas as gpd
import pyarrow.parquet as pq
import pytest
from pyproj import CRS
from shapely.geometry import box

from openplaces.geo import get_crs


def _write(path, crs):
    gdf = gpd.GeoDataFrame({'geometry': [box(0, 0, 1, 1)]}, crs=crs)
    gdf.to_parquet(path)
    return path


def test_null_crs_warns_and_returns_none(tmp_path):
    path = _write(tmp_path / 'naive.parquet', None)

    with pytest.warns(UserWarning, match='No CRS'):
        assert get_crs(path) is None


def test_absent_crs_key_reads_as_crs84(tmp_path):
    path = _write(tmp_path / 'crs84.parquet', 'OGC:CRS84')
    table = pq.read_table(path)
    meta = dict(table.schema.metadata)
    geo = json.loads(meta[b'geo'])
    for col_meta in geo['columns'].values():
        col_meta.pop('crs', None)
    meta[b'geo'] = json.dumps(geo).encode()
    pq.write_table(table.replace_schema_metadata(meta), path)

    assert get_crs(path) == CRS.from_user_input('OGC:CRS84')
