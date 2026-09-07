"""Geometry handling in `_aggregate_to_file`, the shared merge core.

Chunks are not guaranteed to be uniform: a process-level chunk can be
written without geometry while its siblings carry it. The aggregate must
keep the geometry that exists, whichever chunk it arrives in.
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from openplaces.io import read_parquet, save_parquet
from openplaces.io.aggregate import _aggregate_to_file


def _plain(keys):
    return pd.DataFrame(
        {'geo_id': keys, 'n_dwellings': [1] * len(keys)},
        index=pd.Index(keys, name='parcel_id'),
    )


def _with_geometry(keys):
    return gpd.GeoDataFrame(
        {'geo_id': keys, 'n_dwellings': [2] * len(keys)},
        geometry=[box(i, i, i + 1, i + 1) for i in range(len(keys))],
        crs='EPSG:4326',
        index=pd.Index(keys, name='parcel_id'),
    )


@pytest.mark.parametrize('geometry_first', [False, True])
def test_geometry_survives_a_chunk_written_without_it(tmp_path, geometry_first):
    bare_path = tmp_path / 'bare.parquet'
    geo_path = tmp_path / 'geo.parquet'
    save_parquet(_plain(['a']), bare_path)
    save_parquet(_with_geometry(['b']), geo_path)

    inputs = [('bare', bare_path), ('geo', geo_path)]
    if geometry_first:
        inputs = inputs[::-1]

    out_path = tmp_path / 'merged.parquet'
    _aggregate_to_file(out_path, inputs, keep_original=True)

    assert out_path.with_stem('merged_geo').exists()
    result = read_parquet(out_path, geom=True)
    assert sorted(result.index) == ['a', 'b']
    assert result.loc['b', 'geometry'] is not None
    assert result.geometry.isna().loc['a']
    assert result.crs == 'EPSG:4326'
