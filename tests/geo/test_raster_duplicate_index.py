"""Per-entity raster statistics stay per row when an id repeats.

Both backends used to key on the entity id: the rasterized one collapsed
two rows sharing a label into one dict entry, and the exactextract one
joined its stats back by label, multiplying the frame. Both now align on
row position, so each copy keeps its own value.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from openplaces.geo.raster import sample_rasterized, zonal_stats_with_exactextract


@pytest.fixture
def column_raster(tmp_path):
    """4x4 raster whose value equals the column index, 1 unit per pixel."""
    data = np.tile(np.arange(4, dtype='float32'), (4, 1))
    path = tmp_path / 'columns.tif'
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=4,
        width=4,
        count=1,
        dtype='float32',
        crs='EPSG:6933',
        transform=from_origin(0, 4, 1, 1),
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)
    return path


def _three_pixels(index):
    return gpd.GeoDataFrame(
        {'geometry': [box(i, 3, i + 1, 4) for i in range(3)]},
        index=pd.Index(index, name='footprint_id'),
        crs='epsg:6933',
    )


def test_sample_rasterized_keeps_each_copy_of_a_repeated_label(column_raster):
    values = sample_rasterized(
        _three_pixels(['a', 'a', 'b']), column_raster, 'columns', 'mean'
    )
    assert values.index.tolist() == ['a', 'a', 'b']
    assert values.tolist() == [0.0, 1.0, 2.0]


def test_sample_rasterized_unchanged_for_unique_labels(column_raster):
    values = sample_rasterized(
        _three_pixels(['a', 'b', 'c']), column_raster, 'columns', 'mean'
    )
    assert values.tolist() == [0.0, 1.0, 2.0]


def test_zonal_stats_with_exactextract_does_not_multiply_a_repeated_label(
    column_raster,
):
    out = zonal_stats_with_exactextract(
        _three_pixels(['a', 'a', 'b']),
        column_raster,
        stats=['mean'],
        reproject=False,
        clean_geometry=False,
    )
    assert len(out) == 3
    assert out.index.tolist() == ['a', 'a', 'b']
    assert out['mean'].tolist() == [0.0, 1.0, 2.0]
    assert out.geometry.notna().all()


def test_zonal_stats_with_exactextract_unique_labels_keep_index_and_columns(
    column_raster,
):
    gdf = _three_pixels(['a', 'b', 'c'])
    gdf['use_group'] = ['residential', 'commercial', 'residential']
    out = zonal_stats_with_exactextract(
        gdf, column_raster, stats=['mean'], reproject=False, clean_geometry=False
    )
    assert out.index.name == 'footprint_id'
    assert out.index.tolist() == ['a', 'b', 'c']
    assert out['use_group'].tolist() == ['residential', 'commercial', 'residential']
    assert out['mean'].tolist() == [0.0, 1.0, 2.0]
    assert isinstance(out, gpd.GeoDataFrame)
    assert out.crs == gdf.crs
