import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from openplaces.geo.raster import sample_raster_at_points


@pytest.fixture
def synthetic_raster(tmp_path):
    """4x4 float raster; pixel value = column index, 1 unit/pixel, north-up."""
    data = np.tile(np.arange(4, dtype='float32'), (4, 1))
    transform = from_origin(0, 4, 1, 1)
    path = tmp_path / 'test.tif'
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=4,
        width=4,
        count=1,
        dtype='float32',
        crs='EPSG:4326',
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)
    return path, data


def test_sample_raster_at_points_pixel_centers(synthetic_raster):
    path, data = synthetic_raster
    xs = [0.5, 1.5, 2.5, 3.5]
    ys = [3.5, 3.5, 3.5, 3.5]
    values = sample_raster_at_points(path, xs, ys)
    np.testing.assert_allclose(values, data[0, :])


def test_sample_raster_at_points_out_of_bounds(synthetic_raster):
    path, _ = synthetic_raster
    values = sample_raster_at_points(path, [-5.0, 100.0], [-5.0, 100.0])
    assert np.isnan(values).all()


def test_sample_raster_at_points_nodata(tmp_path):
    data = np.full((2, 2), -9999.0, dtype='float32')
    transform = from_origin(0, 2, 1, 1)
    path = tmp_path / 'nodata.tif'
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=2,
        width=2,
        count=1,
        dtype='float32',
        crs='EPSG:4326',
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)
    values = sample_raster_at_points(path, [0.5], [1.5])
    assert np.isnan(values[0])
