"""`fetch_raster_from_tiles` clips a product shipped as several local
tiles (an Earth Engine export grid) to one boundary, keeping every band.

Fixture: two fabricated adjoining 4 x 4 tiles with three bands each; the
boundary straddles their shared edge, so the clip has to mosaic both.
"""

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from openplaces.io.ingester.raster_ingester import fetch_raster_from_tiles


def _tile(path, x0, fill):
    data = np.stack([np.full((4, 4), fill + b, dtype='uint8') for b in range(3)])
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        width=4,
        height=4,
        count=3,
        dtype='uint8',
        crs='EPSG:4326',
        transform=from_origin(x0, 4.0, 1.0, 1.0),
    ) as dst:
        dst.write(data)
        for b in range(3):
            dst.set_band_description(b + 1, f'LC_{1997 + b}-07-01')
    return path


@pytest.fixture
def two_tiles(tmp_path):
    return [_tile(tmp_path / 'a.tif', 0.0, 10), _tile(tmp_path / 'b.tif', 4.0, 20)]


def test_clip_mosaics_both_tiles_and_keeps_bands(two_tiles, tmp_path):
    boundary = gpd.GeoDataFrame(geometry=[box(2.0, 1.0, 6.0, 3.0)], crs='EPSG:4326')
    out = fetch_raster_from_tiles(boundary, tmp_path / 'clip.tif', two_tiles, nodata=0)
    with rasterio.open(out) as src:
        assert src.count == 3
        assert src.descriptions[0] == 'LC_1997-07-01'
        band2 = src.read(2)
    # Band 2 holds fill + 1: 11 from tile a, 21 from tile b.
    assert set(np.unique(band2)) - {0} == {11, 21}


def test_no_overlapping_tile_raises(two_tiles, tmp_path):
    far = gpd.GeoDataFrame(geometry=[box(50.0, 50.0, 51.0, 51.0)], crs='EPSG:4326')
    with pytest.raises(ValueError, match='No tile intersects'):
        fetch_raster_from_tiles(far, tmp_path / 'clip.tif', two_tiles)
