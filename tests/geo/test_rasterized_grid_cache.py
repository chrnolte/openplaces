"""The rasterized back end burns a grid once and reuses it, unchanged.

`sample_rasterized` keeps the polygons burned onto a raster window in a
cache the caller passes, so every further raster on the same grid (the
next year of an annual layer) skips the burn, and `mean`, `sum` and
`count` are taken by one bincount where that is provably exact. Every
value must equal what the original burn-and-groupby returned; the
reference below is that implementation, kept verbatim.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.windows import Window
from shapely.geometry import Polygon, box

from openplaces.geo.raster import _RASTERIZED_STAT_FN, sample_rasterized

STATS = ['mean', 'sum', 'count', 'max', 'min', 'std']


def _reference(parcels_r, raster_path, stat):
    """The burn-and-groupby as it stood before the cache."""
    with rasterio.open(raster_path) as src:
        win = src.window(*parcels_r.total_bounds)
        win = win.intersection(Window(0, 0, src.width, src.height))
        win_transform = src.window_transform(win)
        raster_data = src.read(1, window=win)
        nodata = src.nodata
    shapes = zip(parcels_r.geometry, range(1, len(parcels_r) + 1), strict=True)
    id_grid = rasterize(
        shapes,
        out_shape=raster_data.shape,
        transform=win_transform,
        fill=0,
        dtype='int32',
    )
    raster_f = raster_data.astype('float32')
    if nodata is not None:
        raster_f[raster_data == nodata] = np.nan
    flat_ids = id_grid.ravel()
    flat_vals = raster_f.ravel()
    mask = flat_ids > 0
    df = pd.DataFrame({'idx': flat_ids[mask], 'val': flat_vals[mask]})
    grouped = df.groupby('idx')['val'].agg(_RASTERIZED_STAT_FN[stat])
    return grouped.reindex(np.arange(1, len(parcels_r) + 1)).to_numpy(dtype='float64')


def _write(path, data, res=1.0, nodata=0):
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs='EPSG:6933',
        transform=from_origin(0, 200, res, res),
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)
    return path


@pytest.fixture
def parcels():
    """Fabricated polygons: off the pixel grid, one large, one too small
    to cover a pixel center, one over nodata only."""
    geoms = [
        box(10.3, 10.7, 40.2, 35.9),
        box(5, 120, 190, 195),
        Polygon([(50, 50), (90, 60), (70, 100)]),
        box(100.1, 100.1, 100.4, 100.4),
        box(150, 20, 170, 40),
    ]
    return gpd.GeoDataFrame(
        {'geometry': geoms},
        index=pd.Index(['p1', 'p2', 'p3', 'p4', 'p5'], name='parcel_id'),
        crs='EPSG:6933',
    )


@pytest.fixture
def annual_rasters(tmp_path):
    """Five 'years' of a uint8 0/1 layer (0 is nodata, as in the
    Colombian derived layers), with year-specific values and 1 values
    and the p5 block always nodata."""
    rng = np.random.default_rng(7)
    paths = []
    for year in range(2001, 2006):
        data = rng.integers(0, 3, size=(200, 200)).astype('uint8')
        data[160:180, 150:170] = 0
        paths.append(_write(tmp_path / f'forest_{year}.tif', data))
    return paths


def _assert_same(got, expected):
    np.testing.assert_array_equal(got.to_numpy(), expected)


@pytest.mark.parametrize('stat', STATS)
def test_cached_values_equal_the_uncached_reference(parcels, annual_rasters, stat):
    cache = {}
    for path in annual_rasters:
        got = sample_rasterized(parcels, path, 'forest', stat, grid_cache=cache)
        assert got.index.equals(parcels.index)
        _assert_same(got, _reference(parcels, path, stat))
    # One burn served all five years.
    assert sum(k[0] == 'burned_grid' for k in cache) == 1


def test_uncached_call_is_unchanged(parcels, annual_rasters):
    for stat in STATS:
        got = sample_rasterized(parcels, annual_rasters[0], 'forest', stat)
        _assert_same(got, _reference(parcels, annual_rasters[0], stat))


def test_burn_is_reused_not_redone(parcels, annual_rasters, monkeypatch):
    import openplaces.geo.raster as raster_mod

    calls = []
    real = raster_mod.rasterize

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(raster_mod, 'rasterize', counting)
    cache = {}
    for path in annual_rasters:
        sample_rasterized(parcels, path, 'forest', 'mean', grid_cache=cache)
    assert len(calls) == 1


def test_raster_on_another_grid_misses_and_stays_correct(
    tmp_path, parcels, annual_rasters
):
    coarse = np.random.default_rng(3).integers(0, 3, size=(100, 100))
    coarse_path = _write(tmp_path / 'coarse.tif', coarse.astype('uint8'), res=2.0)
    cache = {}
    sample_rasterized(parcels, annual_rasters[0], 'forest', 'mean', grid_cache=cache)
    got = sample_rasterized(parcels, coarse_path, 'forest', 'mean', grid_cache=cache)
    _assert_same(got, _reference(parcels, coarse_path, 'mean'))
    assert sum(k[0] == 'burned_grid' for k in cache) == 2


def test_a_new_frame_of_equal_polygons_misses(parcels, annual_rasters):
    cache = {}
    sample_rasterized(parcels, annual_rasters[0], 'forest', 'mean', grid_cache=cache)
    moved = parcels.copy()
    moved['geometry'] = moved.translate(30, 0)
    got = sample_rasterized(
        moved, annual_rasters[1], 'forest', 'mean', grid_cache=cache
    )
    _assert_same(got, _reference(moved, annual_rasters[1], 'mean'))
    # The old frame is held by its entry, so the new one cannot reuse
    # its id; each frame has its own burn.
    assert sum(k[0] == 'burned_grid' for k in cache) == 2


@pytest.mark.parametrize('stat', ['mean', 'sum'])
def test_values_too_large_for_float32_sums_take_the_groupby(tmp_path, parcels, stat):
    """Above 2**24 a float32 sum rounds, so bincount is not used."""
    data = np.random.default_rng(5).integers(1, 60000, size=(200, 200))
    path = _write(tmp_path / 'big.tif', data.astype('uint16'), nodata=None)
    cache = {}
    got = sample_rasterized(parcels, path, 'big', stat, grid_cache=cache)
    _assert_same(got, _reference(parcels, path, stat))


@pytest.mark.parametrize('stat', STATS)
def test_float_rasters_take_the_groupby(tmp_path, parcels, stat):
    data = np.random.default_rng(9).random((200, 200)).astype('float32') * 100
    data[0:50, 0:50] = np.nan
    path = _write(tmp_path / 'float.tif', data, nodata=float('nan'))
    cache = {}
    for _ in range(2):
        got = sample_rasterized(parcels, path, 'f', stat, grid_cache=cache)
        _assert_same(got, _reference(parcels, path, stat))
