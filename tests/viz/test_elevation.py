from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import shapely
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from openplaces.viz import elevation


@pytest.fixture
def gradient_raster(tmp_path):
    """10x10 raster; value = column index (a simple east-west elevation ramp)."""
    size = 10
    data = np.tile(np.arange(size, dtype='float32'), (size, 1))
    transform = from_origin(0, size, 1, 1)
    path = tmp_path / 'dem.tif'
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=size,
        width=size,
        count=1,
        dtype='float32',
        crs='EPSG:32619',
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)
    return path


def _patch_get_dataset(raster_path):
    return patch('openplaces.viz.elevation.get_dataset', return_value=raster_path)


def test_drape_parcel_elevation_shared_corner_matches(gradient_raster):
    # Two adjacent unit squares sharing the edge x=1 -- the shared corner
    # vertices should get identical, correctly-sampled elevation in both.
    poly_a = Polygon([(0, 1), (1, 1), (1, 2), (0, 2)])
    poly_b = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])
    gdf = gpd.GeoDataFrame(
        {'admin3_id': ['US-XX-AA', 'US-XX-AA']},
        geometry=[poly_a, poly_b],
        index=pd.Index(['p1', 'p2'], name='parcel_id'),
        crs='EPSG:32619',
    )

    with _patch_get_dataset(gradient_raster):
        geometry, mean_elevation = elevation.drape_parcel_elevation(
            gdf, 'dummy_recipe', cache=False, silent=True
        )

    def z_at(geom, x, y):
        coords = np.asarray(shapely.get_coordinates(geom, include_z=True))
        match = coords[np.isclose(coords[:, 0], x) & np.isclose(coords[:, 1], y)]
        return match[0, 2]

    assert z_at(geometry.iloc[0], 1, 1) == pytest.approx(z_at(geometry.iloc[1], 1, 1))
    assert z_at(geometry.iloc[0], 1, 2) == pytest.approx(z_at(geometry.iloc[1], 1, 2))
    # gradient raster: value at x=1 is column index 1.
    assert z_at(geometry.iloc[0], 1, 1) == pytest.approx(1.0)

    coords_a = np.asarray(shapely.get_coordinates(geometry.iloc[0], include_z=True))
    assert mean_elevation[0] == pytest.approx(coords_a[:, 2].mean())


def test_drape_parcel_elevation_cache_reuse(gradient_raster, mock_data_root):
    poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly2 = Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
    gdf1 = gpd.GeoDataFrame(
        {'admin3_id': ['US-XX-AA']},
        geometry=[poly1],
        index=pd.Index(['p1'], name='parcel_id'),
        crs='EPSG:32619',
    )
    with _patch_get_dataset(gradient_raster):
        elevation.drape_parcel_elevation(gdf1, 'dummy_recipe', silent=True)

    # Corrupt the on-disk cache for p1 with an obviously-wrong elevation, to
    # prove a second call reuses the cache rather than resampling.
    cache_file = elevation._cache_path('US-XX-AA', 'parcel_elevation')
    cached = gpd.read_parquet(cache_file)
    cached['mean_elevation'] = 12345.0
    cached['geometry'] = [shapely.force_3d(poly1, z=12345.0)]
    gpd.GeoDataFrame(cached, crs='EPSG:32619').to_parquet(cache_file)

    gdf2 = gpd.GeoDataFrame(
        {'admin3_id': ['US-XX-AA', 'US-XX-AA']},
        geometry=[poly1, poly2],
        index=pd.Index(['p1', 'p2'], name='parcel_id'),
        crs='EPSG:32619',
    )
    with _patch_get_dataset(gradient_raster):
        _, mean_elevation = elevation.drape_parcel_elevation(
            gdf2, 'dummy_recipe', silent=True
        )

    assert mean_elevation[0] == pytest.approx(12345.0)  # reused from (corrupted) cache
    assert mean_elevation[1] != pytest.approx(12345.0)  # freshly sampled from raster


def test_drape_parcel_elevation_missing_dem_auto_ingests(gradient_raster, tmp_path):
    # Away from (0, 0): a polygon touching the raster's own bottom/left
    # edge would floor to one-past-the-last-valid-pixel and sample NaN --
    # a real (if rare) edge-of-raster case, but not what this test is
    # about, so keep the geometry safely inside the raster's interior.
    poly = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
    gdf = gpd.GeoDataFrame(
        {'admin3_id': ['US-XX-AA']},
        geometry=[poly],
        index=pd.Index(['p1'], name='parcel_id'),
        crs='EPSG:32619',
    )
    not_yet_ingested_path = tmp_path / 'not_ingested.tif'

    def _fake_ingest(recipe, admin_ids, verbose=False):
        # Simulate a real ingest: the DEM now exists where get_dataset says.
        not_yet_ingested_path.write_bytes(gradient_raster.read_bytes())

    with (
        _patch_get_dataset(not_yet_ingested_path),
        patch('openplaces.io.ingester.ingest', side_effect=_fake_ingest) as mock_ingest,
    ):
        geometry, mean_elevation = elevation.drape_parcel_elevation(
            gdf, 'US_land-elevation-usgs-3dep', cache=False, silent=True
        )

    mock_ingest.assert_called_once_with(
        'US_land-elevation-usgs-3dep', admin_ids='US-XX-AA', verbose=False
    )
    # Ring [(2,2),(3,2),(3,3),(2,3),(2,2)] samples columns [2,3,3,2,2] on
    # the gradient raster (value = column index) -> mean 2.4.
    assert mean_elevation[0] == pytest.approx(2.4)


def test_drape_parcel_elevation_missing_dem_no_coverage_raises(tmp_path):
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    gdf = gpd.GeoDataFrame(
        {'admin3_id': ['US-XX-AA']},
        geometry=[poly],
        index=pd.Index(['p1'], name='parcel_id'),
        crs='EPSG:32619',
    )
    never_ingested_path = tmp_path / 'no_coverage.tif'

    with (
        _patch_get_dataset(never_ingested_path),
        patch('openplaces.io.ingester.ingest'),  # no-op: never creates the file
    ):
        with pytest.raises(FileNotFoundError, match='no 3DEP coverage'):
            elevation.drape_parcel_elevation(
                gdf, 'dummy_recipe', cache=False, silent=True
            )


def test_get_building_elevation_zonal_mean(gradient_raster):
    # A footprint spanning columns 2-3 (values 2 and 3) -> zonal mean ~2.5.
    footprint = Polygon([(2, 2), (4, 2), (4, 4), (2, 4)])
    gdf = gpd.GeoDataFrame(
        {'admin3_id': ['US-XX-AA']},
        geometry=[footprint],
        index=pd.Index(['f1'], name='footprint_id'),
        crs='EPSG:32619',
    )
    with _patch_get_dataset(gradient_raster):
        values = elevation.get_building_elevation(
            gdf, 'dummy_recipe', cache=False, silent=True
        )
    assert values[0] == pytest.approx(2.5, abs=0.5)


def test_add_z_offset_on_2d_geometry():
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    result = elevation.add_z_offset(np.array([poly], dtype=object), np.array([10.0]))
    coords = np.asarray(shapely.get_coordinates(result[0], include_z=True))
    assert np.allclose(coords[:, 2], 10.0)


def test_add_z_offset_adds_on_top_of_existing_z():
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    geom3d = shapely.force_3d(poly, z=0.0)
    xyz = np.asarray(shapely.get_coordinates(geom3d, include_z=True))
    xyz[:, 2] = [1.0, 2.0, 3.0, 4.0, 1.0]
    draped = shapely.set_coordinates(geom3d, xyz)

    result = elevation.add_z_offset(np.array([draped], dtype=object), np.array([10.0]))
    coords = np.asarray(shapely.get_coordinates(result[0], include_z=True))
    np.testing.assert_allclose(coords[:, 2], [11.0, 12.0, 13.0, 14.0, 11.0])
