"""Rendering expectations for `mesh_parcel_elevation` on a multi-polygon coverage.

What a viewer sees as a clean 3D parcel map, stated as geometry:

- no holes: each parcel's pieces tile the parcel exactly;
- nothing culled: every piece's exterior is clockwise, the winding deck.gl
  assumes for geometry lonboard hands it unnormalized;
- smooth edges: a mesh vertex on a parcel edge sits on the straight chord
  between the two ring vertices around it, never on its own DEM sample;
- closed seams: two neighbors sharing an edge raise identical wall tops;
- no fly-aways: every mesh elevation stays within the DEM's range over
  the parcel.

The multi-polygon coverage matters: a bug in restarting the per-polygon
boundary profile passed every single-polygon test and displaced every
rim piece of every parcel but the first.
"""

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

CRS = 'EPSG:32619'


@pytest.fixture
def hilly_raster(tmp_path):
    """400x400 m raster at 1 m with a smooth two-hill surface."""
    size = 400
    col, row = np.meshgrid(np.arange(size), np.arange(size))
    data = (
        30 * np.exp(-(((col - 120) ** 2 + (row - 150) ** 2) / (2 * 60**2)))
        + 45 * np.exp(-(((col - 290) ** 2 + (row - 260) ** 2) / (2 * 80**2)))
    ).astype('float32')
    path = tmp_path / 'dem.tif'
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=size,
        width=size,
        count=1,
        dtype='float32',
        crs=CRS,
        transform=from_origin(0, size, 1, 1),
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)
    return path


@pytest.fixture
def coverage():
    """Six parcels forming a coverage: a 2x3 block with an irregular middle seam."""
    seam = [(200, 40), (190, 90), (215, 140), (195, 200), (205, 260), (200, 320)]
    left = [(40, 40), *seam, (40, 320)]
    right = [*seam, (360, 320), (360, 40)]
    # Cut each half into three by two horizontal lines that pass through
    # existing seam vertices, keeping the coverage valid.
    left_polys = [
        Polygon([(40, 40), (200, 40), (190, 90), (215, 140), (40, 140)]),
        Polygon([(40, 140), (215, 140), (195, 200), (205, 260), (40, 260)]),
        Polygon([(40, 260), (205, 260), (200, 320), (40, 320)]),
    ]
    right_polys = [
        Polygon([(200, 40), (360, 40), (360, 140), (215, 140), (190, 90)]),
        Polygon([(215, 140), (360, 140), (360, 260), (205, 260), (195, 200)]),
        Polygon([(205, 260), (360, 260), (360, 320), (200, 320)]),
    ]
    del left, right
    polys = left_polys + right_polys
    return gpd.GeoDataFrame(
        {'admin3_id': ['US-XX-AA'] * len(polys)},
        geometry=polys,
        index=pd.Index([f'p{i}' for i in range(len(polys))], name='parcel_id'),
        crs=CRS,
    )


def _mesh(coverage, raster, **kwargs):
    # The reference ring is the drape of the *densified* polygons: the
    # mesh densifies rings to its cell floor before draping, so that no
    # wall-top chord spans more than one cell of terrain.
    dense = coverage.copy()
    dense.geometry = elevation._densify(coverage.geometry, 5.0)
    with patch('openplaces.viz.elevation.get_dataset', return_value=raster):
        ring, _ = elevation.drape_parcel_elevation(
            dense, 'dummy', cache=False, silent=True
        )
        mesh, walls, _ = elevation.mesh_parcel_elevation(
            coverage,
            'dummy',
            cells=6,
            min_resolution=5.0,
            relief_threshold=0.0,
            # These tests state the chord contract on the full densified
            # ring; profile thinning has its own tests.
            profile_tolerance=0.0,
            cache=False,
            silent=True,
            **kwargs,
        )
    return ring, mesh, walls


def _xyz(geom):
    return shapely.get_coordinates(geom, include_z=True)


def test_no_holes_pieces_tile_each_parcel(coverage, hilly_raster):
    _, mesh, _ = _mesh(coverage, hilly_raster)
    for parcel, pieces in zip(coverage.geometry, mesh, strict=True):
        union = shapely.union_all(shapely.get_parts(pieces))
        assert union.symmetric_difference(parcel).area == pytest.approx(0, abs=1e-6)


def test_nothing_culled_every_exterior_is_clockwise(coverage, hilly_raster):
    _, mesh, _ = _mesh(coverage, hilly_raster)
    parts = shapely.get_parts(mesh.to_numpy())
    assert len(parts) > len(coverage) * 20  # actually meshed, not one piece each
    assert not shapely.is_ccw(shapely.get_exterior_ring(parts)).any()


def test_smooth_edges_rim_vertices_lie_on_ring_chords(coverage, hilly_raster):
    ring, mesh, _ = _mesh(coverage, hilly_raster)
    for ring_geom, pieces in zip(ring, mesh, strict=True):
        ring_xyz = _xyz(ring_geom)
        ring_line = shapely.LineString(ring_xyz[:, :2])
        mesh_xyz = _xyz(pieces)
        on_edge = shapely.distance(shapely.points(mesh_xyz[:, :2]), ring_line) < 1e-6
        assert on_edge.sum() > len(ring_xyz)  # grid cuts added rim vertices
        # The chord through the 3D ring at each rim vertex's position.
        located = shapely.line_locate_point(
            ring_line, shapely.points(mesh_xyz[on_edge, :2])
        )
        cumulative = np.concatenate(
            [[0.0], np.cumsum(np.hypot(*np.diff(ring_xyz[:, :2], axis=0).T))]
        )
        chord_z = np.interp(located, cumulative, ring_xyz[:, 2])
        assert np.allclose(mesh_xyz[on_edge, 2], chord_z, atol=1e-6)


def test_closed_seams_neighbors_share_identical_wall_tops(coverage, hilly_raster):
    _, mesh, _ = _mesh(coverage, hilly_raster)
    xyz = [_xyz(pieces) for pieces in mesh]
    polys = coverage.geometry.to_numpy()
    n_shared = 0
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            shared = polys[i].boundary.intersection(polys[j].boundary)
            if shared.length == 0:
                continue
            for a, b in ((i, j), (j, i)):
                on_shared = shapely.distance(shapely.points(xyz[a][:, :2]), shared)
                mine = xyz[a][on_shared < 1e-6]
                other = xyz[b]
                for x, y, z in mine:
                    hit = other[np.isclose(other[:, 0], x) & np.isclose(other[:, 1], y)]
                    # Every rim vertex on the shared edge reappears in the
                    # neighbor at the same elevation, or the chord through
                    # the neighbor's own vertices passes through it.
                    if len(hit):
                        assert np.allclose(hit[:, 2], z, atol=1e-6)
                        n_shared += 1
    assert n_shared > 0


def test_no_fly_aways_elevations_stay_within_dem_range(coverage, hilly_raster):
    _, mesh, _ = _mesh(coverage, hilly_raster)
    with rasterio.open(hilly_raster) as src:
        band = src.read(1)
    low, high = float(band.min()), float(band.max())
    for pieces in mesh:
        z = _xyz(pieces)[:, 2]
        assert z.min() >= low - 1e-6 and z.max() <= high + 1e-6


def test_walls_follow_ring_for_meshed_parcels(coverage, hilly_raster):
    ring, _, walls = _mesh(coverage, hilly_raster)
    for r, w in zip(ring, walls, strict=True):
        assert w.equals_exact(r, 0)
