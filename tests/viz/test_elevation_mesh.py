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
def ramp_raster(tmp_path):
    """200x200 m raster at 1 m; value = column index (an east-west ramp)."""
    size = 200
    data = np.tile(np.arange(size, dtype='float32'), (size, 1))
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
        transform=from_origin(0, size, 1, 1),
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)
    return path


def _patch_get_dataset(raster_path):
    return patch('openplaces.viz.elevation.get_dataset', return_value=raster_path)


def _frame(polygons, ids):
    return gpd.GeoDataFrame(
        {'admin3_id': ['US-XX-AA'] * len(polygons)},
        geometry=polygons,
        index=pd.Index(ids, name='parcel_id'),
        crs='EPSG:32619',
    )


def _dense(gdf, max_segment):
    """What the mesh drapes: the rings densified to its cell floor."""
    out = gdf.copy()
    out.geometry = elevation._densify(gdf.geometry, max_segment)
    return out


def test_subdivide_covers_polygon_with_bounded_pieces():
    # A square with a hole: interior cells become triangle pairs, rim
    # and hole-edge cells become clipped pieces, and together they tile
    # the polygon exactly.
    big = Polygon(
        [(0, 0), (100, 0), (100, 100), (0, 100)],
        holes=[[(40, 40), (60, 40), (60, 60), (40, 60)]],
    )
    pieces, owner = elevation.subdivide_polygons([big], cells=4, min_resolution=1.0)

    assert set(owner) == {0}
    assert (shapely.get_type_id(pieces) == shapely.GeometryType.POLYGON).all()
    # deck.gl (unnormalized, as lonboard sends it) culls CCW exteriors.
    assert not shapely.is_ccw(shapely.get_exterior_ring(pieces)).any()
    n_vertices = shapely.get_num_coordinates(pieces)
    # 4x4 grid, 12 cells untouched by the hole: 24 triangles, 4 clipped.
    assert (n_vertices == 4).sum() == 24
    assert (n_vertices > 4).sum() == 4
    assert shapely.union_all(pieces).symmetric_difference(big).area == pytest.approx(0)


def test_subdivide_cell_budget_is_independent_of_size():
    small = Polygon([(0, 0), (40, 0), (40, 40), (0, 40)])
    huge = Polygon([(0, 0), (4000, 0), (4000, 4000), (0, 4000)])
    pieces, owner = elevation.subdivide_polygons(
        [small, huge], cells=4, min_resolution=1.0
    )
    counts = np.bincount(owner)
    # Both are 4x4 grids of interior triangle pairs: the budget scales
    # the cell, not the count.
    assert counts[0] == counts[1] == 32


def test_subdivide_below_min_resolution_is_one_piece():
    tiny = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])
    pieces, owner = elevation.subdivide_polygons([tiny], cells=8, min_resolution=10.0)
    assert len(pieces) == 1
    assert pieces[0].equals(tiny)


def test_mesh_boundary_vertices_match_draped_ring(ramp_raster):
    # A 100 m square on a 1 m/m ramp spans 100 m of relief, so it is
    # meshed. Every ring vertex must reappear in the mesh at the same
    # elevation the draped ring sampled, or the cap would not meet its
    # walls.
    square = Polygon([(10, 10), (110, 10), (110, 110), (10, 110)])
    gdf = _frame([square], ['p1'])

    with _patch_get_dataset(ramp_raster):
        ring, ring_mean = elevation.drape_parcel_elevation(
            _dense(gdf, 1.0), 'dummy', cache=False, silent=True
        )
        mesh, walls, mesh_mean = elevation.mesh_parcel_elevation(
            gdf,
            'dummy',
            cells=4,
            min_resolution=1.0,
            profile_tolerance=0.0,
            cache=False,
            silent=True,
        )

    assert mesh_mean[0] == pytest.approx(ring_mean[0])
    # A meshed row keeps the (densified) draped ring as its wall.
    assert walls.iloc[0].equals_exact(ring.iloc[0], 0)
    assert mesh.iloc[0].geom_type == 'MultiPolygon'
    assert shapely.union_all(mesh.iloc[0]).symmetric_difference(
        square
    ).area == pytest.approx(0, abs=1e-6)

    ring_xyz = shapely.get_coordinates(ring.iloc[0], include_z=True)
    mesh_xyz = shapely.get_coordinates(mesh.iloc[0], include_z=True)
    for x, y, z in ring_xyz:
        at_vertex = mesh_xyz[
            np.isclose(mesh_xyz[:, 0], x) & np.isclose(mesh_xyz[:, 1], y)
        ]
        assert len(at_vertex) > 0
        assert np.allclose(at_vertex[:, 2], z)

    # Interior grid nodes follow the ramp too, not just the rim
    # (bilinear between pixel centers reads x - 0.5).
    interior = mesh_xyz[np.isclose(mesh_xyz[:, 0], 60) & np.isclose(mesh_xyz[:, 1], 60)]
    assert len(interior) > 0
    assert np.allclose(interior[:, 2], 59.5)


@pytest.fixture
def bowl_raster(tmp_path):
    """200x200 m raster at 1 m; value = (column index)**2, so not linear."""
    size = 200
    data = np.tile(np.arange(size, dtype='float32') ** 2, (size, 1))
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
        transform=from_origin(0, size, 1, 1),
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)
    return path


def test_mesh_edge_vertices_interpolate_along_ring(bowl_raster):
    # Where the grid cuts the ring, the new vertex must take the chord
    # between the two ring vertices it lies between, not its own DEM
    # sample: that is what makes a wall top identical from both sides
    # of a shared edge. On a quadratic raster the two differ a lot.
    square = Polygon([(10, 10), (110, 10), (110, 110), (10, 110)])
    gdf = _frame([square], ['p1'])

    # min_resolution=40 densifies each 100 m edge into thirds (vertices
    # at x = 10, 43.3, 76.7, 110) and cuts the bottom edge with grid
    # lines at x = 50 and 90, so every cut lies strictly between two
    # ring vertices and the chord differs clearly from the DEM there.
    with _patch_get_dataset(bowl_raster):
        ring, _ = elevation.drape_parcel_elevation(
            _dense(gdf, 40.0), 'dummy', cache=False, silent=True
        )
        mesh, _, _ = elevation.mesh_parcel_elevation(
            gdf, 'dummy', cells=4, min_resolution=40.0, cache=False, silent=True
        )

    ring_xyz = shapely.get_coordinates(ring.iloc[0], include_z=True)
    bottom = ring_xyz[np.isclose(ring_xyz[:, 1], 10)]
    bottom = bottom[np.argsort(bottom[:, 0])]
    mesh_xyz = shapely.get_coordinates(mesh.iloc[0], include_z=True)
    for x in (50, 90):
        cut = mesh_xyz[np.isclose(mesh_xyz[:, 0], x) & np.isclose(mesh_xyz[:, 1], 10)]
        assert len(cut) > 0
        chord = np.interp(x, bottom[:, 0], bottom[:, 2])
        assert np.allclose(cut[:, 2], chord)
        # ... and is nowhere near the DEM's own value there.
        assert abs(cut[0, 2] - (x - 0.5) ** 2) > 10
    # An interior node still samples the DEM itself.
    inner = mesh_xyz[np.isclose(mesh_xyz[:, 0], 50) & np.isclose(mesh_xyz[:, 1], 50)]
    # Bilinear between pixel centers 49.5 and 50.5, whose values are the
    # column indices squared.
    assert np.allclose(inner[:, 2], (49**2 + 50**2) / 2)


def test_mesh_flat_polygon_is_single_piece_at_boundary_mean(ramp_raster):
    # A square spanning 2 m of relief, well under a 5 m threshold: one
    # flat piece, the polygon's own shape, at the boundary mean.
    square = Polygon([(10, 10), (12, 10), (12, 12), (10, 12)])
    gdf = _frame([square], ['p1'])

    with _patch_get_dataset(ramp_raster):
        mesh, walls, mesh_mean = elevation.mesh_parcel_elevation(
            gdf,
            'dummy',
            cells=4,
            min_resolution=0.1,
            relief_threshold=5.0,
            cache=False,
            silent=True,
        )

    piece = mesh.iloc[0]
    assert piece.geom_type == 'Polygon'
    assert piece.equals(square)
    z = shapely.get_coordinates(piece, include_z=True)[:, 2]
    assert np.allclose(z, mesh_mean[0])
    # ... and the wall ring is flattened to the same elevation.
    wall_z = shapely.get_coordinates(walls.iloc[0], include_z=True)[:, 2]
    assert np.allclose(wall_z, mesh_mean[0])


def test_mesh_geographic_input_returns_in_input_crs(ramp_raster):
    square = Polygon([(10, 10), (110, 10), (110, 110), (10, 110)])
    gdf = _frame([square], ['p1']).to_crs('EPSG:4326')

    with _patch_get_dataset(ramp_raster):
        ring, _ = elevation.drape_parcel_elevation(
            _dense(gdf, 1.0), 'dummy', cache=False, silent=True
        )
        mesh, walls, _ = elevation.mesh_parcel_elevation(
            gdf,
            'dummy',
            cells=4,
            min_resolution=1.0,
            profile_tolerance=0.0,
            cache=False,
            silent=True,
        )

    assert mesh.crs == gdf.crs
    # Ring vertices come back from the metric round trip snapped onto
    # the ring's own xy and z, so wall and cap meet exactly.
    ring_xyz = shapely.get_coordinates(ring.iloc[0], include_z=True)
    mesh_xyz = shapely.get_coordinates(mesh.iloc[0], include_z=True)
    for x, y, z in ring_xyz:
        hit = mesh_xyz[(mesh_xyz[:, 0] == x) & (mesh_xyz[:, 1] == y)]
        assert len(hit) > 0
        assert (hit[:, 2] == z).all()
    back = gpd.GeoSeries([shapely.union_all(mesh.iloc[0])], crs=gdf.crs).to_crs(
        'EPSG:32619'
    )
    # Round-trip reprojection noise only: a few mm2 out of 10,000 m2.
    assert back.iloc[0].symmetric_difference(square).area < 1e-2


def test_mesh_vertices_beyond_dem_take_nearest_sampled_elevation(ramp_raster):
    """A parcel straddling the DEM's edge gets no 0 m vertices.

    The DEM is clipped to the admin unit, so a boundary parcel has
    vertices on nodata routinely. Those used to be set to 0 m, and the
    rim interpolation then dragged the whole edge toward sea level.
    """
    # 200 m wide raster; this parcel runs from x=150 well past x=200.
    parcel = Polygon([(150, 50), (260, 50), (260, 150), (150, 150)])
    gdf = gpd.GeoDataFrame(
        {'admin3_id': ['US-XX-AA']},
        geometry=[parcel],
        index=pd.Index(['p1'], name='parcel_id'),
        crs='EPSG:32619',
    )
    with _patch_get_dataset(ramp_raster):
        mesh, ring, mean_elevation = elevation.mesh_parcel_elevation(
            gdf, 'dummy-recipe', cache=False, relief_threshold=0.0
        )
    for geoms in (mesh.to_numpy(), ring.to_numpy()):
        z = shapely.get_coordinates(geoms, include_z=True)[:, 2]
        # The ramp's valid range over the parcel is 150..199 (value =
        # column index, 149.5 at x=150); nothing may fall below it, and the far edge
        # borrows the last sampled column.
        assert z.min() >= 149.5 - 1e-6
        assert z.max() <= 200 + 1e-6
    assert 149.5 <= mean_elevation[0] <= 200


def test_thin_profile_keeps_corners_and_relief_only():
    # A densified square whose bottom edge has one bump at x=50 (z=5)
    # and is flat elsewhere: only the bump's vertex survives there, the
    # corners always do, and the flat run on the top edge collapses.
    square = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    dense = elevation._densify(gpd.GeoSeries([square], crs='EPSG:32619'), 10.0)
    xyz = shapely.get_coordinates(dense.iloc[0], include_z=False)
    z = np.zeros(len(xyz))
    z[np.isclose(xyz[:, 0], 50) & np.isclose(xyz[:, 1], 0)] = 5.0
    ring3d = gpd.GeoSeries(
        [shapely.Polygon(np.column_stack([xyz, z]))], crs='EPSG:32619'
    )
    thinned = elevation.thin_profile(ring3d, 0.5)
    out = shapely.get_coordinates(thinned.iloc[0], include_z=True)
    bottom = out[np.isclose(out[:, 1], 0)]
    # Douglas-Peucker keeps the spike and its two shoulders (the chord
    # from a corner to the spike's top passes 4 m above them) and drops
    # the rest of the flat run.
    assert set(bottom[:, 0].round(6).tolist()) == {0.0, 40.0, 50.0, 60.0, 100.0}
    assert bottom[np.isclose(bottom[:, 0], 50)][0, 2] == pytest.approx(5.0)
    top = out[np.isclose(out[:, 1], 100)]
    assert set(top[:, 0].round(6).tolist()) == {0.0, 100.0}
    assert thinned.iloc[0].is_valid


def test_thin_profile_is_identical_for_both_sides_of_a_shared_edge():
    # Two squares sharing x=100; a wavy profile on that edge must thin to
    # the same vertices from either polygon's ring direction.
    left = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    right = Polygon([(100, 0), (200, 0), (200, 100), (100, 100)])
    dense = elevation._densify(gpd.GeoSeries([left, right], crs='EPSG:32619'), 5.0)
    rings = []
    for geom in dense:
        xy = shapely.get_coordinates(geom)
        z = np.where(np.isclose(xy[:, 0], 100), np.sin(xy[:, 1] / 7.0) * 3, 0.0)
        rings.append(shapely.Polygon(np.column_stack([xy, z])))
    thinned = elevation.thin_profile(gpd.GeoSeries(rings, crs='EPSG:32619'), 0.4)
    shared = []
    for geom in thinned:
        xyz = shapely.get_coordinates(geom, include_z=True)
        on_edge = xyz[np.isclose(xyz[:, 0], 100)]
        shared.append(set(map(tuple, on_edge.round(9))))
    assert shared[0] == shared[1]
    assert 3 < len(shared[0]) < 21  # thinned, but the waves survive
