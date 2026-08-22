"""Tests for the DEM-draped basemap mesh and draped admin boundaries."""

from __future__ import annotations

from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pytest
import shapely
from lonboard import PathLayer, PolygonLayer, SolidPolygonLayer
from shapely.geometry import box

from openplaces.viz.interactive import (
    _fill_missing_elevation,
    get_admin_boundary_layer,
    get_terrain_basemap_layer,
)

DEM = 'US_land-elevation-usgs-3dep'


@pytest.fixture
def town():
    """One square admin unit near 42N, indexed the way `get_admin` returns."""
    gdf = gpd.GeoDataFrame(
        {'name': ['Testville']},
        geometry=[box(-71.70, 42.40, -71.68, 42.42)],
        index=['US-MA-LAN'],
        crs='EPSG:4326',
    )
    gdf.index.name = 'admin4_id'
    return gdf


def _z(frame):
    return shapely.get_coordinates(frame.geometry.to_numpy(), include_z=True)[:, 2]


class TestFillMissingElevation:
    def test_fills_holes_from_nearest_neighbor(self):
        grid = np.array([[10.0, 20.0, 30.0], [40.0, np.nan, 60.0], [70.0, 80.0, 90.0]])
        filled = _fill_missing_elevation(grid)
        # A hole takes a real neighbor's value, never 0 -- a sea-level spike
        # punched through a hillside is the artifact this exists to avoid.
        assert filled[1, 1] in {20.0, 40.0, 60.0, 80.0}
        assert not np.isnan(filled).any()

    def test_untouched_when_complete(self):
        grid = np.arange(9.0).reshape(3, 3)
        assert np.array_equal(_fill_missing_elevation(grid), grid)

    def test_all_missing_raises(self):
        with pytest.raises(ValueError, match='no data anywhere'):
            _fill_missing_elevation(np.full((3, 3), np.nan))


def _ramp_drape(gdf, _recipe, **_kwargs):
    """Stand-in for `drape_parcel_elevation`: Z rises going north."""
    geometry = shapely.force_3d(gdf.geometry.to_numpy(), z=0.0)
    xyz = np.asarray(shapely.get_coordinates(geometry, include_z=True))
    xyz[:, 2] = (xyz[:, 1] - 42.40) * 1e5
    draped = shapely.set_coordinates(geometry, xyz)
    return gpd.GeoSeries(draped, crs=gdf.crs), np.zeros(len(gdf))


class TestDrapedAdminBoundary:
    def _capture(self, town, **kwargs):
        captured = []
        real_path, real_poly = PathLayer.from_geopandas, PolygonLayer.from_geopandas

        def spy_path(frame, **kw):
            captured.append(frame.copy())
            return real_path(frame, **kw)

        def spy_poly(frame, **kw):
            captured.append(frame.copy())
            return real_poly(frame, **kw)

        with (
            patch(
                'openplaces.viz.elevation.drape_parcel_elevation',
                side_effect=_ramp_drape,
            ),
            patch.object(PathLayer, 'from_geopandas', staticmethod(spy_path)),
            patch.object(PolygonLayer, 'from_geopandas', staticmethod(spy_poly)),
        ):
            get_admin_boundary_layer(gdf=town, **kwargs)
        return captured[-1]

    def test_without_recipe_stays_flat(self, town):
        frame = self._capture(town, elevation=5, mode='floating_line')
        assert set(np.unique(_z(frame))) == {5.0}

    def test_drapes_each_vertex(self, town):
        frame = self._capture(
            town, elevation=0, mode='floating_line', elevation_recipe=DEM
        )
        # On a ramp the boundary can no longer sit at a single height.
        assert np.ptp(_z(frame)) > 0

    def test_fence_base_is_draped(self, town):
        frame = self._capture(town, elevation=10, mode='fence', elevation_recipe=DEM)
        # The wall footprint carries the terrain; `get_elevation` raises a
        # constant-height wall from it, so the fence follows the ground
        # instead of being sliced by it.
        assert np.ptp(_z(frame)) > 0

    def test_exaggeration_scales_terrain(self, town):
        one = _z(
            self._capture(
                town,
                elevation=0,
                mode='fence',
                elevation_recipe=DEM,
                terrain_exaggeration=1,
            )
        )
        three = _z(
            self._capture(
                town,
                elevation=0,
                mode='fence',
                elevation_recipe=DEM,
                terrain_exaggeration=3,
            )
        )
        assert np.allclose(three, one * 3)

    def test_elevation_is_clearance_above_terrain(self, town):
        ground = _z(
            self._capture(town, elevation=0, mode='floating_line', elevation_recipe=DEM)
        )
        lifted = _z(
            self._capture(
                town, elevation=50, mode='floating_line', elevation_recipe=DEM
            )
        )
        # Added to the terrain, not substituted for it -- force_3d would
        # silently leave already-3D geometry alone and drop the offset.
        assert np.allclose(lifted - ground, 50)

    def test_non_admin_index_raises(self, town):
        renamed = town.rename(index={'US-MA-LAN': 'not-an-admin-id'})
        with pytest.raises(ValueError, match='not indexed by admin id'):
            self._capture(renamed, elevation=0, elevation_recipe=DEM)

    def test_datum_references_and_clamps(self, town):
        ground = _z(
            self._capture(town, elevation=0, mode='fence', elevation_recipe=DEM)
        )
        referenced = _z(
            self._capture(
                town,
                elevation=0,
                mode='fence',
                elevation_recipe=DEM,
                elevation_datum=ground.max() / 2,
            )
        )
        # Referenced down by the datum, and never below the basemap plane.
        assert referenced.max() < ground.max()
        assert referenced.min() >= 0

    def test_datum_above_the_extent_clamps_flat(self, town):
        referenced = _z(
            self._capture(
                town,
                elevation=0,
                mode='fence',
                elevation_recipe=DEM,
                elevation_datum=1e6,
            )
        )
        # A datum above everything flattens onto the plane rather than
        # sinking the whole scene under the basemap.
        assert np.allclose(referenced, 0)


class TestTerrainBasemapLayer:
    """Mesh geometry and color assembly, with the DEM and tile fetch faked."""

    def _build(self, town, **kwargs):
        captured = []
        real = SolidPolygonLayer.from_geopandas

        def spy(frame, **kw):
            captured.append((frame.copy(), kw))
            return real(frame, **kw)

        # A dark image with one bright east-west band across its middle.
        image = np.full((64, 64, 4), 40, dtype=np.uint8)
        image[28:36, :, :3] = 220
        west, south, east, north = town.to_crs(3857).total_bounds
        extent = (west, east, south, north)

        def corners(grid_x, grid_y, *_args, **_kwargs):
            del grid_x
            return (grid_y - grid_y.min()) / 10.0

        with (
            patch(
                'openplaces.viz.interactive._sample_corner_elevation',
                side_effect=corners,
            ),
            patch('contextily.bounds2img', return_value=(image, extent)),
            patch.object(SolidPolygonLayer, 'from_geopandas', staticmethod(spy)),
        ):
            get_terrain_basemap_layer(gdf=town, elevation_recipe=DEM, **kwargs)
        return captured[-1]

    def test_refuses_an_oversized_mesh(self, town):
        with pytest.raises(ValueError, match='over the'):
            self._build(town, resolution=1.0, max_cells=1000)

    def test_oversize_error_names_a_workable_resolution(self, town):
        with pytest.raises(ValueError, match=r'resolution >= \d+'):
            self._build(town, resolution=1.0, max_cells=1000)

    def test_builds_one_quad_per_cell(self, town):
        mesh, kwargs = self._build(town, resolution=100.0, clip=False)
        assert len(mesh) == len(kwargs['get_fill_color'])
        assert (mesh.geom_type == 'Polygon').all()
        # Four corners plus the closing vertex.
        assert {len(g.exterior.coords) for g in mesh.geometry} == {5}

    def test_clipping_drops_cells_outside_the_extent(self, town):
        narrow = town.copy()
        narrow.geometry = [box(-71.70, 42.40, -71.695, 42.42)]
        clipped, _ = self._build(narrow, resolution=100.0, clip=True)
        unclipped, _ = self._build(narrow, resolution=100.0, clip=False)
        assert len(clipped) < len(unclipped)

    def test_exaggeration_scales_the_mesh(self, town):
        one, _ = self._build(town, resolution=100.0, clip=False, terrain_exaggeration=1)
        three, _ = self._build(
            town, resolution=100.0, clip=False, terrain_exaggeration=3
        )
        assert np.allclose(_z(three), _z(one) * 3)

    def test_colors_land_on_the_right_cells(self, town):
        """The bright band comes back as a bright band, not mirrored."""
        mesh, kwargs = self._build(town, resolution=100.0, clip=False)
        brightness = kwargs['get_fill_color'][:, :3].astype(float).mean(axis=1)
        # Bounds rather than centroids: no reprojection warning, and a quad
        # is small enough that its bbox center is its center.
        bounds = mesh.bounds
        latitude = ((bounds['miny'] + bounds['maxy']) / 2).to_numpy()

        bright = brightness > 130
        assert bright.any(), 'the bright band vanished from the mesh'
        # Image row 0 is the *north* edge, so rows 28-36 of 64 sit just
        # north of center. A flipped mapping would land south of it.
        span = latitude.max() - latitude.min()
        expected = latitude.max() - span * (32 / 64)
        assert abs(latitude[bright].mean() - expected) < span * 0.12

    def test_averaging_keeps_a_subcell_feature_visible(self, town):
        """A feature thinner than a cell still tints every cell it crosses."""
        _, kwargs = self._build(town, resolution=300.0, clip=False)
        brightness = kwargs['get_fill_color'][:, :3].astype(float).mean(axis=1)
        # Point sampling would return only 40 or only 220; averaging leaves
        # intermediate values, and that is what keeps roads continuous
        # instead of breaking them into dots.
        assert ((brightness > 45) & (brightness < 215)).any()

    def test_datum_references_and_clamps_the_mesh(self, town):
        base, _ = self._build(town, resolution=100.0, clip=False)
        peak = _z(base).max()
        referenced, _ = self._build(
            town, resolution=100.0, clip=False, elevation_datum=peak / 2
        )
        assert _z(referenced).max() < peak
        # The mesh is the ground plane itself; it must never go under the
        # basemap it replaces.
        assert _z(referenced).min() >= 0

    def test_datum_above_the_extent_flattens_the_mesh(self, town):
        referenced, _ = self._build(
            town, resolution=100.0, clip=False, elevation_datum=1e6
        )
        assert np.allclose(_z(referenced), 0)

    def test_rejects_unknown_provider(self, town):
        with pytest.raises(ValueError, match='Unknown basemap provider'):
            self._build(town, provider='not-a-provider', resolution=100.0)
