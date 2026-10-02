"""Tests for admin boundaries draped over the DEM."""

from __future__ import annotations

from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pytest
import shapely
from lonboard import PathLayer, PolygonLayer
from shapely.geometry import box

from openplaces.viz.interactive import get_admin_boundary_layer

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
        # Referenced down by the datum, and never below the ground plane.
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
        # sinking the whole scene under it.
        assert np.allclose(referenced, 0)
