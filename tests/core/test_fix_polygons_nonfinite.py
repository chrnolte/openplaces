"""Tests for `fix_polygons` on geometries that cannot be repaired.

`shapely.make_valid` raises rather than returning a usable shape when a
geometry carries a non-finite coordinate, so one such row used to abort a
whole ingest. Measured on the Texas Shovels permit export: 9 rows in the
first 400,000 carry `POINT (lat NaN)` -- a source with one of a point's two
coordinates but not the other -- while the North Carolina export has none,
which is why the failure looked source-specific rather than generic.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from openplaces.geo.polygon import fix_polygons, has_geometry


def _frame(geoms):
    return gpd.GeoDataFrame({'v': range(len(geoms))}, geometry=geoms, crs='EPSG:4326')


def test_point_with_a_nan_coordinate_is_blanked_not_raised():
    gdf = _frame([Point(1, 1), Point(2, float('nan'))])
    out = fix_polygons(gdf)
    assert out.geometry.iloc[0].equals(Point(1, 1))
    assert out.geometry.iloc[1] is None


def test_one_broken_row_does_not_stop_the_rest_being_repaired():
    # The bowtie is genuinely repairable; the NaN point is not. Both are
    # invalid, so they travel through the same branch -- which is exactly
    # how one unrepairable row used to take the repairable ones with it.
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
    gdf = _frame([bowtie, Point(3, float('nan')), Point(5, 5)])
    out = fix_polygons(gdf)

    assert out.geometry.iloc[0].is_valid
    assert out.geometry.iloc[1] is None
    assert out.geometry.iloc[2].equals(Point(5, 5))


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf')])
def test_every_non_finite_coordinate_is_treated_the_same(bad):
    out = fix_polygons(_frame([Point(1, bad)]))
    assert out.geometry.iloc[0] is None


def test_blanked_geometry_is_filtered_by_has_geometry():
    # Blanking is only lossless because the missing-geometry filter picks
    # it up; if that stopped being true, blanking would silently ship a
    # row claiming a location it does not have.
    out = fix_polygons(_frame([Point(1, 1), Point(2, float('nan'))]))
    assert list(has_geometry(out)) == [True, False]


def test_all_valid_input_is_returned_unchanged():
    gdf = _frame([Point(1, 1), Point(2, 2)])
    out = fix_polygons(gdf)
    assert out.geometry.equals(gdf.geometry)


def test_existing_missing_geometry_is_left_alone():
    # A row that already had no geometry must not be routed through the
    # repair path at all -- `None` is not invalid, it is absent.
    gdf = _frame([Point(1, 1), None])
    out = fix_polygons(gdf)
    assert out.geometry.iloc[1] is None
    assert out.geometry.iloc[0].equals(Point(1, 1))


def test_repairable_geometry_is_still_repaired():
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
    assert not bowtie.is_valid
    out = fix_polygons(_frame([bowtie]))
    assert out.geometry.iloc[0].is_valid
    assert np.isfinite(out.geometry.iloc[0].area)
