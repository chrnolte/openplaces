"""Overlays must keep the polygonal part of a GeometryCollection intersection.

Two polygons that share an area and also touch along an edge intersect in
a GeometryCollection (a Polygon plus a LineString). Filtering the overlay
result down to Polygon/MultiPolygon rows drops that pair outright, so a
real overlap disappears from the result and, in identity/union mode, the
left polygon is emitted as if nothing matched it.
"""

from __future__ import annotations

import geopandas as gpd
import pytest
import shapely
from shapely.geometry import box

from openplaces.geo.polygon import overlay_polygons


def _gdfs():
    """Left square, right polygon overlapping it and touching its top edge."""
    left = gpd.GeoDataFrame(
        {'id_1': ['A'], 'geometry': [box(0, 0, 2, 2)]}, crs='epsg:4326'
    ).set_index('id_1')
    # A single polygon: the right half of the left square, plus a band
    # sitting on top of it. The intersection is that right half (an
    # area) plus the shared top edge (a line).
    touching = shapely.union_all([box(1, 0, 3, 2), box(-1, 2, 3, 4)])
    right = gpd.GeoDataFrame(
        {'id_2': ['B'], 'geometry': [touching]}, crs='epsg:4326'
    ).set_index('id_2')
    return left, right


def test_geometry_collection_intersection_is_kept():
    left, right = _gdfs()

    result = overlay_polygons(left, right, how='intersection', geom=True)

    assert list(result.index) == [('A', 'B')]
    assert result.geometry.iloc[0].area == pytest.approx(2.0)


def test_identity_overlay_does_not_report_the_pair_as_unmatched():
    left, right = _gdfs()

    result = overlay_polygons(left, right, how='identity', geom=True)

    matched = result[result.index.get_level_values('id_2').notna()]
    assert len(matched) == 1
    assert matched.geometry.iloc[0].area == pytest.approx(2.0)
    # The other half of the left square is a leftover fragment, not a
    # fully unmatched copy of the whole square.
    leftover = result[result.index.get_level_values('id_2').isna()]
    assert leftover.geometry.area.sum() == pytest.approx(2.0)


def test_union_overlay_does_not_report_the_pair_as_unmatched():
    left, right = _gdfs()

    result = overlay_polygons(left, right, how='union', geom=True)

    matched = result[
        result.index.get_level_values('id_1').notna()
        & result.index.get_level_values('id_2').notna()
    ]
    assert len(matched) == 1
    assert matched.geometry.iloc[0].area == pytest.approx(2.0)
