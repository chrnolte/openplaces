"""Tests for `find_corridors`, the shape-only right-of-way detector."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

from openplaces.geo.polygon import find_corridors


def _strip(length_m, width_m, lat=42.0):
    """A rectangle of the given meter dimensions, in geographic coords."""
    deg_per_m_y = 1 / 111_320
    deg_per_m_x = deg_per_m_y / np.cos(np.radians(lat))
    dx, dy = length_m * deg_per_m_x, width_m * deg_per_m_y
    return Polygon([(0, lat), (dx, lat), (dx, lat + dy), (0, lat + dy)])


def test_flags_long_thin_strip_not_compact_block():
    gdf = gpd.GeoDataFrame(
        geometry=[
            _strip(5000, 10),  # a 5 km road: elongation ~500
            _strip(200, 200),  # an ordinary lot: elongation 4
        ],
        crs='EPSG:4326',
    )
    assert find_corridors(gdf).tolist() == [True, False]


def test_ignores_short_slivers():
    # A sliver is as elongated as a road but spans no distance; it is
    # degenerate geometry, not a right-of-way, so the length guard excludes
    # it rather than the elongation test letting it through.
    gdf = gpd.GeoDataFrame(
        geometry=[_strip(20, 0.04), _strip(5000, 10)], crs='EPSG:4326'
    )
    assert find_corridors(gdf).tolist() == [False, True]


def test_thresholds_are_tunable():
    gdf = gpd.GeoDataFrame(geometry=[_strip(2000, 40)], crs='EPSG:4326')
    # Elongation here is ~50: under the default 200, over an explicit 25.
    assert find_corridors(gdf).tolist() == [False]
    assert find_corridors(gdf, max_elongation=25).tolist() == [True]
    # The length guard vetoes it regardless of how elongated it is.
    assert find_corridors(gdf, max_elongation=25, min_length=5000).tolist() == [False]


def test_zero_area_is_not_a_corridor():
    degenerate = Polygon([(0, 42), (0, 42), (0, 42), (0, 42)])
    gdf = gpd.GeoDataFrame(geometry=[degenerate, _strip(5000, 10)], crs='EPSG:4326')
    # No division-by-zero blowup, and no false positive out of it either.
    assert find_corridors(gdf).tolist() == [False, True]


def test_accepts_a_geoseries():
    series = gpd.GeoSeries([_strip(5000, 10), _strip(200, 200)], crs='EPSG:4326')
    assert find_corridors(series).tolist() == [True, False]


def test_preserves_index_and_handles_empty():
    gdf = gpd.GeoDataFrame(
        geometry=[_strip(5000, 10), _strip(200, 200)],
        index=['road', 'lot'],
        crs='EPSG:4326',
    )
    flagged = find_corridors(gdf)
    assert flagged['road'] and not flagged['lot']

    assert find_corridors(gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')).empty
