"""
Tests for Microsoft global-footprint quadkey selection.

The covering set used to be sampled on a fixed 60x60 lattice of points,
which steps over whole tiles on any wide bounding box, and an
antimeridian-crossing unit asked for a globe-wide box. No network: these
exercise the pure quadkey functions.
"""

import math

import geopandas as gpd
import pytest
from shapely.geometry import MultiPolygon, box

from openplaces.io.scrapers import microsoft_global_scraper as scraper

ZOOM = 9

# Roughly the conterminous United States: the box the review measured a
# 26 percent shortfall on.
CONUS = (-125.0, 24.5, -66.9, 49.4)


def _tile_bounds(quadkey: str) -> tuple[float, float, float, float]:
    """Return a quadkey's own (minx, miny, maxx, maxy) in EPSG:4326."""
    tile_x = tile_y = 0
    for digit in quadkey:
        tile_x <<= 1
        tile_y <<= 1
        value = int(digit)
        if value & 1:
            tile_x |= 1
        if value & 2:
            tile_y |= 1
    n = 1 << len(quadkey)

    def _lat(row):
        angle = math.pi - 2 * math.pi * row / n
        return math.degrees(math.atan(math.sinh(angle)))

    return (
        tile_x / n * 360 - 180,
        _lat(tile_y + 1),
        (tile_x + 1) / n * 360 - 180,
        _lat(tile_y),
    )


def test_every_point_in_a_wide_box_lands_in_a_selected_tile():
    """No tile inside the box may be missing from the covering set."""
    keys = scraper.quadkeys_for_bounds(CONUS, ZOOM)

    minx, miny, maxx, maxy = CONUS
    steps = 200
    missing = set()
    for i in range(steps + 1):
        lon = minx + (maxx - minx) * i / steps
        for j in range(steps + 1):
            lat = miny + (maxy - miny) * j / steps
            point_key = scraper.quadkey(lat, lon, ZOOM)
            if point_key not in keys:
                missing.add(point_key)

    assert not missing


def test_no_selected_tile_lies_outside_the_box():
    """The covering set must not fetch tiles the box does not touch."""
    keys = scraper.quadkeys_for_bounds(CONUS, ZOOM)
    minx, miny, maxx, maxy = CONUS

    for key in keys:
        t_minx, t_miny, t_maxx, t_maxy = _tile_bounds(key)
        assert t_minx <= maxx and t_maxx >= minx
        assert t_miny <= maxy and t_maxy >= miny


def test_a_small_box_still_resolves_to_a_handful_of_tiles():
    """The exact enumeration is not a regression for small units."""
    keys = scraper.quadkeys_for_bounds((-71.2, 42.2, -70.9, 42.5), ZOOM)

    assert 1 <= len(keys) <= 4
    assert scraper.quadkey(42.35, -71.05, ZOOM) in keys


def test_antimeridian_unit_does_not_ask_for_the_whole_globe():
    """
    An Aleutians-shaped unit has a globe-wide total_bounds.

    Selecting per geometry part instead keeps the request to the two
    narrow strips that actually hold land.
    """
    west = box(179.0, 51.0, 179.9, 52.0)
    east = box(-179.9, 51.0, -179.0, 52.0)
    geometries = gpd.GeoSeries([MultiPolygon([west, east])], crs='EPSG:4326')

    keys = scraper.quadkeys_for_geometries(geometries, ZOOM)
    whole_box = scraper.quadkeys_for_bounds(geometries.total_bounds, ZOOM)

    assert len(keys) < 20
    assert len(whole_box) > 1000
    # Nothing in the mid-Atlantic or over Europe may be requested.
    assert scraper.quadkey(51.5, 0.0, ZOOM) not in keys
    assert scraper.quadkey(51.5, -30.0, ZOOM) in whole_box
    # Both real strips are still covered.
    assert scraper.quadkey(51.5, 179.5, ZOOM) in keys
    assert scraper.quadkey(51.5, -179.5, ZOOM) in keys


def test_empty_geometry_selects_nothing():
    geometries = gpd.GeoSeries([], dtype='geometry', crs='EPSG:4326')

    assert scraper.quadkeys_for_geometries(geometries, ZOOM) == set()


@pytest.mark.parametrize('lat', [-89.0, 89.0])
def test_polar_latitudes_are_clamped_not_undefined(lat):
    """Web Mercator is undefined at the poles; the row must still resolve."""
    key = scraper.quadkey(lat, 0.0, ZOOM)

    assert len(key) == ZOOM
