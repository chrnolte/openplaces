"""subdivide_polygons: bounded pieces that cover the original exactly."""

import geopandas as gpd
import numpy as np
import pytest
import shapely

from openplaces.geo.polygon import subdivide_polygons


def _lake(n=4000):
    """A wiggly ring polygon with an island: many vertices, one hole."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 1000 + 50 * np.sin(17 * t)
    outer = np.c_[r * np.cos(t), r * np.sin(t)]
    island = shapely.Point(300, 0).buffer(100, quad_segs=64)
    return shapely.Polygon(outer, [island.exterior.coords[::-1]])


def test_pieces_are_small_and_cover_the_polygon():
    lake = _lake()
    src = gpd.GeoSeries([lake, shapely.box(5000, 0, 5010, 10)], index=['a', 'b'])
    pieces = subdivide_polygons(src, max_vertices=128)

    assert (shapely.get_num_coordinates(pieces.values) <= 128).all()
    assert set(pieces.index) == {'a', 'b'}
    assert len(pieces.loc[['b']]) == 1  # a small polygon is returned whole
    assert pieces.loc['a'].area.sum() == pytest.approx(lake.area, rel=1e-9)
    rebuilt = shapely.union_all(pieces.loc['a'].values)
    assert rebuilt.symmetric_difference(lake).area < 1e-6 * lake.area


def test_extent_bound():
    pieces = subdivide_polygons(
        gpd.GeoSeries([shapely.box(0, 0, 1000, 100)]),
        max_vertices=10**6,
        max_extent=150,
    )
    b = shapely.bounds(pieces.values)
    assert (np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]) <= 150 + 1e-9).all()
    assert pieces.area.sum() == pytest.approx(100_000)
