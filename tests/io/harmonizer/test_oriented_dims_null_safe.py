"""`get_oriented_dims` must survive blank and degenerate geometry.

Blank geometry is a legitimate state: `fix_polygons` blanks a polygon
whose coordinates are non-finite rather than handing it to
`shapely.make_valid`, which raises. One such row used to abort a whole
county's geospine with
`AttributeError: 'NoneType' object has no attribute
'minimum_rotated_rectangle'`.
"""

import geopandas as gpd
import pytest
from shapely.geometry import GeometryCollection, LineString, Point, Polygon, box

from openplaces.io.harmonizer.spine import get_oriented_dims

NEUTRAL = (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    'geom',
    [
        None,
        Polygon(),
        Point(0, 0),
        LineString([(0, 0), (1, 1)]),
        GeometryCollection(),
    ],
    ids=['none', 'empty-polygon', 'point', 'line', 'empty-collection'],
)
def test_degenerate_geometry_is_neutral(geom):
    assert get_oriented_dims(geom) == NEUTRAL


def test_a_real_rectangle_still_measures():
    angle, length, width = get_oriented_dims(box(0, 0, 4, 1))
    assert length == pytest.approx(4.0)
    assert width == pytest.approx(1.0)
    assert angle % 180 == pytest.approx(0.0)


def test_blank_row_does_not_stop_a_column_of_real_ones():
    """The failure mode was one bad row aborting the whole geospine."""
    geoms = gpd.GeoSeries([box(0, 0, 4, 1), None, box(0, 0, 1, 3)])
    dims = geoms.map(get_oriented_dims)
    assert dims.iloc[1] == NEUTRAL
    assert dims.iloc[0][1] == pytest.approx(4.0)
    assert dims.iloc[2][1] == pytest.approx(3.0)


def test_neutral_row_reads_as_not_elongated():
    """Zero width makes the aspect ratio zero, so the row is filtered out.

    This is the property the callers rely on: they divide length by
    width clipped at 1e-6 and compare against an aspect-ratio floor.
    """
    _, length, width = get_oriented_dims(None)
    assert length / max(width, 1e-6) == 0.0
