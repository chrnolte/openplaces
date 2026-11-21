"""The prepared polygon test of resolve_spine's neighbor handling.

Fabricated outlines around a concave unit polygon. The helper has to
answer exactly as the unprepared GeoSeries test it replaced.
"""

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, box

from openplaces.io.harmonizer.spine import _reaching

# A unit shaped like a C: the notch is where an unprepared and a
# bounding-box test would disagree.
UNIT = Polygon([(0, 0), (10, 0), (10, 3), (4, 3), (4, 7), (10, 7), (10, 10), (0, 10)])


def _outlines():
    return gpd.GeoSeries(
        [
            box(1, 1, 2, 2),  # inside
            box(6, 4, 7, 5),  # in the notch: inside the bounds, outside the unit
            box(9.5, 2.5, 11, 3.5),  # crosses the boundary
            box(10, 8, 11, 9),  # touches the boundary along an edge
            box(20, 20, 21, 21),  # far away
            None,  # a missing geometry
            Polygon(),  # an empty geometry
        ],
        crs='epsg:6933',
    )


def test_reaching_answers_as_the_unprepared_test_does():
    outlines = _outlines()
    expected = outlines.intersects(UNIT).to_numpy()
    found = _reaching(outlines, UNIT)
    assert found.dtype == bool
    assert np.array_equal(found, expected)
    assert found.tolist() == [True, False, True, True, False, False, False]


def test_reaching_on_no_rows():
    assert _reaching(_outlines()[:0], UNIT).tolist() == []
