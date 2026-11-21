"""Shape indicators measure each geometry once, with the same numbers.

plans/slow-steps-same-output-faster.md, R6: `get_oriented_dims_of`
measures many geometries from one `shapely.oriented_envelope` call, and
`derive_indicators` keeps the measurement for all its shape specs. Both
are compared exactly with `get_oriented_dims` per geometry, the
previous `_derive_shape_metric` kept below, on fabricated geometry:
blank, empty, degenerate, multi-part and a seeded random set.
"""

from __future__ import annotations

import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box

from openplaces.io.curator import CurateState, inferers
from openplaces.io.curator.inferers import derive_indicators
from openplaces.io.harmonizer.spine import get_oriented_dims, get_oriented_dims_of


def _geometries():
    rng = np.random.default_rng(31)
    random_polygons = [
        shapely.convex_hull(shapely.multipoints(rng.uniform(0, 40, size=(7, 2))))
        for _ in range(500)
    ]
    return [
        None,
        Polygon(),
        box(0, 0, 4, 1),
        box(0, 0, 1, 4),
        box(0, 0, 3, 3),
        Polygon([(0, 0), (4, 1), (3, 4), (-1, 3)]),
        Polygon([(0, 0), (1, 0), (2, 0), (0, 0)]),
        Point(1, 2).buffer(0),
        LineString([(0, 0), (3, 4)]),
        MultiPolygon([box(0, 0, 1, 1), box(5, 5, 7, 6)]),
        *random_polygons,
    ]


def test_many_equals_one_by_one():
    geometries = _geometries()
    new = get_oriented_dims_of(geometries)
    old = [get_oriented_dims(g) for g in geometries]
    assert new == old
    assert [tuple(map(type, t)) for t in new] == [tuple(map(type, t)) for t in old]


def test_rectangles_are_the_same_function():
    geometries = [g for g in _geometries() if g is not None]
    one_by_one = [g.minimum_rotated_rectangle for g in geometries]
    vectorized = shapely.oriented_envelope(np.asarray(geometries, dtype=object))
    assert list(shapely.to_wkb(one_by_one)) == list(shapely.to_wkb(vectorized))


def _old_shape_metric(curated, spec):
    from openplaces.geo.polygon import local_metric_crs

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        geometry = curated.geometry.to_crs(local_metric_crs(curated))
    dims = geometry.map(get_oriented_dims)
    length = dims.map(lambda d: d[1])
    width = dims.map(lambda d: d[2])
    metric = spec.get('metric', 'aspect_ratio')
    if metric == 'aspect_ratio':
        return length / width.clip(lower=1e-6)
    if metric == 'length':
        return length
    if metric == 'width':
        return width
    return dims.map(lambda d: d[0])


def _curated():
    rng = np.random.default_rng(37)
    x = -78.9 + rng.uniform(0, 0.05, 400)
    y = 35.9 + rng.uniform(0, 0.05, 400)
    shapes = list(shapely.box(x, y, x + rng.uniform(1e-5, 3e-4, 400), y + 1e-4))
    shapes[3] = None
    shapes[7] = Polygon()
    return gpd.GeoDataFrame(
        {'n': range(400)},
        geometry=shapes,
        crs='EPSG:4326',
        index=pd.Index([f'b{i:04d}' for i in range(400)], name='building_id'),
    )


@pytest.mark.parametrize(
    'metrics', [['aspect_ratio'], ['aspect_ratio', 'length', 'width', 'orientation']]
)
def test_derive_indicators_shape_specs(metrics):
    curated = _curated()
    specs = [
        {'output': f'shape_{m}', 'type': 'shape_metric', 'metric': m} for m in metrics
    ]
    state = CurateState(
        recipe={},
        entity_recipe={},
        admin_id=None,
        verbose=False,
        timer=None,
        curated=curated.copy(),
    )
    new = derive_indicators(state, specs).curated
    # The measurement is kept only while derive_indicators runs.
    assert inferers._SHAPE_DIMS.get() is None
    old = curated.copy()
    for spec in specs:
        old[spec['output']] = _old_shape_metric(old, spec)
    pd.testing.assert_frame_equal(
        pd.DataFrame(new.drop(columns='geometry')),
        pd.DataFrame(old.drop(columns='geometry')),
        check_exact=True,
        check_dtype=True,
    )
