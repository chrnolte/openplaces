"""The park rule on the building side: aligned long homes, or many.

plans/occupancy-and-dwellings-one-decider.md, "Park rule research" and
the "Aligned-homes park rule build": a lot is a park where it holds 8
or more manufactured homes, or where 3 or more of its homes with an
aspect of 2.5 or more have long-axis bearings inside one 15-degree
window, modulo 180. Every lot, outline, class and bearing below is
fabricated.
"""

from __future__ import annotations

import math

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.estimators.manufactured_homes import (
    aligned_homes,
    flag_manufactured_home_communities,
)
from openplaces.io.curator.inferers import derive_indicators
from openplaces.recipe import get_recipe_by_id

MH = 'Manufactured Home'
RULE = {
    'homes_at_least': 8,
    'aligned_homes_at_least': 3,
    'min_aspect': 2.5,
    'bearing_window_degrees': 15,
}
SHAPES = [
    {'output': 'aspect_ratio', 'type': 'shape_metric', 'metric': 'aspect_ratio'},
    {'output': 'long_axis_bearing', 'type': 'shape_metric', 'metric': 'bearing'},
]
# Outline sizes in meters: a single-wide (aspect 4), a home under the
# aspect cut (2.3), and a short home (2).
LONG = (20.0, 5.0)
BELOW = (11.5, 5.0)
SHORT = (12.0, 6.0)
# UTM 15N on its central meridian, where grid north is true north.
CRS = 'EPSG:32615'
X0, Y0 = 500_000.0, 3_300_000.0


def _state(curated):
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US', 'XX', 'YY'),
        verbose=False,
        timer=None,
        curated=curated,
    )


def _outline(x, y, size, bearing):
    """A rectangle centered on (x, y), its long side along *bearing*."""
    length, width = size
    b = math.radians(bearing)
    ux, uy = math.sin(b), math.cos(b)
    vx, vy = uy, -ux
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    return Polygon(
        [
            (
                x + s * length / 2 * ux + t * width / 2 * vx,
                y + s * length / 2 * uy + t * width / 2 * vy,
            )
            for s, t in corners
        ]
    )


def _lots(lots):
    """Rows from {lot: [(size, bearing, class), ...]}, 40 m apart."""
    rows, geometry = [], []
    for j, (lot, homes) in enumerate(lots.items()):
        for i, (size, bearing, occupancy) in enumerate(homes):
            rows.append({'parcel_id': lot, 'occupancy_type': occupancy})
            geometry.append(_outline(X0 + 40.0 * i, Y0 + 200.0 * j, size, bearing))
    return gpd.GeoDataFrame(rows, geometry=geometry, crs=CRS)


def _flag(lots, **params):
    frame = derive_indicators(_state(_lots(lots)), SHAPES).curated
    out = flag_manufactured_home_communities(_state(frame), **params).curated
    return out.groupby('parcel_id')['manufactured_home_community'].first()


def _homes(size, bearings, occupancy=MH):
    return [(size, b, occupancy) for b in bearings]


@pytest.mark.parametrize(
    ('homes', 'park'),
    [
        (_homes(LONG, [10, 15, 22]), True),
        (_homes(LONG, [0, 40, 80]), False),
        (_homes(LONG, [175, 2, 8]), True),
        (_homes(LONG, [30, 32]), False),
        (_homes(SHORT, [0, 50, 100, 150, 20, 70, 120, 170]), True),
        (_homes(SHORT, [0, 50, 100, 150, 20, 70, 120]), False),
        (_homes(BELOW, [40, 41, 42]), False),
        (_homes(LONG, [40, 41]) + _homes(BELOW, [42]), False),
        (_homes(LONG, [60, 64]) + _homes(LONG, [61, 63, 62], 'Secondary'), False),
    ],
    ids=[
        '3 long within 15',
        '3 long at 0 40 80',
        'window across 0 and 180',
        '2 aligned',
        '8 of any shape',
        '7 of any shape',
        '3 aligned under the aspect cut',
        '2 long and 1 under the cut',
        'long sheds do not count',
    ],
)
def test_the_park_rule(homes, park):
    assert _flag({'L': homes}, **RULE)['L'] == park


def test_the_bearing_is_the_long_side_clockwise_from_north():
    frame = _lots({'L': _homes(LONG, [30, 120, 179.5]) + _homes((8.0, 8.0), [10])})
    out = derive_indicators(_state(frame), SHAPES).curated
    bearing = out['long_axis_bearing'].tolist()
    assert bearing[:3] == pytest.approx([30, 120, 179.5], abs=0.05)
    # A square has no long side: aspect 1, and any bearing it gets is
    # below the cut it is read with.
    assert out['aspect_ratio'].iloc[3] == pytest.approx(1.0)
    assert out['aspect_ratio'].iloc[0] == pytest.approx(4.0)


def test_aligned_homes_are_counted_per_lot_only():
    # Two homes on each of two lots share one bearing: no lot has 3.
    lot = pd.Series(['a', 'a', 'b', 'b'])
    group = aligned_homes(
        lot,
        pd.Series([4.0] * 4),
        pd.Series([10.0] * 4),
        min_aspect=2.5,
        window_degrees=15,
    )
    assert group.tolist() == [2, 2, 2, 2]


def test_aligned_homes_wrap_and_skip_missing_values():
    lot = pd.Series(['a'] * 5 + [None])
    group = aligned_homes(
        lot,
        pd.Series([4.0, 4.0, 4.0, None, 1.0, 4.0]),
        pd.Series([179.0, 1.0, 13.9, 5.0, 5.0, 5.0]),
        min_aspect=2.5,
        window_degrees=15,
    )
    # 179, 1 and 13.9 fit 14.9 degrees; the home without an aspect,
    # the short home and the home without a lot take no part.
    assert group.tolist() == [3, 3, 3, 3, 3, 0]


def test_a_window_of_180_or_more_is_refused():
    with pytest.raises(ValueError, match='window_degrees'):
        aligned_homes(
            pd.Series(['a']),
            pd.Series([4.0]),
            pd.Series([0.0]),
            min_aspect=2.5,
            window_degrees=180,
        )


def test_the_arm_refuses_to_run_without_its_columns():
    frame = pd.DataFrame({'parcel_id': ['L'] * 3, 'occupancy_type': [MH] * 3})
    with pytest.raises(ValueError, match='long_axis_bearing'):
        flag_manufactured_home_communities(_state(frame), **RULE)


def test_without_the_arms_the_old_count_rule_holds():
    assert _flag({'L': _homes(SHORT, [0, 90, 0, 90])})['L']
    assert not _flag({'L': _homes(LONG, [0, 0, 0])})['L']


def test_the_national_recipe_sets_the_rule_and_derives_the_bearing():
    pipeline = get_recipe_by_id('US_building-openplaces-2026')['pipeline']
    step = next(
        e for e in pipeline if e.get('step') == 'flag_manufactured_home_communities'
    )
    assert {k: step[k] for k in RULE} == RULE
    assert 'min_median_aspect' not in step and 'aspect_min_homes' not in step
    specs = [
        spec
        for e in pipeline
        if e.get('step') == 'derive_indicators'
        for spec in e.get('indicators', [])
    ]
    assert {
        'output': 'long_axis_bearing',
        'type': 'shape_metric',
        'metric': 'bearing',
    } in specs
    dropped = next(e for e in pipeline if e.get('step') == 'order_columns')['drop']
    assert 'long_axis_bearing' in dropped
