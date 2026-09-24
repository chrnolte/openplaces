"""A source lot id is only unioned where its polygons actually touch.

The same signature, one lot id on several polygons, covers three
different things (measured 2026-09-21 over the repeated improvement
value whose group unions to one shape): a lot drawn in pieces
(Haywood NC 0.917, Waller TX 0.933, Vilas WI 0.979), separate parcels
under one account across a street (Galveston TX 0.006), and an id
collision (Hyde NC 0.000, 397 of 400 groups a median of 28.9 km apart).
Only the first is a lot.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from openplaces.io.stacked_units import split_stacked_units


def _square(x0, y0, side=10.0):
    return Polygon([(x0, y0), (x0, y0 + side), (x0 + side, y0 + side), (x0 + side, y0)])


def _frame(rows, geometries):
    return gpd.GeoDataFrame(pd.DataFrame(rows), geometry=geometries, crs='EPSG:3857')


def _row(lot, **overrides):
    row = {'lot_id': lot, 'parcel_id_local': lot, 'land_value': 1_000}
    row.update(overrides)
    return row


def test_touching_pieces_of_one_lot_are_unioned_into_one_row():
    # Two squares sharing an edge: one lot, drawn in two pieces.
    df = _frame(
        [_row('L1'), _row('L1')],
        [_square(0, 0), _square(10, 0)],
    )
    result = split_stacked_units(df, lot_key='lot_id')
    assert len(result.parcels) == 1
    assert result.n_multipart_lots == 1
    assert result.n_parts_merged == 1
    assert result.n_lots_not_unioned == 0
    # The surviving row carries the whole outline, 200 m2, not one piece.
    assert result.parcels.geometry.iloc[0].area == 200.0


def test_polygons_that_do_not_touch_are_left_as_separate_parcels():
    # The same record under one lot id on two squares 50 m apart: an
    # account over separate parcels, or an id collision. Either way the
    # union would invent an outline the cadastre does not draw.
    df = _frame(
        [_row('L1'), _row('L1')],
        [_square(0, 0), _square(60, 0)],
    )
    result = split_stacked_units(df, lot_key='lot_id')
    assert len(result.parcels) == 2
    assert result.n_lots_not_unioned == 1
    assert result.n_parts_merged == 0
    assert result.n_multipart_lots == 0
    assert [g.area for g in result.parcels.geometry] == [100.0, 100.0]


def test_pieces_meeting_at_a_single_point_are_not_one_lot():
    # Corner to corner is not a lot drawn in pieces, and treating it as
    # one would be the first step towards a distance tolerance.
    df = _frame(
        [_row('L1'), _row('L1')],
        [_square(0, 0), _square(10, 10)],
    )
    result = split_stacked_units(df, lot_key='lot_id')
    assert result.n_lots_not_unioned == 1
    assert len(result.parcels) == 2


def test_a_scattered_lot_and_a_touching_one_are_judged_separately():
    df = _frame(
        [_row('L1'), _row('L1'), _row('L2'), _row('L2')],
        [_square(0, 0), _square(10, 0), _square(0, 100), _square(0, 200)],
    )
    result = split_stacked_units(df, lot_key='lot_id')
    assert result.n_multipart_lots == 1
    assert result.n_lots_not_unioned == 1
    # L1 collapses to one row, L2 keeps both.
    assert len(result.parcels) == 3


def test_distinct_records_on_one_lot_still_become_properties():
    # The gate must not touch the stacked-units path: two different
    # records on one outline are units, not pieces of a lot.
    df = _frame(
        [_row('L1', land_value=1_000), _row('L1', land_value=2_000)],
        [_square(0, 0), _square(0, 0)],
    )
    result = split_stacked_units(df, lot_key='lot_id')
    assert len(result.parcels) == 1
    assert result.properties is not None
    assert len(result.properties) == 2
    assert result.n_lots_not_unioned == 0


def test_the_summary_says_when_a_lot_id_was_left_alone():
    df = _frame(
        [_row('L1'), _row('L1')],
        [_square(0, 0), _square(60, 0)],
    )
    result = split_stacked_units(df, lot_key='lot_id')
    assert 'do not touch' in result.summary()
