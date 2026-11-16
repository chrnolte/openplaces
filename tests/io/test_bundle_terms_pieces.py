"""A bundle of buildings cut from an OBM outline still reports ODbL.

plans/buildings-from-parcel-pieces.md, step 3: a piece keeps its
outline's `geometry_source` token, which `bundle_terms` reads, so a
piece of an ODbL outline carries the share-alike licence and shares
are counted per building row. Fabricated geometry
(docs/5_contribute/no-personal-data.rst).
"""

from __future__ import annotations

import pandas as pd
import pytest
from shapely.geometry import box

from openplaces.io.delivery.terms import bundle_terms
from tests.io.harmonizer.test_parcel_pieces import (
    _lots,
    _row_of_four,
    _spine,
    _split,
)

RECIPE = 'US_footprint-cheer-2026'


def test_a_piece_of_an_obm_outline_reports_odbl():
    spine, lots, points, values = _row_of_four()
    out, _, stats = _split(spine, lots, points, values)
    assert stats['outlines_cut'] == 1
    terms = bundle_terms(RECIPE, out['geometry_source'])
    assert terms['share_alike'] == {'ODbL-1.0': pytest.approx(1.0)}
    assert terms['unknown_share'] == pytest.approx(0.0)


def test_shares_count_the_pieces_as_rows():
    """Four OBM pieces and one uncut Microsoft outline: OBM's share is
    4 of 5 rows, each piece a row of its own."""
    spine, lots, points, values = _row_of_four()
    spine = pd.concat(
        [spine, _spine(sources={'m1': 'microsoft'}, m1=box(100, 0, 110, 10))]
    )
    lots = pd.concat([lots, _lots(Q=box(95, -5, 115, 15))])
    out, _, _ = _split(spine, lots, points, values)
    assert out['geometry_source'].tolist().count('obm') == 4
    terms = bundle_terms(RECIPE, out['geometry_source'])
    shares = {entry['source_id']: entry['share'] for entry in terms['sources']}
    assert shares['obm'] == pytest.approx(0.8)
    assert shares['microsoft'] == pytest.approx(0.2)
