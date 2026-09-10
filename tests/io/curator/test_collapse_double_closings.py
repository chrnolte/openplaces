"""Tests for the `collapse_double_closings` curate step."""

from __future__ import annotations

import pandas as pd
import pytest

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.transactions import collapse_double_closings


def _state(df: pd.DataFrame) -> CurateState:
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US'),
        verbose=False,
        timer=None,
        curated=df,
    )


def _sale(parcel, year, month, price, book, page):
    return {
        'parcel_id_assessor': parcel,
        'sale_year': year,
        'sale_month': month,
        'price': price,
        'sale_book': book,
        'sale_page': page,
    }


def test_earlier_leg_dropped_when_same_price_and_close_in_time():
    df = pd.DataFrame(
        [
            _sale('A', 2020, 1, 100000, '100', '1'),
            _sale('A', 2020, 2, 100000, '101', '1'),
        ]
    )
    out = collapse_double_closings(_state(df), key_column='parcel_id_assessor').curated
    assert len(out) == 1
    assert out['sale_month'].iloc[0] == 2


def test_different_price_both_survive():
    df = pd.DataFrame(
        [
            _sale('A', 2020, 1, 100000, '100', '1'),
            _sale('A', 2020, 2, 150000, '101', '1'),
        ]
    )
    out = collapse_double_closings(_state(df), key_column='parcel_id_assessor').curated
    assert len(out) == 2


def test_gap_beyond_window_both_survive():
    df = pd.DataFrame(
        [
            _sale('A', 2020, 1, 100000, '100', '1'),
            _sale('A', 2020, 6, 100000, '101', '1'),
        ]
    )
    out = collapse_double_closings(
        _state(df), key_column='parcel_id_assessor', max_gap_months=1
    ).curated
    assert len(out) == 2


def test_same_document_is_not_a_double_closing():
    # Same book/page: this is dedup_transactions's concern, not this
    # step's; collapse_double_closings requires a genuinely different
    # recorded document.
    df = pd.DataFrame(
        [
            _sale('A', 2020, 1, 100000, '100', '1'),
            _sale('A', 2020, 2, 100000, '100', '1'),
        ]
    )
    out = collapse_double_closings(_state(df), key_column='parcel_id_assessor').curated
    assert len(out) == 2


def test_different_parcels_are_independent():
    df = pd.DataFrame(
        [
            _sale('A', 2020, 1, 100000, '100', '1'),
            _sale('B', 2020, 2, 100000, '101', '1'),
        ]
    )
    out = collapse_double_closings(_state(df), key_column='parcel_id_assessor').curated
    assert len(out) == 2


def test_missing_key_column_is_noop():
    df = pd.DataFrame({'value': [1.0, 2.0]})
    out = collapse_double_closings(_state(df), key_column='parcel_id_assessor').curated
    assert list(out['value']) == [1.0, 2.0]


def test_near_miss_price_both_survive():
    df = pd.DataFrame(
        [
            _sale('A', 2020, 1, 100000, '100', '1'),
            _sale('A', 2020, 2, 100500, '101', '1'),
        ]
    )
    out = collapse_double_closings(_state(df), key_column='parcel_id_assessor').curated
    assert len(out) == 2


def test_keep_first_not_implemented():
    df = pd.DataFrame([_sale('A', 2020, 1, 100000, '100', '1')])
    with pytest.raises(NotImplementedError):
        collapse_double_closings(
            _state(df), key_column='parcel_id_assessor', keep='first'
        )
