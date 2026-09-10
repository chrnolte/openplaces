"""Tests for the `dedup_transactions` curate step."""

from __future__ import annotations

import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.transactions import dedup_transactions

KEY = [
    'parcel_id_assessor',
    'sale_year',
    'sale_month',
    'price',
    'sale_book',
    'sale_page',
]


def _state(df: pd.DataFrame) -> CurateState:
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US'),
        verbose=False,
        timer=None,
        curated=df,
    )


def test_exact_duplicate_dropped_keeping_first():
    # SDF's rolling window can list the same recorded document in two
    # adjacent year-files -- same key, same row, twice.
    df = pd.DataFrame(
        {
            'parcel_id_assessor': ['A', 'A'],
            'sale_year': [2020, 2020],
            'sale_month': [6, 6],
            'price': [100000, 100000],
            'sale_book': ['100', '100'],
            'sale_page': ['1', '1'],
            'source_file': ['2020', '2021'],
        }
    )
    out = dedup_transactions(_state(df), key_columns=KEY).curated
    assert len(out) == 1
    assert out['source_file'].iloc[0] == '2020'


def test_different_documents_both_survive():
    df = pd.DataFrame(
        {
            'parcel_id_assessor': ['A', 'A'],
            'sale_year': [2019, 2020],
            'sale_month': [1, 6],
            'price': [100000, 150000],
            'sale_book': ['90', '100'],
            'sale_page': ['1', '1'],
        }
    )
    out = dedup_transactions(_state(df), key_columns=KEY).curated
    assert len(out) == 2


def test_null_in_key_columns_does_not_collapse_distinct_rows():
    # sale_clerk_instrument can be entirely null for a county that only
    # uses book/page -- a null-filled key must not make every row of a
    # county look identical.
    key = KEY + ['sale_clerk_instrument']
    df = pd.DataFrame(
        {
            'parcel_id_assessor': ['A', 'B'],
            'sale_year': [2020, 2020],
            'sale_month': [1, 1],
            'price': [100000, 200000],
            'sale_book': ['100', '200'],
            'sale_page': ['1', '1'],
            'sale_clerk_instrument': [None, None],
        }
    )
    out = dedup_transactions(_state(df), key_columns=key).curated
    assert len(out) == 2
