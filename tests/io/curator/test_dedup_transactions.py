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


def _respelled_pair(second_id: str) -> pd.DataFrame:
    # One deed in two adjacent SDF year files; the second file writes the
    # parcel number with other punctuation (fabricated values).
    return pd.DataFrame(
        {
            'parcel_id_assessor': ['12-34-56-0007-000-00080', second_id],
            'sale_year': [2021, 2021],
            'sale_month': [6, 6],
            'price': [250000, 250000],
            'sale_book': ['100', '100'],
            'sale_page': ['7', '7'],
            'source_file': ['2021', '2022'],
        }
    )


def test_respelled_parcel_number_is_one_document_when_normalized():
    df = _respelled_pair('123456000700000080')
    out = dedup_transactions(
        _state(df), key_columns=KEY, normalize_columns=['parcel_id_assessor']
    ).curated
    assert out['source_file'].tolist() == ['2021']
    # The surviving row keeps its own spelling; nothing is rewritten.
    assert out['parcel_id_assessor'].iloc[0] == '12-34-56-0007-000-00080'


def test_respelled_parcel_number_survives_twice_without_normalization():
    df = _respelled_pair('123456000700000080')
    assert len(dedup_transactions(_state(df), key_columns=KEY).curated) == 2


def test_normalization_does_not_merge_different_parcels():
    # Exact after the rewrite: a different number is a different row.
    df = _respelled_pair('123456000700000081')
    out = dedup_transactions(
        _state(df), key_columns=KEY, normalize_columns=['parcel_id_assessor']
    ).curated
    assert len(out) == 2
