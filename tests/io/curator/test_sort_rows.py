"""Tests for the generic `sort_rows` curate formatter."""

from __future__ import annotations

import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.formatters import sort_rows


def _state(df: pd.DataFrame) -> CurateState:
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US'),
        verbose=False,
        timer=None,
        curated=df,
    )


def test_sorted_ascending_by_all_keys():
    df = pd.DataFrame(
        {
            'parcel_id_assessor': ['B', 'A', 'A'],
            'sale_year': [2020, 2021, 2020],
            'sale_month': [1, 1, 6],
        }
    )
    out = sort_rows(
        _state(df), by=['parcel_id_assessor', 'sale_year', 'sale_month']
    ).curated
    assert list(
        zip(out['parcel_id_assessor'], out['sale_year'], out['sale_month'])
    ) == [
        ('A', 2020, 6),
        ('A', 2021, 1),
        ('B', 2020, 1),
    ]


def test_missing_sort_columns_ignored():
    df = pd.DataFrame({'parcel_id_assessor': ['B', 'A'], 'sale_year': [2020, 2021]})
    out = sort_rows(_state(df), by=['parcel_id_assessor', 'no_such_column']).curated
    assert list(out['parcel_id_assessor']) == ['A', 'B']


def test_no_matching_columns_is_noop():
    df = pd.DataFrame({'value': [2, 1]})
    out = sort_rows(_state(df), by=['no_such_column']).curated
    assert list(out['value']) == [2, 1]
