"""Tests for the generic `exclude_by_value` curate filter."""

from __future__ import annotations

import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.filters import exclude_by_value


def _state(df: pd.DataFrame) -> CurateState:
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US'),
        verbose=False,
        timer=None,
        curated=df,
    )


def test_matching_rows_dropped():
    df = pd.DataFrame(
        {
            'parcel_id_assessor': ['WATER', 'ROW', '12345'],
            'value': [1.0, 2.0, 3.0],
        }
    )
    out = exclude_by_value(
        _state(df), column='parcel_id_assessor', values=['WATER', 'ROW']
    ).curated
    assert list(out['parcel_id_assessor']) == ['12345']


def test_non_matching_rows_survive_unchanged():
    df = pd.DataFrame({'parcel_id_assessor': ['12345', '67890'], 'value': [1.0, 2.0]})
    out = exclude_by_value(
        _state(df), column='parcel_id_assessor', values=['WATER', 'ROW']
    ).curated
    assert list(out['parcel_id_assessor']) == ['12345', '67890']
    assert list(out['value']) == [1.0, 2.0]


def test_missing_column_is_noop():
    df = pd.DataFrame({'value': [1.0, 2.0]})
    out = exclude_by_value(
        _state(df), column='parcel_id_assessor', values=['WATER']
    ).curated
    assert list(out['value']) == [1.0, 2.0]
