"""Tests for the generic `keep_by_value` curate filter."""

from __future__ import annotations

import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.filters import keep_by_value


def _state(df: pd.DataFrame) -> CurateState:
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US'),
        verbose=False,
        timer=None,
        curated=df,
    )


def test_matching_rows_kept():
    df = pd.DataFrame(
        {
            'sale_qualification_code': ['Qualified', 'Disqualified', 'Qualified'],
            'value': [1.0, 2.0, 3.0],
        }
    )
    out = keep_by_value(
        _state(df), column='sale_qualification_code', values=['Qualified']
    ).curated
    assert list(out['value']) == [1.0, 3.0]


def test_non_matching_rows_dropped():
    df = pd.DataFrame({'code': ['A', 'B', 'C'], 'value': [1.0, 2.0, 3.0]})
    out = keep_by_value(_state(df), column='code', values=['A', 'C']).curated
    assert list(out['code']) == ['A', 'C']


def test_missing_column_is_noop():
    df = pd.DataFrame({'value': [1.0, 2.0]})
    out = keep_by_value(_state(df), column='code', values=['A']).curated
    assert list(out['value']) == [1.0, 2.0]
