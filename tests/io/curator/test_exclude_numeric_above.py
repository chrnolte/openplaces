"""Tests for the generic `exclude_numeric_above` curate filter."""

from __future__ import annotations

import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.filters import exclude_numeric_above


def _state(df: pd.DataFrame) -> CurateState:
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US'),
        verbose=False,
        timer=None,
        curated=df,
    )


def test_rows_above_max_dropped():
    df = pd.DataFrame({'n_dwellings': [1, 2, 4]})
    out = exclude_numeric_above(_state(df), column='n_dwellings', max=1).curated
    assert list(out['n_dwellings']) == [1]


def test_missing_value_is_kept():
    df = pd.DataFrame({'n_dwellings': [1, None, 2]})
    out = exclude_numeric_above(_state(df), column='n_dwellings', max=1).curated
    assert out['n_dwellings'].isna().sum() == 1
    assert len(out) == 2


def test_missing_column_is_noop():
    df = pd.DataFrame({'value': [1.0, 2.0]})
    out = exclude_numeric_above(_state(df), column='n_dwellings', max=1).curated
    assert list(out['value']) == [1.0, 2.0]
