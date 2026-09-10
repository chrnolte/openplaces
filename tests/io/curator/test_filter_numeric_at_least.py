"""Tests for the generic `filter_numeric_at_least` curate filter."""

from __future__ import annotations

import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.filters import filter_numeric_at_least


def _state(df: pd.DataFrame) -> CurateState:
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US'),
        verbose=False,
        timer=None,
        curated=df,
    )


def test_rows_below_threshold_dropped():
    df = pd.DataFrame({'sale_year': [2014, 2016, 2020]})
    out = filter_numeric_at_least(_state(df), column='sale_year', min=2016).curated
    assert list(out['sale_year']) == [2016, 2020]


def test_non_numeric_values_dropped():
    df = pd.DataFrame({'sale_year': [2020, 'not_a_year', 2010]})
    out = filter_numeric_at_least(_state(df), column='sale_year', min=2016).curated
    assert list(out['sale_year']) == [2020]


def test_missing_column_is_noop():
    df = pd.DataFrame({'value': [1.0, 2.0]})
    out = filter_numeric_at_least(_state(df), column='sale_year', min=2016).curated
    assert list(out['value']) == [1.0, 2.0]
