"""Tests for the generic `fill_missing_numeric` output-shaping step."""

from __future__ import annotations

import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.imputers import fill_missing_numeric


def _state(df: pd.DataFrame) -> CurateState:
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US'),
        verbose=False,
        timer=None,
        curated=df,
    )


def test_zero_fills_and_casts_to_int64():
    df = pd.DataFrame({'n_dwellings_overture': [2.0, None, 1.0]})
    out = fill_missing_numeric(_state(df), columns=['n_dwellings_overture']).curated
    assert out['n_dwellings_overture'].dtype == 'int64'
    assert out['n_dwellings_overture'].tolist() == [2, 0, 1]


def test_custom_fill_value_and_dtype():
    df = pd.DataFrame({'value': [1.5, None]})
    out = fill_missing_numeric(
        _state(df), columns=['value'], fill_value=-1, dtype='float64'
    ).curated
    assert out['value'].dtype == 'float64'
    assert out['value'].tolist() == [1.5, -1.0]


def test_missing_column_is_noop():
    df = pd.DataFrame({'other': [1.0]})
    out = fill_missing_numeric(_state(df), columns=['n_dwellings_overture']).curated
    assert 'n_dwellings_overture' not in out.columns
