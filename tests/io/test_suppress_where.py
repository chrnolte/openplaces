"""Tests for the generic `suppress_where` evidence-validity gate."""

from __future__ import annotations

import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.reconcilers import suppress_where


def _state(df: pd.DataFrame) -> CurateState:
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US'),
        verbose=False,
        timer=None,
        curated=df,
    )


def test_nulls_column_where_condition_matches():
    df = pd.DataFrame(
        {
            'n_dwellings_overture': [2.0, 3.0, 1.0],
            'land_use_class_parcel': ['Vacant', 'Single-Family', 'Vacant'],
        }
    )
    out = suppress_where(
        _state(df),
        column='n_dwellings_overture',
        condition_column='land_use_class_parcel',
        condition_value='Vacant',
    ).curated
    assert pd.isna(out['n_dwellings_overture'].iloc[0])
    assert out['n_dwellings_overture'].iloc[1] == 3.0
    assert pd.isna(out['n_dwellings_overture'].iloc[2])


def test_default_condition_value_is_true():
    df = pd.DataFrame({'value': [1.0, 2.0], 'flag': [True, False]})
    out = suppress_where(_state(df), column='value', condition_column='flag').curated
    assert pd.isna(out['value'].iloc[0])
    assert out['value'].iloc[1] == 2.0


def test_already_missing_values_stay_missing():
    df = pd.DataFrame({'value': [None], 'flag': [True]})
    out = suppress_where(_state(df), column='value', condition_column='flag').curated
    assert pd.isna(out['value'].iloc[0])


def test_missing_columns_are_noop():
    df = pd.DataFrame({'value': [1.0]})
    out = suppress_where(_state(df), column='value', condition_column='flag').curated
    assert out['value'].iloc[0] == 1.0
