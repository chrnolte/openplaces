"""Tests for the curate step filter_by_range."""

from __future__ import annotations

import pandas as pd
import pytest

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.filters import filter_by_range


def _state(df):
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('CO'),
        verbose=False,
        timer=None,
        curated=df,
    )


def _df():
    return pd.DataFrame(
        {
            'year': [1990, 1993, 2000, 2020, 2021, None],
            'price': [1.0, 0.0, 5.0, -1.0, 2.0, 3.0],
        },
        index=list('abcdef'),
    )


def test_inclusive_window_drops_null_by_default():
    out = filter_by_range(
        _state(_df()), filters=[{'column': 'year', 'min': 1993, 'max': 2020}]
    ).curated
    assert list(out.index) == ['b', 'c', 'd']


def test_exclusive_bound_and_keep_null():
    out = filter_by_range(
        _state(_df()),
        filters=[
            {'column': 'price', 'min': 0, 'inclusive': False},
            {'column': 'year', 'max': 2020, 'keep_null': True},
        ],
    ).curated
    # price > 0 keeps a, c, e, f; year <= 2020 or null keeps a, c, f.
    assert list(out.index) == ['a', 'c', 'f']


def test_absent_column_warns_and_skips():
    with pytest.warns(UserWarning, match='absent'):
        out = filter_by_range(
            _state(_df()), filters=[{'column': 'nope', 'min': 0}]
        ).curated
    assert len(out) == 6


def test_filter_without_bounds_raises():
    with pytest.raises(ValueError, match='min and/or max'):
        filter_by_range(_state(_df()), filters=[{'column': 'year'}])


def test_unknown_key_raises():
    with pytest.raises(ValueError, match='unknown key'):
        filter_by_range(_state(_df()), filters=[{'column': 'year', 'mn': 1}])
