"""Tests for the generic `score_relative_to_group` statistical primitive."""

from __future__ import annotations

import math

import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.inferers import score_relative_to_group


def _state(df: pd.DataFrame) -> CurateState:
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US'),
        verbose=False,
        timer=None,
        curated=df,
    )


def test_zscore_zero_for_group_mean():
    # Three equal values in one cohort: mean == every value, so std is 0 and
    # the score is missing (no variance to score against), not a divide error.
    df = pd.DataFrame({'grp': ['a', 'a', 'a'], 'val': [10.0, 10.0, 10.0]})
    out = score_relative_to_group(
        _state(df),
        group_column='grp',
        value_column='val',
        output='z',
        transform=None,
    ).curated
    assert out['z'].isna().all()


def test_zscore_separates_low_and_high_within_cohort():
    df = pd.DataFrame({'grp': ['a', 'a', 'a'], 'val': [1.0, 10.0, 100.0]})
    out = score_relative_to_group(
        _state(df),
        group_column='grp',
        value_column='val',
        output='z',
        transform=None,
    ).curated
    assert out['z'].iloc[0] < out['z'].iloc[1] < out['z'].iloc[2]


def test_zscore_compares_within_cohort_not_globally():
    # Cohort 'a' is tiny values, cohort 'b' is huge values; a mid-sized 'a'
    # value should score high within its own cohort despite being globally
    # tiny, since scoring is per-group.
    df = pd.DataFrame(
        {
            'grp': ['a', 'a', 'a', 'b', 'b', 'b'],
            'val': [1.0, 5.0, 9.0, 1000.0, 5000.0, 9000.0],
        }
    )
    out = score_relative_to_group(
        _state(df),
        group_column='grp',
        value_column='val',
        output='z',
        transform=None,
    ).curated
    assert out['z'].iloc[2] > 0  # 9.0 is the top of cohort 'a'
    assert out['z'].iloc[3] < 0  # 1000.0 is the bottom of cohort 'b'


def test_missing_value_scores_missing_not_extreme():
    df = pd.DataFrame({'grp': ['a', 'a', 'a'], 'val': [1.0, 10.0, None]})
    out = score_relative_to_group(
        _state(df),
        group_column='grp',
        value_column='val',
        output='z',
        transform=None,
    ).curated
    assert out['z'].isna().iloc[2]


def test_log1p_transform_compresses_skew():
    df = pd.DataFrame({'grp': ['a', 'a', 'a'], 'val': [1.0, 10.0, 1000.0]})
    out = score_relative_to_group(
        _state(df),
        group_column='grp',
        value_column='val',
        output='z',
        transform='log1p',
    ).curated
    # Order preserved, but the extreme value's z-score is far less extreme
    # in log space than it would be on the raw scale.
    assert out['z'].iloc[0] < out['z'].iloc[1] < out['z'].iloc[2]
    assert out['z'].iloc[2] < 5.0


def test_percentile_statistic():
    df = pd.DataFrame({'grp': ['a', 'a', 'a'], 'val': [1.0, 2.0, 3.0]})
    out = score_relative_to_group(
        _state(df),
        group_column='grp',
        value_column='val',
        output='pct',
        transform=None,
        statistic='percentile',
    ).curated
    assert math.isclose(out['pct'].iloc[2], 1.0)
    assert math.isclose(out['pct'].iloc[0], 1.0 / 3.0)


def test_missing_columns_are_noop():
    df = pd.DataFrame({'val': [1.0]})
    out = score_relative_to_group(
        _state(df),
        group_column='grp',
        value_column='val',
        output='z',
    ).curated
    assert 'z' not in out.columns
