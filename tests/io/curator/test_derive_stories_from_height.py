"""Tests for `derive_stories_from_height` (FEMA has no story count, only a
LiDAR-derived building height)."""

from __future__ import annotations

import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.estimators.stories import derive_stories_from_height


def _state(df: pd.DataFrame) -> CurateState:
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US'),
        verbose=False,
        timer=None,
        curated=df,
    )


def test_rounds_to_nearest_story():
    df = pd.DataFrame({'height_footprint_fema': [3.5, 6.5, 9.5]})
    out = derive_stories_from_height(_state(df)).curated
    # 3.5/3.5=1.0 -> 1; 6.5/3.5=1.86 -> 2; 9.5/3.5=2.71 -> 3
    assert out['n_stories_footprint_fema'].tolist() == [1.0, 2.0, 3.0]


def test_half_stories_round_up():
    # 8.75/3.5 = 2.5 and 15.75/3.5 = 4.5: half up, not half to even.
    df = pd.DataFrame({'height_footprint_fema': [8.75, 15.75]})
    out = derive_stories_from_height(_state(df)).curated
    assert out['n_stories_footprint_fema'].tolist() == [3.0, 5.0]


def test_floors_at_one_story():
    df = pd.DataFrame({'height_footprint_fema': [0.5]})
    out = derive_stories_from_height(_state(df)).curated
    assert out['n_stories_footprint_fema'].iloc[0] == 1.0


def test_non_positive_or_missing_height_yields_missing_stories():
    df = pd.DataFrame({'height_footprint_fema': [0.0, -1.0, None]})
    out = derive_stories_from_height(_state(df)).curated
    assert out['n_stories_footprint_fema'].isna().all()


def test_missing_height_column_is_a_noop():
    df = pd.DataFrame({'other': [1.0]})
    out = derive_stories_from_height(_state(df)).curated
    assert 'n_stories_footprint_fema' not in out.columns


def test_custom_columns_and_floor_height():
    df = pd.DataFrame({'h': [10.0]})
    out = derive_stories_from_height(
        _state(df), column='stories', height_column='h', floor_height_m=5.0
    ).curated
    assert out['stories'].iloc[0] == 2.0
