"""Tests for `select_value_source_by_admin_unit`."""

from __future__ import annotations

import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.reconcilers import select_value_source_by_admin_unit


def _state(df: pd.DataFrame) -> CurateState:
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US', 'MA', 'MI'),
        verbose=False,
        timer=None,
        curated=df,
    )


def _call(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return select_value_source_by_admin_unit(
        _state(df),
        output='structure_value',
        parcel_column='structure_value',
        other_column='structure_value_building_nsi',
        **kwargs,
    ).curated


def test_high_coverage_group_keeps_parcel_even_for_its_own_gap():
    # Town A: 7/8 eligible rows have a real parcel value (coverage >= 0.5) --
    # stays on the parcel source, including the one row individually missing
    # one (no per-row NSI top-off).
    df = pd.DataFrame(
        {
            'admin4_id': ['A'] * 8,
            'structure_value': [100.0] * 7 + [None],
            'structure_value_building_nsi': [999.0] * 8,
            'priority_on_parcel': ['primary'] * 8,
        }
    )
    out = _call(df, min_group_size=1)
    assert out['structure_value'].iloc[:7].eq(100.0).all()
    assert pd.isna(out['structure_value'].iloc[7])
    assert (out['structure_value_source'] == 'parcel').all()


def test_low_coverage_group_switches_wholesale_even_for_present_parcel_value():
    # Town B: only 2/8 eligible rows have a real parcel value (coverage <
    # 0.5) -- the whole town switches to NSI, discarding the 2 real parcel
    # values too.
    df = pd.DataFrame(
        {
            'admin4_id': ['B'] * 8,
            'structure_value': [100.0] * 2 + [None] * 6,
            'structure_value_building_nsi': [999.0] * 8,
            'priority_on_parcel': ['primary'] * 8,
        }
    )
    out = _call(df, min_group_size=1)
    assert out['structure_value'].eq(999.0).all()
    assert (out['structure_value_source'] == 'nsi').all()


def test_secondary_priority_rows_excluded_from_coverage_denominator():
    # 4 primary rows all with real parcel values (coverage 4/4 = 1.0) plus 4
    # secondary rows that structurally never get an apportioned parcel value.
    # Counting them in the denominator would read as 4/8 = 0.5-ish coverage;
    # they must not count at all -- true coverage among eligible rows is
    # 1.0, so the town stays on parcel, and the secondary rows switch with
    # it (they were already null either way).
    df = pd.DataFrame(
        {
            'admin4_id': ['C'] * 8,
            'structure_value': [100.0] * 4 + [None] * 4,
            'structure_value_building_nsi': [999.0] * 8,
            'priority_on_parcel': ['primary'] * 4 + ['secondary'] * 4,
        }
    )
    out = _call(df, min_group_size=1)
    assert out['structure_value'].iloc[:4].eq(100.0).all()
    assert pd.isna(out['structure_value'].iloc[4:]).all()
    assert (out['structure_value_source'] == 'parcel').all()


def test_small_group_falls_back_to_chunk_wide_coverage():
    # Town D has only 2 eligible rows, both missing a parcel value (0
    # coverage on its own) -- but the rest of the chunk (town E) is
    # well-covered, so with min_group_size=5, D's decision should follow
    # the chunk-wide coverage (parcel) rather than its own tiny sample.
    df = pd.DataFrame(
        {
            'admin4_id': ['D'] * 2 + ['E'] * 8,
            'structure_value': [None] * 2 + [100.0] * 8,
            'structure_value_building_nsi': [999.0] * 10,
            'priority_on_parcel': ['primary'] * 10,
        }
    )
    out = _call(df, min_group_size=5)
    assert (out['structure_value_source'].iloc[:2] == 'parcel').all()
    assert pd.isna(out['structure_value'].iloc[:2]).all()


def test_missing_source_columns_skip_without_raising():
    df = pd.DataFrame({'structure_value': [100.0]})
    out = _call(df)
    assert 'structure_value_source' not in out.columns


def test_admin_level_at_or_above_chunk_level_treats_whole_chunk_as_one_group():
    # admin_level=3 matches the chunk's own level (AdminId('US', 'MA', 'MI')
    # is level 3), so the admin4_id column present on the data must be
    # ignored -- the whole chunk is one group regardless of it.
    df = pd.DataFrame(
        {
            'admin4_id': ['A'] * 4 + ['B'] * 4,
            'structure_value': [100.0] * 6 + [None] * 2,
            'structure_value_building_nsi': [999.0] * 8,
            'priority_on_parcel': ['primary'] * 8,
        }
    )
    out = _call(df, admin_level=3, min_group_size=1)
    # Chunk-wide coverage = 6/8 = 0.75 >= 0.5 -> parcel everywhere.
    assert (out['structure_value_source'] == 'parcel').all()
    assert out['structure_value'].iloc[:6].eq(100.0).all()
