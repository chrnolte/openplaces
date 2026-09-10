"""Tests for the generic `normalize_id_column` curate step."""

from __future__ import annotations

import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.filters import normalize_id_column


def _state(df: pd.DataFrame) -> CurateState:
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US'),
        verbose=False,
        timer=None,
        curated=df,
    )


def test_punctuation_stripped():
    df = pd.DataFrame(
        {'parcel_id_assessor': ['11-20-26-0300-000-12800', '011728010000000800']}
    )
    out = normalize_id_column(_state(df), column='parcel_id_assessor').curated
    assert list(out['parcel_id_assessor']) == [
        '112026030000012800',
        '011728010000000800',
    ]


def test_dashed_and_bare_ids_match_once_normalized():
    # The same physical parcel, one row from a dashed-format year and one
    # from a bare-digit year -- distinct strings before normalizing.
    df = pd.DataFrame({'parcel_id_assessor': ['11-20-26-0300-000-12800']})
    dashed = normalize_id_column(_state(df), column='parcel_id_assessor').curated
    bare = normalize_id_column(
        _state(pd.DataFrame({'parcel_id_assessor': ['112026030000012800']})),
        column='parcel_id_assessor',
    ).curated
    assert dashed['parcel_id_assessor'].iloc[0] == bare['parcel_id_assessor'].iloc[0]


def test_missing_column_is_noop():
    df = pd.DataFrame({'value': [1.0, 2.0]})
    out = normalize_id_column(_state(df), column='parcel_id_assessor').curated
    assert list(out['value']) == [1.0, 2.0]
