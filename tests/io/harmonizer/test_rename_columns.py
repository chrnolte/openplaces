from __future__ import annotations

import pandas as pd

from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer.attributes import rename_columns


def make_state(df: pd.DataFrame) -> HarmonizeState:
    return HarmonizeState(
        recipe={}, admin_id='US-MA-MI', verbose=False, timer=None, spine=df
    )


def test_rename_columns_renames_existing_column():
    df = pd.DataFrame({'address': ['1 SAMPLE AVE'], 'other': [1]})
    state = rename_columns(make_state(df), columns={'address': 'address_original'})

    assert list(state.spine.columns) == ['address_original', 'other']
    assert state.spine.loc[0, 'address_original'] == '1 SAMPLE AVE'


def test_rename_columns_skips_missing_column():
    df = pd.DataFrame({'other': [1]})
    state = rename_columns(make_state(df), columns={'address': 'address_original'})

    assert list(state.spine.columns) == ['other']


def test_rename_columns_spine_none_is_noop():
    state = HarmonizeState(
        recipe={}, admin_id='US-MA-MI', verbose=False, timer=None, spine=None
    )
    result = rename_columns(state, columns={'address': 'address_original'})
    assert result.spine is None
