"""Tests for the derive_use_classes harmonize step (use_group_combined)."""

import pandas as pd

from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer.attributes import derive_use_classes


def _state(spine):
    return HarmonizeState(
        recipe={'recipe_id': 'US_parcel-spine-2026'},
        admin_id=None,
        verbose=False,
        timer=None,
        spine=spine,
    )


def test_combines_both_columns():
    spine = pd.DataFrame(
        {
            'use_group': ['Residential', None],
            'use_subgroup': ['Single Family', 'Vacant'],
        }
    )
    state = derive_use_classes(_state(spine))
    combined = state.spine['use_group_combined'].astype(str)
    assert combined[0] == 'Residential | Single Family'
    assert combined[1] == 'n/a | Vacant'


def test_falls_back_to_use_group_only():
    spine = pd.DataFrame({'use_group': ['Residential', 'Commercial']})
    state = derive_use_classes(_state(spine))
    assert list(state.spine['use_group_combined']) == ['Residential', 'Commercial']


def test_falls_back_to_use_subgroup_only():
    spine = pd.DataFrame({'use_subgroup': ['Single Family', 'Retail']})
    state = derive_use_classes(_state(spine))
    assert list(state.spine['use_group_combined']) == ['Single Family', 'Retail']


def test_noop_when_neither_present():
    spine = pd.DataFrame({'parcel_id': ['P1', 'P2']})
    state = derive_use_classes(_state(spine))
    assert 'use_group_combined' not in state.spine.columns


def test_noop_when_spine_is_none():
    state = derive_use_classes(_state(None))
    assert state.spine is None
