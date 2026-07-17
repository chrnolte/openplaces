"""Tests for the derive_use_classes harmonize step (use_group_combined)."""

import pandas as pd

import openplaces.io.harmonizer.attributes as attrs
from openplaces.io.harmonizer import HarmonizeState
from openplaces.recipe import get_recipe_by_id


def _state(spine):
    return HarmonizeState(
        recipe={'recipe_id': 'US_parcel-spine-2026'},
        admin_id='US-MA-MI',
        verbose=False,
        timer=None,
        spine=spine,
    )


def test_ma_use_code_remap_103_is_manufactured_home():
    table = get_recipe_by_id(
        'US-MA_parcel-massgis-2025_use-group-code-remap', dtype=str
    )
    by_code = table.set_index(table.columns[0])
    assert by_code.loc['103', 'use_group'] == 'residential'
    # Case-insensitive 'MOBILE|MANUFACTURED' keyword vote depends on this.
    assert 'manufactured' in by_code.loc['103', 'use_subgroup'].lower()
    assert by_code.loc['101', 'use_subgroup'].lower().startswith('single')


def test_combines_both_columns():
    spine = pd.DataFrame(
        {
            'use_group': ['residential', 'residential', None],
            'use_subgroup': ['manufactured home', 'single family', 'Vacant'],
        }
    )
    state = attrs.derive_use_classes(_state(spine))
    combined = state.spine['use_group_combined'].astype(str)
    assert combined[0] == 'residential | manufactured home'
    assert combined[1] == 'residential | single family'
    assert combined[2] == 'n/a | Vacant'


def test_falls_back_to_use_group_only():
    spine = pd.DataFrame({'use_group': ['residential', 'commercial']})
    state = attrs.derive_use_classes(_state(spine))
    assert list(state.spine['use_group_combined']) == ['residential', 'commercial']


def test_falls_back_to_use_subgroup_only():
    spine = pd.DataFrame({'use_subgroup': ['Single Family', 'Retail']})
    state = attrs.derive_use_classes(_state(spine))
    assert list(state.spine['use_group_combined']) == ['Single Family', 'Retail']


def test_noop_when_neither_present():
    spine = pd.DataFrame({'parcel_id_local': ['A']})
    state = attrs.derive_use_classes(_state(spine))
    assert 'use_group_combined' not in state.spine.columns


def test_noop_when_spine_is_none():
    state = attrs.derive_use_classes(_state(None))
    assert state.spine is None
