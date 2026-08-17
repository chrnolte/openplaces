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
    # use_group is genuinely missing here (None), not empty -- falls back to
    # use_subgroup alone, no 'n/a' filler.
    assert combined[2] == 'Vacant'


def test_empty_strings_are_treated_as_missing():
    spine = pd.DataFrame(
        {
            'use_group': ['', 'residential', '  '],
            'use_subgroup': ['', '', 'single family'],
        }
    )
    state = attrs.derive_use_classes(_state(spine))
    combined = state.spine['use_group_combined']
    assert pd.isna(combined[0])  # both empty -> missing, not ' | '
    assert combined[1] == 'residential'  # subgroup empty -> group alone
    assert combined[2] == 'single family'  # group whitespace-only -> subgroup alone


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


def test_extra_column_extends_the_label():
    """A third source column joins the label without displacing the first two.

    Motivating case: a county whose land-use column is a land *segment*
    type carrying no occupancy signal, while its structure description
    names the building type.
    """
    spine = pd.DataFrame(
        {
            'use_group': ['HOMESITE', 'CROPLAND', 'LOT'],
            'use_subgroup': [None, None, None],
            'building_style': ['DOUBLE WIDE MOHO', None, 'RANCH'],
        }
    )
    out = (
        attrs.derive_use_classes(
            _state(spine), columns=['use_group', 'use_subgroup', 'building_style']
        )
        .spine['use_group_combined']
        .astype('string')
    )

    assert out.iloc[0] == 'HOMESITE | DOUBLE WIDE MOHO'
    assert out.iloc[1] == 'CROPLAND'  # no style -> land use alone
    assert out.iloc[2] == 'LOT | RANCH'


def test_extra_column_alone_still_labels():
    """A row with only the extra column populated is not dropped."""
    spine = pd.DataFrame(
        {
            'use_group': [None],
            'use_subgroup': [None],
            'building_style': ['DOUBLE WIDE MOHO'],
        }
    )
    out = (
        attrs.derive_use_classes(
            _state(spine), columns=['use_group', 'use_subgroup', 'building_style']
        )
        .spine['use_group_combined']
        .astype('string')
    )
    assert out.iloc[0] == 'DOUBLE WIDE MOHO'


def test_default_columns_unchanged_when_extra_absent():
    """Omitting `columns` keeps the historical two-column behaviour."""
    spine = pd.DataFrame(
        {'use_group': ['residential'], 'use_subgroup': ['single family']}
    )
    out = attrs.derive_use_classes(_state(spine)).spine['use_group_combined']
    assert out.astype('string').iloc[0] == 'residential | single family'
