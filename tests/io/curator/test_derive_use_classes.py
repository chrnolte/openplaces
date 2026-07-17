"""Tests for the parcel-spine use-code derivation.

`derive_use_classes` builds the `use_group_combined` label from whatever
`use_group`/`use_subgroup` already reached the spine. Mapping a source's raw
use_group_code to that vocabulary is a separate, earlier step: an
auto-discovered `*-remap.csv` crosswalk applied by `link_by_id` (see
`openplaces.io.harmonizer.links._apply_remap_csvs` and
`tests/test_link_by_id_auto_discover.py`). The Massachusetts crosswalk encodes
the DOR property type classification codes (notably 103 = manufactured home,
the manufactured-home signal).
"""

import pandas as pd

import openplaces.io.harmonizer.attributes as attrs
from openplaces.io.harmonizer import HarmonizeState
from openplaces.recipe import get_recipe_by_id


def _state(spine):
    return HarmonizeState(
        recipe={}, admin_id='US-MA-MI', verbose=False, timer=None, spine=spine
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


def test_derive_use_classes_builds_combined():
    spine = pd.DataFrame(
        {
            'use_group': ['residential', 'residential', None],
            'use_subgroup': ['manufactured home', 'single family', None],
        }
    )
    state = attrs.derive_use_classes(_state(spine))
    out = state.spine

    assert out['use_group_combined'].iloc[0] == 'residential | manufactured home'
    assert out['use_group_combined'].iloc[1] == 'residential | single family'
    # Unmapped code -> n/a side, still a valid combined label.
    assert out['use_group_combined'].iloc[2] == 'n/a | n/a'


def test_derive_use_classes_noop_without_use_group_or_use_subgroup():
    # use_subgroup missing entirely -> no-op.
    spine = pd.DataFrame({'use_group': ['residential']})
    state = attrs.derive_use_classes(_state(spine))
    assert 'use_group_combined' not in state.spine.columns

    # Neither present -> no-op.
    spine2 = pd.DataFrame({'parcel_id_local': ['A']})
    state2 = attrs.derive_use_classes(_state(spine2))
    assert 'use_group_combined' not in state2.spine.columns
