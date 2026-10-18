"""Harris County's two-character state class crosswalk.

The statewide TxGIO crosswalk keys on the first letter and reads both
F1 (commercial) and F2 (industrial) as "commercial and industrial". The
HCAD crosswalk splits them and repeats the statewide label for every
other code it names.
"""

import pandas as pd

import openplaces.io.harmonizer.links as links
from openplaces.io.harmonizer import HarmonizeState
from openplaces.path import recipe_path
from openplaces.recipe import get_recipe_by_id

HCAD = 'US-TX-HAR_property-hcad-2026'
TXGIO = 'US-TX_parcel-txgio-2025'


def _table(recipe_id):
    recipe = get_recipe_by_id(recipe_id)
    folder = recipe_path(recipe['admin_id'], recipe['entity'], as_dir=True)
    return pd.read_csv(folder / f'{recipe_id}_use-group-code-remap.csv', dtype=str)


def test_hcad_crosswalk_splits_commercial_from_industrial_over_statewide():
    codes = ['A1'] * 6 + ['F1', 'F2', 'X1', 'Z4']
    spine = pd.DataFrame(
        {
            'use_group_code': codes,
            'use_group': ['residential'] * 6
            + ['commercial', 'commercial', 'exempt', None],
            'use_subgroup': ['single family'] * 6
            + ['commercial and industrial'] * 2
            + ['totally exempt', None],
        },
        index=pd.Index([f'p{i}' for i in range(len(codes))], name='parcel_id'),
    )
    state = HarmonizeState(
        recipe={}, admin_id=None, verbose=False, timer=None, spine=spine
    )
    out = links._apply_remap_csvs(state, HCAD).spine.set_index('use_group_code')

    assert out.loc['F1', 'use_group'] == 'commercial'
    assert out.loc['F1', 'use_subgroup'] == 'commercial'
    assert out.loc['F2', 'use_group'] == 'industrial'
    assert out.loc['F2', 'use_subgroup'] == 'industrial and manufacturing'
    assert out.loc['X1', 'use_subgroup'] == 'totally exempt'
    # A code the HCAD table does not name keeps what it had.
    assert pd.isna(out.loc['Z4', 'use_group'])


def test_hcad_crosswalk_repeats_statewide_labels_except_category_f():
    hcad = _table(HCAD)
    state = _table(TXGIO)
    state = state[state['admin_id'].fillna('') == ''].set_index('use_group_code')

    assert hcad['use_group_code'].str.len().eq(2).all()
    assert hcad['use_group_code'].is_unique
    other = hcad[~hcad['use_group_code'].str.startswith('F')]
    letter = other['use_group_code'].str[0]
    for col in ('use_group', 'use_subgroup'):
        assert (other[col].to_numpy() == letter.map(state[col]).to_numpy()).all()
