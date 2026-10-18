"""Texas state code crosswalks: two characters where the Comptroller does.

The statewide table names the categories the Comptroller's Property
Classification Guide (March 2026) defines at two characters (Exhibit 1,
and J1 to J9 in Exhibit 5) beside the one-letter rows, which stay as
the fallback for codes a district defines on its own. County copies
repeat the statewide table.
"""

from pathlib import Path

import pandas as pd

import openplaces

TX = Path(openplaces.__file__).parent / 'recipes' / 'US' / 'TX'
STATEWIDE = (
    TX / '_all/parcel/txgio/2025/US-TX_parcel-txgio-2025_use-group-code-remap.csv'
)
COMPTROLLER = {
    'C1',
    'C2',
    'D1',
    'D2',
    'F1',
    'F2',
    'G1',
    'G2',
    'G3',
    'L1',
    'L2',
    'M1',
    'M2',
    *(f'J{i}' for i in range(1, 10)),
}


def _unscoped():
    table = pd.read_csv(STATEWIDE, dtype=str)
    return table[table['admin_id'].fillna('') == ''].drop(columns='admin_id')


def test_statewide_two_character_keys_are_the_comptrollers():
    keys = _unscoped()['use_group_code']
    assert keys.is_unique
    assert set(keys[keys.str.len() == 2]) == COMPTROLLER
    assert set(keys[keys.str.len() == 1]) >= set('ABCDEFGJLMOX')


def test_commercial_and_industrial_are_split():
    table = _unscoped().set_index('use_group_code')
    assert table.loc['F1', 'use_subgroup'] == 'commercial'
    assert 'industrial' in table.loc['F2', 'use_subgroup']
    assert 'industrial' not in table.loc['F1', 'use_subgroup']


def test_cameron_stays_on_the_letter_labels_until_its_building_types_land():
    # Maintainer's decision 2026-10-18: Cameron's Office or Retail is to
    # come from its own building data, not from the cohort's mode.
    table = pd.read_csv(STATEWIDE, dtype=str)
    letters = _unscoped().set_index('use_group_code')
    held = table[table['admin_id'] == 'US-TX-CAM'].set_index('use_group_code')
    assert set(held.index) == COMPTROLLER
    for code, row in held.iterrows():
        assert row['use_subgroup'] == letters.loc[code[0], 'use_subgroup'], code


def test_county_copies_repeat_the_statewide_table():
    statewide = _unscoped().reset_index(drop=True)
    copies = [
        p
        for p in TX.glob('*/_all/*/*/*/*use-group-code-remap.csv')
        if p.parts[-6] not in ('_all', 'HAR')
    ]
    assert copies
    for path in copies:
        pd.testing.assert_frame_equal(
            pd.read_csv(path, dtype=str), statewide, obj=path.name
        )
