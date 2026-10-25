"""The CHEER Inventory v0 recipes translate NSI letter codes at ingest."""

from pathlib import Path

import pandas as pd
import pytest

import openplaces

RECIPES = Path(openplaces.__file__).parent / 'recipes'
FOLDERS = {
    'US-NC': RECIPES / 'US' / 'NC' / '_all' / 'building' / 'cheer' / 'v0',
    'US-TX': RECIPES / 'US' / 'TX' / '_all' / 'building' / 'cheer' / 'v0',
}


@pytest.mark.parametrize('column', ['construction-type', 'foundation-type'])
def test_both_states_carry_the_same_label_table(column):
    # Without the table, the ingester keeps the source's letter codes
    # (Harris and Jefferson shipped them in the TX bundle of 2026-10-15).
    tables = {
        state: pd.read_csv(folder / f'{state}_building-cheer-v0_{column}-labels.csv')
        for state, folder in FOLDERS.items()
    }
    assert tables['US-NC'].equals(tables['US-TX'])
    assert tables['US-TX'].iloc[:, 0].str.len().eq(1).all()
