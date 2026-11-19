"""A 0 improvement the account division placed stays a stated 0.

`divide_account_values` marks a lot whose 0 is where its accounts put
it (`on_other_lots_flag`), and `impute_land_value` keeps such a 0 where
a total-only roll's divided total and the layer's land cannot confirm
it (plans/account-values-one-deciding-step.md, V3 deviation 5).

All values are fabricated.
"""

import numpy as np
import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.account_values import on_other_lots_flag
from openplaces.io.curator.land_value import impute_land_value


def _contributions():
    return pd.DataFrame(
        {
            # A: one account on lots 1 and 2, its improvement all on 1.
            # B: one account on lot 3 that states 0 itself.
            # C: lot 4 shares account A's 0 with D, which puts 50 there.
            # E: lot 5 is reached by an account stating no improvement.
            'account': ['A', 'A', 'B', 'A2', 'A2', 'D', 'E'],
            'lot': ['1', '2', '3', '4', '6', '4', '5'],
            'improvement_value': [100.0, 0.0, 0.0, 0.0, 70.0, 50.0, np.nan],
        }
    )


def test_a_zero_whose_account_put_its_value_elsewhere_is_marked():
    flag = on_other_lots_flag(_contributions(), 'improvement_value')

    assert bool(flag['2'])
    assert not bool(flag['1'])


def test_an_account_stating_zero_itself_is_left_to_the_readers_test():
    flag = on_other_lots_flag(_contributions(), 'improvement_value')

    assert not bool(flag['3'])


def test_a_lot_another_account_puts_value_on_is_not_marked():
    flag = on_other_lots_flag(_contributions(), 'improvement_value')

    assert not bool(flag['4'])


def test_an_account_stating_nothing_marks_nothing():
    flag = on_other_lots_flag(_contributions(), 'improvement_value')

    assert not bool(flag['5'])


def _state(df):
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US'),
        verbose=False,
        timer=None,
        curated=df,
    )


def _lots():
    # A total-only roll over a statewide layer: the lot holds the
    # layer's land and the roll's total by area, which exceeds it.
    base = {
        'land_value': 10_000.0,
        'improvement_value': 0.0,
        'total_value': 40_000.0,
        'footprint_area_m2_in_parcel': np.nan,
        'area_ha': 1.0,
        'land_use_class': 'Single-Family',
        'use_group_combined': None,
        'address_street': None,
        'city': None,
    }
    df = pd.DataFrame([base, base])
    df['improvement_value_source'] = 'roll'
    df['improvement_value_on_other_lots'] = [True, False]
    return df


def test_a_placed_zero_stays_a_stated_zero():
    out = impute_land_value(
        _state(_lots()),
        min_group_size=1,
        stated_zero_column='improvement_value_on_other_lots',
    ).curated

    assert out['improvement_value_imputed'].iloc[0] == 0
    # An unmarked 0 under a larger total states nothing, as before.
    assert pd.isna(out['improvement_value_imputed'].iloc[1])


def test_without_the_column_the_total_test_decides_alone():
    out = impute_land_value(_state(_lots()), min_group_size=1).curated

    assert out['improvement_value_imputed'].isna().all()
