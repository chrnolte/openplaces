"""A value outside plausibility bounds is unknown, not a figure.

NC OneMap computes improvement value as total minus land, which goes
negative wherever assessed land exceeds the taxable total: 6,845 of
Duplin County's 43,425 parcels, -984.6M in total. Shipped as-is, the
negatives surfaced downstream as a misleading "over-allocated" error in
the apportionment conservation check, which assumes non-negative
references.
"""

import numpy as np
import pandas as pd
import pytest

from openplaces.io.curator.reconcilers import null_out_of_range


class _State:
    def __init__(self, curated):
        self.curated = curated
        self.verbose = False


def _state():
    curated = pd.DataFrame(
        {
            'improvement_value': [-289_900.0, 0.0, 125_000.0, np.nan, -1.0],
            'improvement_value_source': ['nconemap'] * 5,
            'land_value': [349_400.0, 10.0, 20.0, 30.0, 40.0],
        },
        index=pd.Index(list('abcde'), name='parcel_id'),
    )
    return _State(curated)


def test_negative_values_are_nulled_and_lose_their_source():
    state = null_out_of_range(_state(), columns='improvement_value', minimum=0)

    out = state.curated
    assert out.loc['a', 'improvement_value'] is np.nan or np.isnan(
        out.loc['a', 'improvement_value']
    )
    assert np.isnan(out.loc['e', 'improvement_value'])
    assert pd.isna(out.loc['a', 'improvement_value_source'])
    assert pd.isna(out.loc['e', 'improvement_value_source'])


def test_in_range_values_and_their_source_are_untouched():
    state = null_out_of_range(_state(), columns='improvement_value', minimum=0)

    out = state.curated
    # Zero is inclusive: a parcel with no buildings holds zero, not unknown.
    assert out.loc['b', 'improvement_value'] == 0.0
    assert out.loc['c', 'improvement_value'] == 125_000.0
    assert out.loc['b', 'improvement_value_source'] == 'nconemap'
    assert out.loc['c', 'improvement_value_source'] == 'nconemap'
    # An already-missing value stays missing and keeps whatever it had.
    assert np.isnan(out.loc['d', 'improvement_value'])


def test_several_columns_and_an_absent_one():
    state = null_out_of_range(
        _state(), columns=['improvement_value', 'land_value', 'total_value'], minimum=0
    )

    out = state.curated
    assert out['land_value'].tolist() == [349_400.0, 10.0, 20.0, 30.0, 40.0]
    assert 'total_value' not in out.columns


def test_a_maximum_bound_works_too():
    state = null_out_of_range(_state(), columns='land_value', maximum=100.0)

    out = state.curated
    assert np.isnan(out.loc['a', 'land_value'])
    assert out.loc['b', 'land_value'] == 10.0


def test_no_bounds_is_a_recipe_error():
    with pytest.raises(ValueError, match='minimum, a maximum, or both'):
        null_out_of_range(_state(), columns='improvement_value')
