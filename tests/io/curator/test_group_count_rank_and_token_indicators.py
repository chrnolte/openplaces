"""Tests for the parcel indicators and predicates of the second occupancy pass.

`derive_group_count` and `derive_group_rank` are groupbys on an id the
row carries; `has_token`, `none_of`, `numeric_below` and
`column_greater_than` are the voting predicates the pass reads them
with. All values are fabricated.
"""

from __future__ import annotations

import pandas as pd
import pytest

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.indicators import evaluate_indicator
from openplaces.io.curator.inferers import derive_group_count, derive_group_rank


def _state(frame: pd.DataFrame) -> CurateState:
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US'),
        verbose=False,
        timer=None,
        curated=frame,
    )


def test_has_token_matches_whole_parts_only():
    frame = pd.DataFrame(
        {
            'src': [
                'keyword_probability+morphology',
                'morphology+keyword',
                'nsi',
                None,
                'imputed_keyword',
            ]
        }
    )
    matched = evaluate_indicator(
        frame, {'type': 'has_token', 'column': 'src', 'values': ['keyword']}
    )
    assert matched.tolist() == [False, True, False, False, False]


def test_has_token_reads_a_categorical_column():
    frame = pd.DataFrame({'src': pd.Categorical(['fema+nsi', 'fema', None])})
    matched = evaluate_indicator(
        frame, {'type': 'has_token', 'column': 'src', 'values': ['nsi']}
    )
    assert matched.tolist() == [True, False, False]


def test_none_of_negates_and_holds_over_an_absent_column():
    frame = pd.DataFrame({'a': [1, 5, None]})
    indicator = {
        'type': 'none_of',
        'indicators': [{'type': 'numeric_at_least', 'column': 'a', 'min': 3}],
    }
    assert evaluate_indicator(frame, indicator).tolist() == [True, False, True]
    absent = {
        'type': 'none_of',
        'indicators': [{'type': 'equals', 'column': 'missing', 'value': 1}],
    }
    assert evaluate_indicator(frame, absent).all()


def test_numeric_below_is_strict():
    frame = pd.DataFrame({'area_m2': [39.9, 40.0, None]})
    matched = evaluate_indicator(
        frame, {'type': 'numeric_below', 'column': 'area_m2', 'max': 40}
    )
    assert matched.tolist() == [True, False, False]


def test_column_greater_than_is_false_on_missing():
    frame = pd.DataFrame({'rank': [3, 2, 1, None], 'units': [2, 2, None, 1]})
    matched = evaluate_indicator(
        frame, {'type': 'column_greater_than', 'column': 'rank', 'other': 'units'}
    )
    assert matched.tolist() == [True, False, False, False]


def test_group_count_counts_qualifying_rows_of_each_group():
    frame = pd.DataFrame(
        {
            'parcel_id': ['p1', 'p1', 'p1', 'p2', None],
            'priority_on_parcel': [
                'primary',
                'primary',
                'secondary',
                'primary',
                'primary',
            ],
            'occupancy_type': ['Manufactured Home'] * 2 + ['Single-Family'] * 3,
        }
    )
    state = derive_group_count(
        _state(frame),
        group_column='parcel_id',
        output='n_primary_mh',
        where=[
            {'type': 'equals', 'column': 'priority_on_parcel', 'value': 'primary'},
            {
                'type': 'equals',
                'column': 'occupancy_type',
                'value': 'Manufactured Home',
            },
        ],
    )
    out = state.curated['n_primary_mh']
    assert out.iloc[:4].tolist() == [2, 2, 2, 0]
    assert pd.isna(out.iloc[4])


def test_group_count_skips_an_absent_group_column():
    frame = pd.DataFrame({'a': [1]})
    state = derive_group_count(_state(frame), group_column='parcel_id', output='n')
    assert 'n' not in state.curated.columns


def test_group_rank_orders_by_value_then_id():
    frame = pd.DataFrame(
        {
            'parcel_id': ['p1', 'p1', 'p1', 'p1', 'p2', None],
            'area_m2': [30.0, 80.0, 30.0, 500.0, 10.0, 90.0],
            'occupancy_type': [
                'Manufactured Home',
                'Manufactured Home',
                'Manufactured Home',
                'Single-Family',
                'Manufactured Home',
                'Manufactured Home',
            ],
        },
        index=pd.Index(['f9', 'f5', 'f2', 'f1', 'f7', 'f8'], name='footprint_id'),
    )
    state = derive_group_rank(
        _state(frame),
        group_column='parcel_id',
        output='rank',
        rank_by='area_m2',
        where=[
            {'type': 'equals', 'column': 'occupancy_type', 'value': 'Manufactured Home'}
        ],
    )
    rank = state.curated['rank']
    # f5 is largest; f2 and f9 tie on area and break on id.
    assert rank.loc['f5'] == 1
    assert rank.loc['f2'] == 2
    assert rank.loc['f9'] == 3
    assert rank.loc['f7'] == 1
    # Not qualifying, or no group: no rank.
    assert pd.isna(rank.loc['f1'])
    assert pd.isna(rank.loc['f8'])


def test_group_rank_is_independent_of_row_order():
    frame = pd.DataFrame(
        {
            'parcel_id': ['p1'] * 3,
            'area_m2': [20.0, 20.0, 40.0],
        },
        index=pd.Index(['b', 'a', 'c']),
    )
    forward = derive_group_rank(
        _state(frame.copy()), group_column='parcel_id', output='r', rank_by='area_m2'
    ).curated['r']
    backward = derive_group_rank(
        _state(frame.iloc[::-1].copy()),
        group_column='parcel_id',
        output='r',
        rank_by='area_m2',
    ).curated['r']
    assert forward.to_dict() == backward.to_dict()
    assert forward.to_dict() == {'b': 3, 'a': 2, 'c': 1}


@pytest.mark.parametrize('missing', ['parcel_id', 'area_m2'])
def test_group_rank_skips_when_an_input_is_absent(missing):
    frame = pd.DataFrame({'parcel_id': ['p1'], 'area_m2': [1.0]}).drop(columns=missing)
    state = derive_group_rank(
        _state(frame), group_column='parcel_id', output='r', rank_by='area_m2'
    )
    assert 'r' not in state.curated.columns
