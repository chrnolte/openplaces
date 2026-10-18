"""The `group` transformation: one value chosen across a key's rows."""

import pandas as pd
import pytest

from openplaces.io.transform import apply_transformation

STEP = {
    'type': 'group',
    'operation': 'weighted_mode',
    'inputs': ['kind', 'value'],
    'args': {'by': 'account'},
    'output': 'kind',
}


def _apply(rows):
    df = pd.DataFrame(rows, columns=['account', 'kind', 'value'])
    return apply_transformation(df, STEP, silent=True)['kind'].tolist()


def test_the_kind_holding_most_value_wins_over_the_first_row():
    # A complex listed clubhouse first: its apartments hold the value.
    out = _apply(
        [
            ['A', 'club', '200'],
            ['A', 'apartments', '900'],
            ['A', 'apartments', '800'],
            ['B', 'office', '50'],
        ]
    )
    assert out == ['apartments', 'apartments', 'apartments', 'office']


def test_weights_sum_per_kind_before_comparing():
    out = _apply(
        [
            ['A', 'warehouse', '400'],
            ['A', 'office', '300'],
            ['A', 'office', '300'],
        ]
    )
    assert set(out) == {'office'}


def test_no_weight_counts_each_row_once_and_ties_keep_file_order():
    assert set(_apply([['A', 'x', None], ['A', 'y', ''], ['A', 'y', None]])) == {'y'}
    assert set(_apply([['A', 'x', '5'], ['A', 'y', '5']])) == {'x'}


def test_a_row_without_a_kind_gets_its_group_value_and_no_key_gets_none():
    out = _apply([['A', None, '10'], ['A', 'shop', '1'], [None, 'shop', '9']])
    assert out[:2] == ['shop', 'shop']
    assert pd.isna(out[2])


def test_unknown_group_operation_raises():
    df = pd.DataFrame({'account': ['A'], 'kind': ['x'], 'value': [1]})
    with pytest.raises(RuntimeError, match='Unknown group operation'):
        apply_transformation(df, {**STEP, 'operation': 'median'}, silent=True)
