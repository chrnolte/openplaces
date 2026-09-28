"""The `coalesce` string operation.

A source that splits one concept over mutually exclusive columns needs
them read as one: a Philippine city carries either a province code or,
in Metro Manila, a region code, and the unused one arrives as the JSON
literal `false` rather than as null.
"""

import pandas as pd
import pytest

from openplaces.io.transform import apply_transformation


def _coalesce(frame, inputs, output='parent', **args):
    config = {
        'output': output,
        'type': 'string',
        'operation': 'coalesce',
        'inputs': inputs,
    }
    if args:
        config['args'] = args
    return apply_transformation(frame, config, silent=True)[output]


def test_the_first_present_value_wins():
    frame = pd.DataFrame({'a': ['x', None, None], 'b': ['y', 'y', None]})
    assert list(_coalesce(frame, ['a', 'b'])) == ['x', 'y', pd.NA]


def test_a_blank_string_counts_as_absent():
    frame = pd.DataFrame({'a': ['', '   ', 'x'], 'b': ['y', 'y', 'y']})
    assert list(_coalesce(frame, ['a', 'b'])) == ['y', 'y', 'x']


def test_a_declared_sentinel_counts_as_absent():
    """The case this exists for: JSON `false` meaning 'not applicable'."""
    frame = pd.DataFrame(
        {'province': [False, '060400000'], 'region': ['130000000', '060000000']}
    )
    parent = _coalesce(frame, ['province', 'region'], empty_values=[False])
    assert list(parent) == ['130000000', '060400000']


def test_without_the_sentinel_the_false_would_have_won():
    """Pins why `empty_values` is needed rather than relying on nulls."""
    frame = pd.DataFrame(
        {'province': [False, '060400000'], 'region': ['130000000', '060000000']}
    )
    parent = _coalesce(frame, ['province', 'region'])
    assert parent.iloc[0] == 'False'


def test_an_id_column_read_as_float_does_not_gain_a_decimal_point():
    """Shares `_to_string_series`, so '001' does not become '1.0'."""
    frame = pd.DataFrame({'a': [None, 604.0], 'b': [1300.0, None]})
    assert list(_coalesce(frame, ['a', 'b'])) == ['1300', '604']


def test_a_zero_row_frame_yields_a_zero_row_column():
    frame = pd.DataFrame({'a': pd.Series([], dtype='object'), 'b': []})
    assert len(_coalesce(frame, ['a', 'b'])) == 0


def test_one_missing_input_column_is_an_error_naming_the_operation():
    """All inputs absent is skipped upstream; some absent is a recipe bug."""
    frame = pd.DataFrame({'a': ['x']})
    with pytest.raises(RuntimeError, match='Missing columns for coalesce'):
        _coalesce(frame, ['a', 'absent'])
