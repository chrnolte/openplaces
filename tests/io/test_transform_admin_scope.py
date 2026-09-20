"""Admin-scoped transformations (`admin_ids`) and the `set_null` op.

A statewide source can be wrong in one county only. A transformation
entry listing `admin_ids` runs on a chunk inside one of those units and
nowhere else, and is skipped with a warning where the chunk is coarser
than the scope. All values here are fabricated.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from openplaces.io.transform import apply_transformations

SCOPED_NULL = {
    'type': 'unary',
    'operation': 'set_null',
    'input': 'use_subgroup',
    'output': 'use_subgroup',
    'admin_ids': ['XX-AA-BBB'],
}


def _frame():
    return pd.DataFrame(
        {
            'use_subgroup': ['alpha', 'beta', 'gamma'],
            'land_value': [100, 200, 300],
            'improvement_value': [10.0, np.nan, 30.0],
        }
    )


@pytest.mark.parametrize('chunk', ['XX-AA-BBB', 'XX-AA-BBB-CC'])
def test_scoped_step_runs_inside_scope(chunk):
    out = apply_transformations(
        _frame(), {'transformations': [SCOPED_NULL]}, process_admin_id=chunk
    )
    assert out['use_subgroup'].isna().all()


def test_scoped_step_skips_other_unit_silently():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        out = apply_transformations(
            _frame(),
            {'transformations': [SCOPED_NULL]},
            process_admin_id='XX-AA-BBC',
        )
    assert out['use_subgroup'].tolist() == ['alpha', 'beta', 'gamma']


def test_prefix_sharing_unit_is_not_a_descendant():
    # 'XX-AA-BBBX' starts with the listed id's text but is a sibling.
    out = apply_transformations(
        _frame(),
        {'transformations': [SCOPED_NULL]},
        process_admin_id='XX-AA-BBBX',
    )
    assert out['use_subgroup'].notna().all()


@pytest.mark.parametrize('chunk', ['XX-AA', None])
def test_scoped_step_skips_with_warning_when_chunk_cannot_be_scoped(chunk):
    with pytest.warns(UserWarning, match='admin-scoped'):
        out = apply_transformations(
            _frame(), {'transformations': [SCOPED_NULL]}, process_admin_id=chunk
        )
    assert out['use_subgroup'].notna().all()


def test_unscoped_steps_still_run_everywhere():
    recipe = {
        'transformations': [
            SCOPED_NULL,
            {
                'type': 'unary',
                'operation': 'to_numeric',
                'input': 'land_value',
                'output': 'land_value',
            },
        ]
    }
    out = apply_transformations(_frame(), recipe, process_admin_id='XX-ZZ')
    assert out['use_subgroup'].notna().all()
    assert pd.api.types.is_numeric_dtype(out['land_value'])


def test_process_admin_id_defaults_to_admin_id():
    out = apply_transformations(
        _frame(), {'transformations': [SCOPED_NULL]}, admin_id='XX-AA-BBB'
    )
    assert out['use_subgroup'].isna().all()


def test_scoped_pattern_entry():
    recipe = {
        'transformation_patterns': [
            {
                'type': 'unary',
                'operation': 'set_null',
                'pattern': '{column}',
                'apply_to_columns': ['land_value', 'improvement_value'],
                'admin_ids': ['XX-AA-BBB'],
            }
        ]
    }
    inside = apply_transformations(_frame(), recipe, process_admin_id='XX-AA-BBB')
    outside = apply_transformations(_frame(), recipe, process_admin_id='XX-AA-CCC')
    assert inside[['land_value', 'improvement_value']].isna().all().all()
    assert outside['land_value'].notna().all()


def test_scoped_total_recomputed_from_parts():
    recipe = {
        'transformations': [
            {
                'type': 'aggregate',
                'operation': 'sum',
                'inputs': ['land_value', 'improvement_value'],
                'output': 'total_value',
                'admin_ids': ['XX-AA-BBB'],
            }
        ]
    }
    out = apply_transformations(_frame(), recipe, process_admin_id='XX-AA-BBB')
    assert out['total_value'].tolist() == [110.0, 200.0, 330.0]


@pytest.mark.parametrize(
    'values, expected_dtype',
    [
        (pd.Series([1, 2], dtype='int64'), 'Int64'),
        (pd.Series([True, False]), 'boolean'),
        (pd.Series([1.5, 2.5]), 'float64'),
        (pd.Series(['a', 'b'], dtype='string'), 'string'),
        (pd.Series([1, 2], dtype='Int64'), 'Int64'),
        (pd.Series(pd.to_datetime(['2020-01-01', '2021-01-01'])), None),
    ],
)
def test_set_null_keeps_a_dtype_that_can_hold_missing(values, expected_dtype):
    # None: the input's own dtype, whose resolution varies by pandas.
    expected_dtype = expected_dtype or str(values.dtype)
    df = pd.DataFrame({'x': values})
    config = {
        'type': 'unary',
        'operation': 'set_null',
        'input': 'x',
        'output': 'x',
    }
    out = apply_transformations(df, {'transformations': [config]})
    assert out['x'].isna().all()
    assert str(out['x'].dtype) == expected_dtype


def test_set_null_keeps_categories():
    df = pd.DataFrame({'x': pd.Categorical(['a', 'b', 'a'])})
    config = {'type': 'unary', 'operation': 'set_null', 'input': 'x', 'output': 'x'}
    out = apply_transformations(df, {'transformations': [config]})
    assert isinstance(out['x'].dtype, pd.CategoricalDtype)
    assert list(out['x'].cat.categories) == ['a', 'b']
    assert out['x'].isna().all()
