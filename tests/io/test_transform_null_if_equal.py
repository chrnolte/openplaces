"""Tests for the `null_if_equal` unary transform.

Nulls out a literal placeholder value (e.g. NC OneMap's `year_built`
field uses 0, not a genuine null, wherever a county never recorded a
construction year) while leaving every other value -- including
pre-existing NaNs -- untouched.
"""

import numpy as np
import pandas as pd

from openplaces.io.transform import apply_transformations


def _recipe(value):
    return {
        'transformations': [
            {
                'type': 'unary',
                'operation': 'null_if_equal',
                'input': 'x',
                'output': 'x',
                'args': {'value': value},
            }
        ]
    }


def test_matching_value_becomes_null():
    df = pd.DataFrame({'x': [0, 1964, 0, np.nan]})
    result = apply_transformations(df, _recipe(0))
    assert result['x'].isna().tolist() == [True, False, True, True]
    assert result['x'].dropna().tolist() == [1964]


def test_non_matching_values_pass_through_unchanged():
    df = pd.DataFrame({'x': [1900, 2020]})
    result = apply_transformations(df, _recipe(0))
    assert result['x'].tolist() == [1900, 2020]
