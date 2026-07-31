"""Tests for the `resolve_century` unary transform.

Expands a 2-digit year to 4 digits using the strptime('%y') pivot convention
(used, e.g., to fix up New Hanover County, NC's fixed-width `last_sale_year`
field).
"""

import pandas as pd

from openplaces.io.transform import apply_transformations


def _recipe(pivot=None):
    config = {
        'type': 'unary',
        'operation': 'resolve_century',
        'input': 'x',
        'output': 'x',
    }
    if pivot is not None:
        config['args'] = {'pivot': pivot}
    return {'transformations': [config]}


def test_low_two_digit_year_maps_to_2000s():
    df = pd.DataFrame({'x': ['06']})
    result = apply_transformations(df, _recipe())
    assert result['x'].tolist() == [2006]


def test_high_two_digit_year_maps_to_1900s():
    df = pd.DataFrame({'x': ['99']})
    result = apply_transformations(df, _recipe())
    assert result['x'].tolist() == [1999]


def test_pivot_boundary():
    df = pd.DataFrame({'x': ['68', '69']})
    result = apply_transformations(df, _recipe())
    assert result['x'].tolist() == [2068, 1969]


def test_already_four_digit_value_passes_through():
    df = pd.DataFrame({'x': ['1985']})
    result = apply_transformations(df, _recipe())
    assert result['x'].tolist() == [1985]


def test_custom_pivot():
    df = pd.DataFrame({'x': ['30', '31']})
    result = apply_transformations(df, _recipe(pivot=30))
    assert result['x'].tolist() == [2030, 1931]
