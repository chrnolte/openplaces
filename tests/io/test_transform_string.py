"""Tests for string transformations, focused on row-wise `concat`.

The `concat` string op assembles a value from several columns row by row (used,
e.g., to build the New Hanover County, NC parcel id from its map/block/lot
fixed-width fields). A regression guard: an earlier implementation joined whole
Series instead of per-row values.
"""

import warnings

import pandas as pd
import pytest

from openplaces.io.transform import apply_transformations


def test_concat_joins_per_row_with_separator():
    recipe = {
        'transformations': [
            {
                'type': 'string',
                'operation': 'concat',
                'inputs': ['a', 'b', 'c'],
                'args': {'sep': '-'},
                'output': 'joined',
            }
        ]
    }
    df = pd.DataFrame({'a': ['R1', 'R2'], 'b': ['001', '002'], 'c': ['000', '009']})

    result = apply_transformations(df, recipe)

    assert result['joined'].tolist() == ['R1-001-000', 'R2-002-009']


def test_concat_chained_builds_dashed_parcel_id():
    # Mirrors the parcel recipe: first concat with no separator, then a dashed
    # concat that consumes its own output.
    recipe = {
        'transformations': [
            {
                'type': 'string',
                'operation': 'concat',
                'inputs': ['parcel_type', 'parcel_mapnumber'],
                'output': 'parcel_id',
            },
            {
                'type': 'string',
                'operation': 'concat',
                'inputs': [
                    'parcel_id',
                    'parcel_blocknumber',
                    'parcel_lotnumber',
                    'parcel_sublotnumber',
                ],
                'args': {'sep': '-'},
                'output': 'parcel_id',
            },
        ]
    }
    df = pd.DataFrame(
        {
            'parcel_type': ['R'],
            'parcel_mapnumber': ['06116'],
            'parcel_blocknumber': ['006'],
            'parcel_lotnumber': ['001'],
            'parcel_sublotnumber': ['000'],
        }
    )

    result = apply_transformations(df, recipe)

    assert result.loc[0, 'parcel_id'] == 'R06116-006-001-000'


def test_concat_treats_nan_as_empty():
    # A blank continuation line (NaN) must not contribute the literal 'nan'.
    recipe = {
        'transformations': [
            {
                'type': 'string',
                'operation': 'concat',
                'inputs': ['legal1', 'legal2'],
                'output': 'legal_description',
            }
        ]
    }
    df = pd.DataFrame(
        {
            'legal1': ['NORTH COUNTY SQ', '80 20 CAROLINA BEACH'],
            'legal2': ['UARE', pd.NA],
        }
    )

    result = apply_transformations(df, recipe)

    assert result['legal_description'].tolist() == [
        'NORTH COUNTY SQUARE',
        '80 20 CAROLINA BEACH',
    ]


def test_to_datetime_parses_and_coerces():
    # Used to normalize a text date column to datetime; a leaked header label
    # ('SALE DATE') coerces to NaT rather than poisoning the column dtype.
    recipe = {
        'transformations': [
            {
                'type': 'unary',
                'operation': 'to_datetime',
                'input': 'recorded_date',
                'output': 'recorded_date',
            }
        ]
    }
    df = pd.DataFrame({'recorded_date': ['2022-07-08 00:00:00', 'SALE DATE']})

    result = apply_transformations(df, recipe)

    assert str(result['recorded_date'].dtype).startswith('datetime64')
    assert result['recorded_date'].iloc[0] == pd.Timestamp('2022-07-08')
    assert pd.isna(result['recorded_date'].iloc[1])


def test_lstrip_strips_leading_zeros():
    # Used to clean zero-padded fixed-width street numbers (e.g. '001705').
    recipe = {
        'transformations': [
            {
                'type': 'string',
                'operation': 'lstrip',
                'input': 'address_number',
                'args': {'chars': '0'},
                'output': 'address_number',
            }
        ]
    }
    df = pd.DataFrame({'address_number': ['001705', '000416', '000000']})

    result = apply_transformations(df, recipe)

    assert result['address_number'].tolist() == ['1705', '416', '']


def test_in_place_transform_does_not_warn():
    # An output column that is also the transformation's own input (a strip/
    # cast refining a column in place, or a concat chain consuming its own
    # prior output) is the standard cleaning idiom, not an accidental
    # collision, and must not raise the 'already exists' warning.
    recipe = {
        'transformations': [
            {
                'type': 'string',
                'operation': 'strip',
                'input': 'address_number',
                'output': 'address_number',
            }
        ]
    }
    df = pd.DataFrame({'address_number': [' 1705 ', ' 416 ']})

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = apply_transformations(df, recipe)

    assert result['address_number'].tolist() == ['1705', '416']


def test_unrelated_column_overwrite_still_warns():
    # A transformation whose output collides with an existing, unrelated
    # column (not one of its own inputs) is a genuine accidental overwrite and
    # should still be flagged.
    recipe = {
        'transformations': [
            {
                'type': 'string',
                'operation': 'strip',
                'input': 'raw_value',
                'output': 'other_column',
            }
        ]
    }
    df = pd.DataFrame({'raw_value': [' 1705 '], 'other_column': ['placeholder']})

    with pytest.warns(UserWarning, match="Column 'other_column' already exists"):
        apply_transformations(df, recipe)
