"""Dtype and not-a-time hygiene across the transformation ops.

Every op here has to survive the four shapes a partition's column really
arrives in: object, float64 (an integer column that blanks turned into
floats), nullable Int64, and datetimes carrying NaT. The failures these
guard were silent value corruption ('001.0', '2020.0-4.0') or a raised
exception that took the whole partition down.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from openplaces.io.transform import (
    DATETIME_OPS,
    _apply_remap_conditional,
    _resolve_century,
    apply_transformations,
)


def _string_recipe(operation, **args):
    return {
        'transformations': [
            {
                'type': 'string',
                'operation': operation,
                'input': 'parcel_number',
                'output': 'parcel_id_local',
                'args': args,
            }
        ]
    }


@pytest.mark.parametrize(
    'values, dtype',
    [
        (['1', '2'], object),
        ([1.0, 2.0], 'float64'),
        ([1, 2], 'Int64'),
        ([1, 2], 'int64'),
    ],
)
def test_zfill_never_renders_a_float_point(values, dtype):
    df = pd.DataFrame({'parcel_number': pd.Series(values, dtype=dtype)})

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = apply_transformations(df, _string_recipe('zfill', width=3))

    assert result['parcel_id_local'].tolist() == ['001', '002']


def test_add_prefix_leaves_a_missing_value_missing():
    # 'P-nan' is a wrong id, not a missing one: it matches nothing and
    # reads as present.
    df = pd.DataFrame({'parcel_number': pd.Series([1.0, np.nan])})

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = apply_transformations(df, _string_recipe('add_prefix', prefix='P-'))

    assert result['parcel_id_local'][0] == 'P-1'
    assert pd.isna(result['parcel_id_local'][1])


def test_string_conversion_warning_respects_silent():
    df = pd.DataFrame({'parcel_number': pd.Series([1.0, 2.0])})

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        apply_transformations(df, _string_recipe('zfill', width=3), silent=True)


def test_concat_on_an_empty_frame_returns_a_column():
    recipe = {
        'transformations': [
            {
                'type': 'string',
                'operation': 'concat',
                'inputs': ['map', 'lot'],
                'args': {'sep': '-'},
                'output': 'parcel_id_local',
            }
        ]
    }
    df = pd.DataFrame({'map': pd.Series([], dtype=object), 'lot': []})

    result = apply_transformations(df, recipe)

    assert result['parcel_id_local'].tolist() == []


def test_concat_of_a_float_column_drops_the_float_point():
    recipe = {
        'transformations': [
            {
                'type': 'string',
                'operation': 'concat',
                'inputs': ['map', 'lot'],
                'args': {'sep': '-'},
                'output': 'parcel_id_local',
            }
        ]
    }
    df = pd.DataFrame({'map': ['R1', 'R2'], 'lot': pd.Series([7.0, np.nan])})

    result = apply_transformations(df, recipe)

    assert result['parcel_id_local'].tolist() == ['R1-7', 'R2-']


@pytest.mark.parametrize('dtype', ['Int64', 'float64', object])
def test_resolve_century_handles_missing_values(dtype):
    values = pd.Series([5, 99, None, 1994], dtype=dtype)

    result = _resolve_century(values)

    assert result[0] == 2005
    assert result[1] == 1999
    assert pd.isna(result[2])
    assert result[3] == 1994


def _datetimes():
    return pd.Series(pd.to_datetime(['2020-04-17', None]))


def test_year_month_formats_year_and_month():
    result = DATETIME_OPS['year_month'](_datetimes())

    assert result[0] == '2020-04'
    assert pd.isna(result[1])


def test_year_quarter_is_stable_when_a_partition_holds_nat():
    with_nat = DATETIME_OPS['year_quarter'](_datetimes())
    without_nat = DATETIME_OPS['year_quarter'](
        pd.Series(pd.to_datetime(['2020-04-17']))
    )

    assert with_nat[0] == '2020-2'
    assert with_nat[0] == without_nat[0]
    assert pd.isna(with_nat[1])


def test_datetime_op_coerces_an_unparsable_value_instead_of_failing():
    recipe = {
        'transformations': [
            {
                'type': 'datetime',
                'operation': 'year',
                'input': 'sale_date',
                'output': 'sale_year',
            }
        ]
    }
    df = pd.DataFrame({'sale_date': ['2020-04-17', 'SALE DATE']})

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = apply_transformations(df, recipe)

    assert result['sale_year'][0] == 2020
    assert pd.isna(result['sale_year'][1])


def test_datetime_conversion_warning_respects_silent():
    recipe = {
        'transformations': [
            {
                'type': 'datetime',
                'operation': 'year',
                'input': 'sale_date',
                'output': 'sale_year',
            }
        ]
    }
    df = pd.DataFrame({'sale_date': ['2020-04-17']})

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        apply_transformations(df, recipe, silent=True)


def test_remap_conditional_holds_string_outputs_without_a_default():
    series = pd.Series(['MH-01', 'SF-02'])
    conditions = [{'condition': 'startswith', 'value': 'MH', 'output': 'manufactured'}]

    result = _apply_remap_conditional(series, conditions)

    assert result[0] == 'manufactured'
    assert pd.isna(result[1])
