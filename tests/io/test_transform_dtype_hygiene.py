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


def _postal_code_recipe():
    # The pair of steps a recipe declares for a ZIP stored in a numeric
    # field (the Florida parcel and roll recipes).
    return {
        'transformations': [
            {
                'type': 'unary',
                'operation': 'null_if_equal',
                'input': 'postal_code',
                'args': {'value': 0},
                'output': 'postal_code',
            },
            {
                'type': 'string',
                'operation': 'zfill',
                'input': 'postal_code',
                'args': {'width': 5},
                'output': 'postal_code',
            },
        ]
    }


def test_a_float_postal_code_becomes_a_five_digit_code():
    # 0 is the field's "no ZIP"; a blank made the column float; 2134 is
    # a ZIP whose leading zero the numeric field dropped.
    df = pd.DataFrame({'postal_code': [32801.0, 0.0, np.nan, 2134.0]})

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = apply_transformations(df, _postal_code_recipe())

    codes = result['postal_code']
    assert codes[0] == '32801'
    assert pd.isna(codes[1])
    assert pd.isna(codes[2])
    assert codes[3] == '02134'


def test_a_text_postal_code_keeps_its_leading_zero():
    df = pd.DataFrame({'postal_code': ['02134', '32801', None]})

    result = apply_transformations(df, _postal_code_recipe())

    assert result['postal_code'][0] == '02134'
    assert result['postal_code'][1] == '32801'
    assert pd.isna(result['postal_code'][2])


@pytest.mark.parametrize(
    'recipe_id, values',
    [
        # A numeric geodatabase field: floats, 0 for "no ZIP".
        ('US-FL_parcel-floridagio-2026', [32801.0, 0.0, np.nan, 2134.0]),
        ('US-FL_parcel-floridagio-2025', [32801.0, 0.0, np.nan, 2134.0]),
        # A CSV the recipe reads as text (see the csv_dtype test below).
        ('US-FL_property-fldor-2026', ['32801', '0', None, '2134']),
    ],
)
@pytest.mark.parametrize('column', ['postal_code', 'owner_postal_code'])
def test_florida_recipes_ingest_postal_codes_as_codes(recipe_id, values, column):
    # Both sources store the ZIP as a number. Ingested as a float, it
    # reached address assembly as '32801.0' and came out as '328010'.
    from openplaces.recipe.loading import get_recipe_by_id

    recipe = get_recipe_by_id(recipe_id)
    steps = [t for t in recipe['transformations'] if t.get('output') == column]
    df = pd.DataFrame({column: values})

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = apply_transformations(df, {'transformations': steps})

    codes = result[column]
    assert codes[0] == '32801'
    assert pd.isna(codes[1])
    assert pd.isna(codes[2])
    assert codes[3] == '02134'


def test_florida_roll_reads_its_postal_code_fields_as_text():
    from openplaces.recipe.loading import get_recipe_by_id

    recipe = get_recipe_by_id('US-FL_property-fldor-2026')

    for column in ('postal_code', 'owner_postal_code'):
        assert recipe['csv_dtype'][recipe['columns'][column]] == 'str'


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
