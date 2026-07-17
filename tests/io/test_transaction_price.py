import pandas as pd

from openplaces.io.transform import apply_legacy_columns, apply_transformations
from openplaces.recipe import get_recipe_by_id


def test_transaction_recipes_map_and_parse_price():
    for recipe_id, raw_value, expected in [
        ('US-MA_transaction-masslandrecords-v1', '820000.00', 820000.0),
        ('US-WI_transaction-widor-2026', '$99,350.00', 99350.0),
    ]:
        recipe = get_recipe_by_id(recipe_id)
        source_column = recipe['columns']['price_raw']
        df = pd.DataFrame({source_column: [raw_value]})
        df = df.rename(columns={source_column: 'price_raw'})

        result = apply_transformations(df, recipe)

        assert result.loc[0, 'price_raw'] == raw_value
        assert result.loc[0, 'price'] == expected


def test_parse_currency_handles_missing_invalid_and_parenthesized_values():
    recipe = {
        'transformations': [
            {
                'type': 'unary',
                'operation': 'parse_currency',
                'input': 'price_raw',
                'output': 'price',
            }
        ]
    }
    df = pd.DataFrame({'price_raw': ['$0.00', '', None, 'unknown', '($1,234.50)']})

    result = apply_transformations(df, recipe)

    assert result['price'].iloc[0] == 0.0
    assert result['price'].iloc[4] == -1234.5
    assert result['price'].iloc[1:4].isna().all()


def test_transaction_recipes_migrate_legacy_consideration_column():
    recipe = get_recipe_by_id('US-WI_transaction-widor-2026')
    legacy = pd.DataFrame({'consideration_raw': ['$10.00']})

    result = apply_legacy_columns(legacy, recipe)
    result = apply_transformations(result, recipe)

    assert 'consideration_raw' not in result
    assert result.loc[0, 'price_raw'] == '$10.00'
    assert result.loc[0, 'price'] == 10.0
