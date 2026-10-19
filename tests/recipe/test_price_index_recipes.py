"""Published price-index reference tables: FHFA county HPI and BLS CPI-U.

Both recipes ingest an index as its agency publishes it. Their columns
have to be registry names, and their sources public-domain US
government work with the terms page recorded.
"""

import pytest

from openplaces.core.attribute_registry import load_registry
from openplaces.recipe import get_recipe_by_id

RECIPES = {
    'US_economy-price-house-fhfa-2026': {
        'county_fips',
        'year',
        'house_price_index',
        'house_price_index_base_2000',
        'house_price_index_annual_change',
    },
    'US_economy-price-house-fhfa-2026_state': {
        'year',
        'house_price_index',
        'house_price_index_base_2000',
        'house_price_index_annual_change',
    },
    'US_economy-price-consumer-bls-2026': {
        'series_id_bls',
        'year',
        'period_bls',
        'consumer_price_index',
    },
}


@pytest.mark.parametrize('recipe_id', sorted(RECIPES))
def test_columns_are_registered(recipe_id):
    recipe = get_recipe_by_id(recipe_id)
    columns = set(recipe['columns'])
    assert columns == RECIPES[recipe_id]
    assert columns <= set(load_registry().index)


@pytest.mark.parametrize('recipe_id', sorted(RECIPES))
def test_source_terms_are_recorded(recipe_id):
    source = get_recipe_by_id(recipe_id)['dataset'].source
    assert source.license.startswith('public-domain')
    assert source.terms_url
    assert source.redistribution_restricted is False


def test_cpi_keeps_only_the_all_items_city_average_series():
    recipe = get_recipe_by_id('US_economy-price-consumer-bls-2026')
    assert recipe['query'] == 'series_id_bls == "CUUR0000SA0"'


def test_fhfa_state_table_is_keyed_by_admin_id_and_saved_apart():
    recipe = get_recipe_by_id('US_economy-price-house-fhfa-2026_state')
    (prefix,) = [t for t in recipe['transformations'] if t['output'] == 'admin2_id']
    assert prefix['operation'] == 'add_prefix'
    assert prefix['args'] == {'prefix': 'US-'}
    assert recipe['save_to']['filename'] == 'state'


def test_fhfa_keeps_the_fips_code_as_text():
    recipe = get_recipe_by_id('US_economy-price-house-fhfa-2026')
    assert recipe['csv_dtype'] == {'FIPS code': 'str'}
