"""NCES EDGE school and school district recipes load and use the registry."""

import pytest

from openplaces.core.attribute_registry import get_attributes
from openplaces.recipe import get_recipe_by_id

RECIPE_IDS = [
    'US_built-school-nces-sy2425_district',
    'US_built-school-nces-sy2425_school',
]


@pytest.mark.parametrize('recipe_id', RECIPE_IDS)
def test_recipe_loads_with_registered_columns(recipe_id):
    recipe = get_recipe_by_id(recipe_id)
    registered = get_attributes().index
    assert set(recipe['columns']) <= set(registered)
    assert recipe['set_index'] in recipe['columns']


@pytest.mark.parametrize('recipe_id', RECIPE_IDS)
def test_recipe_records_public_domain_terms(recipe_id):
    source = get_recipe_by_id(recipe_id)['dataset'].source
    assert 'public-domain' in source.license
    assert source.terms_url
    assert source.redistribution_restricted is False


def test_school_and_district_share_the_nces_district_key():
    district, school = (get_recipe_by_id(r) for r in RECIPE_IDS)
    assert district['set_index'] == 'school_district_id_nces'
    assert 'school_district_id_nces' in school['columns']
