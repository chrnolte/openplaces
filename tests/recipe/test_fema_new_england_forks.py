"""Every New England state chunks FEMA's structures by town, spatially.

FEMA's FIPS column is a county code, and New England's level 3 is the
town, so the national recipe's FIPS crosswalk finds no town there.
Each of the six states has a fork that cuts the towns spatially.
"""

import pytest

from openplaces.recipe import get_recipe_by_id

NEW_ENGLAND = ['CT', 'MA', 'ME', 'NH', 'RI', 'VT']


@pytest.mark.parametrize('state', NEW_ENGLAND)
def test_the_state_fork_cuts_towns_spatially(state):
    recipe = get_recipe_by_id(f'US-{state}_footprint-fema-2023')
    national = get_recipe_by_id('US_footprint-fema-2023')

    assert recipe['process_by'] == {'admin_level': 3, 'use_spatial_mask': True}
    assert recipe.get('coverage') == 'complete'
    assert recipe['columns'] == national['columns']
    assert recipe['transformations'] == national['transformations']
