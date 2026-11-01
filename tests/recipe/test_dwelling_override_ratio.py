"""The building spine keeps a large NSI footprint primary from 2x.

North Carolina and the Florida counties where the large footprint
matched the roll's living area on under half the parcels keep 3x;
their patches repeat the whole thresholds mapping.
"""

import pytest

from openplaces.core.schema import AdminId
from openplaces.recipe import apply_recipe_patches, get_recipe_by_id


def _thresholds(admin):
    recipe, _ = apply_recipe_patches(
        get_recipe_by_id('US_building-spine-2026'), AdminId(admin)
    )
    (step,) = [
        s for s in recipe['pipeline'] if s['step'] == 'classify_footprint_priority'
    ]
    return step['thresholds']


@pytest.mark.parametrize('admin', ['US-TX-HAR', 'US-FL-PO', 'US-MA-BOS'])
def test_national_override_is_two(admin):
    assert _thresholds(admin)['dwelling_override_ratio'] == 2.0


@pytest.mark.parametrize('admin', ['US-NC-WAK', 'US-FL-LA', 'US-FL-SU'])
def test_patched_units_keep_three_and_the_sibling_rule(admin):
    thresholds = _thresholds(admin)
    assert thresholds['dwelling_override_ratio'] == 3.0
    assert thresholds['building_point_sibling_ratio'] == 0.5
