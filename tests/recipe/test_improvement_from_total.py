"""Total minus land is the improvement only where a county opts in."""

from openplaces.core.schema import AdminId
from openplaces.recipe import apply_recipe_patches, get_recipe_by_id


def _step(admin):
    recipe, _ = apply_recipe_patches(
        get_recipe_by_id('US_parcel-openplaces-2026'), AdminId(admin)
    )
    (step,) = [s for s in recipe['pipeline'] if s['step'] == 'impute_land_value']
    return step


def test_northampton_takes_total_minus_land():
    assert _step('US-NC-NOR')['improvement_from_total'] is True


def test_other_counties_keep_the_default():
    assert not _step('US-NC-WAK').get('improvement_from_total', False)
