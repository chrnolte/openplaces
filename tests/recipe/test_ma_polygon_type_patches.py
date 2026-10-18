"""MassGIS polygon type: the US-MA spine and curate patches apply in MA only."""

from openplaces.core.attribute_registry import get_attributes
from openplaces.core.schema import AdminId
from openplaces.recipe import apply_recipe_patches, get_recipe_by_id


def _steps(recipe_id, admin):
    recipe, _ = apply_recipe_patches(get_recipe_by_id(recipe_id), AdminId(admin))
    return recipe['pipeline']


def test_spine_links_polygon_type_after_the_checkpoint_in_massachusetts():
    steps = _steps('US_parcel-spine-2026', 'US-MA-WIN')
    names = [s['step'] for s in steps]
    link = next(s for s in steps if s.get('columns') == ['polygon_type'])
    i = steps.index(link)
    assert names[i + 1] == 'estimate_property_counts'
    checkpoint = max(j for j, s in enumerate(steps) if s.get('checkpoint'))
    assert i > checkpoint
    assert link['spine_key'] == link['ref_key'] == 'parcel_id_admin2'


def test_curate_blanks_land_use_after_the_vote_in_massachusetts_only():
    steps = _steps('US_parcel-openplaces-2026', 'US-MA-WIN')
    names = [s['step'] for s in steps]
    i = names.index('reconcile_land_use')
    assert names[i + 1] == 'suppress_where'
    assert steps[i + 1]['column'] == ['land_use_class']
    other = [s['step'] for s in _steps('US_parcel-openplaces-2026', 'US-NC-NHA')]
    assert other[other.index('reconcile_land_use') + 1] != 'suppress_where'


def test_polygon_type_is_registered():
    assert 'polygon_type' in get_attributes('parcel').index
