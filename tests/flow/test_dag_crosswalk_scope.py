"""A statewide source gets no job for a county its crosswalk does not list.

Pennsylvania's PASDA parcel layer carries 38 of the state's 67 counties
and says which in a `recipe_id`-form `process_by.admin_id_crosswalk`.
Huntingdon County is not among them (it has its own county recipes). The
graph used to derive a PASDA ingest for it anyway, which raised in the
ingester and stopped the building-spine wave; the August run worked only
because the recipe was excluded by hand.
"""

from openplaces.flow.dag import RecipeDAG
from openplaces.io.transform import get_crosswalk
from openplaces.recipe import get_recipe_by_id

TARGET = 'US_footprint-cheer-2026'
PASDA = 'US-PA_parcel-pasda-2026'
UNCOVERED = 'US-PA-HU'


def _covered_county():
    recipe = get_recipe_by_id(PASDA)
    spec = dict(recipe['process_by']['admin_id_crosswalk'])
    spec['admin_id'] = 'US-PA'
    covered = sorted(str(u) for u in get_crosswalk(spec, flip=True).iloc[:, 0])
    assert UNCOVERED not in covered
    return covered[0]


def _pasda_admins(admin_id):
    dag = RecipeDAG(TARGET, admin_ids=[admin_id], deliver=False)
    return {n.admin_id for n in dag._nodes if n.recipe_id == PASDA}


def test_uncovered_county_gets_no_statewide_job():
    assert _pasda_admins(UNCOVERED) == set()


def test_covered_county_keeps_its_statewide_job():
    county = _covered_county()
    assert county in {a for a in _pasda_admins(county) if a}
