"""The MA occupancy-group table replaces the cohort fill in Massachusetts."""

from pathlib import Path

import pytest

from openplaces.core.schema import AdminId
from openplaces.io.curator.translation import first_match, load_translation_table
from openplaces.path import recipe_path
from openplaces.recipe import apply_recipe_patches, get_recipe_by_id

RID = 'US_parcel-openplaces-2026'


def _step(admin):
    recipe, _ = apply_recipe_patches(get_recipe_by_id(RID), AdminId(admin))
    names = [s['step'] for s in recipe['pipeline']]
    assert 'impute_from_group_statistic' not in names
    return next(
        s
        for s in recipe['pipeline']
        if s['step'] == 'translate_descriptions' and s['output'] == 'group_parcel'
    )


def _table(suffix):
    recipe = get_recipe_by_id(RID)
    path = Path(
        recipe_path(recipe['admin_id'], recipe['entity'], filename=f'{suffix}.csv')
    )
    return load_translation_table(str(path), 'group_parcel')


def _group(table, code):
    rule = first_match(code, table)
    return None if rule is None else rule.target


@pytest.mark.parametrize(
    ('admin', 'suffix'),
    [('US-MA-SOM', 'group-parcel-ma'), ('US-MA-BOS', 'group-parcel-ma-bos')],
)
def test_massachusetts_reads_its_own_table(admin, suffix):
    assert _step(admin)['keywords'] == suffix


def test_other_states_keep_the_cohort_fill():
    recipe, _ = apply_recipe_patches(get_recipe_by_id(RID), AdminId('US-NC-NHA'))
    assert 'impute_from_group_statistic' in [s['step'] for s in recipe['pipeline']]


def test_codes_translate_from_their_definitions():
    state = _table('group-parcel-ma')
    assert _group(state, '970C') == 'Multi Family'  # housing authority
    assert _group(state, '1124') == 'Multi Family'  # apartments
    assert _group(state, '330') == 'Retail'  # auto sales, not an office
    assert _group(state, '931') == 'Government Services'  # improved municipal
    assert _group(state, '922') == 'Government Services'  # state police etc.
    assert _group(state, '930') is None  # vacant municipal land
    boston = _table('group-parcel-ma-bos')
    assert _group(boston, '970') == 'Church'
    assert _group(boston, '908') == 'Multi Family'
