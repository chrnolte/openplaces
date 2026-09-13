"""Recipes that declare `supplements:` against the committed recipe tree.

A supplement details another recipe's entities (an improvement-detail
table beside its appraisal roll). Spine discovery and single-recipe
resolution skip it; link_by_id still joins it. These tests pin the
committed declarations and the resolution that used to pick the wrong
table.
"""

import pytest

from openplaces.core.schema import ENTITY_TYPES, admin_scope_covers
from openplaces.diagnostics import find_recipes
from openplaces.recipe import (
    find_entity_recipe_id,
    get_recipe_by_id,
    get_recipe_dependencies,
    get_supplemented_table,
)

MASSGIS = 'US-MA_parcel-massgis-2025'
MA_CITY_TABLES = [
    ('BOS', 'US-MA-BOS_property-bostongov-2026'),
    ('CAM', 'US-MA-CAM_property-cambridgema-2026'),
    ('ARL', 'US-MA-ARL_property-arlingtonma-2026'),
    ('SOM', 'US-MA-SOM_property-somervillema-2022'),
    ('FAR', 'US-MA-FAR_property-fallriverma-2023'),
]


@pytest.mark.parametrize(('town', 'recipe_id'), MA_CITY_TABLES)
def test_massachusetts_city_tables_supplement_the_massgis_property_layer(
    town, recipe_id
):
    recipe = get_recipe_by_id(recipe_id)
    assert recipe['supplements'] == MASSGIS
    assert str(get_supplemented_table(recipe)['entity']) == 'property-massgis-2025'
    # A property key: on a parcel key every condo unit of a building
    # would receive the whole building's rooms.
    assert recipe['supplements_key'] == 'property_id_assessor'
    # Never "the" property recipe of its town.
    assert (
        find_entity_recipe_id(f'US-MA-{town}', 'property', stage='ingest', silent=True)
        != recipe_id
    )

    by_step: dict[str, set] = {}
    edges = get_recipe_dependencies('US_property-spine-2026', admin_id=f'US-MA-{town}')
    for edge in edges:
        by_step.setdefault(edge.step, set()).add(edge.upstream_recipe_id)
    assert MASSGIS in by_step['union_spine_sources']
    assert recipe_id not in by_step['union_spine_sources']
    assert recipe_id in by_step['link_by_id']


TX_DETAIL_TABLES = [
    ('VIC', 'victoriacad'),
    ('HAN', 'hardincad'),
    ('LAV', 'lavacacad'),
]


@pytest.mark.parametrize('entity_type', sorted(ENTITY_TYPES))
def test_every_supplement_names_a_roll_of_the_same_entity_within_scope(entity_type):
    # The roll may be a recipe or a host's additional layer, and its
    # scope may contain the supplement's (a city table detailing a
    # statewide layer). get_supplemented_table raises on anything
    # else: an unknown layer, another supplement, a scope outside.
    df = find_recipes(entity_type, stage='ingest')
    if df.empty:
        return
    for recipe_id in df.loc[df['supplements'] != '', 'recipe_id']:
        recipe = get_recipe_by_id(recipe_id)
        table = get_supplemented_table(recipe)
        assert table['stage'] == 'ingest', recipe_id
        assert str(table['entity'].entity_type) == entity_type, recipe_id
        assert admin_scope_covers(table['admin_id'], recipe['admin_id']), recipe_id


@pytest.mark.parametrize(('county', 'source_id'), TX_DETAIL_TABLES)
def test_texas_improvement_detail_tables_are_supplements(county, source_id):
    roll = f'US-TX-{county}_property-{source_id}-2026'
    by_id = find_recipes('property', stage='ingest').set_index('recipe_id')
    assert by_id.loc[f'{roll}_improvement-detail', 'supplements'] == roll


@pytest.mark.parametrize(('county', 'source_id'), TX_DETAIL_TABLES)
def test_single_best_property_recipe_is_the_roll(county, source_id):
    # The detail table used to win the tie-break on its longer filename.
    found = find_entity_recipe_id(
        f'US-TX-{county}', 'property', stage='ingest', silent=True
    )
    assert found == f'US-TX-{county}_property-{source_id}-2026'


def test_a_supplement_is_still_found_by_its_filename():
    # `filename` is the recipe id's trailing `_{filename}` part.
    found = find_entity_recipe_id(
        'US-TX-VIC',
        'property',
        stage='ingest',
        filename='improvement-detail',
        silent=True,
    )
    assert found == 'US-TX-VIC_property-victoriacad-2026_improvement-detail'


def test_the_property_spine_joins_supplements_as_columns_only():
    # Skipped by the union (no rows), joined by link_by_id (columns), and
    # with no record-count column: the property is where these attributes
    # live, once.
    from openplaces.recipe import get_recipe_by_id

    steps = get_recipe_by_id('US_property-spine-2026')['pipeline']
    union = next(i for i, s in enumerate(steps) if s['step'] == 'union_spine_sources')
    join = next(
        i
        for i, s in enumerate(steps)
        if s['step'] == 'link_by_id' and s.get('supplements_only')
    )
    assert join > union
    assert steps[join]['entity_type'] == 'property'
    assert steps[join]['count_as'] is False
