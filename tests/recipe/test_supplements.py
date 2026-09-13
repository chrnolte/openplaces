"""Recipes that declare `supplements:` against the committed recipe tree.

A supplement details another recipe's entities (an improvement-detail
table beside its appraisal roll). Spine discovery and single-recipe
resolution skip it; link_by_id still joins it. These tests pin the
committed declarations and the resolution that used to pick the wrong
table.
"""

import pytest

from openplaces.core.schema import ENTITY_TYPES
from openplaces.diagnostics import find_recipes
from openplaces.recipe import find_entity_recipe_id

TX_DETAIL_TABLES = [
    ('VIC', 'victoriacad'),
    ('HRD', 'hardincad'),
    ('LAV', 'lavacacad'),
]


@pytest.mark.parametrize('entity_type', sorted(ENTITY_TYPES))
def test_every_supplement_names_a_roll_of_the_same_entity_covering_it(entity_type):
    df = find_recipes(entity_type, stage='ingest')
    if df.empty:
        return
    by_id = df.set_index('recipe_id')
    for recipe_id, row in by_id[by_id['supplements'] != ''].iterrows():
        target = row['supplements']
        assert target in by_id.index, (
            f'{recipe_id} supplements {target}, which is not an ingest '
            f'recipe of entity type {entity_type!r}'
        )
        # A roll covers its supplement's scope: a county appraiser's
        # building table details the state roll's rows for that county.
        roll_admin = str(by_id.loc[target, 'admin_id'])
        assert str(row['admin_id']).startswith(roll_admin), (
            f'{recipe_id} ({row["admin_id"]}) supplements {target} '
            f'({roll_admin}), which does not cover it'
        )
        assert by_id.loc[target, 'supplements'] == '', (
            f'{recipe_id} supplements another supplement ({target})'
        )


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


def test_craven_building_cards_supplement_the_property_roll():
    # Craven's county CSV is its property roll (a parcel recipe until
    # 2026-09-12); the building cards detail its properties, and parcel
    # geometry stays with NC OneMap.
    roll = 'US-NC-CRA_property-cravencounty-2026'
    by_id = find_recipes('property', stage='ingest').set_index('recipe_id')
    assert by_id.loc[f'{roll}_building-cards', 'supplements'] == roll
    found = find_entity_recipe_id('US-NC-CRA', 'property', stage='ingest', silent=True)
    assert found == roll
    parcel = find_entity_recipe_id('US-NC-CRA', 'parcel', stage='ingest', silent=True)
    assert parcel == 'US-NC_parcel-nconemap-2025'


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
