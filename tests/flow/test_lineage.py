"""Declared step columns resolved across the recipe tree (flow.lineage)."""

from functools import cache
from types import SimpleNamespace

import pytest

from openplaces.core.attribute_registry import load_registry
from openplaces.flow import lineage
from openplaces.flow.lineage import StepUse
from openplaces.recipe import apply_recipe_patches


def test_every_declaration_resolves_in_every_recipe():
    # Raises on an unknown step name, on a template naming a parameter
    # the step no longer has, and on a mapping read without its part.
    steps = lineage.all_steps()
    assert steps
    assert {use.stage for use in steps} <= {'harmonize', 'enrich', 'curate'}


def test_declarations_resolve_with_a_county_patch_applied():
    # US-NC-PIT patches the building curate (no Overture heights).
    steps = lineage.all_steps('US-NC-PIT')
    assert lineage.repeated_deciders(steps) == {}


def test_no_recipe_decides_a_column_twice():
    assert lineage.repeated_deciders(lineage.all_steps()) == {}


def test_one_step_decides_n_stories_in_the_building_curate():
    # plans/n-stories-one-deciding-step.md: decide_stories is the only
    # declared writer of n_stories there, so nothing rewrites it later.
    recipe_id = 'US_building-openplaces-2026'
    building = [use for use in lineage.all_steps() if use.recipe_id == recipe_id]
    found = lineage.writers('n_stories', building)
    assert [use.step for use in found] == ['decide_stories']
    assert 'n_stories' in found[0].decides


#: The recipes a building's structure_value passes through
#: (plans/findings/architecture-review/structure-value-chain-map.md).
_VALUE_CHAIN_RECIPES = (
    'US_property-spine-2026',
    'US_parcel-geospine-2026',
    'US_parcel-spine-2026',
    'US_footprint-geospine-2026',
    'US_building-geospine-2026',
    'US_building-spine-2026',
    'US_parcel-openplaces-2026',
    'US_building-openplaces-2026',
    'US_footprint-cheer-2026',
)

#: The value columns of that chain, as the recipes name them.
_VALUE_COLUMNS = frozenset(
    {
        'land_value',
        'improvement_value',
        'total_value',
        'land_value_imputed',
        'improvement_value_imputed',
        'structure_value',
        'structure_value_zero_reason',
        'improvement_value_parcel',
        'land_value_parcel',
        'total_value_parcel',
        'n_dwellings_parcel',
    }
)


def _value_chain_steps():
    return [u for u in lineage.all_steps() if u.recipe_id in _VALUE_CHAIN_RECIPES]


def test_structure_value_chain_declares_its_deciding_steps():
    # plans/account-values-one-deciding-step.md, V1. A step that starts
    # deciding a value column, or stops, changes this map on purpose.
    deciders = {}
    for use in _value_chain_steps():
        decided = use.decides & _VALUE_COLUMNS
        if decided:
            deciders[(use.recipe_id, use.step)] = decided
    assert deciders == {
        ('US_parcel-geospine-2026', 'link_by_id'): {
            'land_value',
            'improvement_value',
            'total_value',
        },
        # V3: one step decides the account values on a lot.
        ('US_parcel-openplaces-2026', 'divide_account_values'): {
            'land_value',
            'improvement_value',
            'total_value',
        },
        ('US_parcel-openplaces-2026', 'impute_land_value'): {
            'land_value_imputed',
            'improvement_value_imputed',
        },
        ('US_building-openplaces-2026', 'split_curated_values'): {
            'structure_value',
            'improvement_value_parcel',
            'land_value',
            'land_value_parcel',
            'total_value_parcel',
            'n_dwellings_parcel',
        },
        ('US_building-openplaces-2026', 'derive_structure_value_zero_reason'): {
            'structure_value_zero_reason',
        },
    }


def test_structure_value_is_decided_once_in_the_building_curate():
    recipe_id = 'US_building-openplaces-2026'
    building = [use for use in _value_chain_steps() if use.recipe_id == recipe_id]
    found = lineage.writers('structure_value', building)
    assert [use.step for use in found] == [
        'split_curated_values',
        'empty_selected_cells',
    ]
    assert [use.step for use in found if 'structure_value' in use.decides] == [
        'split_curated_values'
    ]
    # The chain from the curated lot: the improvement value the split
    # divides is the one impute_land_value decides.
    parcel_id = 'US_parcel-openplaces-2026'
    parcel = [use for use in _value_chain_steps() if use.recipe_id == parcel_id]
    assert [
        use.step for use in lineage.writers('improvement_value_imputed', parcel)
    ] == ['impute_land_value']


def test_one_step_decides_the_account_values_on_a_lot():
    # plans/account-values-one-deciding-step.md, V3: in the parcel
    # curate divide_account_values is the only decider of the values
    # it divides; the other writers carry, fill or gate.
    parcel_id = 'US_parcel-openplaces-2026'
    parcel = [use for use in lineage.all_steps() if use.recipe_id == parcel_id]
    for column in (
        'land_value',
        'improvement_value',
        'total_value',
        'other_value',
        'n_dwellings',
    ):
        deciders = [use.step for use in parcel if column in use.decides]
        assert deciders == ['divide_account_values'], column


def test_the_parcel_geospine_takes_property_values_back_last():
    # plans/account-values-one-deciding-step.md, V4: the link passes
    # copy a roll's values in memory, and withhold_property_values,
    # the geospine's last writer of them, returns each lot to the
    # parcel layer's own statement. It decides nothing.
    geospine_id = 'US_parcel-geospine-2026'
    geospine = [use for use in lineage.all_steps() if use.recipe_id == geospine_id]
    for column in (
        'land_value',
        'improvement_value',
        'total_value',
        'n_dwellings',
        'tax_year',
        'building_style',
    ):
        found = lineage.writers(column, geospine)
        assert found[-1].step == 'withhold_property_values', column
        assert column not in found[-1].decides
    last = [use for use in geospine if use.step == 'withhold_property_values']
    assert 'n_dwellings_preliminary' in last[0].writes


def test_the_value_curates_add_no_undeclared_step_unseen():
    # Lineage cannot see what an undeclared step writes, so a new step
    # in the two curates that decide structure_value fails here until
    # it declares its columns or is added to this list as touching no
    # value column (plans/account-values-one-deciding-step.md, V1).
    parcel = 'US_parcel-openplaces-2026'
    building = 'US_building-openplaces-2026'
    undeclared = {
        (use.recipe_id, use.step)
        for use in _value_chain_steps()
        if not use.declared and use.recipe_id in (parcel, building)
    }
    assert undeclared == set()


#: The columns of the occupancy and dwelling chain
#: (plans/occupancy-and-dwellings-one-decider.md, O1).
_OCCUPANCY_COLUMNS = (
    'occupancy_type',
    'n_dwellings',
    'land_use_class',
    'group_parcel',
    'manufactured_home_community',
)


def _chain_writers(admin_id=None) -> dict:
    """(recipe, column) -> [(step, decides)] in the two curates."""
    curates = ('US_parcel-openplaces-2026', 'US_building-openplaces-2026')
    steps = lineage.all_steps(admin_id) if admin_id else lineage.all_steps()
    found = {}
    for recipe_id in curates:
        of_recipe = [use for use in steps if use.recipe_id == recipe_id]
        for column in _OCCUPANCY_COLUMNS:
            writers = lineage.writers(column, of_recipe)
            if writers:
                found[(recipe_id, column)] = [
                    (use.step, column in use.decides) for use in writers
                ]
    return found


def test_occupancy_chain_declares_its_writers_and_deciders():
    # O1: every step that writes a column of the chain declares it, and
    # a step decides a column only where it rebuilds every row of it.
    # The building class has one decider (the base class) and four
    # correcting writers; the dwelling count is decided before the
    # class. A step that starts or stops deciding one of these columns
    # changes this map on purpose. cast_categoricals declares every
    # registry categorical it may cast, present in the recipe or not.
    parcel = 'US_parcel-openplaces-2026'
    building = 'US_building-openplaces-2026'
    assert _chain_writers() == {
        (parcel, 'occupancy_type'): [('cast_categoricals', False)],
        (parcel, 'n_dwellings'): [
            ('aggregate_from_entities', False),
            ('divide_account_values', True),
        ],
        (parcel, 'land_use_class'): [
            ('resolve_by_vote', True),
            ('reconcile_land_use', False),
            ('cast_categoricals', False),
        ],
        (parcel, 'group_parcel'): [
            ('impute_from_group_statistic', True),
            ('cast_categoricals', False),
        ],
        (parcel, 'manufactured_home_community'): [('resolve_by_vote', True)],
        (building, 'occupancy_type'): [
            ('classify_occupancy_by_source_vote', True),
            ('override_occupancy_by_keywords', False),
            ('resolve_by_vote', False),
            ('resolve_by_vote', False),
            ('resolve_by_vote', False),
            ('cast_categoricals', False),
        ],
        (building, 'n_dwellings'): [
            ('decide_dwellings', True),
            ('round_to_integers', False),
        ],
        (building, 'land_use_class'): [('cast_categoricals', False)],
        (building, 'group_parcel'): [
            ('copy_from_curated_entity', False),
            ('cast_categoricals', False),
        ],
        (building, 'manufactured_home_community'): [
            ('copy_from_curated_entity', False),
            ('flag_manufactured_home_communities', True),
        ],
    }


@pytest.mark.parametrize(
    'admin_id', [None, 'US-MA-SOM', 'US-MA-CAM', 'US-WI-DA', 'US-NC-PIT']
)
def test_one_step_decides_the_building_dwelling_count(admin_id):
    # O2 to O4: the lanes write a preliminary count before the class,
    # and decide_dwellings, after the last class writer, is the only
    # decider of n_dwellings, with the state patches applied too.
    building = 'US_building-openplaces-2026'
    steps = lineage.all_steps(admin_id) if admin_id else lineage.all_steps()
    uses = [use for use in steps if use.recipe_id == building]
    deciders = [use for use in uses if 'n_dwellings' in use.decides]
    assert [use.step for use in deciders] == ['decide_dwellings']
    class_writers = [use.position for use in lineage.writers('occupancy_type', uses)]
    # A reviewed correction may set any cell, and the format cast
    # changes the class's dtype, not a value: neither is a class
    # writer here.
    corrections = [
        use.position
        for use in uses
        if use.step in ('apply_reviewed_corrections', 'cast_categoricals')
    ]
    last_class_writer = max(p for p in class_writers if p not in corrections)
    assert deciders[0].position > last_class_writer
    preliminary = [use.step for use in uses if 'n_dwellings_preliminary' in use.decides]
    assert preliminary == ['pick_first_value']


def test_a_reviewed_correction_is_listed_as_writing_the_chain():
    # Cambridge's sidecar names the columns it sets, so the step
    # declares that it may write any column and decides none.
    writers = _chain_writers('US-MA-CAM')
    building = 'US_building-openplaces-2026'
    for column in ('occupancy_type', 'n_dwellings'):
        assert ('apply_reviewed_corrections', False) in writers[(building, column)]


@cache
def _every_pipeline() -> tuple:
    """Every recipe's steps, unpatched and as each patch scope runs it."""
    return tuple(lineage.all_steps() + lineage.patched_steps())


#: Registry columns with more than one deciding step for one entity
#: type in recipes some unit runs together, with those steps and why
#: they are allowed for now
#: (plans/one-decider-per-column-test.md). Only shrinks: an entry that
#: is no longer needed fails the test as a new violation does.
_MULTI_DECIDER_ALLOWLIST = {
    ('parcel', column): (
        frozenset(
            {
                'US_parcel-geospine-2026:link_by_id',
                'US_parcel-openplaces-2026:divide_account_values',
            }
        ),
        'V1 to V4: the roll pass decides the harmonized value, the curate '
        'divides it per lot; one has to become a named second pass',
    )
    for column in ('improvement_value', 'land_value', 'total_value')
}


def test_one_decider_per_registry_column_and_entity():
    # Target architecture 2.1: a delivered registry column has one
    # deciding step per entity. Recipes no unit runs together (US and
    # CO) are not compared.
    found = lineage.entity_deciders(_every_pipeline(), load_registry().index)
    allowed = {key: steps for key, (steps, _) in _MULTI_DECIDER_ALLOWLIST.items()}
    new = {key: found[key] for key in found.keys() - allowed.keys()}
    resolved = sorted(allowed.keys() - found.keys())
    changed = {
        key: found[key]
        for key in found.keys() & allowed.keys()
        if found[key] != allowed[key]
    }
    assert not new, f'new columns with several deciders: {new}'
    assert not resolved, f'resolved, remove from the allowlist: {resolved}'
    assert not changed, f'deciders changed, update the allowlist: {changed}'


#: Steps used by a recipe (patches included) that declare no columns,
#: and why. Only shrinks: a step that starts declaring, or leaves every
#: recipe, fails the test until its entry is removed.
_UNDECLARED_STEP_ALLOWLIST = {
    ('harmonize', 'merge_global'): 'the spine becomes the sources tables',
    ('harmonize', 'union_spine_sources'): 'the spine becomes the sources tables',
    ('harmonize', 'load_geospine'): 'the spine becomes the geospine table',
    ('harmonize', 'assign_entity_ids'): 'merges rows; a source column on any column',
    ('harmonize', 'append_last_sales'): 'appends rows carrying the roll columns',
    ('harmonize', 'aggregate_multi_parcel_sales'): 'rewrites folded rows whole',
    ('harmonize', 'assign_transaction_ids'): 'writes the index only',
    ('harmonize', 'link_geographic_ids'): 'nested output columns not checked yet',
    ('harmonize', 'carve_lumped_territories'): 'edits geometry, drops rows',
    ('harmonize', 'resolve_overlaps'): 'rewrites geometry; not checked yet',
    ('enrich', 'enrich_footprints_from_reference_buildings'): (
        'without columns, every reference column'
    ),
    ('enrich', 'overlay_reference_polygons'): 'year columns and suffix not checked',
    ('enrich', 'enrich_parcels_from_reference_crosswalk'): (
        'names come from a column-notes file'
    ),
}


def test_every_used_step_declares_its_columns():
    # A step that declares nothing is invisible to lineage: it may
    # write a decided column and no test would see it.
    found = {(use.stage, use.step) for use in _every_pipeline() if not use.declared}
    new = sorted(found - _UNDECLARED_STEP_ALLOWLIST.keys())
    resolved = sorted(_UNDECLARED_STEP_ALLOWLIST.keys() - found)
    assert not new, f'undeclared steps in use: {new}'
    assert not resolved, (
        f'declared or unused now, remove from the allowlist: {resolved}'
    )


#: Patch recipes whose anchor names no step of the pipeline they
#: patch, or which insert an unregistered step. Only shrinks.
_PATCH_ERROR_ALLOWLIST: dict[str, str] = {}


def test_patch_anchors_name_existing_steps():
    # Without this a broken anchor fails only when the unit runs.
    errors = lineage.patch_errors()
    assert lineage.patch_recipes()
    new = {k: v for k, v in errors.items() if k not in _PATCH_ERROR_ALLOWLIST}
    resolved = sorted(_PATCH_ERROR_ALLOWLIST.keys() - errors.keys())
    assert not new, f'patches that do not apply: {new}'
    assert not resolved, f'patches apply now, remove from the allowlist: {resolved}'


def test_patch_errors_names_a_missing_anchor(monkeypatch):
    patch = {
        'recipe_id': 'fabricated',
        'patches': 'US_building-openplaces-2026',
        'admin_id': 'US-MA-SOM',
        'stage': 'curate',
        'entity': SimpleNamespace(entity_type='building'),
        'pipeline_patch': [{'remove': 'no_such_step'}],
    }

    def apply_fabricated(recipe, admin_id):
        return apply_recipe_patches(recipe, admin_id, patches=[patch])

    monkeypatch.setattr(lineage, 'patch_recipes', lambda: {'fabricated': patch})
    monkeypatch.setattr(lineage, 'apply_recipe_patches', apply_fabricated)
    errors = lineage.patch_errors()
    assert list(errors) == ['fabricated']
    assert 'no_such_step' in errors['fabricated']


def test_entity_deciders_compare_only_overlapping_scopes():
    def decider(recipe_id, step):
        return StepUse(
            recipe_id,
            'harmonize',
            1,
            step,
            True,
            writes=frozenset({'x'}),
            decides=frozenset({'x'}),
            entity_type='parcel',
        )

    us = decider('US_parcel-spine-2026', 'a')
    co = decider('CO_parcel-spine-2026', 'b')
    florida = decider('US-FL_parcel-roll-2026', 'c')
    assert lineage.entity_deciders([us, co], ['x']) == {}
    assert lineage.entity_deciders([us, florida, co], ['x']) == {
        ('parcel', 'x'): frozenset(
            {
                'US_parcel-spine-2026:a',
                'US-FL_parcel-roll-2026:c',
                'CO_parcel-spine-2026:b',
            }
        )
    }
    assert lineage.entity_deciders([us, florida], ['y']) == {}


def test_reconcile_attributes_declares_its_evidence_columns():
    # The step names its columns after the link the recipe builds and
    # the spine's entity type; the declaration reads both off the
    # recipe. A point reference writes suffixed evidence, a polygon
    # overlay also its two relational counts, which it decides.
    steps = [use for use in lineage.all_steps() if use.step == 'add_source_evidence']
    by_recipe = {}
    for use in steps:
        by_recipe.setdefault(use.recipe_id, set()).update(use.writes)
    building = by_recipe['US_building-spine-2026']
    assert {
        'occupancy_type_building_nsi',
        'occupancy_type_building_nsi_all',
        'n_buildings_nsi',
        'n_dwellings_overture',
        'n_dwellings_nsi',
        'use_group_combined_parcel',
        'n_parcels_per_building',
        'n_buildings_per_parcel',
        'parcel_id_local',
    } <= building
    parcel = by_recipe['US_parcel-spine-2026']
    assert {
        'address_street_dwelling_overture',
        'group_building_nsi',
        'occupancy_type_footprint_fema',
        'height_footprint_fema',
        'n_footprints_fema_per_parcel',
    } <= parcel
    assert 'height_footprint_fema_all' not in parcel
    decided = {c for use in steps for c in use.decides}
    assert decided == {
        'n_parcels_per_building',
        'n_buildings_per_parcel',
        'n_footprints_fema_per_parcel',
        'n_parcels_per_footprint_fema',
    }


def test_a_dropped_column_is_not_written_and_says_where_it_leaves():
    # order_columns removes scratch columns; that is no write, so it
    # follows no decider, and lineage still names where they leave.
    building = 'US_building-openplaces-2026'
    uses = [use for use in lineage.all_steps() if use.recipe_id == building]
    scratch = 'p_manufactured_home'
    assert [use.step for use in lineage.droppers(scratch, uses)] == ['order_columns']
    assert 'order_columns' not in [use.step for use in lineage.writers(scratch, uses)]
    later = lineage.writes_after_decider(_every_pipeline(), load_registry().index)
    assert not [found for found in later if found[3] == 'order_columns']


def test_keep_registered_columns_drops_by_its_rule():
    # The allow-list's removals are stated as a rule over the
    # registry: an unregistered header leaves, a registered attribute
    # and a column the recipe keeps by name stay.
    property_curate = 'US_property-openplaces-2026'
    uses = [use for use in lineage.all_steps() if use.recipe_id == property_curate]
    (keep,) = [use for use in uses if use.step == 'keep_registered_columns']
    assert keep.declared and not keep.writes
    assert [name for name, _ in keep.drop_rules] == ['unregistered or withheld']
    assert lineage.droppers('fabricated_raw_header', [keep]) == [keep]
    assert lineage.droppers('year_built', [keep]) == []
    assert lineage.droppers('construction_class', [keep]) == []


def test_droppers_match_names_patterns_and_rules():
    steps = [
        StepUse('r', 'curate', 1, 'a', True, drops=frozenset({'x', 'tmp_*'})),
        StepUse(
            'r',
            'curate',
            2,
            'b',
            True,
            drop_rules=(('starts with z', lambda column: column.startswith('z')),),
        ),
        StepUse('r', 'curate', 3, 'c', True, writes=frozenset({'x'})),
    ]
    assert [use.step for use in lineage.droppers('x', steps)] == ['a']
    assert [use.step for use in lineage.droppers('tmp_1', steps)] == ['a']
    assert [use.step for use in lineage.droppers('zeta', steps)] == ['b']
    assert lineage.droppers('y', steps) == []
    assert [use.step for use in lineage.writers('x', steps)] == ['c']


def test_writers_readers_and_repeated_deciders():
    steps = [
        StepUse('r1', 'curate', 1, 'a', True, writes=frozenset({'x'})),
        StepUse(
            'r1',
            'curate',
            2,
            'b',
            True,
            reads=frozenset({'x'}),
            writes=frozenset({'y', 'z_*'}),
            decides=frozenset({'y'}),
        ),
        StepUse(
            'r2',
            'curate',
            1,
            'c',
            True,
            writes=frozenset({'y'}),
            decides=frozenset({'y'}),
            reads_from=(('r1', frozenset({'y'})),),
        ),
        StepUse(
            'r2',
            'curate',
            2,
            'd',
            True,
            writes=frozenset({'y'}),
            decides=frozenset({'y'}),
        ),
        StepUse('r2', 'curate', 3, 'e', False),
    ]
    assert [use.step for use in lineage.writers('x', steps)] == ['a']
    assert [use.step for use in lineage.readers('x', steps)] == ['b']
    assert [use.step for use in lineage.writers('z_roll', steps)] == ['b']
    assert [use.step for use in lineage.readers('y', steps)] == ['c']
    assert lineage.repeated_deciders(steps) == {('r2', 'y'): [1, 2]}
