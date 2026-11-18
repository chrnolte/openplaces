"""Declared step columns resolved across the recipe tree (flow.lineage)."""

from openplaces.flow import lineage
from openplaces.flow.lineage import StepUse


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
        ('US_building-openplaces-2026', 'apportion_curated_values'): {
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
    assert [use.step for use in found] == ['apportion_curated_values', 'suppress_where']
    assert [use.step for use in found if 'structure_value' in use.decides] == [
        'apportion_curated_values'
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
    assert undeclared == {
        (parcel, 'declare_columns'),
        (parcel, 'merge_enrichments'),
        (parcel, 'apportion_common_land'),
        (parcel, 'impute_from_group_statistic'),
        (parcel, 'derive_indicators'),
        (parcel, 'resolve_by_vote'),
        (parcel, 'reconcile_land_use'),
        (parcel, 'cast_categoricals'),
        (parcel, 'order_columns'),
        (building, 'declare_columns'),
        (building, 'collect_link_ids'),
        (building, 'keep_curated_ids'),
        (building, 'merge_enrichments'),
        (building, 'derive_admin_attribute'),
        (building, 'resolve_by_vote'),
        (building, 'fill_missing_numeric'),
        (building, 'derive_indicators'),
        (building, 'impute_n_dwellings'),
        (building, 'impute_occupancy_type'),
        (building, 'resolve_occupancy'),
        (building, 'classify_manufactured_homes'),
        (building, 'derive_group_count'),
        (building, 'derive_group_rank'),
        (building, 'flag_manufactured_home_communities'),
        (building, 'flag_unconfirmed_height'),
        (building, 'report_lot_value_sharing'),
        (building, 'cast_categoricals'),
        (building, 'order_columns'),
    }


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
