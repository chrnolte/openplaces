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
