"""A step name belongs to one function.

Registering a second function under a taken name used to replace the
first silently. On 2026-09-30 an admin geometry step registered as
'resolve_overlaps' displaced the footprint step of that name, and the
level-1 run then applied the footprint step to countries and dropped
46 of them. The registry now refuses the clash.
"""

import pytest

from openplaces.io.steps import make_register


def _first(state):
    return state


def _second(state):
    return state


def test_a_taken_name_is_refused():
    register = make_register({})
    register('step')(_first)

    with pytest.raises(ValueError, match="'step' is already registered"):
        register('step')(_second)


def test_the_same_function_may_register_again():
    registry = {}
    register = make_register(registry)
    register('step')(_first)
    register('step')(_first)

    assert registry['step'] is _first


@pytest.mark.parametrize(
    'stage', ['harmonizer', 'enricher', 'curator'], ids=lambda s: s
)
def test_every_stage_loads_its_steps_without_a_clash(stage):
    from importlib import import_module

    package = import_module(f'openplaces.io.{stage}')
    package._load_steps()
    assert package._STEP_REGISTRY


def test_the_admin_carve_and_the_footprint_overlap_step_are_distinct():
    from openplaces.io import harmonizer
    from openplaces.io.harmonizer import discover
    from openplaces.io.harmonizer.links import overlaps

    harmonizer._load_steps()
    registry = harmonizer._STEP_REGISTRY
    assert registry['carve_lumped_territories'] is discover.carve_lumped_territories
    assert registry['resolve_overlaps'] is overlaps.resolve_overlaps
