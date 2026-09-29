"""Every curate step declares a phase, and a curate recipe's pipeline
reads as gather, reconcile, standardize, infer, format.

plans/core-schema-and-stage-contracts-review.md, section 2. The phase
vocabulary is core.constants.CURATE_PHASES; each step's phase is the
contract sentence it answers to, and a recipe whose phases never
decrease can be read top to bottom as one of each. The order check
starts as a warning that lists today's violations (none of the four
national recipes follows the order yet, measured 2026-09-28); it turns
into a refusal once the recipes are reordered against the oracle. A
step that must run out of order for a dependency reason declares
``phase_override: <reason>`` in the recipe and is not reported.
"""

import warnings

import pytest

from openplaces.core.constants import CURATE_PHASES
from openplaces.diagnostics import find_recipes
from openplaces.io import curator
from openplaces.recipe import get_recipe_by_id

ENTITY_TYPES = ['parcel', 'footprint', 'property', 'transaction']


def _curate_recipe_ids():
    ids = []
    for entity_type in ENTITY_TYPES:
        found = find_recipes(entity_type, stage='curate')
        ids.extend(found['recipe_id'].tolist())
    return sorted(set(ids))


def test_every_registered_curate_step_declares_a_phase():
    curator._load_steps()
    untagged = sorted(set(curator._STEP_REGISTRY) - set(curator._STEP_PHASES))
    assert not untagged, untagged
    bad = {n: p for n, p in curator._STEP_PHASES.items() if p not in CURATE_PHASES}
    assert not bad, bad


def test_a_phase_is_mandatory_for_a_new_curate_step():
    with pytest.raises(TypeError, match='requires a phase'):
        curator._register('fabricated_step_without_a_phase')(lambda state: state)
    assert 'fabricated_step_without_a_phase' not in curator._STEP_REGISTRY


def test_an_unknown_phase_is_refused():
    with pytest.raises(ValueError, match='phase must be one of'):
        curator._register('fabricated_step', phase='guess')(lambda state: state)


@pytest.mark.parametrize('recipe_id', _curate_recipe_ids())
def test_every_pipeline_step_is_registered_with_a_phase(recipe_id):
    recipe = get_recipe_by_id(recipe_id)
    for step_cfg in recipe.get('pipeline') or []:
        assert curator.step_phase(step_cfg['step']) in CURATE_PHASES


@pytest.mark.parametrize('recipe_id', _curate_recipe_ids())
def test_phase_order_violations_are_listed(recipe_id):
    """Warning level until the recipes are reordered; then an assertion."""
    violations = curator.phase_order_violations(get_recipe_by_id(recipe_id))
    if violations:
        warnings.warn(
            f'{recipe_id}: {len(violations)} step(s) out of phase order: '
            + '; '.join(violations),
            stacklevel=1,
        )
    assert isinstance(violations, list)


def test_phase_override_silences_a_reported_step():
    recipe = {
        'stage': 'curate',
        'pipeline': [
            {'step': 'derive_indicators'},
            {'step': 'merge_enrichments'},
        ],
    }
    assert curator.phase_order_violations(recipe) == [
        'merge_enrichments (gather) runs after infer steps'
    ]
    recipe['pipeline'][1]['phase_override'] = 'reads the vote, tested'
    assert curator.phase_order_violations(recipe) == []
