"""reconcile_addresses documents a curate registration it did not have."""

from __future__ import annotations


def test_reconcile_addresses_is_a_dispatchable_curate_step():
    """Its docstring promises the registration; the decorator was missing."""
    from openplaces.io.curator import _STEP_REGISTRY, _load_steps
    from openplaces.io.curator.reconcilers import reconcile_addresses

    _load_steps()
    assert _STEP_REGISTRY.get('reconcile_addresses') is reconcile_addresses


def test_registering_it_changes_no_shipping_recipe():
    """No shipping curate recipe declares the step, so nothing moves."""
    from pathlib import Path

    import yaml

    import openplaces

    recipes_dir = Path(openplaces.__file__).parent / 'recipes'
    declared = []
    for path in recipes_dir.rglob('*.yaml'):
        text = path.read_text(encoding='utf-8')
        if 'reconcile_addresses' not in text:
            continue
        recipe = yaml.safe_load(text) or {}
        if recipe.get('stage') != 'curate':
            continue
        steps = {s.get('step') for s in (recipe.get('pipeline') or []) if s}
        if 'reconcile_addresses' in steps:
            declared.append(path.name)
    assert declared == []
