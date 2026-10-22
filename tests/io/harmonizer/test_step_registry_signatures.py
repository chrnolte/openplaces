"""Every registered harmonize step takes the pipeline state first.

The runner calls each step as ``fn(state, **params)``. A decorator left
on a helper above its step registers the helper under the step's name,
and the first run of that step fails (`link_entities_by_id`,
2026-10-18).
"""

import inspect

from openplaces.io.harmonizer import _STEP_REGISTRY, _load_steps


def test_every_registered_step_takes_state_first():
    _load_steps()
    wrong = {
        name: fn.__name__
        for name, fn in _STEP_REGISTRY.items()
        if next(iter(inspect.signature(fn).parameters), None) != 'state'
    }
    assert not wrong, f'steps whose first parameter is not state: {wrong}'
