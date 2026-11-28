"""A renamed step keeps its old name working, with a warning.

The ten steps of the documented facts' chains were renamed on
2026-11-27 (plans/step-naming-rules.md). Each old name stays
registered as deprecated: a recipe that still writes it runs the same
function, warns once per process, and the timer carries the new name.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pandas as pd
import pytest

import openplaces.io.curator as cur
from openplaces.core.schema import AdminId
from openplaces.io import steps
from openplaces.io.steps import (
    DeprecatedStepNameWarning,
    current_step_name,
    find_step,
    make_register,
)

RENAMED = {
    'reconcile_attributes': ('harmonizer', 'add_source_evidence'),
    'declare_columns': ('curator', 'add_empty_columns'),
    'suppress_where': ('curator', 'empty_selected_cells'),
    'reconcile_values': ('curator', 'pick_first_value'),
    'impute_occupancy_type': ('curator', 'classify_occupancy_by_source_vote'),
    'link_curated_entity': ('curator', 'copy_from_curated_entity'),
    'apportion_curated_values': ('curator', 'split_curated_values'),
    'keep_curated_ids': ('curator', 'empty_unknown_ids'),
    'cast_integers': ('curator', 'round_to_integers'),
    'resolve_occupancy': ('curator', 'override_occupancy_by_keywords'),
    'detect_nfloors': ('enricher', 'detect_n_stories'),
}

RECIPES = Path(__file__).resolve().parents[2] / 'src' / 'openplaces' / 'recipes'


@pytest.fixture
def fresh_map(monkeypatch):
    """An empty map of retired names and an empty warned set."""
    monkeypatch.setattr(steps, 'DEPRECATED_STEP_NAMES', {})
    monkeypatch.setattr(steps, '_WARNED_STEP_NAMES', set())
    return steps


def _step(state):
    return state


def _other(state):
    return state


def test_old_name_registers_the_same_function(fresh_map):
    registry = {}
    make_register(registry)('new_name', deprecated=('old_name',))(_step)

    assert registry['old_name'] is registry['new_name'] is _step
    assert fresh_map.DEPRECATED_STEP_NAMES == {'old_name': 'new_name'}


def test_old_name_warns_once_and_names_the_new_one(fresh_map):
    make_register({})('new_name', deprecated=('old_name',))(_step)

    with pytest.warns(DeprecatedStepNameWarning, match="'old_name'.*'new_name'"):
        assert current_step_name('old_name') == 'new_name'
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert current_step_name('old_name') == 'new_name'


def test_the_warning_says_the_old_name_will_be_removed(fresh_map):
    make_register({})('new_name', deprecated=('old_name',))(_step)

    with pytest.warns(DeprecatedStepNameWarning) as caught:
        current_step_name('old_name')
    assert 'will be removed' in str(caught[0].message)


def test_new_name_does_not_warn(fresh_map):
    make_register({})('new_name', deprecated=('old_name',))(_step)

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert current_step_name('new_name') == 'new_name'


def test_a_deprecated_name_held_by_another_function_is_refused(fresh_map):
    register = make_register({})
    register('old_name')(_other)

    with pytest.raises(ValueError, match="'old_name' is already registered"):
        register('new_name', deprecated=('old_name',))(_step)


def test_a_name_retired_for_two_replacements_is_refused(fresh_map):
    register = make_register({})
    register('first', deprecated=('old_name',))(_step)

    with pytest.raises(ValueError, match='already retired'):
        register('second', deprecated=('old_name',))(_other)


def test_find_step_translates_after_a_lazy_load(fresh_map):
    registry = {}

    def load():
        make_register(registry)('new_name', deprecated=('old_name',))(_step)

    with pytest.warns(DeprecatedStepNameWarning):
        name, fn = find_step(registry, load, 'old_name')
    assert (name, fn) == ('new_name', _step)


@pytest.mark.parametrize('old', sorted(RENAMED), ids=lambda s: s)
def test_every_renamed_step_keeps_its_old_name(old):
    from importlib import import_module

    stage, new = RENAMED[old]
    package = import_module(f'openplaces.io.{stage}')
    package._load_steps()

    assert steps.DEPRECATED_STEP_NAMES[old] == new
    assert package._STEP_REGISTRY[old] is package._STEP_REGISTRY[new]
    assert package._STEP_REGISTRY[new].__name__ == new


def test_no_recipe_uses_a_deprecated_step_name():
    from importlib import import_module

    for stage in ('harmonizer', 'enricher', 'curator'):
        import_module(f'openplaces.io.{stage}')._load_steps()
    retired = sorted(steps.DEPRECATED_STEP_NAMES)
    assert set(RENAMED) <= set(retired)
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, retired)) + r')\b')

    found = []
    for path in sorted(RECIPES.rglob('*.y*ml')):
        for number, line in enumerate(path.read_text(encoding='utf-8').splitlines()):
            match = pattern.search(line)
            if match:
                found.append(f'{path.relative_to(RECIPES)}:{number + 1} {match[1]}')
    assert not found, 'recipes use retired step names:\n' + '\n'.join(found)


def test_curator_runs_an_old_name_and_times_the_new_one(monkeypatch):
    ran = []
    marks = []

    def step(state, **kwargs):
        ran.append(kwargs)
        return state

    class _Timer:
        def mark(self, label, **metadata):
            marks.append(label)

    monkeypatch.setitem(cur._STEP_REGISTRY, 't_new', step)
    monkeypatch.setitem(cur._STEP_REGISTRY, 't_old', step)
    monkeypatch.setitem(steps.DEPRECATED_STEP_NAMES, 't_old', 't_new')
    monkeypatch.setattr(steps, '_WARNED_STEP_NAMES', set())
    monkeypatch.setattr(cur, 'get_entities', lambda *a, **k: pd.DataFrame({'x': [1]}))
    monkeypatch.setattr(cur, 'save_parquet', lambda *a, **k: None)
    monkeypatch.setattr(cur, 'get_output_path', lambda *a, **k: 'dummy')

    curator = cur.Curator.__new__(cur.Curator)
    curator.recipe = {'pipeline': [{'step': 't_old', 'foo': 1}], 'save_to': {}}
    curator.entity_recipe = {}
    curator.verbose = False
    curator.save_statistics = False
    curator.skip_steps = set()
    curator.audit_steps = False
    curator._timer = _Timer()
    with pytest.warns(DeprecatedStepNameWarning, match="'t_old'"):
        curator._curate_one(AdminId('US'))

    assert ran == [{'foo': 1}]
    assert marks == ['step 00 t_new']
