"""Tests for skipping curate pipeline steps.

A step can be turned off persistently in the recipe (``enabled: false``) or for a
single run via ``curate(..., skip_steps=[...])``. Skipped steps must not execute
and must not write their columns; ``enabled`` is a control key, never passed to
the step.
"""

from __future__ import annotations

import pandas as pd

import openplaces.io.curator as cur
from openplaces.core.schema import AdminId
from openplaces.io.curator import _STEP_REGISTRY, Curator


def _run_pipeline(monkeypatch, pipeline, skip_steps=None, save_to=None):
    """Run *pipeline* through a Curator with stubbed I/O.

    Returns (ran, seen_kwargs, saved_df, save_kwargs): the ordered names that
    executed, the kwargs each received, the saved GeoDataFrame, and the
    kwargs save_parquet was called with (e.g. ``combined``).
    """
    ran: list[str] = []
    seen_kwargs: dict[str, dict] = {}

    def make_step(name):
        def step(state, **kwargs):
            ran.append(name)
            seen_kwargs[name] = kwargs
            state.curated[name] = 1
            return state

        return step

    for name in ('t_a', 't_b', 't_c'):
        _STEP_REGISTRY[name] = make_step(name)

    saved = {}
    save_kwargs = {}

    def _fake_save_parquet(gdf, path, **kwargs):
        saved['df'] = gdf
        save_kwargs.update(kwargs)

    monkeypatch.setattr(cur, 'get_entities', lambda *a, **k: pd.DataFrame({'x': [1]}))
    monkeypatch.setattr(cur, 'save_parquet', _fake_save_parquet)
    monkeypatch.setattr(cur, 'get_output_path', lambda *a, **k: 'dummy')

    curator = Curator.__new__(Curator)
    curator.recipe = {'pipeline': pipeline, 'save_to': save_to or {}}
    curator.entity_recipe = {}
    curator.verbose = False
    curator.save_statistics = False
    curator.skip_steps = set(skip_steps or [])
    curator._timer = None
    try:
        curator._curate_one(AdminId('US'))
    finally:
        for name in ('t_a', 't_b', 't_c'):
            _STEP_REGISTRY.pop(name, None)
    return ran, seen_kwargs, saved['df'], save_kwargs


def test_enabled_false_skips_step(monkeypatch):
    ran, _, out, _ = _run_pipeline(
        monkeypatch,
        [{'step': 't_a'}, {'step': 't_b', 'enabled': False}, {'step': 't_c'}],
    )
    assert ran == ['t_a', 't_c']
    assert 't_b' not in out.columns


def test_skip_steps_skips_named_step(monkeypatch):
    ran, _, out, _ = _run_pipeline(
        monkeypatch,
        [{'step': 't_a'}, {'step': 't_b'}, {'step': 't_c'}],
        skip_steps=['t_c'],
    )
    assert ran == ['t_a', 't_b']
    assert 't_c' not in out.columns


def test_control_keys_not_passed_as_kwargs(monkeypatch):
    # An enabled: true step runs and the control key is stripped from kwargs;
    # real params still pass through.
    ran, seen, _, _ = _run_pipeline(
        monkeypatch, [{'step': 't_a', 'enabled': True, 'foo': 7}]
    )
    assert ran == ['t_a']
    assert seen['t_a'] == {'foo': 7}
    assert 'enabled' not in seen['t_a']
    assert 'step' not in seen['t_a']


def test_save_to_combined_true_passed_to_save_parquet(monkeypatch):
    _, _, _, save_kwargs = _run_pipeline(
        monkeypatch, [{'step': 't_a'}], save_to={'data_dir': 'share', 'combined': True}
    )
    assert save_kwargs == {'combined': True}


def test_save_to_combined_defaults_to_false(monkeypatch):
    _, _, _, save_kwargs = _run_pipeline(monkeypatch, [{'step': 't_a'}])
    assert save_kwargs == {'combined': False}
