"""Tests for RecipeDAG.plan(), to_mermaid(), and the submit helpers."""

import os
import subprocess
import sys
import time

import pandas as pd
import pytest

from openplaces.config import cfg
from openplaces.flow import RecipeDAG
from openplaces.flow import submit as submit_mod
from openplaces.recipe import get_output_path

TARGET = 'US_footprint-openplaces-2026'
SPINE = 'US_footprint-spine-2026'
NSI = 'US_building-nsi-2026'
COUNTY = 'US-NC-BR'


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    dirs = dict(cfg.config['directories'])
    dirs['data_root'] = tmp_path
    for name in ('core', 'external', 'raw', 'cache', 'out', 'share'):
        dirs[name] = tmp_path / 'data' / name
    dirs['heap'] = tmp_path / 'data/cache/_heap'
    dirs['logs'] = tmp_path / 'data/cache/_logs'
    monkeypatch.setitem(cfg.config, 'directories', dirs)
    return tmp_path


@pytest.fixture(scope='module')
def dag():
    # Built against the real data root (module scope, so it is constructed
    # before any function-scoped data_root monkeypatch): expanding per-town
    # image jobs requires ingested admin boundaries. plan() then stats
    # outputs under whatever root is configured at call time.
    return RecipeDAG(TARGET, admin_ids=[COUNTY])


def _write_parquet(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'geo_id': ['a'], 'value': [1.0]}).to_parquet(path)
    return path


def test_plan_all_missing(dag, data_root):
    plan = dag.plan()
    assert plan['will_run'].all()
    assert (plan['reason'] == 'output missing').all()
    assert not plan['exists'].any()


def test_plan_upstream_propagation(dag, data_root):
    # The spine exists, but its ingest inputs are missing: the spine job
    # re-runs because its upstreams are scheduled, not because it's absent
    _write_parquet(get_output_path(SPINE, admin_id=COUNTY))
    plan = dag.plan().set_index(['recipe_id', 'admin_id'])
    spine = plan.loc[(SPINE, COUNTY)]
    assert bool(spine['exists'])
    assert bool(spine['will_run'])
    assert spine['reason'].startswith('upstream scheduled')
    # And the propagation cascades to the terminal curate job
    target = plan.loc[(TARGET, COUNTY)]
    assert target['reason'].startswith('output missing')


def test_plan_inputs_newer(dag, data_root):
    spine_path = _write_parquet(get_output_path(SPINE, admin_id=COUNTY))
    old = time.time() - 3600
    os.utime(spine_path, (old, old))
    # An input of the spine, newer than the spine output: the direct mtime
    # reason wins over the propagated upstream-scheduled one
    _write_parquet(get_output_path(NSI, admin_id=COUNTY))
    plan = dag.plan().set_index(['recipe_id', 'admin_id'])
    spine = plan.loc[(SPINE, COUNTY)]
    assert bool(spine['will_run'])
    assert spine['reason'] == 'inputs newer than output'


def test_plan_columns_and_order(dag, data_root):
    plan = dag.plan()
    assert list(plan.columns) == [
        'stage',
        'recipe_id',
        'admin_id',
        'output',
        'exists',
        'size_mb',
        'will_run',
        'reason',
    ]
    stages = plan['stage'].drop_duplicates().tolist()
    assert stages == sorted(
        stages, key=['ingest', 'harmonize', 'enrich', 'curate'].index
    )


def test_mermaid_full_detail(dag):
    mermaid = dag.to_mermaid(collapse_admin=False)
    assert 'flowchart LR' in mermaid
    assert 'US_footprint_openplaces_2026_US_NC_BR[' in mermaid
    assert '-->' in mermaid
    for stage in ('ingest', 'harmonize', 'enrich', 'curate'):
        assert f'classDef {stage}' in mermaid


def test_mermaid_collapse(dag):
    # The default auto-collapses if and only if the DAG exceeds the threshold
    assert dag.to_mermaid() == dag.to_mermaid(collapse_admin=len(dag.nodes()) > 30)
    # Explicit collapse folds per-admin jobs into recipe-level nodes
    collapsed = dag.to_mermaid(collapse_admin=True)
    assert 'admin units)' in collapsed
    assert 'US_footprint_openplaces_2026_all[' in collapsed
    assert len(collapsed) < len(dag.to_mermaid(collapse_admin=False))


def test_dry_run_command_and_stored_file(data_root, monkeypatch):
    calls = {}

    def _fake_run(command, **kwargs):
        calls['command'] = command
        calls['kwargs'] = kwargs
        return subprocess.CompletedProcess(command, 0, stdout='DRY', stderr='')

    monkeypatch.setattr(submit_mod.subprocess, 'run', _fake_run)
    returncode, output, stored = submit_mod.dry_run(
        config={'recipe': TARGET, 'admin_ids': [COUNTY]},
        extra_args=('--forcerun', 'some_rule'),
        verbose=False,
    )
    assert returncode == 0
    assert output == 'DRY'
    assert calls['command'][:3] == [sys.executable, '-m', 'snakemake']
    assert '-n' in calls['command']
    snakefile = calls['command'][calls['command'].index('--snakefile') + 1]
    assert snakefile == str(cfg.code_root / 'workflow' / 'Snakefile')
    config_pairs = calls['command'][calls['command'].index('--config') + 1 :][:2]
    assert config_pairs == [f'recipe={TARGET}', f'admin_ids={COUNTY}']
    assert '--forcerun' in calls['command']
    assert calls['kwargs']['cwd'] == cfg.code_root
    assert stored.exists()
    assert stored.read_text(encoding='utf-8') == 'DRY'


def test_deploy_profile_and_cores(data_root, monkeypatch):
    commands = []
    monkeypatch.setattr(
        submit_mod.subprocess,
        'run',
        lambda command, **kwargs: (
            commands.append(command),
            subprocess.CompletedProcess(command, 0),
        )[1],
    )
    submit_mod.deploy(profile='workflow/profiles/scc', config={'recipe': TARGET})
    assert '--profile' in commands[0]
    profile = commands[0][commands[0].index('--profile') + 1]
    assert profile == str(cfg.code_root / 'workflow' / 'profiles' / 'scc')

    submit_mod.deploy(cores=2, config={'recipe': TARGET})
    assert commands[1][commands[1].index('--cores') + 1] == '2'


# Consolidated from test_flow_imports.py
"""The notebook import contract of the flow package must keep working."""


def test_notebook_imports():
    from openplaces.flow import convert_to_script, test_script  # noqa: F401


def test_run_subprocess_import():
    from openplaces.flow import run_subprocess  # noqa: F401


def test_caller_path_helpers_import():
    from openplaces.flow import (  # noqa: F401
        get_caller_path,
        get_caller_path_in_code_directory,
    )
