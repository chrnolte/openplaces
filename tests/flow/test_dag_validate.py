"""The `validate` node kind: derived from delivery, scoped like it."""

from __future__ import annotations

import json

import pytest

from openplaces.flow import RecipeDAG, run_stage
from openplaces.flow.dag import (
    node_key,
    parse_deliver_config,
    rule_name,
    validation_manifest_path,
)
from openplaces.io.curator.validation import validation_notebooks
from openplaces.io.delivery import delivery_accuracy_dir, delivery_paths

TARGET = 'US_footprint-openplaces-2026'
COUNTY = 'US-NC-BRU'
# The committed recipe declares the survey notebooks for this region;
# permit notebooks join them only where the untracked sidecar exists.
REGION = 'cheer-eastern-nc'
SURVEY_NOTEBOOK = 'notebooks/05_curate/validate_occupancy.ipynb'


@pytest.fixture(scope='module')
def shipping_dag():
    return RecipeDAG(TARGET, admin_ids=[COUNTY], deliver=True)


def _validate_nodes(dag):
    return [node for node, _ in dag.validation_nodes]


def test_the_survey_region_declares_its_notebooks():
    assert SURVEY_NOTEBOOK in validation_notebooks(TARGET, REGION)
    # A region nothing scores gets no notebooks, hence no node.
    assert validation_notebooks(TARGET, 'boston-inner-core') == []


def test_a_shipped_region_is_validated_after_it_ships(shipping_dag):
    nodes = _validate_nodes(shipping_dag)
    assert [(n.stage, n.admin_id, n.region) for n in nodes] == [
        ('validate', 'US-NC', REGION)
    ]
    node = nodes[0]
    assert node in shipping_dag.nodes()

    deliver_key = (TARGET, 'US-NC', REGION)
    assert (deliver_key, node_key(node)) in shipping_dag._edges
    # Same recipe, unit and region as the delivery: the stage must keep
    # the two apart everywhere a job is identified.
    assert node_key(node) != deliver_key
    deliver = next(n for n, _ in shipping_dag.delivery_nodes if n.region == REGION)
    assert rule_name(node) != rule_name(deliver)


def test_the_team_twin_is_not_validated(shipping_dag):
    assert all(not n.region.endswith('.team') for n in _validate_nodes(shipping_dag))


def test_validate_paths(shipping_dag):
    job = ('validate', TARGET, 'US-NC', REGION)
    manifest = shipping_dag.output_path(*job)
    assert manifest == validation_manifest_path(TARGET, REGION)
    assert manifest.parent == delivery_accuracy_dir(TARGET, region=REGION)
    assert shipping_dag.input_paths(*job) == [
        delivery_paths(TARGET, region=REGION)['canonical']
    ]
    assert shipping_dag.extra_outputs(*job) == []
    assert manifest in shipping_dag.target_paths()


def test_plan_and_mermaid_carry_the_validate_job(shipping_dag):
    plan = shipping_dag.plan()
    assert (plan['stage'] == 'validate').sum() == 1
    assert (plan['stage'] == 'deliver').sum() == len(shipping_dag.delivery_nodes)
    diagram = shipping_dag.to_mermaid(collapse_admin=False)
    assert 'classDef validate' in diagram


def test_a_scoped_run_neither_ships_nor_validates():
    dag = RecipeDAG(TARGET, admin_ids=[COUNTY])
    assert dag.validation_nodes == []
    assert all(node.stage != 'validate' for node in dag.nodes())


def test_validate_false_keeps_the_delivery_only():
    dag = RecipeDAG(TARGET, admin_ids=[COUNTY], deliver=True, validate=False)
    assert dag.validation_nodes == []
    assert dag.delivery_nodes
    assert all(p.suffix != '.json' for p in dag.target_paths())


def test_validate_true_rescores_a_shipped_bundle_without_reshipping():
    dag = RecipeDAG(TARGET, admin_ids=[COUNTY], deliver=False, validate=True)
    assert dag.delivery_nodes == []
    nodes = _validate_nodes(dag)
    assert [n.region for n in nodes] == [REGION]
    # No delivery in the graph, so no edge into the job.
    assert not [edge for edge in dag._edges if edge[1] == node_key(nodes[0])]
    assert dag.target_paths() == [validation_manifest_path(TARGET, REGION)]


@pytest.mark.parametrize(('raw', 'expected'), [('false', False), (0, False)])
def test_validate_config_reads_like_deliver(raw, expected):
    assert parse_deliver_config(raw) is expected


def test_notebook_arguments():
    assert run_stage.notebook_arguments('r', 'g') == '--recipe_id r --region g'
    assert run_stage.notebook_arguments('r', 'g', True).endswith(' --verbose')


def test_run_validation_executes_every_notebook_and_records_them(monkeypatch, tmp_path):
    import openplaces.io.curator.validation as validation_mod

    monkeypatch.setattr(
        validation_mod,
        'validation_notebooks',
        lambda recipe, region: ['nb/one.ipynb', 'nb/two.ipynb'],
    )
    calls = []

    class _Result:
        def __init__(self, returncode):
            self.returncode = returncode

    def _fake_run(command, cwd=None, env=None, check=False):
        calls.append((command, env))
        # The first notebook fails; the second must still run.
        return _Result(1 if 'one.ipynb' in ' '.join(command) else 0)

    monkeypatch.setattr(run_stage.subprocess, 'run', _fake_run)
    manifest = tmp_path / 'manifest.json'

    with pytest.raises(SystemExit, match='one.ipynb'):
        run_stage.run_validation(
            TARGET,
            REGION,
            log_dir=tmp_path / 'logs',
            manifest_path=manifest,
        )

    executed = [c for c, _ in calls if 'nbconvert' in c]
    assert len(executed) == 2
    assert all('--execute' in command for command in executed)
    assert all(
        env[run_stage.NOTEBOOK_ARGS_VARIABLE]
        == f'--recipe_id {TARGET} --region {REGION}'
        for command, env in calls
        if 'nbconvert' in command
    )
    record = json.loads(manifest.read_text(encoding='utf-8'))
    assert [run['returncode'] for run in record['notebooks']] == [1, 0]
