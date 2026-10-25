"""Tests for flow.freshness.out_of_date."""

import os
import time
from types import SimpleNamespace
from typing import NamedTuple

import pandas as pd
import pytest
import yaml

from openplaces.config import cfg
from openplaces.core.held_recipes import EXCLUDE_ENV
from openplaces.flow import RecipeDAG, freshness
from openplaces.flow.freshness import out_of_date
from openplaces.recipe import get_output_path

TARGET = 'US_footprint-openplaces-2026'
BUILDING_SPINE = 'US_building-spine-2026'
NSI = 'US_building-nsi-2026'
COUNTY = 'US-NC-BRU'


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
    # Built against the real data root before data_root patches it, as
    # in test_plan; out_of_date stats files under the root at call time.
    return RecipeDAG(TARGET, admin_ids=[COUNTY], deliver=False, validate=False)


def _write(path, age_s=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'geo_id': ['a'], 'value': [1.0]}).to_parquet(path)
    stamp = time.time() - age_s
    os.utime(path, (stamp, stamp))
    return path


def _consumer_of(dag, path):
    """A job of the graph that reads path, with its output path."""
    for node in dag.nodes():
        job = (node.stage, node.recipe_id, node.admin_id, node.region)
        if path in dag.input_paths(*job):
            return node.recipe_id, dag.output_path(*job)
    raise AssertionError(f'nothing in the graph reads {path}')


def test_nothing_built_is_nothing_stale(dag, data_root):
    report = out_of_date(dag)
    assert report.empty
    assert report.attrs['checked'] == 0


def test_missing_upstream_does_not_mark_a_built_job(dag, data_root):
    # plan() calls this job "upstream scheduled"; out_of_date must not,
    # since a missing input may be held, opt-in or consumed.
    _write(get_output_path(BUILDING_SPINE, admin_id=COUNTY))
    assert out_of_date(dag).empty


def test_input_newer_than_output(dag, data_root):
    _write(get_output_path(BUILDING_SPINE, admin_id=COUNTY), age_s=3600)
    _write(get_output_path(NSI, admin_id=COUNTY))
    stale = out_of_date(dag).set_index('recipe_id')
    assert stale.loc[BUILDING_SPINE, 'reason'].startswith('input newer:')


def test_consumer_of_a_stale_output_is_stale(dag, data_root):
    spine = _write(get_output_path(BUILDING_SPINE, admin_id=COUNTY), age_s=7200)
    _write(get_output_path(NSI, admin_id=COUNTY), age_s=3600)
    consumer_id, consumer_out = _consumer_of(dag, spine)
    # Newer than the stale spine it read, so only propagation finds it
    _write(consumer_out)
    stale = out_of_date(dag).set_index('recipe_id')
    assert stale.loc[consumer_id, 'reason'].startswith('built from stale:')


def test_current_outputs_are_not_reported(dag, data_root):
    _write(get_output_path(NSI, admin_id=COUNTY), age_s=3600)
    _write(get_output_path(BUILDING_SPINE, admin_id=COUNTY))
    assert BUILDING_SPINE not in set(out_of_date(dag)['recipe_id'])


@pytest.fixture(scope='module')
def dag_without_nsi():
    return RecipeDAG(
        TARGET,
        admin_ids=[COUNTY],
        exclude_recipe_ids={NSI},
        deliver=False,
        validate=False,
    )


def test_an_excluded_recipe_never_marks_a_consumer_stale(
    dag, dag_without_nsi, data_root
):
    # The same files that make the spine stale in the full graph
    _write(get_output_path(BUILDING_SPINE, admin_id=COUNTY), age_s=3600)
    _write(get_output_path(NSI, admin_id=COUNTY))
    assert BUILDING_SPINE in set(out_of_date(dag)['recipe_id'])

    report = out_of_date(dag_without_nsi)
    assert report.empty
    assert NSI not in {node.recipe_id for node in dag_without_nsi.nodes()}


class _Node(NamedTuple):
    stage: str
    recipe_id: str
    admin_id: str
    region: str | None = None


class _LaneDag:
    """A spine, an enrich adding one source to it, and the curate.

    The source itself has no job: the run excluded it, or it is held,
    and RecipeDAG gives such a recipe none while the enrich, named by
    the curation directly, keeps its own.
    """

    exclude_recipe_ids = frozenset({'ZZ_parcel-optional-2026'})

    def __init__(self, root, enrich_id):
        self.files = {
            'ZZ_parcel-spine-2026': root / 'spine.parquet',
            enrich_id: root / 'spine_optional.parquet',
            'ZZ_parcel-curated-2026': root / 'curated.parquet',
        }
        self.enrich_id = enrich_id

    def nodes(self):
        stages = ('harmonize', 'enrich', 'curate')
        return [
            _Node(stage, recipe_id, 'ZZ-AA-ALP')
            for stage, recipe_id in zip(stages, self.files, strict=True)
        ]

    def output_path(self, stage, recipe_id, admin_id=None, region=None):
        return self.files[recipe_id]

    def input_paths(self, stage, recipe_id, admin_id=None, region=None):
        spine, enrich, _ = self.files.values()
        return {'enrich': [spine], 'curate': [spine, enrich]}.get(stage, [])


EDGES = {
    'ZZ_parcel-enrich-optional-2026': [
        'ZZ_parcel-spine-2026',
        'ZZ_parcel-optional-2026',
    ],
    'ZZ_parcel-enrich-held-2026': ['ZZ_parcel-spine-2026', 'ZZ_parcel-heldsrc-2026'],
}


@pytest.fixture
def lane_recipes(monkeypatch):
    def recipe(recipe_id):
        return {'recipe_id': recipe_id, 'held': recipe_id == 'ZZ_parcel-heldsrc-2026'}

    def dependencies(recipe, admin_id=None, exclude_recipe_ids=None):
        return [
            SimpleNamespace(upstream_recipe_id=upstream)
            for upstream in EDGES.get(recipe['recipe_id'], [])
        ]

    monkeypatch.setattr(freshness, 'get_recipe_by_id', recipe)
    monkeypatch.setattr(freshness, 'get_recipe_dependencies', dependencies)


@pytest.mark.parametrize('enrich_id', list(EDGES))
def test_an_enrich_of_a_left_out_source_never_marks_the_curate_stale(
    tmp_path, lane_recipes, enrich_id
):
    dag = _LaneDag(tmp_path, enrich_id)
    spine, enrich, curated = dag.files.values()
    # The enrich output predates the rebuilt spine; the curate is newer
    # than both, so only the enrich job and propagation would mark it.
    _write(enrich, age_s=7200)
    _write(spine, age_s=3600)
    _write(curated)
    assert set(out_of_date(dag)['recipe_id']) == {enrich_id, 'ZZ_parcel-curated-2026'}

    graph = freshness.freshness_graph(dag)

    assert out_of_date(graph).empty
    assert graph.left_out == {('enrich', enrich_id, 'ZZ-AA-ALP', None)}


def test_a_curate_older_than_its_spine_stays_stale_beside_a_left_out_lane(
    tmp_path, lane_recipes
):
    dag = _LaneDag(tmp_path, 'ZZ_parcel-enrich-optional-2026')
    spine, enrich, curated = dag.files.values()
    _write(enrich, age_s=7200)
    _write(curated, age_s=3600)
    _write(spine)

    stale = out_of_date(freshness.freshness_graph(dag)).set_index('recipe_id')

    assert list(stale.index) == ['ZZ_parcel-curated-2026']
    assert stale.loc['ZZ_parcel-curated-2026', 'reason'].startswith('input newer:')


def test_default_exclusions_come_from_the_job_then_the_profile(monkeypatch, tmp_path):
    profile = tmp_path / 'defaults.yaml'
    profile.write_text('exclude_recipe_ids:\n  - ZZ_parcel-optional-lane-2026\n')
    monkeypatch.setattr(freshness, 'profile_defaults_path', lambda: profile)
    monkeypatch.delenv(EXCLUDE_ENV, raising=False)
    assert freshness.default_exclude_recipe_ids() == {'ZZ_parcel-optional-lane-2026'}

    # Inside a job, exactly what the workflow passed, even nothing
    monkeypatch.setenv(EXCLUDE_ENV, '')
    assert freshness.default_exclude_recipe_ids() == set()
    monkeypatch.setenv(EXCLUDE_ENV, 'ZZ_lane-a-2026')
    assert freshness.default_exclude_recipe_ids() == {'ZZ_lane-a-2026'}


def test_the_shipped_profile_excludes_the_placeslab_lane():
    assert 'US_parcel-placeslab-fmv2026' in yaml.safe_load(
        freshness.profile_defaults_path().read_text()
    ).get('exclude_recipe_ids')
