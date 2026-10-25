"""Tests for flow.freshness.out_of_date."""

import os
import time

import pandas as pd
import pytest
import yaml

import tests.io.harmonizer.test_held_recipes_never_discovered as lane
from openplaces.config import cfg
from openplaces.core.held_recipes import EXCLUDE_ENV
from openplaces.flow import RecipeDAG, freshness
from openplaces.flow.freshness import out_of_date
from openplaces.recipe import get_output_path

# The fabricated recipe root of the held-recipe tests
held_root = lane.held_root

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


def _lane(root, exclude):
    """The fabricated enrich lane's graph and the files its jobs write.

    A spine, one enrich per source (open, held, excluded by the run)
    and the curate merging all three (`lane.write_enrich_lane`).
    """
    enriches = lane.write_enrich_lane(root)
    dag = RecipeDAG(
        lane.LANE_CURATE_ID,
        [lane.UNIT],
        exclude_recipe_ids=set(exclude),
        deliver=False,
        validate=False,
    )

    def out(recipe_id, stage):
        # Asked of a graph excluding nothing, so that a left-out
        # enrich's path resolves too
        return RecipeDAG(
            recipe_id, [lane.UNIT], deliver=False, validate=False
        ).output_path(stage, recipe_id, lane.UNIT)

    files = {
        'spine': out(lane.LANE_SPINE_ID, 'harmonize'),
        'curated': out(lane.LANE_CURATE_ID, 'curate'),
        **{source: out(enrich, 'enrich') for source, enrich in enriches.items()},
    }
    return dag, enriches, files


@pytest.mark.parametrize('source_id', [lane.HELD_BY_KEY_ID, lane.EXCLUDED_ID])
def test_an_enrich_of_a_left_out_source_never_marks_the_curate_stale(
    data_root, held_root, source_id
):
    dag, enriches, files = _lane(held_root, exclude={lane.EXCLUDED_ID})
    # The left-out enrich output predates the rebuilt spine; the curate
    # is newer than both, so only that enrich job and propagation from
    # it could mark the curate.
    _write(files[source_id], age_s=7200)
    _write(files['spine'], age_s=3600)
    _write(files[lane.OPEN_ID])
    _write(files['curated'])

    assert out_of_date(dag).empty
    assert ('enrich', enriches[source_id], lane.UNIT, None) in dag.left_out

    # The open source's enrich still counts
    _write(files[lane.OPEN_ID], age_s=7200)
    stale = set(out_of_date(dag)['recipe_id'])
    assert stale == {enriches[lane.OPEN_ID], lane.LANE_CURATE_ID}


def test_a_curate_older_than_its_spine_stays_stale_beside_a_left_out_lane(
    data_root, held_root
):
    dag, _, files = _lane(held_root, exclude={lane.EXCLUDED_ID})
    _write(files[lane.EXCLUDED_ID], age_s=7200)
    _write(files['curated'], age_s=3600)
    _write(files['spine'])

    stale = out_of_date(dag).set_index('recipe_id')

    assert list(stale.index) == [lane.LANE_CURATE_ID]
    assert stale.loc[lane.LANE_CURATE_ID, 'reason'].startswith('input newer:')


def test_the_profile_exclusions_plan_no_placeslab_enrich():
    """The real tree, as the local profile runs it, on a DC county."""
    dag = RecipeDAG(
        'US_parcel-openplaces-2026',
        admin_ids=['US-VA-FA'],
        exclude_recipe_ids=freshness.default_exclude_recipe_ids(),
        deliver=False,
        validate=False,
    )
    enrich_id = 'US_parcel_parcel-placeslab-fmv2026'
    assert enrich_id not in {node.recipe_id for node in dag.nodes()}
    assert ('enrich', enrich_id, 'US-VA-FA', None) in dag.left_out
    inputs = dag.input_paths('curate', 'US_parcel-openplaces-2026', 'US-VA-FA')
    assert not any('placeslab' in p.name for p in inputs)


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
