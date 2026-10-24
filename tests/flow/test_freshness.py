"""Tests for flow.freshness.out_of_date."""

import os
import time

import pandas as pd
import pytest

from openplaces.config import cfg
from openplaces.flow import RecipeDAG
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
