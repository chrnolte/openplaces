"""`export_delivery` refuses to pool curated files older than their inputs.

The graph is a stand-in with RecipeDAG's three methods over fabricated
files: a spine per county feeding that county's curated file.
"""

from __future__ import annotations

import os
import time
from typing import NamedTuple

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from openplaces.core.schema import AdminId, Entity
from openplaces.io import save_parquet
from openplaces.io.delivery import StaleOutputError, delivery_paths, export_delivery
from openplaces.recipe import get_output_path

UNITS = ['US-NC-AL', 'US-NC-BB']


class _Node(NamedTuple):
    stage: str
    recipe_id: str
    admin_id: str
    region: str | None = None


class _Graph:
    """Spine -> curate, per unit, over files on disk."""

    def __init__(self, spines, curated):
        self.spines = spines
        self.curated = curated

    def nodes(self):
        return [
            *(_Node('harmonize', 'spine', unit) for unit in self.spines),
            *(_Node('curate', 'curated', unit) for unit in self.curated),
        ]

    def output_path(self, stage, recipe_id, admin_id=None, region=None):
        files = self.spines if stage == 'harmonize' else self.curated
        return files[admin_id]

    def input_paths(self, stage, recipe_id, admin_id=None, region=None):
        return [self.spines[admin_id]] if stage == 'curate' else []


def _recipe():
    return {
        'recipe_id': 'US_footprint-test-2026',
        'admin_id': AdminId('US'),
        'stage': 'curate',
        'entity': Entity('footprint', 'test', '2026'),
        'process_by': {'admin_level': 3},
        'save_to': {'data_dir': 'core', 'combined': True},
        'share': {
            'columns': ['admin3_id', 'lat', 'long', 'year_built'],
            'delivery': {'admin_level': 2, 'admin_ids': UNITS},
        },
    }


def _age(path, seconds):
    stamp = time.time() - seconds
    os.utime(path, (stamp, stamp))


def _county(footprint_ids, admin_id):
    n = len(footprint_ids)
    frame = gpd.GeoDataFrame(
        {
            'lat': [40.0 + i for i in range(n)],
            'long': [-80.0 - i for i in range(n)],
            'year_built': [1990] * n,
            'geometry_source': ['obm'] * n,
        },
        geometry=[box(i, i, i + 1, i + 1) for i in range(n)],
        crs='EPSG:4326',
        index=pd.Index(footprint_ids, name='footprint_id'),
    )
    path = get_output_path(_recipe(), admin_id)
    save_parquet(frame, path, combined=True)
    return path


@pytest.fixture
def graph(mock_data_root):
    """Both counties curated an hour after their spines were built."""
    spines, curated = {}, {}
    for unit, ids in zip(UNITS, (['a', 'b'], ['c', 'd']), strict=True):
        spine = mock_data_root / 'data' / 'core' / 'spine' / f'{unit}.parquet'
        spine.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'footprint_id': ids}).to_parquet(spine)
        _age(spine, 7200)
        curated[unit] = _county(ids, unit)
        _age(curated[unit], 3600)
        spines[unit] = spine
    return _Graph(spines, curated)


def test_a_fresh_bundle_exports(graph):
    paths = export_delivery(_recipe(), 'US-NC', admin_ids=UNITS, graph=graph)

    assert paths['canonical'].exists()


def test_a_newer_input_refuses_the_export(graph):
    _age(graph.spines['US-NC-BB'], 0)

    with pytest.raises(StaleOutputError) as raised:
        export_delivery(_recipe(), 'US-NC', admin_ids=UNITS, graph=graph)

    message = str(raised.value)
    assert '1 of 2 pooled curated files' in message
    assert 'US-NC-BB.parquet' in message.split('input newer: ')[1]
    assert 'allow_stale=True' in message
    for path in delivery_paths(_recipe(), 'US-NC').values():
        assert not path.exists()


def test_the_override_lets_a_stale_bundle_through(graph):
    _age(graph.spines['US-NC-BB'], 0)

    paths = export_delivery(
        _recipe(), 'US-NC', admin_ids=UNITS, graph=graph, allow_stale=True
    )

    assert paths['canonical'].exists()


def test_no_graph_refuses_unless_overridden(graph):
    with pytest.raises(ValueError, match='delivery_graph'):
        export_delivery(_recipe(), 'US-NC', admin_ids=UNITS)

    assert export_delivery(_recipe(), 'US-NC', admin_ids=UNITS, allow_stale=True)


def test_a_graph_that_misses_a_pooled_file_refuses(graph):
    del graph.curated['US-NC-BB']

    with pytest.raises(ValueError, match='does not build 1 of the 2 pooled'):
        export_delivery(_recipe(), 'US-NC', admin_ids=UNITS, graph=graph)
