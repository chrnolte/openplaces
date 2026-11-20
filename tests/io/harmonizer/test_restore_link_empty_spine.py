"""A geospine with no rows leaves its attribute recipe nothing to build.

A unit whose sources load nothing saves an empty geospine. The
attribute recipe then stops at `load_geospine` and saves the empty
table a pipeline without a spine saves, so no later step has to guard
against a frame without rows, and no link sidecar is owed. Fabricated
recipes and frames throughout.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from openplaces.core.schema import AdminId, Entity
from openplaces.io.harmonizer import NOTHING_TO_BUILD_KEY, HarmonizeState
from openplaces.io.harmonizer import load as load_mod
from openplaces.recipe import get_output_path, get_recipe_by_id

COUNTY = 'US-XX-YY'
GEOSPINE = {
    'recipe_id': 'US_parcel-fabricated-2026',
    'pipeline': [
        {'step': 'link_to_reference', 'recipe_id': 'ref-a', 'join': 'spatial_point'}
    ],
}


def _geospine_rows(n_rows):
    return gpd.GeoDataFrame(
        index=pd.Index([f'p{i}' for i in range(n_rows)], name='parcel_id'),
        geometry=[box(i, 0, i + 1, 1) for i in range(n_rows)],
        crs='EPSG:4326',
    )


def _patch_load(monkeypatch, n_rows, restored):
    monkeypatch.setattr(load_mod, 'apply_recipe_patches', lambda r, a: (r, []))
    monkeypatch.setattr(
        load_mod, 'get_entities', lambda *a, **k: _geospine_rows(n_rows)
    )
    monkeypatch.setattr(load_mod, '_restore_handoff_metadata', lambda *a: None)
    monkeypatch.setattr(load_mod, 'get_recipe_id', lambda r: r['recipe_id'])

    def _record(state, geospine, step_index, step_cfg, spine_id_col):
        restored.append(step_cfg['recipe_id'])
        return state

    monkeypatch.setattr(load_mod, '_restore_link', _record)


def test_an_empty_geospine_restores_no_spine_and_no_link(monkeypatch):
    restored = []
    _patch_load(monkeypatch, 0, restored)
    state = HarmonizeState(
        recipe={}, admin_id=AdminId(COUNTY), verbose=False, timer=None
    )
    with pytest.warns(UserWarning, match='nothing to build'):
        state = load_mod.load_geospine(state, entity_recipe_id=GEOSPINE)
    assert state.spine is None
    assert state.metadata[NOTHING_TO_BUILD_KEY] == 'US_parcel-fabricated-2026'
    assert restored == []


def test_a_geospine_with_rows_restores_its_links(monkeypatch):
    restored = []
    _patch_load(monkeypatch, 1, restored)
    state = HarmonizeState(
        recipe={}, admin_id=AdminId(COUNTY), verbose=False, timer=None
    )
    state = load_mod.load_geospine(state, entity_recipe_id=GEOSPINE)
    assert len(state.spine) == 1
    assert NOTHING_TO_BUILD_KEY not in state.metadata
    assert restored == ['ref-a']


def test_the_run_skips_the_steps_after_nothing_to_build(mock_data_root, monkeypatch):
    from openplaces.io.harmonizer import _STEP_REGISTRY, Harmonizer

    recipe = dict(get_recipe_by_id('US_footprint-spine-2026'))
    recipe['entity'] = Entity('footprint', 'nothingtobuild', '2026')
    recipe['save_to'] = {'data_dir': 'core', 'geometry': False}
    recipe['pipeline'] = [{'step': '_nothing_here'}, {'step': '_must_not_run'}]

    def _nothing(state):
        state.metadata[NOTHING_TO_BUILD_KEY] = 'US_parcel-fabricated-2026'
        return state

    def _fail(state):
        raise AssertionError('a step ran after nothing to build')

    monkeypatch.setitem(_STEP_REGISTRY, '_nothing_here', _nothing)
    monkeypatch.setitem(_STEP_REGISTRY, '_must_not_run', _fail)
    with pytest.warns(UserWarning, match='saved an empty table'):
        Harmonizer(recipe, admin_ids=[COUNTY])._harmonize_one(AdminId(COUNTY))

    saved = pd.read_parquet(get_output_path(recipe, AdminId(COUNTY)))
    assert saved.empty
