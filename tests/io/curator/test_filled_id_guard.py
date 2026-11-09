"""A curate refuses to save an id column empty that its entity filled.

An all-empty column is written with the Arrow null type, and every
downstream curate and delivery inherits it without a word, as the census
tract and block-group ids of 70 Massachusetts towns did in November 2026.
"""

import pandas as pd
import pytest

import openplaces.io.curator as cur
from openplaces.core.schema import AdminId
from openplaces.io.curator import (
    _STEP_REGISTRY,
    Curator,
    EmptiedIdColumnError,
    require_filled_ids_kept,
)

# Fabricated ids, shaped like a tract and a block-group geoid.
ENTITY = pd.DataFrame(
    {
        'census_tract_id': ['99001000100', '99001000200'],
        'census_blockgroup_id': ['990010001001', None],
        'admin4_id': [None, None],
        'area_ha': [0.1, 0.2],
    },
    index=pd.Index(['p1', 'p2'], name='parcel_id'),
)


def _curate(monkeypatch, step):
    saved = []
    monkeypatch.setattr(cur, 'get_entities', lambda *a, **k: ENTITY.copy())
    monkeypatch.setattr(cur, 'save_parquet', lambda frame, *a, **k: saved.append(frame))
    monkeypatch.setattr(cur, 'get_output_path', lambda *a, **k: 'out')
    monkeypatch.setitem(_STEP_REGISTRY, 't_step', step)
    curator = Curator.__new__(Curator)
    curator.recipe = {'recipe_id': 'US_parcel-x-2026', 'pipeline': [{'step': 't_step'}]}
    curator.entity_recipe = {}
    curator.verbose = False
    curator.save_statistics = False
    curator.skip_steps = set()
    curator._timer = None
    curator._curate_one(AdminId('US-MA-LAN'))
    return saved


def test_a_step_emptying_a_filled_id_column_fails_before_saving(monkeypatch):
    def empty_tracts(state):
        state.curated['census_tract_id'] = pd.NA
        return state

    with pytest.raises(EmptiedIdColumnError, match='census_tract_id') as excinfo:
        _curate(monkeypatch, empty_tracts)
    assert 'curate US_parcel-x-2026 for US-MA-LAN' in str(excinfo.value)


def test_partly_empty_id_column_is_saved(monkeypatch):
    def empty_one(state):
        state.curated.loc['p1', 'census_tract_id'] = None
        return state

    (saved,) = _curate(monkeypatch, empty_one)
    assert saved['census_tract_id'].notna().sum() == 1


def test_an_id_column_the_entity_left_empty_is_not_refused(monkeypatch):
    # admin4_id arrives empty (a New England town has no admin4); the
    # guard is about ids lost in curate, not coverage gaps upstream.
    (saved,) = _curate(monkeypatch, lambda state: state)
    assert saved['admin4_id'].isna().all()


def test_a_dropped_id_column_is_not_refused(monkeypatch):
    def drop(state):
        state.curated = state.curated.drop(columns='census_blockgroup_id')
        return state

    (saved,) = _curate(monkeypatch, drop)
    assert 'census_blockgroup_id' not in saved.columns


def test_an_empty_curated_table_is_not_refused():
    require_filled_ids_kept(['census_tract_id'], ENTITY.iloc[0:0], 'curate X')
