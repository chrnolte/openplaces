"""A curate saves an empty entity table as it is and runs no step.

Harmonize saves a table without rows or columns for a unit that has
nothing to build from. No curate step has an entity to work on there,
so none is run and none has to guard against an absent column.
"""

import pandas as pd
import pytest

import openplaces.io.curator as cur
from openplaces.core.schema import AdminId
from openplaces.io.curator import _STEP_REGISTRY, Curator


def _curate(monkeypatch, entity, step):
    saved = []
    monkeypatch.setattr(cur, 'get_entities', lambda *a, **k: entity.copy())
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
    curator._curate_one(AdminId('US-XX-YY'))
    return saved


def test_an_empty_entity_table_is_saved_without_running_a_step(monkeypatch):
    def needs_area(state):
        state.curated['area_m2'] = state.curated['area_ha'] * 10_000
        return state

    with pytest.warns(UserWarning, match='has no rows; saved an empty table'):
        (saved,) = _curate(monkeypatch, pd.DataFrame(), needs_area)
    assert saved.empty
    assert list(saved.columns) == []


def test_a_table_with_rows_runs_its_steps(monkeypatch):
    entity = pd.DataFrame({'area_ha': [0.1]}, index=pd.Index(['p1'], name='parcel_id'))

    def needs_area(state):
        state.curated['area_m2'] = state.curated['area_ha'] * 10_000
        return state

    (saved,) = _curate(monkeypatch, entity, needs_area)
    assert saved.loc['p1', 'area_m2'] == 1_000
