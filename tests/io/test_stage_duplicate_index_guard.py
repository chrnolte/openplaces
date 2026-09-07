"""Each pipeline stage refuses an entity frame whose index repeats.

Within one output file a duplicate index cannot happen (the ingester
raises on one), but per-admin files are concatenated on read, so an
entity curated by two neighboring counties can arrive twice. Harmonize,
enrich and curate all align on the entity id, so each names the admin
unit, the recipe and the offending labels instead of failing later
inside a step's reindex.
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from openplaces.core.schema import AdminId
from openplaces.io import curator as curator_module
from openplaces.io import enricher as enricher_module
from openplaces.io import harmonizer as harmonizer_module
from openplaces.io.curator import Curator
from openplaces.io.enricher import Enricher
from openplaces.io.harmonizer import Harmonizer

ADMIN_ID = AdminId('US-NC-WAK')


def _frame(labels):
    return gpd.GeoDataFrame(
        {'geometry': [box(i, 0, i + 1, 1) for i in range(len(labels))]},
        index=pd.Index(labels, name='footprint_id'),
        crs='epsg:6933',
    )


def _instance(cls, **attributes):
    instance = object.__new__(cls)
    instance.__dict__.update(attributes)
    return instance


def _curator(labels, monkeypatch, steps):
    monkeypatch.setattr(curator_module, 'get_entities', lambda *a, **k: _frame(labels))
    monkeypatch.setattr(curator_module, 'save_parquet', lambda *a, **k: None)
    monkeypatch.setattr(curator_module, 'get_output_path', lambda *a, **k: 'out')
    monkeypatch.setitem(curator_module._STEP_REGISTRY, 'noop', steps)
    return _instance(
        Curator,
        recipe={'recipe_id': 'US_footprint-cheer-2026', 'pipeline': [{'step': 'noop'}]},
        entity_recipe={},
        verbose=False,
        save_statistics=False,
        skip_steps=set(),
        _timer=None,
    )


def test_curate_refuses_a_repeated_entity_id(monkeypatch):
    ran = []
    curator = _curator(['a', 'a', 'b'], monkeypatch, lambda state: ran.append(1))
    with pytest.raises(ValueError) as excinfo:
        curator._curate_one(ADMIN_ID)
    message = str(excinfo.value)
    assert 'curate US_footprint-cheer-2026 for US-NC-WAK' in message
    assert 'footprint_id' in message
    assert "'a'" in message
    assert not ran, 'the guard must fire before any curate step runs'


def test_curate_runs_when_the_index_is_unique(monkeypatch):
    ran = []

    def step(state):
        ran.append(1)
        return state

    _curator(['a', 'b', 'c'], monkeypatch, step)._curate_one(ADMIN_ID)
    assert ran == [1]


def _enricher(labels, monkeypatch):
    monkeypatch.setattr(enricher_module, 'get_entities', lambda *a, **k: _frame(labels))
    return _instance(
        Enricher,
        recipe={
            'recipe_id': 'US_footprint-brails-2026',
            'pipeline': [{'step': 'noop'}],
        },
        entity_recipe={},
        verbose=False,
        _timer=None,
    )


def test_enrich_refuses_a_repeated_entity_id(monkeypatch):
    enricher = _enricher(['a', 'a', 'b'], monkeypatch)
    with pytest.raises(ValueError) as excinfo:
        enricher._enrich_one(ADMIN_ID)
    message = str(excinfo.value)
    assert 'enrich US_footprint-brails-2026 for US-NC-WAK' in message
    assert "'a'" in message


def test_enrich_gets_past_the_guard_when_the_index_is_unique(monkeypatch):
    enricher = _enricher(['a', 'b', 'c'], monkeypatch)
    # The unknown step is reached only once the guard has let the spine
    # through, so its message is what a unique index looks like here.
    with pytest.raises(ValueError, match='Unknown'):
        enricher._enrich_one(ADMIN_ID)


def _harmonizer(labels, monkeypatch):
    def build_spine(state):
        state.spine = _frame(labels)
        return state

    monkeypatch.setitem(harmonizer_module._STEP_REGISTRY, 'build_spine', build_spine)
    monkeypatch.setattr(harmonizer_module, 'save_parquet', lambda *a, **k: None)
    monkeypatch.setattr(harmonizer_module, 'get_output_path', lambda *a, **k: 'out')
    monkeypatch.setattr(harmonizer_module, 'saves_geometry', lambda recipe: True)
    return _instance(
        Harmonizer,
        recipe={
            'recipe_id': 'US_footprint-spine-2026',
            'pipeline': [{'step': 'build_spine'}],
        },
        verbose=False,
        save_statistics=False,
        _timer=None,
    )


def test_harmonize_refuses_a_spine_whose_index_repeats(monkeypatch):
    harmonizer = _harmonizer(['a', 'a', 'b'], monkeypatch)
    with pytest.raises(ValueError) as excinfo:
        harmonizer._harmonize_one(ADMIN_ID)
    message = str(excinfo.value)
    assert 'harmonize US_footprint-spine-2026 for US-NC-WAK' in message
    assert 'build_spine' in message
    assert "'a'" in message


def test_harmonize_saves_a_unique_spine(monkeypatch):
    saved = []
    harmonizer = _harmonizer(['a', 'b', 'c'], monkeypatch)
    monkeypatch.setattr(
        harmonizer_module, 'save_parquet', lambda spine, *a, **k: saved.append(spine)
    )
    harmonizer._harmonize_one(ADMIN_ID)
    assert len(saved) == 1
    assert saved[0].index.tolist() == ['a', 'b', 'c']
