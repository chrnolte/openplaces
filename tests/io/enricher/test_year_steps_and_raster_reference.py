"""Tests for the Enricher's years expansion and recipe-addressed rasters."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from openplaces.core.schema import AdminId
from openplaces.io.enricher import EnrichState, expand_year_steps
from openplaces.io.enricher import zonal as zonal_module
from openplaces.io.enricher.zonal import resolve_raster_reference


def test_expand_year_steps_unrolls_and_substitutes():
    pipeline = [
        {'step': 'a'},
        {
            'step': 'zonal_stats',
            'years': [2001, 2003],
            'layer': 'forest',
            'column': 'forest_share_{year}',
            'stat': 'mean',
        },
        {'step': 'b'},
    ]
    out = expand_year_steps(pipeline)
    assert [s['step'] for s in out] == ['a'] + ['zonal_stats'] * 3 + ['b']
    assert [s['year'] for s in out[1:4]] == [2001, 2002, 2003]
    assert out[2]['column'] == 'forest_share_2002'
    assert out[2]['layer'] == 'forest'
    assert 'years' not in out[2]
    # The originals are untouched.
    assert 'year' not in pipeline[1]


def _state(admin='CO-AN-ABC'):
    return EnrichState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId(admin),
        verbose=False,
        timer=None,
        spine=pd.DataFrame(),
        evidence=pd.DataFrame(),
    )


@pytest.fixture
def stub_recipe(monkeypatch):
    calls = {}
    monkeypatch.setattr(zonal_module, 'get_recipe_by_id', lambda rid: {'id': rid})
    monkeypatch.setattr(zonal_module, 'get_save_admin_level', lambda r: 2)

    def fake_get_dataset(recipe, admin_id, partition_id=None):
        calls['admin_id'] = str(admin_id)
        calls['partition_id'] = partition_id
        return Path(f'/x/{admin_id}_{partition_id}.tif')

    monkeypatch.setattr(zonal_module, 'get_dataset', fake_get_dataset)
    return calls


def test_recipe_reference_truncates_to_save_level(stub_recipe):
    path = resolve_raster_reference(
        _state('CO-AN-ABC'), raster_recipe_id='r', layer='forest', year=2005
    )
    assert stub_recipe['admin_id'] == 'CO-AN'
    assert stub_recipe['partition_id'] == 'forest_2005'
    assert path.name == 'CO-AN_forest_2005.tif'


def test_recipe_reference_needs_explicit_unit_when_coarser(stub_recipe):
    with pytest.raises(ValueError, match='raster_admin_id'):
        resolve_raster_reference(_state('CO'), raster_recipe_id='r')
    resolve_raster_reference(
        _state('CO'), raster_recipe_id='r', raster_admin_id='CO-AN'
    )
    assert stub_recipe['admin_id'] == 'CO-AN'
    assert stub_recipe['partition_id'] is None


def test_reference_arguments_are_exclusive_and_paired(stub_recipe):
    with pytest.raises(ValueError, match='exactly one'):
        resolve_raster_reference(_state(), raster_path='a.tif', raster_recipe_id='r')
    with pytest.raises(ValueError, match='exactly one'):
        resolve_raster_reference(_state())
    with pytest.raises(ValueError, match='go together'):
        resolve_raster_reference(_state(), raster_recipe_id='r', layer='forest')


def test_path_reference_resolves_under_raster_root(monkeypatch):
    monkeypatch.setattr(
        zonal_module, 'resolve_raster_path', lambda p: Path('/root') / p
    )
    assert resolve_raster_reference(_state(), raster_path='x/y.tif') == Path(
        '/root/x/y.tif'
    )
