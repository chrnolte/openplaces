"""A spine source that covers the unit must have been ingested for it.

An ingest that ran for a unit always leaves a file: rows, or an empty
table for a unit the source does not publish. A missing file of a
source covering the unit is an unfinished build, and the spine steps
used to skip it unseen (a parcel geospine built before a county's roll
was ingested lacked the roll's values). Discovery also stops listing a
source whose crosswalk sidecar leaves the unit out, since the DAG never
schedules its ingest there.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

import openplaces.io.harmonizer.source_outputs as source_outputs
import openplaces.io.harmonizer.spine as spine_module
import openplaces.io.ingester.scope as scope_module
import openplaces.recipe as recipe_module
from openplaces.core.schema import AdminId, Entity
from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer.source_outputs import (
    SourceNotIngestedError,
    require_source_ingested,
)

ROLL = 'US-ZZ_property-roll-2026'
LAYER = 'US-ZZ_parcel-layer-2026'


def _state(entity_type='property', admin_id='US-ZZ-AAA'):
    return HarmonizeState(
        recipe={
            'admin_id': AdminId('US'),
            'entity': Entity(entity_type, 'spine', '2026'),
        },
        admin_id=AdminId(admin_id),
        verbose=False,
        timer=None,
    )


def _ingest_recipe(recipe_id, crosswalk=False):
    entity_type = recipe_id.split('_')[1].split('-')[0]
    recipe = {
        'admin_id': AdminId('US-ZZ'),
        'entity': Entity(entity_type, recipe_id.split('-')[2], '2026'),
        'save_to': {'admin_level': 3},
        'process_by': {'admin_level': 3},
    }
    if crosswalk:
        recipe['process_by']['admin_id_crosswalk'] = {'recipe_id': 'zz-crosswalk'}
    return recipe


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Fabricated recipes whose outputs live under *tmp_path*."""
    recipes = {ROLL: _ingest_recipe(ROLL), LAYER: _ingest_recipe(LAYER)}
    paths = {}

    def output_path(recipe, admin_id, partition_id=None):
        rid = next(k for k, v in recipes.items() if v is recipe)
        return tmp_path / f'{admin_id}_{rid}.parquet'

    def get_recipe(recipe_id, **kwargs):
        if recipe_id not in recipes:
            raise OSError(recipe_id)
        return recipes[recipe_id]

    monkeypatch.setattr(recipe_module, 'get_recipe_by_id', get_recipe)
    monkeypatch.setattr(spine_module, 'get_recipe_by_id', get_recipe)
    monkeypatch.setattr(scope_module, 'get_output_path', output_path)
    monkeypatch.setattr(
        scope_module,
        '_get_output_admin_ids',
        lambda recipe, admin_id: ([AdminId(str(admin_id))], [], []),
    )
    monkeypatch.setattr(scope_module, 'crosswalk_coverage', lambda recipe: None)
    monkeypatch.setattr(source_outputs, 'is_undiscoverable', lambda rid: False)
    return {'recipes': recipes, 'paths': paths, 'dir': tmp_path}


def _write(world, recipe_id, admin_id='US-ZZ-AAA', rows=1):
    frame = pd.DataFrame({'parcel_id_local': [f'P{i}' for i in range(rows)]})
    frame.to_parquet(world['dir'] / f'{admin_id}_{recipe_id}.parquet')


def test_missing_output_of_covering_source_raises(world):
    with pytest.raises(SourceNotIngestedError) as err:
        require_source_ingested(ROLL, 'US-ZZ-AAA', 'link_by_id')
    message = str(err.value)
    assert ROLL in message and 'US-ZZ-AAA' in message and 'link_by_id' in message


def test_saved_table_without_rows_is_an_observed_none(world):
    _write(world, ROLL, rows=0)
    require_source_ingested(ROLL, 'US-ZZ-AAA', 'link_by_id')


def test_unit_outside_the_crosswalk_is_not_owed(world, monkeypatch):
    world['recipes'][ROLL] = _ingest_recipe(ROLL, crosswalk=True)
    monkeypatch.setattr(
        scope_module, 'crosswalk_coverage', lambda recipe: {'US-ZZ-BBB'}
    )
    require_source_ingested(ROLL, 'US-ZZ-AAA', 'link_by_id')
    with pytest.raises(SourceNotIngestedError):
        require_source_ingested(ROLL, 'US-ZZ-BBB', 'link_by_id')


def test_held_or_unloadable_recipes_are_not_checked(world, monkeypatch):
    require_source_ingested('US-ZZ_property-unknown-2026', 'US-ZZ-AAA', 'x')
    monkeypatch.setattr(source_outputs, 'is_undiscoverable', lambda rid: True)
    require_source_ingested(ROLL, 'US-ZZ-AAA', 'x')


def _discover(monkeypatch, recipe_ids):
    rows = [
        {
            'recipe_id': rid,
            'admin_id': 'US-ZZ',
            'source_id': rid.split('-')[2],
            'version': '2026',
            'exclude_from_auto_discover': False,
        }
        for rid in recipe_ids
    ]
    monkeypatch.setattr(
        spine_module, 'find_recipes', lambda *a, **k: pd.DataFrame(rows)
    )
    monkeypatch.setattr(
        recipe_module, 'find_additional_layer_recipes', lambda *a, **k: []
    )


def _absent(*args, **kwargs):
    raise FileNotFoundError('1 recipe output file(s) do not exist')


def test_union_spine_sources_raises_on_an_uningested_roll(world, monkeypatch):
    _discover(monkeypatch, [ROLL])
    # missing='ignore' reads an absent file as no rows.
    monkeypatch.setattr(spine_module, 'get_entities', lambda *a, **k: pd.DataFrame())
    with pytest.raises(SourceNotIngestedError):
        spine_module.union_spine_sources(
            _state(), sources=[{'auto_discover': True, 'entity_type': 'property'}]
        )


def test_union_spine_sources_reads_an_empty_roll(world, monkeypatch):
    _discover(monkeypatch, [ROLL])
    _write(world, ROLL, rows=0)
    monkeypatch.setattr(spine_module, 'get_entities', lambda *a, **k: pd.DataFrame())
    with pytest.warns(UserWarning, match='no rows loaded'):
        state = spine_module.union_spine_sources(
            _state(), sources=[{'auto_discover': True, 'entity_type': 'property'}]
        )
    assert state.spine is None


def test_resolve_spine_raises_on_an_uningested_layer(world, monkeypatch):
    _discover(monkeypatch, [LAYER])
    monkeypatch.setattr(spine_module, 'get_entities', _absent)
    with pytest.raises(SourceNotIngestedError):
        spine_module.resolve_spine(
            _state('parcel'),
            sources=[{'auto_discover': True, 'entity_type': 'parcel'}],
        )


def test_resolve_spine_builds_from_an_ingested_layer(world, monkeypatch):
    _discover(monkeypatch, [LAYER])
    _write(world, LAYER)
    layer = gpd.GeoDataFrame(
        {'parcel_id_local': ['P0']},
        geometry=[box(0, 0, 0.001, 0.001)],
        index=pd.Index(['g0'], name='parcel_id'),
        crs=None,
    )
    monkeypatch.setattr(spine_module, 'get_entities', lambda *a, **k: layer)
    state = spine_module.resolve_spine(
        _state('parcel'), sources=[{'auto_discover': True, 'entity_type': 'parcel'}]
    )
    assert len(state.spine) == 1


def test_auto_discovered_roll_join_raises_on_an_uningested_roll(world, monkeypatch):
    import openplaces.io.harmonizer.links.by_id as by_id_module

    match = {
        'recipe_id': ROLL,
        'layer': None,
        'key': 'parcel_id_local',
        'aggregation_function': None,
        'supplements': None,
        'supplements_key': None,
    }
    monkeypatch.setattr(by_id_module, '_discover_link_sources', lambda s, e: [match])
    monkeypatch.setattr(by_id_module, 'get_entities', _absent)
    state = _state('parcel')
    state.spine = pd.DataFrame(
        {'parcel_id_local': ['P0']}, index=pd.Index(['g0'], name='parcel_id')
    )

    with pytest.raises(SourceNotIngestedError, match='link_by_id'):
        by_id_module.link_by_id(
            state, auto_discover=True, entity_type='property', mode='aggregate'
        )


def test_discovery_skips_a_source_whose_crosswalk_leaves_the_unit_out(
    world, monkeypatch
):
    world['recipes'][ROLL] = _ingest_recipe(ROLL, crosswalk=True)
    monkeypatch.setattr(
        scope_module, 'crosswalk_coverage', lambda recipe: {'US-ZZ-BBB'}
    )
    _discover(monkeypatch, [ROLL])

    listed_a = spine_module._expand_auto_discover(
        [{'auto_discover': True, 'entity_type': 'property'}], _state()
    )
    listed_b = spine_module._expand_auto_discover(
        [{'auto_discover': True, 'entity_type': 'property'}],
        _state(admin_id='US-ZZ-BBB'),
    )

    assert [s['recipe_id'] for s in listed_a] == []
    assert [s['recipe_id'] for s in listed_b] == [ROLL]


def test_crosswalk_keyed_to_another_level_claims_nothing(monkeypatch):
    recipe = _ingest_recipe(ROLL, crosswalk=True)
    assert scope_module.crosswalk_lists_unit(recipe, 'US-ZZ-AAA', {'US-ZZ'})
    assert scope_module.crosswalk_lists_unit(recipe, 'US-ZZ', {'US-ZZ-AAA'})
    assert not scope_module.crosswalk_lists_unit(recipe, 'US-ZZ-AAA-001', {'US-ZZ-BBB'})
