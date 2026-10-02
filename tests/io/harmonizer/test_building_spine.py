"""The building spine, slice b1: one building per footprint, its own id,
the footprint id kept on the row, and a link table keyed on it.

plans/core-schema-and-stage-contracts-review.md, deliverable 3b. Fabricated
frames throughout; the recipes are read from the tree.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from openplaces.core.schema import Entity
from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer.entity_links import build_id_links
from openplaces.io.harmonizer.spine import adopt_source_entity_id
from openplaces.recipe import get_recipe_by_id, get_recipe_dependencies


def _state(spine):
    return HarmonizeState(
        recipe={'entity': Entity('building', 'geospine', '2026')},
        admin_id='US-XX-YY',
        verbose=False,
        timer=None,
        spine=spine,
    )


def _footprints():
    return gpd.GeoDataFrame(
        {'admin4_id': ['US-XX-YY-A', 'US-XX-YY-A']},
        geometry=[box(0, 0, 1, 1), box(2, 0, 3, 1)],
        crs='epsg:6933',
        index=pd.Index(['f1', 'f2'], name='footprint_id'),
    )


def test_the_source_id_becomes_a_column_and_the_index_is_the_entity_id():
    state = adopt_source_entity_id(
        _state(_footprints()), source_id_column='footprint_id'
    )
    assert state.spine.index.name == 'building_id'
    assert state.spine.index.tolist() == ['f1', 'f2']
    assert state.spine['footprint_id'].tolist() == ['f1', 'f2']
    assert state.spine['admin4_id'].tolist() == ['US-XX-YY-A', 'US-XX-YY-A']


def test_a_source_id_column_already_on_the_spine_is_refused():
    spine = _footprints()
    spine['footprint_id'] = ['x', 'y']
    with pytest.raises(ValueError, match='already carries'):
        adopt_source_entity_id(_state(spine), source_id_column='footprint_id')


def test_running_twice_is_refused():
    state = adopt_source_entity_id(
        _state(_footprints()), source_id_column='footprint_id'
    )
    with pytest.raises(ValueError, match='already indexed'):
        adopt_source_entity_id(state, source_id_column='footprint_id_2')


def test_a_link_key_may_be_the_other_sides_index():
    """The footprint spine's id is its index, not a column."""
    buildings = pd.DataFrame(
        {'footprint_id': ['f1', 'f2', 'f9']},
        index=pd.Index(['f1', 'f2', 'f9'], name='building_id'),
    )
    footprints = pd.DataFrame(
        {'area_m2': [10.0, 20.0]}, index=pd.Index(['f1', 'f2'], name='footprint_id')
    )
    links = build_id_links(
        buildings,
        footprints,
        'footprint_id',
        'footprint_id',
        'building_id',
        'footprint_id',
        'footprint_outline',
        source_column=None,
    )
    assert links[['building_id', 'footprint_id']].to_numpy().tolist() == [
        ['f1', 'f1'],
        ['f2', 'f2'],
    ]
    assert set(links['link_method']) == {'footprint_outline'}


def test_a_key_neither_side_has_yields_no_links():
    frame = pd.DataFrame({'a': [1]}, index=pd.Index(['x'], name='building_id'))
    other = pd.DataFrame({'b': [1]}, index=pd.Index(['x'], name='footprint_id'))
    assert build_id_links(frame, other, 'nope', 'nope', 'b', 'f', 'm').empty


def test_the_building_recipes_load_and_depend_on_the_footprint_geospine():
    geospine = get_recipe_by_id('US_building-geospine-2026')
    spine = get_recipe_by_id('US_building-spine-2026')
    assert str(geospine['entity'].entity_type) == 'building'
    assert spine['entity_recipe'] == 'US_building-geospine-2026'
    assert [s['step'] for s in geospine['pipeline']] == [
        'resolve_spine',
        'adopt_source_entity_id',
        'derive_geometry_attributes',
    ]
    upstream = {e.upstream_recipe_id for e in get_recipe_dependencies(geospine)}
    assert 'US_footprint-geospine-2026' in upstream
    upstream = {e.upstream_recipe_id for e in get_recipe_dependencies(spine)}
    assert {'US_building-geospine-2026', 'US_footprint-geospine-2026'} <= upstream
    # The footprint spine is downstream of the building spine (it projects
    # its attributes), so the building spine must not read it.
    assert 'US_footprint-spine-2026' not in upstream


def test_the_footprint_spine_projects_the_building_spine():
    """Slice b2: the attribute steps run once, on the building spine, and
    the footprint spine adopts the result under footprint-named counts."""
    footprint = get_recipe_by_id('US_footprint-spine-2026')
    building = get_recipe_by_id('US_building-spine-2026')
    assert [s['step'] for s in footprint['pipeline']] == [
        'load_geospine',
        'adopt_entity_attributes',
    ]
    adopt = footprint['pipeline'][1]
    assert adopt['recipe_id'] == 'US_building-spine-2026'
    assert adopt['key'] == 'footprint_id'
    building_steps = [s['step'] for s in building['pipeline']]
    assert building_steps[:2] == ['load_geospine', 'link_entities_by_id']
    assert building['pipeline'][0]['links_recipe_id'] == 'US_footprint-geospine-2026'
    for moved in (
        'classify_footprint_priority',
        'reconcile_attributes',
        'reconcile_addresses',
        'reconcile_postal_code',
        'impute_postal_city',
        'derive_address_id_local',
        'link_by_id',
    ):
        assert moved in building_steps
    # Every building-named count the building spine writes is renamed on
    # projection, and only those.
    count_as = {s['count_as'] for s in building['pipeline'] if 'count_as' in s}
    assert count_as <= set(adopt['rename'])
    assert all(
        v.endswith(('_footprint', '_footprint_address', '_per_parcel'))
        for v in adopt['rename'].values()
    )
    upstream = {e.upstream_recipe_id for e in get_recipe_dependencies(footprint)}
    assert {'US_footprint-geospine-2026', 'US_building-spine-2026'} <= upstream
