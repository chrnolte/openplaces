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


def test_a_rekeyed_sidecar_keeps_its_rows_labels_and_footer(tmp_path):
    """b3 step 1: the building spine persists the footprint geospine's
    sidecars under building ids, with the link label the curate readers
    need and the footer fingerprint intact."""
    import json

    from openplaces.io import to_parquet
    from openplaces.io.aggregate import read_file_metadata
    from openplaces.io.harmonizer.links._shared import _LINK_METADATA_KEY
    from openplaces.io.harmonizer.load import (
        LINK_REKEYED_FROM_KEY,
        _persist_rekeyed_sidecar,
    )

    source = tmp_path / 'fp_parcel.parquet'
    to_parquet(
        pd.DataFrame(
            {
                'footprint_id': ['f1', 'f2', 'f9'],
                'parcel_id': ['p1', 'p2', 'p3'],
                'area_intersection_m2': [10.0, 20.0, 1.0],
                'link': ['unique parcel', 'dominant parcel', None],
            }
        ),
        source,
        file_metadata={_LINK_METADATA_KEY: json.dumps({'format': 2})},
        index=False,
    )
    mapping = pd.Series(['b1', 'b2'], index=['f1', 'f2'])
    target = tmp_path / 'nested' / 'bld_parcel.parquet'
    _persist_rekeyed_sidecar(
        source,
        target,
        'US_footprint-geospine-2026',
        'footprint_id',
        'building_id',
        mapping,
    )
    out = pd.read_parquet(target)
    # Renamed and re-keyed; the row no spine row names keeps its id.
    assert list(out.columns) == [
        'building_id',
        'parcel_id',
        'area_intersection_m2',
        'link',
    ]
    assert out['building_id'].tolist() == ['b1', 'b2', 'f9']
    assert out['link'].tolist()[:2] == ['unique parcel', 'dominant parcel']
    footer = read_file_metadata(target)
    assert json.loads(footer[_LINK_METADATA_KEY]) == {'format': 2}
    assert footer[LINK_REKEYED_FROM_KEY] == 'US_footprint-geospine-2026'
    # A missing source writes nothing.
    _persist_rekeyed_sidecar(
        tmp_path / 'absent.parquet', tmp_path / 'never.parquet', 'x', 'a', 'b', mapping
    )
    assert not (tmp_path / 'never.parquet').exists()


def test_a_point_link_after_load_links_is_fingerprinted_by_the_parcel_overlay(
    mock_data_root,
):
    """b3 step 1: the building geospine's point joins read the footprint
    geospine's parcel overlay, a harmonize product no ingest entry of the
    fingerprint covers, so a rewritten overlay must make them stale."""
    from openplaces.geo.link import get_entity_link_path
    from openplaces.io.harmonizer.links.sidecars import _link_fingerprint

    unit = 'US-NC-BRU'
    building = get_recipe_by_id('US_building-geospine-2026')
    steps = [s['step'] for s in building['pipeline']]
    nsi_index = steps.index('link_to_reference')
    assert steps.index('load_links') < nsi_index

    def fingerprint(recipe, step_index, ref):
        state = HarmonizeState(recipe=recipe, admin_id=unit, verbose=False, timer=None)
        state.step_index = step_index
        return _link_fingerprint(state, ref, {'join': 'spatial_point'})

    overlay = get_entity_link_path(
        'US_footprint-geospine-2026', 'US-NC_parcel-nconemap-2025', unit
    )
    absent = fingerprint(building, nsi_index, 'US_building-nsi-2026')
    assert absent['restored_links'] == {overlay.name: None}

    overlay.parent.mkdir(parents=True, exist_ok=True)
    overlay.write_bytes(b'one')
    first = fingerprint(building, nsi_index, 'US_building-nsi-2026')
    overlay.write_bytes(b'two')
    second = fingerprint(building, nsi_index, 'US_building-nsi-2026')
    assert first['restored_links'][overlay.name] is not None
    assert first['restored_links'] != second['restored_links']

    # No other link gets the key: the footprint geospine's own overlay
    # fingerprint is the one already stored in every sidecar on disk.
    footprint = get_recipe_by_id('US_footprint-geospine-2026')
    overlay_index = [s['step'] for s in footprint['pipeline']].index(
        'link_to_reference'
    )
    plain = fingerprint(footprint, overlay_index, 'US-NC_parcel-nconemap-2025')
    assert 'restored_links' not in plain


def test_the_building_recipes_load_and_depend_on_the_footprint_geospine():
    geospine = get_recipe_by_id('US_building-geospine-2026')
    spine = get_recipe_by_id('US_building-spine-2026')
    assert str(geospine['entity'].entity_type) == 'building'
    assert spine['entity_recipe'] == 'US_building-geospine-2026'
    assert [s['step'] for s in geospine['pipeline']] == [
        'resolve_spine',
        'adopt_source_entity_id',
        'derive_geometry_attributes',
        'load_links',
        'link_to_reference',
        'link_to_reference',
    ]
    upstream = {e.upstream_recipe_id for e in get_recipe_dependencies(geospine)}
    assert 'US_footprint-geospine-2026' in upstream
    # b3 step 1: the point joins run on the building's own polygon.
    assert {'US_building-nsi-2026', 'dwelling-overture-2025'} <= upstream
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
    # The re-keyed sidecars are written beside the building geospine for
    # the curate steps that read a link by the curated entity's geospine.
    assert building['pipeline'][0]['persist_links'] is True
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
