"""Slice b2 of the building spine: the footprint geospine's links re-keyed
to building ids, and the footprint spine as a projection of the building
spine.

plans/core-schema-and-stage-contracts-review.md, deliverable 3b.
Fabricated frames throughout.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from openplaces.core.schema import Entity
from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer import load as load_module
from openplaces.io.harmonizer.load import (
    _rekey_frame,
    _restore_foreign_links,
    adopt_entity_attributes,
)
from openplaces.io.harmonizer.spine import _stamp_geometry_source


def _mapping():
    return pd.Series(['b1', 'b2'], index=['f1', 'f2'])


def test_rekey_renames_a_multiindex_level_and_keeps_unmapped_rows():
    """A sidecar pair whose footprint a later geometry step removed keeps
    its id: main's attribute steps compute over those rows too."""
    frame = pd.DataFrame(
        {'area_intersection_m2': [1.0, 2.0, 3.0]},
        index=pd.MultiIndex.from_tuples(
            [('f1', 'p1'), ('f2', 'p1'), ('f9', 'p2')],
            names=['footprint_id', 'parcel_id'],
        ),
    )
    out = _rekey_frame(frame, 'footprint_id', 'building_id', _mapping())
    assert out.index.names == ['building_id', 'parcel_id']
    assert out.index.tolist() == [('b1', 'p1'), ('b2', 'p1'), ('f9', 'p2')]
    assert out['area_intersection_m2'].tolist() == [1.0, 2.0, 3.0]


def test_rekey_maps_a_column_and_keeps_nulls_null():
    frame = pd.DataFrame(
        {'footprint_id': ['f1', None, 'f9'], 'n_dwellings': [1, 2, 3]},
        index=pd.Index(['n1', 'n2', 'n3'], name='point_id'),
    )
    out = _rekey_frame(frame, 'footprint_id', 'building_id', _mapping())
    assert list(out.columns) == ['building_id', 'n_dwellings']
    assert out['building_id'].tolist()[0] == 'b1'
    assert pd.isna(out['building_id'].tolist()[1])
    assert out['building_id'].tolist()[2] == 'f9'
    assert out['n_dwellings'].tolist() == [1, 2, 3]


def test_rekey_leaves_a_frame_without_the_key_alone():
    frame = pd.DataFrame({'x': [1]}, index=pd.Index(['a'], name='other_id'))
    assert _rekey_frame(frame, 'footprint_id', 'building_id', _mapping()) is frame


def _building_state(footprint_ids):
    spine = gpd.GeoDataFrame(
        {'footprint_id': footprint_ids},
        geometry=[box(i, 0, i + 1, 1) for i in range(len(footprint_ids))],
        crs='epsg:6933',
        index=pd.Index(
            [f'b{i}' for i in range(len(footprint_ids))], name='building_id'
        ),
    )
    return HarmonizeState(
        recipe={'entity': Entity('building', 'spine', '2026')},
        admin_id='US-XX-YY',
        verbose=False,
        timer=None,
        spine=spine,
    )


def test_foreign_links_refuse_rows_sharing_a_source_id(monkeypatch):
    monkeypatch.setattr(
        load_module,
        'get_recipe_by_id',
        lambda rid: {'entity': Entity('footprint', 'geospine', '2026'), 'pipeline': []},
    )
    state = _building_state(['f1', 'f1'])
    with pytest.raises(ValueError, match='share a footprint_id'):
        _restore_foreign_links(state, 'US_footprint-geospine-2026', None, 'building_id')


def test_foreign_links_are_restored_with_the_source_key_and_rekeyed(monkeypatch):
    """The other geospine's link steps are restored keyed by its own id,
    then every crosswalk and overlay that arrived is re-keyed."""
    monkeypatch.setattr(
        load_module,
        'get_recipe_by_id',
        lambda rid: {
            'recipe_id': rid,
            'entity': Entity('footprint', 'geospine', '2026'),
            'pipeline': [
                {'step': 'resolve_spine'},
                {'step': 'link_to_reference', 'entity_type': 'parcel'},
                {'step': 'link_to_reference', 'recipe_id': 'x', 'save_link': False},
            ],
        },
    )
    calls = []

    def fake_restore(state, geospine, step_index, step_cfg, spine_id_col):
        calls.append((step_index, spine_id_col))
        state.crosswalks['US_parcel-x-2026'] = pd.DataFrame(
            {'link': ['unique parcel']},
            index=pd.MultiIndex.from_tuples(
                [('f1', 'p1')], names=['footprint_id', 'parcel_id']
            ),
        )
        state.overlays['US_parcel-x-2026'] = pd.DataFrame(
            {'area_intersection_m2': [5.0]},
            index=pd.MultiIndex.from_tuples(
                [('f1', 'p1')], names=['footprint_id', 'parcel_id']
            ),
        )
        return state

    monkeypatch.setattr(load_module, '_restore_link', fake_restore)
    state = _building_state(['f1', None])
    state.crosswalks['kept'] = pd.DataFrame({'footprint_id': ['f1']})
    state = _restore_foreign_links(
        state, 'US_footprint-geospine-2026', None, 'building_id'
    )
    assert calls == [(1, 'footprint_id')]
    assert state.crosswalks['US_parcel-x-2026'].index.tolist() == [('b0', 'p1')]
    assert state.crosswalks['US_parcel-x-2026'].index.names == [
        'building_id',
        'parcel_id',
    ]
    assert state.overlays['US_parcel-x-2026'].index.names == [
        'building_id',
        'parcel_id',
    ]
    # A crosswalk that was there before is not this recipe's to re-key.
    assert list(state.crosswalks['kept'].columns) == ['footprint_id']


def _footprint_state():
    spine = gpd.GeoDataFrame(
        {'geometry_source': ['obm', 'parcel.spine'], 'lat': [1.0, 2.0]},
        geometry=[box(0, 0, 1, 1), box(2, 0, 3, 1)],
        crs='epsg:6933',
        index=pd.Index(['f1', 'f2'], name='footprint_id'),
    )
    return HarmonizeState(
        recipe={'entity': Entity('footprint', 'spine', '2026')},
        admin_id='US-XX-YY',
        verbose=False,
        timer=None,
        spine=spine,
    )


def _buildings(footprint_ids=('f2', 'f1', None)):
    n = len(footprint_ids)
    return pd.DataFrame(
        {
            'footprint_id': list(footprint_ids),
            'lat': [20.0, 10.0, 30.0][:n],
            'n_parcels_per_building': [2, 1, 0][:n],
            'priority_on_parcel': pd.Categorical(
                ['primary', 'secondary', 'unknown'][:n],
                categories=['primary', 'secondary', 'unknown'],
            ),
        },
        index=pd.Index([f'b{i}' for i in range(n)], name='building_id'),
    )


def test_adopt_projects_every_column_through_the_key(monkeypatch):
    monkeypatch.setattr(load_module, 'get_entities', lambda *a, **k: _buildings())
    state = adopt_entity_attributes(
        _footprint_state(),
        'US_building-spine-2026',
        key='footprint_id',
        rename={'n_parcels_per_building': 'n_parcels_per_footprint'},
    )
    spine = state.spine
    # Existing columns keep their position, overwritten; new ones follow
    # in the building spine's order; the key is not copied.
    assert list(spine.columns) == [
        'geometry_source',
        'lat',
        'geometry',
        'n_parcels_per_footprint',
        'priority_on_parcel',
    ]
    assert spine['lat'].tolist() == [10.0, 20.0]
    assert spine['n_parcels_per_footprint'].tolist() == [1, 2]
    assert spine['priority_on_parcel'].tolist() == ['secondary', 'primary']
    assert isinstance(spine['priority_on_parcel'].dtype, pd.CategoricalDtype)


def test_adopt_refuses_a_footprint_no_building_names(monkeypatch):
    monkeypatch.setattr(
        load_module, 'get_entities', lambda *a, **k: _buildings(('f1',))
    )
    with pytest.raises(ValueError, match='have no row'):
        adopt_entity_attributes(_footprint_state(), 'US_building-spine-2026')


def test_adopt_refuses_two_buildings_on_one_footprint(monkeypatch):
    monkeypatch.setattr(
        load_module, 'get_entities', lambda *a, **k: _buildings(('f1', 'f1', 'f2'))
    )
    with pytest.raises(ValueError, match='share a footprint_id'):
        adopt_entity_attributes(_footprint_state(), 'US_building-spine-2026')


def test_adopt_ignores_a_rename_of_a_column_the_source_lacks(monkeypatch):
    """A permit count is absent where no permit source applies to the unit."""
    monkeypatch.setattr(load_module, 'get_entities', lambda *a, **k: _buildings())
    state = adopt_entity_attributes(
        _footprint_state(),
        'US_building-spine-2026',
        rename={'n_permits_per_building': 'n_permits_per_footprint'},
    )
    assert 'n_permits_per_footprint' not in state.spine.columns
    assert 'n_parcels_per_building' in state.spine.columns


def test_geometry_source_is_kept_from_the_row_only_when_listed():
    frame = pd.DataFrame({'geometry_source': ['obm', None]})
    _stamp_geometry_source(frame, 'footprint', ['geometry_source'])
    assert frame['geometry_source'].tolist() == ['obm', 'footprint']
    frame = pd.DataFrame({'geometry_source': ['obm', None]})
    _stamp_geometry_source(frame, 'footprint', [])
    assert frame['geometry_source'].tolist() == ['footprint', 'footprint']
