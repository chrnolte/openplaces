"""Harmonize steps declare the columns they write, truthfully.

plans/step-column-declarations-harmonize-enrich.md. Each test runs a
declared step on a fabricated fixture and checks that every column it
adds is declared, and that a column the declaration says it decides
is rebuilt on every row. Every id, label and count below is fabricated.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

import openplaces.io.readers as readers
from openplaces.core.schema import SourceGeometryType
from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer.attributes.dwelling_address import (
    attribute_dwelling_address,
)
from openplaces.io.harmonizer.attributes.morphology import (
    MORPHOLOGY_COLUMNS,
    summarize_footprint_morphology,
)
from openplaces.io.harmonizer.attributes.priority import classify_footprint_priority
from openplaces.io.harmonizer.attributes.property_counts import (
    estimate_property_counts,
)
from openplaces.io.harmonizer.attributes.reconcile import rename_columns
from openplaces.io.harmonizer.attributes.shared_land import (
    detect_shared_land_groups,
)
from openplaces.io.harmonizer.attributes.use_classes import derive_use_classes
from openplaces.io.steps import resolve_step_columns, undeclared_new_columns


def _state(spine, **fields):
    return HarmonizeState(
        recipe={}, admin_id='US-XX-YY', verbose=False, timer=None, spine=spine, **fields
    )


def _added(step, params, before, after):
    """Columns *step* added that its declaration does not list."""
    return undeclared_new_columns(step, params, before, after)


def _priority_state():
    spine = gpd.GeoDataFrame(
        {
            'geometry_source': ['src_a', 'src_a', 'src_a'],
            'area_ha': [0.02, 0.005, 0.01],
        },
        index=pd.Index(['b1', 'b2', 'b3'], name='building_id'),
        geometry=[box(0, 0, 1, 1), box(2, 0, 3, 1), box(4, 0, 5, 1)],
        crs='EPSG:32617',
    )
    overlay = pd.DataFrame(
        {
            'building_id': ['b1', 'b2', 'b3'],
            'parcel_id': ['p1', 'p1', 'p2'],
            'area_intersection_m2': [200.0, 50.0, 100.0],
        }
    ).set_index(['building_id', 'parcel_id'])
    points = pd.DataFrame({'building_id': ['b2']}, index=pd.Index([7], name='pt'))
    return _state(
        spine,
        crosswalks={'lots_x': overlay, 'points_x': points},
        reference_types={'lots_x': 'parcel', 'points_x': 'dwelling'},
        source_geometry_types={
            'points_x': SourceGeometryType.single_dwelling_point,
        },
    )


def test_priority_declares_and_decides_both_columns_it_writes():
    resolved = resolve_step_columns(classify_footprint_priority, {})
    assert resolved.decides == {'priority_on_parcel', 'parcel_id_secondary'}
    assert resolved.writes == resolved.decides

    state = _priority_state()
    before = list(state.spine.columns)
    out = classify_footprint_priority(state).spine
    assert _added(classify_footprint_priority, {}, before, out.columns) == []
    # The dwelling point makes b2 primary and b1 secondary on p1.
    assert out['priority_on_parcel'].astype(str).tolist() == [
        'secondary',
        'primary',
        'primary',
    ]
    assert out['priority_on_parcel'].notna().all()


def test_priority_names_its_secondary_column_after_the_entity_type():
    resolved = resolve_step_columns(classify_footprint_priority, {'entity_type': 'lot'})
    assert 'lot_id_secondary' in resolved.decides


def _footprints():
    return gpd.GeoDataFrame(
        {
            'parcel_id': ['A', 'A', 'B'],
            'geometry_source': ['src_a', 'src_a', 'src_a'],
            'area_intersection_m2_parcel': [100.0, 60.0, 80.0],
            'priority_on_parcel': ['primary', 'secondary', 'primary'],
            'n_dwellings_overture': [2, 0, 1],
            'n_parcels_per_building': [1, 1, 1],
        },
        geometry=[box(0, 0, 10, 10), box(20, 0, 26, 10), box(100, 0, 109, 9)],
        crs='EPSG:32617',
    )


def test_morphology_decides_every_column_it_writes(monkeypatch):
    footprints = _footprints()
    monkeypatch.setattr(readers, 'get_entities', lambda *a, **k: footprints)
    spine = gpd.GeoDataFrame(
        index=pd.Index(['A', 'B', 'C'], name='parcel_id'),
        geometry=[box(-1, -1, 30, 11), box(99, -1, 110, 10), box(200, 0, 210, 10)],
        crs='EPSG:32617',
    )
    params = {'footprint_recipe_id': 'fp_x'}
    resolved = resolve_step_columns(summarize_footprint_morphology, params)
    assert resolved.writes == resolved.decides == set(MORPHOLOGY_COLUMNS)
    assert {'priority_on_parcel', 'n_dwellings_overture'} <= resolved.reads_from['fp_x']

    state = _state(spine)
    before = list(spine.columns)
    out = summarize_footprint_morphology(state, **params).spine
    assert _added(summarize_footprint_morphology, params, before, out.columns) == []
    assert set(MORPHOLOGY_COLUMNS) <= set(out.columns)
    # Decided: no row is left unset, the parcel without buildings included.
    for column in MORPHOLOGY_COLUMNS:
        if column.startswith(('n_', 'max_dwellings', 'max_parcels')):
            assert out[column].notna().all(), column
    assert out['n_buildings_per_parcel'].tolist() == [2, 1, 0]
    assert out['n_primary_buildings_per_parcel'].tolist() == [1, 1, 0]


def test_use_classes_decide_both_labels_and_read_every_alternative():
    resolved = resolve_step_columns(derive_use_classes, {})
    assert resolved.decides == {'use_group_combined', 'use_group_combined_labeled'}
    assert resolved.reads == {
        'use_group',
        'use_group_code',
        'use_subgroup',
        'use_subgroup_code',
    }
    none_labeled = resolve_step_columns(derive_use_classes, {'labeled_column': None})
    assert none_labeled.writes == {'use_group_combined'}

    spine = pd.DataFrame(
        {'use_group': ['Kind A', None], 'use_subgroup_code': ['c1', 'c2']},
        index=pd.Index(['p1', 'p2'], name='parcel_id'),
    )
    before = list(spine.columns)
    out = derive_use_classes(_state(spine)).spine
    assert _added(derive_use_classes, {}, before, out.columns) == []
    assert out['use_group_combined'].astype(str).tolist() == ['Kind A | c1', 'c2']


def test_property_counts_write_and_keep_an_earlier_source():
    resolved = resolve_step_columns(estimate_property_counts, {})
    assert resolved.writes == {'n_properties_per_parcel', 'property_source'}
    assert resolved.decides == frozenset()

    spine = pd.DataFrame(
        {
            'property_source': ['shared_land_group', None, None],
            'n_properties_per_parcel': [4, None, None],
            'n_dwellings': [0, 3, 0],
        },
        index=pd.Index(['p1', 'p2', 'p3'], name='parcel_id'),
    )
    out = estimate_property_counts(_state(spine.copy())).spine
    assert _added(estimate_property_counts, {}, spine.columns, out.columns) == []
    assert out['property_source'].astype(str).tolist() == [
        'shared_land_group',
        'estimated',
        'none',
    ]


def _shared_land_spine():
    # A compact valueless lot (L) between two valued units (U1, U2),
    # and one valued lot far away (F).
    return gpd.GeoDataFrame(
        {
            'land_value': [0.0, 10.0, 10.0, 10.0],
            'improvement_value': [0.0, 50.0, 50.0, 50.0],
            'area_ha': [0.01, 0.01, 0.01, 0.01],
        },
        index=pd.Index(['L', 'U1', 'U2', 'F'], name='parcel_id'),
        geometry=[
            box(10, 0, 20, 10),
            box(0, 0, 10, 10),
            box(20, 0, 30, 10),
            box(500, 0, 510, 10),
        ],
        crs='EPSG:32617',
    )


def test_shared_land_decides_the_group_id_and_writes_the_count():
    resolved = resolve_step_columns(detect_shared_land_groups, {})
    assert resolved.decides == {'shared_land_parcel_id'}
    assert resolved.writes == {
        'shared_land_parcel_id',
        'n_properties_per_parcel',
        'property_source',
    }

    spine = _shared_land_spine()
    before = list(spine.columns)
    out = detect_shared_land_groups(_state(spine)).spine
    assert _added(detect_shared_land_groups, {}, before, out.columns) == []
    assert out['shared_land_parcel_id'].tolist()[1:3] == ['L', 'L']
    assert pd.isna(out.loc['F', 'property_source'])


def test_dwelling_address_declares_its_default_relay(monkeypatch):
    footprints = pd.DataFrame(
        {
            'parcel_id': ['A'],
            'priority_on_parcel': ['primary'],
            'area_intersection_m2_parcel': [100.0],
            'address_street_dwelling_overture': ['SAMPLE ST'],
            'city_dwelling_overture': ['Sampletown'],
        }
    )
    monkeypatch.setattr(readers, 'get_entities', lambda *a, **k: footprints)
    params = {'footprint_recipe_id': 'fp_x'}
    resolved = resolve_step_columns(attribute_dwelling_address, params)
    assert 'address_street_dwelling_footprint' in resolved.decides
    assert 'address_street_dwelling_overture' in resolved.reads_from['fp_x']

    spine = pd.DataFrame(index=pd.Index(['A', 'B'], name='parcel_id'))
    out = attribute_dwelling_address(_state(spine), **params).spine
    assert _added(attribute_dwelling_address, params, [], out.columns) == []
    assert out['city_dwelling_footprint'].tolist()[0] == 'Sampletown'


def test_rename_declares_the_new_names():
    params = {'columns': {'address': 'address_original', 'absent': 'never'}}
    resolved = resolve_step_columns(rename_columns, params)
    assert resolved.writes == {'address_original', 'never'}
    assert resolved.decides == frozenset()
    spine = pd.DataFrame({'address': ['1 SAMPLE ST']})
    out = rename_columns(_state(spine), **params).spine
    assert _added(rename_columns, params, ['address'], out.columns) == []
