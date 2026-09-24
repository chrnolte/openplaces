"""A parcel table's stacked ownership units become a property layer.

One row per lot, every source record kept as a property, nothing
double-counted: a stack's parcel row keeps only what its members agree
on, exact duplicates collapse, singletons pass through untouched.
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from openplaces.io.stacked_units import split_stacked_units


def _table():
    # Lot A: two condo units sharing one outline (a stack). Lot B: one
    # house (a singleton). Lot C: the same record shipped twice (exact
    # duplicates). One row without a lot key.
    rows = pd.DataFrame(
        {
            'geo_id': ['A', 'A', 'B', 'C', 'C', None],
            'parcel_id_local': ['A1', 'A2', 'B1', 'C1', 'C1', 'Z'],
            'parcel_id_assessor': ['A-1', 'A-2', 'B-1', 'C-1', 'C-1', 'Z'],
            'land_value': [1000.0, 1000.0, 5000.0, 700.0, 700.0, 10.0],
            'improvement_value': [90000.0, 80000.0, 60000.0, 0.0, 0.0, None],
            'use_group': ['Condo', 'Condo', 'SF', 'SF', 'SF', 'SF'],
            'admin3_id': ['US-XX-YY'] * 6,
        },
        index=pd.Index(['r1', 'r2', 'r3', 'r4', 'r5', 'r6'], name='parcel_id'),
    )
    geoms = [box(0, 0, 1, 1)] * 2 + [box(2, 2, 3, 3)] + [box(4, 4, 5, 5)] * 2
    geoms.append(box(6, 6, 7, 7))
    return gpd.GeoDataFrame(rows, geometry=geoms, crs='EPSG:4326')


def test_a_stack_becomes_one_parcel_and_its_members_become_properties():
    result = split_stacked_units(_table())
    parcels, properties = result.parcels, result.properties

    assert list(parcels.index) == ['r1', 'r3', 'r4', 'r6']
    assert result.n_stacks == 1 and result.n_rows_to_properties == 2
    assert result.n_exact_duplicates == 1 and result.n_lots == 4
    # The stack's parcel keeps what its members agree on and nothing else;
    # an additive column stays empty even where the members agree.
    lot_a = parcels.loc['r1']
    assert lot_a['use_group'] == 'Condo'
    assert pd.isna(lot_a['land_value'])
    assert pd.isna(lot_a['improvement_value'])
    assert pd.isna(lot_a['parcel_id_assessor'])
    # The members disagreed on the link key, so the lot key stands in.
    assert lot_a['parcel_id_local'] == 'A'
    assert list(properties.index) == ['r1', 'r2']
    # A unit keeps its own key and names its lot beside it.
    assert properties['parcel_id_local'].tolist() == ['A1', 'A2']
    assert properties['lot_id_local'].tolist() == ['A', 'A']
    assert properties['improvement_value'].tolist() == [90000.0, 80000.0]
    assert 'geometry' not in properties.columns


def test_identical_units_do_not_stand_in_for_the_lot_total():
    # Three identical condo units on one outline: every member agrees on
    # its own floor area and dwelling count, which is not the lot's sum.
    rows = pd.DataFrame(
        {
            'geo_id': ['L', 'L', 'L'],
            'parcel_id_local': ['L1', 'L2', 'L3'],
            'living_area_sqft': [900.0, 900.0, 900.0],
            'n_dwellings': [1.0, 1.0, 1.0],
            'use_group': ['Condo', 'Condo', 'Condo'],
        },
        index=pd.Index(['u1', 'u2', 'u3'], name='parcel_id'),
    )
    table = gpd.GeoDataFrame(rows, geometry=[box(0, 0, 1, 1)] * 3, crs='EPSG:4326')
    result = split_stacked_units(table)
    lot = result.parcels.loc['u1']
    assert pd.isna(lot['living_area_sqft']) and pd.isna(lot['n_dwellings'])
    assert lot['use_group'] == 'Condo'
    assert result.properties['living_area_sqft'].sum() == 2700.0


def test_singletons_and_the_unkeyed_row_pass_through_unchanged():
    table = _table()
    parcels = split_stacked_units(table).parcels
    for row in ('r3', 'r6'):
        pd.testing.assert_series_equal(
            parcels.loc[row].drop('geometry'), table.loc[row].drop('geometry')
        )
    assert isinstance(parcels, gpd.GeoDataFrame)
    assert parcels.geometry.loc['r3'].equals(box(2, 2, 3, 3))


def test_a_link_key_the_members_share_is_kept():
    table = _table()
    table.loc[['r1', 'r2'], 'parcel_id_local'] = 'A'
    result = split_stacked_units(table)
    assert result.parcels.loc['r1', 'parcel_id_local'] == 'A'
    assert result.properties['parcel_id_local'].tolist() == ['A', 'A']


def test_a_source_lot_id_can_replace_the_geometry_hash():
    table = _table()
    table['lot_id'] = ['L1', 'L1', 'L2', 'L3', 'L3', 'L4']
    result = split_stacked_units(table, lot_key='lot_id')
    assert result.n_stacks == 1
    assert result.properties['lot_id_local'].tolist() == ['L1', 'L1']
    assert result.properties['parcel_id_local'].tolist() == ['A1', 'A2']


def test_no_stack_means_no_property_table():
    table = _table().loc[['r3', 'r6']]
    result = split_stacked_units(table)
    assert result.properties is None and result.n_stacks == 0
    assert list(result.parcels.index) == ['r3', 'r6']


def test_a_missing_lot_key_column_raises():
    with pytest.raises(KeyError):
        split_stacked_units(_table(), lot_key='nope')


def _parts_table():
    # One lot id drawn as three adjoining polygons, all carrying the
    # same record, beside a neighboring lot drawn once.
    rows = pd.DataFrame(
        {
            'lot_id': ['L1', 'L1', 'L1', 'L2'],
            'geo_id': ['g1', 'g2', 'g3', 'g4'],
            'parcel_id_local': ['L1', 'L1', 'L1', 'L2'],
            'land_value': [5000.0, 5000.0, 5000.0, 900.0],
            'use_group': ['SF', 'SF', 'SF', 'SF'],
        },
        index=pd.Index(['p1', 'p2', 'p3', 'p4'], name='parcel_id'),
    )
    geoms = [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1), box(5, 5, 6, 6)]
    return gpd.GeoDataFrame(rows, geometry=geoms, crs='EPSG:4326')


def test_a_lot_drawn_as_several_polygons_keeps_its_whole_outline():
    result = split_stacked_units(_parts_table(), lot_key='lot_id')

    # The parts are one record, not several units, so nothing is a unit.
    assert result.properties is None
    assert result.n_multipart_lots == 1 and result.n_parts_merged == 2
    assert result.n_exact_duplicates == 0
    parcels = result.parcels
    assert list(parcels.index) == ['p1', 'p4']
    lot = parcels.loc['p1']
    assert lot['geometry'].equals(box(0, 0, 3, 1))
    # One record, not a stack: its additive value survives untouched.
    assert lot['land_value'] == 5000.0
    # geo_id labels the outline the row now carries, not the first part.
    assert lot['geo_id'] not in ('g1', 'g2', 'g3')
    assert parcels.loc['p4', 'geo_id'] == 'g4'


def test_a_repeated_row_is_a_duplicate_and_another_polygon_is_a_part():
    table = _parts_table()
    table.loc['p3', 'geometry'] = box(1, 0, 2, 1)  # now repeats p2 exactly
    result = split_stacked_units(table, lot_key='lot_id')

    assert result.n_exact_duplicates == 1
    assert result.n_multipart_lots == 1 and result.n_parts_merged == 1
    assert result.parcels.loc['p1', 'geometry'].equals(box(0, 0, 2, 1))


def test_the_geometry_hash_key_never_unions_what_it_collided_on():
    # Two polygons far apart that the quantized shape hash put in one
    # group. Unioning them would invent an outline no source drew.
    rows = pd.DataFrame(
        {
            'geo_id': ['c', 'c'],
            'parcel_id_local': ['c', 'c'],
            'use_group': ['SF', 'SF'],
        },
        index=pd.Index(['x1', 'x2'], name='parcel_id'),
    )
    table = gpd.GeoDataFrame(
        rows, geometry=[box(0, 0, 1, 1), box(9, 9, 10, 10)], crs='EPSG:4326'
    )
    result = split_stacked_units(table)

    assert result.n_multipart_lots == 0 and result.n_parts_merged == 1
    assert result.parcels.loc['x1', 'geometry'].equals(box(0, 0, 1, 1))


def test_repeated_unit_ids_in_a_stack_are_kept_and_counted():
    # Three accounts on one outline, two of which repeat the lot's id
    # instead of carrying one of their own.
    rows = pd.DataFrame(
        {
            'geo_id': ['S', 'S', 'S'],
            'parcel_id_local': ['S', 'S', 'S'],
            'parcel_id_assessor': ['acct-1', 'acct-1', 'acct-2'],
            'improvement_value': [100.0, 200.0, 300.0],
        },
        index=pd.Index(['u1', 'u2', 'u3'], name='parcel_id'),
    )
    table = gpd.GeoDataFrame(rows, geometry=[box(0, 0, 1, 1)] * 3, crs='EPSG:4326')
    result = split_stacked_units(table)

    assert len(result.properties) == 3
    assert result.unit_key == 'parcel_id_assessor'
    assert result.n_lots_with_repeated_unit_id == 1
    assert result.n_rows_with_repeated_unit_id == 2
    assert 'repeat their parcel_id_assessor' in result.summary()
