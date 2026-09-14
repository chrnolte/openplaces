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
    # The stack's parcel keeps what its members agree on and nothing else.
    lot_a = parcels.loc['r1']
    assert lot_a['land_value'] == 1000.0 and lot_a['use_group'] == 'Condo'
    assert pd.isna(lot_a['improvement_value'])
    assert pd.isna(lot_a['parcel_id_assessor'])
    # The members disagreed on the link key, so the lot key stands in.
    assert lot_a['parcel_id_local'] == 'A'
    assert list(properties.index) == ['r1', 'r2']
    assert properties['parcel_id_local'].tolist() == ['A', 'A']
    assert properties['improvement_value'].tolist() == [90000.0, 80000.0]
    assert 'geometry' not in properties.columns


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
    assert result.properties['parcel_id_local'].tolist() == ['L1', 'L1']


def test_no_stack_means_no_property_table():
    table = _table().loc[['r3', 'r6']]
    result = split_stacked_units(table)
    assert result.properties is None and result.n_stacks == 0
    assert list(result.parcels.index) == ['r3', 'r6']


def test_a_missing_lot_key_column_raises():
    with pytest.raises(KeyError):
        split_stacked_units(_table(), lot_key='nope')
