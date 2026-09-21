"""Links only the ingest-time split knows, and shares across several lots.

Every id and value below is fabricated.
"""

import pandas as pd

from openplaces.io.harmonizer.entity_links import (
    apportion_shares,
    build_id_links,
    stacked_unit_links,
)


def _parcels():
    # Lot A is a stacked lot (its row carries the lot key), B a house,
    # C and D two lots that one account is drawn on.
    return pd.DataFrame(
        {
            'parcel_id_local': ['A', 'B1', 'C', 'D'],
            'area_ha': [1.0, 0.5, 3.0, 1.0],
        },
        index=pd.Index(['pa', 'pb', 'pc', 'pd'], name='parcel_id'),
    )


def _properties():
    return pd.DataFrame(
        {
            # Two roll rows keyed on lot A's units, one house, one account
            # on two lots, and a split unit the roll does not know.
            'parcel_id_local': ['A1', 'A2', 'B1', 'CD', 'A3'],
            'lot_id_local': [None, None, None, None, 'A'],
            'source': ['roll', 'roll', 'roll', 'roll', 'layer:units'],
        },
        index=pd.Index(['r1', 'r2', 'r3', 'r4', 'u3'], name='property_id'),
    )


def _pairs():
    return pd.DataFrame(
        {
            'unit_key': ['A1', 'A2', 'A3', 'CD', 'CD'],
            'lot_key': ['A', 'A', 'A', 'C', 'D'],
        }
    )


def _links():
    properties, parcels = _properties(), _parcels()
    links = build_id_links(
        properties,
        parcels,
        'parcel_id_local',
        'parcel_id_local',
        'property_id',
        'parcel_id',
        'parcel_id_local',
    )
    return stacked_unit_links(
        links,
        properties,
        parcels,
        _pairs(),
        'parcel_id_local',
        'parcel_id_local',
        'property_id',
        'parcel_id',
    )


def test_the_key_pass_alone_cannot_reach_a_stacked_lot():
    links = build_id_links(
        _properties(),
        _parcels(),
        'parcel_id_local',
        'parcel_id_local',
        'property_id',
        'parcel_id',
        'parcel_id_local',
    )
    assert links['property_id'].tolist() == ['r3']


def test_a_roll_row_keyed_on_a_unit_reaches_the_units_lot():
    links = _links().set_index('property_id')
    assert links.loc['r1', 'parcel_id'] == 'pa'
    assert links.loc['r1', 'link_method'] == 'stacked_units_crosswalk'
    assert links.loc['r2', 'parcel_id'] == 'pa'


def test_a_split_unit_reaches_the_lot_it_names():
    links = _links().set_index('property_id')
    assert links.loc['u3', 'parcel_id'] == 'pa'
    assert links.loc['u3', 'link_source'] == 'layer:units'


def test_a_pair_the_key_pass_found_keeps_its_method():
    links = _links().set_index('property_id')
    assert links.loc['r3', 'link_method'] == 'parcel_id_local'


def test_an_account_on_two_lots_links_to_both_and_no_pair_repeats():
    links = _links()
    assert sorted(links.loc[links['property_id'] == 'r4', 'parcel_id']) == ['pc', 'pd']
    assert not links.duplicated(['property_id', 'parcel_id']).any()


def test_a_property_on_several_lots_is_divided_by_lot_area():
    links = apportion_shares(_links(), _parcels()).set_index(
        ['property_id', 'parcel_id']
    )
    assert links.loc[('r4', 'pc'), 'share'] == 0.75
    assert links.loc[('r4', 'pd'), 'share'] == 0.25
    assert links.loc[('r4', 'pc'), 'share_basis'] == 'area'
    # Everything on one lot stays undivided.
    assert pd.isna(links.loc[('r1', 'pa'), 'share'])
    assert pd.isna(links.loc[('r1', 'pa'), 'share_basis'])


def test_a_lot_without_an_area_falls_back_to_an_equal_split():
    parcels = _parcels()
    parcels.loc['pd', 'area_ha'] = None
    links = apportion_shares(_links(), parcels).set_index(['property_id', 'parcel_id'])
    assert links.loc[('r4', 'pc'), 'share'] == 0.5
    assert links.loc[('r4', 'pd'), 'share_basis'] == 'equal'
