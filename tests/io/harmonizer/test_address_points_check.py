"""The address-points check on transaction-to-parcel links (fabricated data)."""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box

from openplaces.core.schema import AdminId
from openplaces.io.harmonizer.transactions import (
    _parcel_point_keys,
    address_points_check,
)
from openplaces.recipe import apply_recipe_patches, get_recipe_by_id


def _links(pairs):
    return pd.DataFrame(pairs, columns=['transaction_id', 'parcel_id'])


def test_each_link_gets_one_fixed_label():
    links = _links(
        [('T1', 'P1'), ('T2', 'P1'), ('T3', 'P2'), ('T4', 'P1'), ('T5', 'P3')]
    )
    sale_keys = pd.Series(
        ['1|main st', '9|elm st', '1|main st', '5|oak st', '2|pine st'],
        index=['T1', 'T2', 'T3', 'T4', 'T5'],
    )
    parcel_keys = pd.Series(
        ['1|main st', '3|main st', '9|elm st'], index=['P1', 'P1', 'P2']
    )
    labels = address_points_check(links, sale_keys, parcel_keys).tolist()
    # T1 agrees; T2's address is a point of P2 only; T3 names P1's point
    # but is linked to P2; T4 is the address of no point; P3 has none.
    assert labels == [
        'agree',
        'other_parcel',
        'other_parcel',
        'no_point_match',
        'no_points',
    ]


def test_a_sale_without_an_address_is_said_so_first():
    links = _links([('T1', 'P9')])
    labels = address_points_check(
        links, pd.Series([None], index=['T1']), pd.Series(dtype='string')
    )
    assert labels.tolist() == ['no_sale_address']


def test_a_folded_deed_agrees_through_any_member_address():
    links = _links([('D1', 'P1'), ('D1', 'P2')])
    sale_keys = pd.Series(['4|birch rd', '6|birch rd'], index=['D1', 'D1'])
    parcel_keys = pd.Series(['6|birch rd'], index=['P2'])
    labels = address_points_check(links, sale_keys, parcel_keys).tolist()
    assert labels == ['no_points', 'agree']


def test_the_check_never_adds_or_drops_a_link():
    links = _links([('T1', 'P1'), ('T2', 'P2')])
    labels = address_points_check(
        links, pd.Series(['1|a st'], index=['T1']), pd.Series(['1|a st'], index=['P2'])
    )
    assert len(labels) == len(links)
    assert labels.index.equals(links.index)


def test_points_count_only_inside_their_own_parcel():
    parcels = gpd.GeoDataFrame(
        geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)],
        index=pd.Index(['P1', 'P2'], name='parcel_id'),
        crs='EPSG:26986',
    )
    points = gpd.GeoDataFrame(
        {
            'address_number': ['12', '14', '99'],
            'address_street': ['Fabricated Street', 'Fabricated Street', 'Nowhere Rd'],
        },
        geometry=[Point(5, 5), Point(15, 5), Point(50, 50)],
        crs='EPSG:26986',
    )
    keys = _parcel_point_keys(points, parcels, AdminId('US-MA-XXX'))
    assert sorted(keys.index.tolist()) == ['P1', 'P2']
    assert keys.loc['P1'] != keys.loc['P2']


def test_massachusetts_checks_against_its_own_points_first():
    ma, _ = apply_recipe_patches(
        get_recipe_by_id('US_transaction-spine-2026'), AdminId('US-MA-SOM')
    )
    step = [
        s for s in ma['pipeline'] if s['step'] == 'check_parcel_links_by_address_points'
    ]
    assert step[0]['dwelling_recipe_ids'][0] == 'US-MA_dwelling-massgis-2026'
    national = get_recipe_by_id('US_transaction-spine-2026')
    names = [s['step'] for s in national['pipeline']]
    assert names.index('check_parcel_links_by_address_points') > names.index(
        'link_transactions'
    )
