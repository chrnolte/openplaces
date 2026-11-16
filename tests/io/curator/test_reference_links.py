"""Point-first record-to-parcel linking and per-parcel reference labels.

Pins the rule order (point, then a unique id key, then a unique address
key), the tie-break for stacked polygons, that an ambiguous key links
nothing, the per-parcel summary the validation notebooks read, and the
record-to-building pairs (outline first, then the parcel's buildings,
every building where there are several). All records, ids, and
addresses are fabricated.
"""

import warnings

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from openplaces.io.curator import reference_links as rl


def _parcels():
    return gpd.GeoDataFrame(
        {
            'parcel_id_local': ['K-1', 'K-2', 'DUP', 'DUP', 'K-5'],
            'address_street': [
                'Sample Street',
                'Example Avenue',
                None,
                None,
                'Fictional Road',
            ],
            'address_number': ['10', '22', None, None, '7'],
        },
        geometry=[
            box(0, 0, 10, 10),
            box(10, 0, 20, 10),
            box(20, 0, 30, 10),
            # A small parcel stacked inside the first one.
            box(2, 2, 4, 4),
            box(40, 0, 50, 10),
        ],
        index=pd.Index(['p1', 'p2', 'p3', 'p4', 'p5'], name='parcel_id'),
        crs='EPSG:3857',
    )


def _records():
    return gpd.GeoDataFrame(
        {
            'parcel_id_local': ['K2', 'K5', 'DUP', None, None, None],
            'street': [None, None, None, 'EXAMPLE AVE', 'FICTIONAL RD', 'NONE ST'],
            'street_no': [None, None, None, '22', '7', '1'],
        },
        geometry=[
            Point(3, 3),  # inside p1 and the stacked p4
            None,  # no point: id key K5 links p5
            None,  # id key held by two parcels: ambiguous
            Point(100, 100),  # outside every parcel: address links p2
            None,
            None,  # nothing matches
        ],
        index=pd.Index([f'r{i}' for i in range(6)], name='record_id'),
        crs='EPSG:3857',
    )


def _link():
    return rl.link_records_to_parcels(
        _records(),
        _parcels(),
        key_columns=(('parcel_id_local', 'parcel_id_local'),),
        record_address={'street_column': 'street', 'number_column': 'street_no'},
        parcel_address={
            'street_column': 'address_street',
            'number_column': 'address_number',
        },
        admin_id='US-XX-AAA',
    )


def test_rules_apply_in_order_and_record_which_linked():
    links = _link()

    assert links['parcel_id'].tolist()[:2] == ['p4', 'p5']
    assert links['matched_via'].tolist()[:2] == ['point', 'parcel_id_local']
    assert pd.isna(links['parcel_id'].iloc[2])
    assert links.loc['r3', ['parcel_id', 'matched_via']].tolist() == ['p2', 'address']
    assert links.loc['r4', ['parcel_id', 'matched_via']].tolist() == ['p5', 'address']
    assert pd.isna(links['parcel_id'].iloc[5])
    assert pd.isna(links['matched_via'].iloc[5])


def test_point_beats_a_conflicting_id_key():
    records = _records().iloc[[0]].copy()
    records['parcel_id_local'] = 'K-5'

    links = rl.link_records_to_parcels(records, _parcels())

    assert links['parcel_id'].iloc[0] == 'p4'
    assert links['matched_via'].iloc[0] == 'point'


def _outlines_and_buildings():
    # f1: one building on p1; f2: a townhome row, two buildings on p1;
    # f3: an outline on p2 with no building. b5 on p2 and b3/b4 on p5
    # have no outline. Fabricated ids.
    outlines = gpd.GeoDataFrame(
        geometry=[box(0.5, 0.5, 1.5, 1.5), box(5, 5, 9, 9), box(12, 2, 14, 4)],
        index=pd.Index(['f1', 'f2', 'f3'], name='footprint_id'),
        crs='EPSG:3857',
    )
    buildings = pd.DataFrame(
        {
            'footprint_id': ['f1', 'f2', 'f2', None, None, None],
            'parcel_id': ['p1', 'p1', 'p1', 'p5', 'p5', 'p2'],
        },
        index=pd.Index(['b1', 'b2a', 'b2b', 'b3', 'b4', 'b5'], name='building_id'),
    )
    return outlines, buildings


def test_records_reach_buildings_by_outline_then_parcel():
    records = gpd.GeoDataFrame(
        {'parcel_id_local': [None, None, None, 'K-5', None, None]},
        geometry=[
            Point(1, 1),  # inside f1: its one building
            Point(6, 6),  # inside f2: both buildings
            Point(13, 3),  # inside f3, which has none: p2's building
            None,  # no point: key K-5 reaches p5's two buildings
            Point(100, 100),  # nothing
            Point(3, 3),  # stacked p4 holds no building: nothing
        ],
        index=pd.Index([f'r{i}' for i in range(6)], name='record_id'),
        crs='EPSG:3857',
    )
    outlines, buildings = _outlines_and_buildings()

    pairs = rl.link_records_to_buildings(records, outlines, buildings, _parcels())

    got = {
        (r, b): m
        for r, b, m in pairs[['record_id', 'building_id', 'link_method']].itertuples(
            index=False
        )
    }
    assert got == {
        ('r0', 'b1'): 'footprint',
        ('r1', 'b2a'): 'footprint, several buildings',
        ('r1', 'b2b'): 'footprint, several buildings',
        ('r2', 'b5'): 'parcel',
        ('r3', 'b3'): 'parcel, several buildings',
        ('r3', 'b4'): 'parcel, several buildings',
    }
    assert list(pairs.columns) == ['record_id', 'building_id', 'link_method']


def test_outline_hit_beats_the_parcel_key():
    # The key names p5, but the point lies in f1: the outline decides.
    records = gpd.GeoDataFrame(
        {'parcel_id_local': ['K-5']},
        geometry=[Point(1, 1)],
        index=pd.Index(['r0'], name='record_id'),
        crs='EPSG:3857',
    )
    outlines, buildings = _outlines_and_buildings()

    pairs = rl.link_records_to_buildings(records, outlines, buildings, _parcels())

    assert pairs['building_id'].tolist() == ['b1']


def test_overlap_tie_break_measures_geographic_layers_in_meters():
    parcels = _parcels().to_crs('EPSG:4326')
    records = _records().iloc[[0]].to_crs('EPSG:4326')

    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        links = rl.link_records_to_parcels(records, parcels)

    assert links['parcel_id'].iloc[0] == 'p4'


def test_summary_mode_share_recency_and_majority_rule():
    records = pd.DataFrame(
        {
            'cls': ['Class A', 'Class A', 'Class B', None, 'Class C'],
            'permit_file_date': [
                '2020-01-01',
                '2021-01-01',
                '2022-01-01',
                '2023-01-01',
                '2020-06-01',
            ],
            'area_sqft': [100, 200, 300, None, 50],
            'street_no': ['1', '1', '1', '1', '9'],
            'street': ['SAMPLE ST'] * 4 + ['OTHER RD'],
        },
        index=pd.Index([f'r{i}' for i in range(5)], name='record_id'),
    )
    links = pd.DataFrame(
        {
            'parcel_id': ['p1', 'p1', 'p1', 'p1', 'p2'],
            'matched_via': ['address', 'point', 'point', 'point', 'address'],
        },
        index=records.index,
    )

    out = rl.summarize_labels_by_parcel(
        records,
        links,
        class_column='cls',
        date_columns=('permit_file_date',),
        area_column='area_sqft',
        address_columns=('street_no', 'street'),
        link_order=('point', 'parcel_id_local', 'address'),
    )

    p1 = out.loc['p1']
    assert p1['occupancy_type_mode'] == 'Class A'
    assert p1['occupancy_type_mode_pct'] == pytest.approx(2 / 3)
    assert p1['occupancy_type_most_recent'] == 'Class B'
    assert p1['n_permits'] == 4
    assert p1['n_permits_with_occupancy_type'] == 3
    assert p1['area_sqft_most_recent'] == 300
    assert p1['address'] == '1 SAMPLE ST'
    assert p1['latest_permit_date'] == pd.Timestamp('2023-01-01')
    # Two label-bearing records by point against one by address.
    assert p1['matched_via'] == 'point'
    assert out.loc['p2', 'matched_via'] == 'address'


def test_build_files_cover_every_parcel_and_footprint(monkeypatch, tmp_path):
    parcels = _parcels()
    records = _records()
    records['occupancy_type_raw'] = ['Class A'] * 6
    # Buildings carry their parcel and the outline they stand under (b3
    # step 6). f5's two buildings name different parcels and f6's two the
    # same one: an outline takes its buildings' parcel only where they
    # agree. Fabricated ids.
    buildings = pd.DataFrame(
        {
            'parcel_id': ['p4', 'p4', 'p1', None, 'p4', 'p1', 'p4', 'p4'],
            'footprint_id': ['f1', 'f2', 'f3', 'f4', 'f5', 'f5', 'f6', 'f6'],
        },
        index=pd.Index(
            ['b1', 'b2', 'b3', 'b4', 'b5a', 'b5b', 'b6a', 'b6b'], name='building_id'
        ),
    )
    tables = {
        'records': records,
        'parcels-geo': parcels.drop(columns=['address_street', 'address_number']),
        'parcels-attr': pd.DataFrame(parcels[['address_street', 'address_number']]),
        'buildings': buildings,
    }

    def fake_get_entities(recipe, admin_id, geom=False, columns=None, **kwargs):
        return tables[recipe].copy()

    import openplaces.io.readers as readers

    monkeypatch.setattr(readers, 'get_entities', fake_get_entities)

    summary = rl.build_parcel_reference_files(
        'US-XX-AAA',
        tmp_path,
        record_recipe_id='records',
        parcel_recipe_id='parcels-geo',
        parcel_address_recipe_id='parcels-attr',
        building_recipe_id='buildings',
    )

    parcel_file = pd.read_parquet(
        tmp_path / 'US-XX-AAA_parcel_occupancy_validation.parquet'
    )
    footprint_file = pd.read_parquet(
        tmp_path / 'US-XX-AAA_footprint_occupancy_validation.parquet'
    )
    assert list(parcel_file.columns) == list(rl.REFERENCE_COLUMNS)
    assert len(parcel_file) == 5
    assert len(footprint_file) == 6
    assert footprint_file.loc[['f1', 'f2', 'f6'], 'n_permits'].tolist() == [1.0] * 3
    assert pd.isna(footprint_file.loc['f3', 'n_permits'])
    assert pd.isna(footprint_file.loc['f5', 'n_permits'])
    assert parcel_file.loc['p1', 'matched_via'] == 'none'
    assert summary['linked'] == 4
    assert summary['footprints_labeled'] == 3


def test_build_building_links_writes_pairs_only(monkeypatch, tmp_path):
    parcels = _parcels()
    outlines, buildings = _outlines_and_buildings()
    records = gpd.GeoDataFrame(
        {'parcel_id_local': [None, 'K-5'], 'cls': ['Class A', 'Class B']},
        geometry=[Point(6, 6), None],
        index=pd.Index(['r0', 'r1'], name='record_id'),
        crs='EPSG:3857',
    )
    tables = {
        'records': records,
        'parcels-geo': parcels.drop(columns=['address_street', 'address_number']),
        'parcels-attr': pd.DataFrame(parcels[['address_street', 'address_number']]),
        'outlines': outlines,
        'buildings': buildings,
    }

    def fake_get_entities(recipe, admin_id, geom=False, columns=None, **kwargs):
        return tables[recipe].copy()

    import openplaces.io.readers as readers

    monkeypatch.setattr(readers, 'get_entities', fake_get_entities)

    summary = rl.build_building_reference_links(
        'US-XX-AAA',
        tmp_path,
        record_recipe_id='records',
        footprint_recipe_id='outlines',
        building_recipe_id='buildings',
        parcel_recipe_id='parcels-geo',
        parcel_address_recipe_id='parcels-attr',
    )

    out = pd.read_parquet(tmp_path / 'US-XX-AAA_building_permit_links.parquet')
    # Ids and the rule only: no record attribute travels with a link.
    assert list(out.columns) == ['record_id', 'building_id', 'link_method']
    assert len(out) == 4
    assert summary == {
        'admin_id': 'US-XX-AAA',
        'records': 2,
        'linked': 2,
        'linked_footprint, several buildings': 1,
        'linked_parcel, several buildings': 1,
    }
