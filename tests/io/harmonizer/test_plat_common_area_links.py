"""Plat link between a townhouse plat's lots and its common areas.

Every id, code and value below is fabricated.
"""

import pandas as pd

from openplaces.io.harmonizer import entity_links
from openplaces.io.harmonizer.entity_links import (
    COMMON_AREA_ID,
    LINK_COLUMNS_TAIL,
    PLAT_KEY_METHOD,
    plat_common_area_links,
)


def _parcels():
    rows = [
        # Townhouse plat 9001 / X-1: three TH lots, one R lot, one
        # common area with no values, one unvalued outlot.
        ('t1', '9001', 'X-1', False, 'TH', 50_000, 150_000),
        ('t2', '9001', 'X-1', False, 'TH', 50_000, 140_000),
        ('t3', '9001', 'X-1', False, 'TH', 50_000, None),
        ('t4', '9001', 'X-1', False, 'R', 60_000, 200_000),
        ('tc', '9001', 'X-1', True, 'R', None, None),
        ('to', '9001', 'X-1', False, 'R', None, None),
        # Single-family plat 9002 / 77: open space, R lots only.
        ('s1', '9002', '77', False, 'R', 90_000, 300_000),
        ('s2', '9002', '77', False, 'R', 90_000, 280_000),
        ('sc', '9002', '77', True, 'R', None, None),
        # Condominium plat 9003 / 12 whose common area records a value.
        ('u1', '9003', '12', False, 'U', 20_000, 100_000),
        ('uc', '9003', '12', True, 'R', 5_000, None),
        # Placeholder subdivision: no key.
        ('z1', '0000', '5', False, 'TH', 40_000, 120_000),
        ('zc', '0000', '5', True, 'R', None, None),
        # Same plat reference, other subdivision: a different plat.
        ('o1', '9004', 'X-1', False, 'TH', 45_000, 130_000),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            'parcel_id',
            'subdivision_code',
            'plat_id_local',
            'is_common_area',
            'use_group_code',
            'land_value',
            'improvement_value',
        ],
    ).set_index('parcel_id')
    frame['geometry_source'] = 'exampleroll'
    return frame


def _links():
    return plat_common_area_links(_parcels(), lot_use_values=['TH', 'U'])


def test_the_table_holds_two_ids_and_the_labels_and_nothing_else():
    links = _links()
    assert list(links.columns) == ['parcel_id', COMMON_AREA_ID, *LINK_COLUMNS_TAIL]
    assert set(links['link_method']) == {PLAT_KEY_METHOD}
    assert PLAT_KEY_METHOD == 'plat key'
    assert links['share'].isna().all()


def test_every_valued_lot_of_a_townhouse_plat_links_to_its_common_area():
    links = _links()
    assert links['parcel_id'].tolist() == ['t1', 't2', 't3', 't4']
    assert set(links[COMMON_AREA_ID]) == {'tc'}
    assert set(links['link_source']) == {'exampleroll'}


def test_an_unvalued_outlot_is_not_a_lot():
    assert 'to' not in set(_links()['parcel_id'])


def test_a_single_family_plat_is_left_unlinked():
    links = _links()
    assert not set(links['parcel_id']) & {'s1', 's2'}
    assert 'sc' not in set(links[COMMON_AREA_ID])


def test_a_common_area_with_a_recorded_value_is_left_unlinked():
    links = _links()
    assert 'uc' not in set(links[COMMON_AREA_ID])
    assert 'u1' not in set(links['parcel_id'])


def test_a_placeholder_key_and_a_half_key_link_nothing():
    links = _links()
    assert not set(links['parcel_id']) & {'z1', 'o1'}


def test_a_plat_of_mostly_single_family_lots_is_left_unlinked():
    parcels = _parcels()
    parcels.loc[['t1', 't2'], 'use_group_code'] = 'R'
    links = plat_common_area_links(parcels, lot_use_values=['TH', 'U'])
    assert links.empty


def test_two_common_areas_of_one_plat_each_link_every_lot():
    parcels = _parcels()
    parcels.loc['tc2'] = parcels.loc['tc']
    links = plat_common_area_links(parcels, lot_use_values=['TH', 'U'])
    assert len(links) == 8
    assert links.groupby('parcel_id').size().eq(2).all()


def test_a_spine_without_the_key_columns_yields_an_empty_table():
    parcels = _parcels().drop(columns=['plat_id_local'])
    links = plat_common_area_links(parcels, lot_use_values=['TH'])
    assert links.empty
    assert list(links.columns) == ['parcel_id', COMMON_AREA_ID, *LINK_COLUMNS_TAIL]


def test_the_step_writes_the_table_and_leaves_the_spine_alone(monkeypatch, tmp_path):
    from openplaces.core.schema import AdminId
    from openplaces.io.harmonizer import HarmonizeState

    written = {}

    def fake_write(path, links, **kwargs):
        written['path'] = path
        written['links'] = links
        written.update(kwargs)

    monkeypatch.setattr(entity_links, 'write_entity_link', fake_write)
    monkeypatch.setattr(
        entity_links, 'get_link_owner_recipe_id', lambda recipe: 'X_parcel-geo-2026'
    )
    monkeypatch.setattr(
        entity_links,
        'get_plat_link_path',
        lambda recipe, admin_id: tmp_path / 'links.parquet',
    )
    spine = _parcels()
    before = spine.copy()
    state = HarmonizeState(
        recipe={'recipe_id': 'X_parcel-spine-2026'},
        admin_id=AdminId('US-XX-YY'),
        verbose=False,
        timer=None,
        spine=spine,
    )
    state = entity_links.link_common_areas_by_plat(state, lot_use_values=['TH', 'U'])
    pd.testing.assert_frame_equal(state.spine, before)
    assert written['path'] == tmp_path / 'links.parquet'
    assert len(written['links']) == 4
    assert written['stamped_recipe_ids'] == ('X_parcel-geo-2026',)
    assert written['step_config']['lot_use_values'] == ['TH', 'U']
