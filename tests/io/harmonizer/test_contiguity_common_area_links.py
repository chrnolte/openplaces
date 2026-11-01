"""Contiguity link between townhouse lots and the common land they touch.

Every id, label and value below is fabricated. Coordinates are meters
in a projected CRS.
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from openplaces.io.harmonizer.entity_links import (
    COMMON_AREA_ID,
    CONTIGUITY_METHOD,
    LINK_COLUMNS_TAIL,
    contiguity_common_area_links,
)

COMMON = r'common|owners?.? assoc|\bhoa\b|unused land'
UNIT = r'town ?(?:home|house)|condo|attached'


def _parcels(
    lot_label='Residential | Townhouse', common_label=None, common_value=1_000
):
    # A 100 m square common lot with five 10 x 30 m lots along its
    # south edge, touching it, and one lot far away.
    rows = [
        (
            'c',
            box(0, 0, 100, 100),
            common_label or 'Residential | Common Area',
            common_value,
        )
    ]
    rows += [
        (f'l{i}', box(i * 10, -30, i * 10 + 10, 0), lot_label, 300_000)
        for i in range(5)
    ]
    rows.append(('far', box(500, 500, 510, 530), lot_label, 300_000))
    frame = gpd.GeoDataFrame(
        {
            'use_group': [r[2].split(' | ')[0] for r in rows],
            'use_subgroup': [r[2].split(' | ')[1] for r in rows],
            'total_value': [float(r[3]) for r in rows],
            'geometry_source': 'exampleroll',
        },
        geometry=[r[1] for r in rows],
        index=pd.Index([r[0] for r in rows], name='parcel_id'),
        crs='EPSG:32617',
    )
    frame['area_ha'] = frame.geometry.area / 10_000
    return frame


def test_townhouse_lots_touching_common_land_are_linked():
    links = contiguity_common_area_links(_parcels(), COMMON, UNIT)
    assert list(links.columns) == ['parcel_id', COMMON_AREA_ID, *LINK_COLUMNS_TAIL]
    assert set(links['parcel_id']) == {f'l{i}' for i in range(5)}
    assert set(links[COMMON_AREA_ID]) == {'c'}
    assert links['link_method'].eq(CONTIGUITY_METHOD).all()
    assert links['share'].isna().all()


def test_single_family_open_space_and_roads_stay_unlinked():
    houses = _parcels(lot_label='Residential | Single Family')
    road = _parcels(common_label='Exempt | Right-of-Way')
    for frame in (houses, road):
        assert contiguity_common_area_links(frame, COMMON, UNIT).empty


def test_a_common_lot_valued_like_its_lots_is_not_common_land():
    frame = _parcels(common_value=2_000_000)
    assert contiguity_common_area_links(frame, COMMON, UNIT).empty


def test_too_few_lots_are_not_linked():
    frame = _parcels().drop(index=['l3', 'l4'])
    assert contiguity_common_area_links(frame, COMMON, UNIT).empty


def test_a_roll_flag_leaves_the_unit_to_the_plat_key():
    frame = _parcels()
    frame['is_common_area'] = frame.index == 'c'
    assert contiguity_common_area_links(frame, COMMON, UNIT).empty
