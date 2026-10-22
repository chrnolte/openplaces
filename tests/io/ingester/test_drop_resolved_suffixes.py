"""A parcel whose geometry hash the stacked-units split made unique is
indexed by that hash, so the footprint overlay (which keys parcels by
`geo_id`) and the parcel table name it alike."""

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from openplaces.io.ingester.table_ids import drop_resolved_suffixes
from openplaces.io.stacked_units import split_stacked_units
from openplaces.table import add_unique_suffix


def _parcels(rows):
    gdf = gpd.GeoDataFrame(
        [{k: v for k, v in r.items() if k != 'geometry'} for r in rows],
        geometry=[r['geometry'] for r in rows],
        crs='EPSG:4326',
    )
    gdf.index = pd.Index(add_unique_suffix(gdf['geo_id']), name='parcel_id')
    return gdf


def test_a_collapsed_stack_takes_its_bare_geo_id():
    lot = box(0, 0, 1, 1)
    gdf = _parcels(
        [
            {'geo_id': 'aaa', 'parcel_id_local': 'U1', 'geometry': lot},
            {'geo_id': 'aaa', 'parcel_id_local': 'U2', 'geometry': lot},
            {'geo_id': 'bbb', 'parcel_id_local': 'P9', 'geometry': box(2, 2, 3, 3)},
        ]
    )
    assert list(gdf.index) == ['aaa-1', 'aaa-2', 'bbb']
    result = split_stacked_units(gdf)
    parcels = drop_resolved_suffixes(result.parcels)
    assert sorted(parcels.index) == ['aaa', 'bbb']
    assert parcels.index.name == 'parcel_id'
    assert (parcels.index == parcels['geo_id']).all()


def test_a_row_shipped_twice_takes_its_bare_geo_id():
    lot = box(0, 0, 1, 1)
    gdf = _parcels(
        [
            {'geo_id': 'aaa', 'parcel_id_local': 'P1', 'geometry': lot},
            {'geo_id': 'aaa', 'parcel_id_local': 'P1', 'geometry': lot},
        ]
    )
    parcels = drop_resolved_suffixes(split_stacked_units(gdf).parcels)
    assert list(parcels.index) == ['aaa']


def test_a_hash_still_shared_keeps_its_suffixed_ids():
    # Two rows left sharing a hash (no split: a source opting out) keep
    # the ids preprocessing gave them; nothing is renumbered.
    gdf = _parcels(
        [
            {'geo_id': 'aaa', 'parcel_id_local': 'P1', 'geometry': box(0, 0, 1, 1)},
            {'geo_id': 'aaa', 'parcel_id_local': 'P2', 'geometry': box(0, 0, 1, 1)},
            {'geo_id': 'ccc', 'parcel_id_local': 'P3', 'geometry': box(4, 4, 5, 5)},
        ]
    )
    parcels = drop_resolved_suffixes(gdf)
    assert list(parcels.index) == ['aaa-1', 'aaa-2', 'ccc']


def test_a_table_under_another_index_is_left_alone():
    gdf = _parcels(
        [{'geo_id': 'aaa', 'parcel_id_local': 'P1', 'geometry': box(0, 0, 1, 1)}]
    )
    gdf.index = pd.Index(['custom-1'], name='lot_id')
    assert list(drop_resolved_suffixes(gdf).index) == ['custom-1']
