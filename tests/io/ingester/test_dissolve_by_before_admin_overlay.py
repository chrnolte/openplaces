"""`dissolve_by` folds a record's polygons before it is placed in a unit.

Florida DEP's state-owned lands layer draws one acquisition as several
polygons, some across a county line. Folded first, the record is placed
whole by the centroid of its outline; placed polygon by polygon, its
price would be saved once in each county it touches. The layer is in
EPSG:6439 and the admin polygons are kept in the source's CRS, so the
overlay also has to bring them to the rows' CRS.
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from openplaces.io.ingester.table_ingester import TableIngester
from openplaces.timing import Timer


def _admin(crs='EPSG:4326'):
    units = gpd.GeoSeries(
        [box(0, 0, 1, 1), box(1, 0, 2, 1)],
        index=pd.Index(['XX-AA-AA', 'XX-AA-BB'], name='admin3_id'),
        crs='EPSG:4326',
    )
    return units.to_crs(crs)


def _records():
    # Deal K1: a small polygon in AA and a large one in BB; K2: in AA.
    return gpd.GeoDataFrame(
        {
            'deal': ['K1', 'K1', 'K2'],
            'price': [900.0, 900.0, 40.0],
        },
        geometry=[
            box(0.8, 0.4, 0.9, 0.5),
            box(1.1, 0.1, 1.9, 0.9),
            box(0.1, 0.1, 0.2, 0.2),
        ],
        crs='EPSG:4326',
    )


def _ingester(admin_crs='EPSG:4326'):
    ingester = TableIngester.__new__(TableIngester)
    ingester.recipe = {
        'entity': None,
        'dataset': 'records',
        'dissolve_by': 'deal',
        'overlay_admin_ids': {'admin_level': 3},
    }
    ingester.download_partition = {'admin_geometries': _admin(admin_crs)}
    ingester.processing_chunk = {'admin_id_to_process': 'XX-AA'}
    ingester.timer = Timer('test')
    ingester.verbose = False
    return ingester


def test_a_record_is_placed_whole_by_its_outline():
    df = _ingester()._preprocess_recipe_data(_records())

    assert len(df) == 2
    by_deal = df.set_index('deal')
    assert by_deal.loc['K1', 'admin3_id'] == 'XX-AA-BB'
    assert by_deal.loc['K2', 'admin3_id'] == 'XX-AA-AA'
    assert df['price'].sum() == 940.0


def test_admin_polygons_in_the_source_crs_still_place_every_row():
    df = _ingester('EPSG:6439')._preprocess_recipe_data(_records())

    assert df['admin3_id'].notna().all()
    assert sorted(df['admin3_id']) == ['XX-AA-AA', 'XX-AA-BB']
