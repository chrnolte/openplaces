"""The ingester scopes `admin_ids` transformations by the process chunk.

A statewide file downloads as the state and is split per county by
`process_by`, so only the process chunk knows the county. The ingester
must hand that unit to the transformation engine, not the download
partition's. Values are fabricated.
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from openplaces.core.schema import Entity
from openplaces.io.ingester.table_ingester import TableIngester
from openplaces.timing import Timer


def _make_ingester(recipe, chunk_admin_id):
    ingester = TableIngester.__new__(TableIngester)
    ingester.recipe = recipe
    ingester.processing_chunk = {'admin_id_to_process': chunk_admin_id}
    ingester.download_partition = {'admin_id_to_download': 'XX-AA'}
    ingester.timer = Timer('test')
    return ingester


def _gdf():
    return gpd.GeoDataFrame(
        {'use_subgroup_code': ['1', '2']},
        geometry=[box(i, i, i + 1, i + 1) for i in range(2)],
        crs='EPSG:4326',
    )


RECIPE = {
    'entity': Entity('parcel'),
    'admin_id': 'XX-AA',
    'transformations': [
        {
            'type': 'unary',
            'operation': 'set_null',
            'input': 'use_subgroup_code',
            'output': 'use_subgroup_code',
            'admin_ids': ['XX-AA-BBB'],
        }
    ],
}


def test_chunk_inside_scope_is_transformed():
    df = _make_ingester(RECIPE, 'XX-AA-BBB')._preprocess_recipe_data(_gdf())
    assert pd.Series(df['use_subgroup_code']).isna().all()


def test_other_chunk_of_the_same_download_is_untouched():
    df = _make_ingester(RECIPE, 'XX-AA-CCC')._preprocess_recipe_data(_gdf())
    assert df['use_subgroup_code'].tolist() == ['1', '2']
