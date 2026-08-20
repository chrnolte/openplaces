"""A recipe declaring only `transformation_patterns` must still transform.

The ingester's guard used to test for the `transformations` key alone, so a
recipe whose only transformation was a pattern block (e.g. txgio's bare
to_numeric cast of its text-typed value columns) was silently skipped -- the
columns stayed str through an entire re-ingest. Either key alone triggers
the call now.
"""

import geopandas as gpd
from shapely.geometry import box

from openplaces.core.schema import Entity
from openplaces.io.ingester.table_ingester import TableIngester
from openplaces.timing import Timer


def _make_ingester(recipe):
    ingester = TableIngester.__new__(TableIngester)
    ingester.recipe = recipe
    ingester.processing_chunk = {}
    ingester.timer = Timer('test')
    return ingester


def _gdf():
    return gpd.GeoDataFrame(
        {'improvement_value': ['0', '125000']},
        geometry=[box(i, i, i + 1, i + 1) for i in range(2)],
        crs='EPSG:4326',
    )


def test_patterns_only_recipe_is_applied():
    recipe = {
        'entity': Entity('parcel'),
        'transformation_patterns': [
            {
                'type': 'unary',
                'operation': 'to_numeric',
                'pattern': '{column}',
                'apply_to_columns': ['improvement_value'],
            }
        ],
    }
    df = _make_ingester(recipe)._preprocess_recipe_data(_gdf())
    import pandas as pd

    assert pd.api.types.is_numeric_dtype(df['improvement_value'])
    assert df['improvement_value'].tolist() == [0, 125000]
