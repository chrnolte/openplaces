"""`entity_type: parcel`'s automatic geo_id indexing (table_ingester.py's
"Set index" step) should defer to a recipe's own `set_index`/
`create_index`/`index_function`, and should not recompute an index that is
already a real `geo_id` -- both needed so a recipe can preserve a shared
join key across several tables meant to be merged later (see
`io.aggregate.join_partitions_by_index`)."""

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from openplaces.core.schema import Entity
from openplaces.io.ingester.table_ingester import TableIngester


def _make_ingester(recipe):
    """Build a bare TableIngester carrying only what `_preprocess_recipe_data`
    needs (no I/O setup)."""
    ingester = TableIngester.__new__(TableIngester)
    ingester.recipe = recipe
    ingester.processing_chunk = {}
    return ingester


def _gdf(n=2):
    return gpd.GeoDataFrame(
        {'value': range(n)},
        geometry=[box(i, i, i + 1, i + 1) for i in range(n)],
        crs='EPSG:4326',
    )


def test_default_parcel_entity_auto_assigns_geo_id():
    ingester = _make_ingester({'entity': Entity('parcel')})
    df = ingester._preprocess_recipe_data(_gdf())

    assert df.index.name == 'parcel_id'
    assert 'geo_id' in df.columns


def test_set_index_overrides_auto_geo_id():
    df_in = _gdf()
    df_in['source_key'] = ['a', 'b']

    ingester = _make_ingester({'entity': Entity('parcel'), 'set_index': 'source_key'})
    df = ingester._preprocess_recipe_data(df_in)

    assert df.index.name == 'source_key'
    assert list(df.index) == ['a', 'b']
    assert 'geo_id' not in df.columns


def test_create_index_function_overrides_auto_geo_id():
    ingester = _make_ingester(
        {
            'entity': Entity('parcel'),
            'create_index': {
                'function': 'openplaces.io.transform.rename_index',
                'args': {'name': 'geo_id_source'},
            },
        }
    )
    df = ingester._preprocess_recipe_data(_gdf())

    assert df.index.name == 'geo_id_source'
    assert 'geo_id' not in df.columns


def test_already_geo_id_index_is_not_recomputed():
    df_in = _gdf()
    df_in.index = pd.Index(['existing_1', 'existing_2'], name='geo_id')

    ingester = _make_ingester({'entity': Entity('parcel')})
    df = ingester._preprocess_recipe_data(df_in)

    # Neither the auto branch (guarded by `df.index.name != 'geo_id'`) nor any
    # custom-index branch fires, so the index passes through unchanged.
    assert df.index.name == 'geo_id'
    assert list(df.index) == ['existing_1', 'existing_2']
    assert 'geo_id' not in df.columns
