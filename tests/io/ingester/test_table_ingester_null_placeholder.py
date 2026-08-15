"""year_built's 0-as-"never recorded" placeholder is nulled automatically at
ingest for every recipe that maps a year_built column -- driven by
attribute_registry.csv's null_placeholder column, not a per-recipe
transformation. See tests/core/test_attribute_registry.py for the registry
accessor itself."""

import geopandas as gpd
from shapely.geometry import box

from openplaces.core.schema import Entity
from openplaces.io.ingester.table_ingester import TableIngester
from openplaces.timing import Timer


def _make_ingester(recipe):
    """Build a bare TableIngester carrying only what `_preprocess_recipe_data`
    needs (no I/O setup)."""
    ingester = TableIngester.__new__(TableIngester)
    ingester.recipe = recipe
    ingester.processing_chunk = {}
    ingester.timer = Timer('test')
    return ingester


def _gdf(year_built_raw, value_raw):
    return gpd.GeoDataFrame(
        {'YEAR_BUILT': year_built_raw, 'ASSESSED_VALUE': value_raw},
        geometry=[box(i, i, i + 1, i + 1) for i in range(len(year_built_raw))],
        crs='EPSG:4326',
    )


def _recipe():
    return {
        'entity': Entity('parcel'),
        'columns': {'year_built': 'YEAR_BUILT', 'value': 'ASSESSED_VALUE'},
    }


def test_year_built_placeholder_zero_is_nulled_without_any_recipe_transform():
    ingester = _make_ingester(_recipe())
    df = ingester._preprocess_recipe_data(_gdf([0, 1964, 0], [0, 100000, 50000]))

    assert df['year_built'].isna().tolist() == [True, False, True]
    assert df['year_built'].dropna().tolist() == [1964]


def test_unregistered_placeholder_column_is_left_alone():
    ingester = _make_ingester(_recipe())
    df = ingester._preprocess_recipe_data(_gdf([1964, 1980, 1990], [0, 100000, 50000]))

    # `value` has no declared null_placeholder -- a genuine $0 assessed
    # value (e.g. vacant land) must NOT be nulled.
    assert df['value'].tolist() == [0, 100000, 50000]
