"""A recipe's `drop_columns` discards scratch columns a `transformations`
step needed only as an intermediate (e.g. a zero-padded value before a
prefix is added), so they never reach the saved output. Without it,
transformation-added columns are unconditionally kept regardless of
`keep_unnamed_columns` (see the CO-AN catastro recipe's
admin3_id_admin2_zfill, used only to derive the country-qualified
admin3_id_admin1)."""

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


def _gdf():
    return gpd.GeoDataFrame(
        {'raw': ['1', '23']},
        geometry=[box(i, i, i + 1, i + 1) for i in range(2)],
        crs='EPSG:4326',
    )


def _recipe(drop_columns=None):
    recipe = {
        'entity': Entity('parcel'),
        'transformations': [
            {
                'type': 'string',
                'operation': 'zfill',
                'input': 'raw',
                'output': 'raw_zfill',
                'args': {'width': 3},
            },
            {
                'type': 'string',
                'operation': 'add_prefix',
                'input': 'raw_zfill',
                'output': 'qualified',
                'args': {'prefix': '05'},
            },
        ],
    }
    if drop_columns is not None:
        recipe['drop_columns'] = drop_columns
    return recipe


def test_drop_columns_removes_scratch_column():
    ingester = _make_ingester(_recipe(drop_columns=['raw_zfill']))
    df = ingester._preprocess_recipe_data(_gdf())

    assert 'raw_zfill' not in df.columns
    assert df['qualified'].tolist() == ['05001', '05023']
    assert df['raw'].tolist() == ['1', '23']


def test_without_drop_columns_scratch_column_is_kept():
    ingester = _make_ingester(_recipe())
    df = ingester._preprocess_recipe_data(_gdf())

    assert 'raw_zfill' in df.columns


def test_drop_columns_ignores_missing_names():
    ingester = _make_ingester(_recipe(drop_columns=['raw_zfill', 'does_not_exist']))
    df = ingester._preprocess_recipe_data(_gdf())

    assert 'raw_zfill' not in df.columns
