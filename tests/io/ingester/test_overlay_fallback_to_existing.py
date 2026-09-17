"""`overlay_admin_ids: fallback_to_existing` keeps a recipe-derived id.

Without the option the overlay writes the admin id column in place, so a
record without coordinates loses the unit its county-name column gave
it. With it, the location decides wherever it resolves and the recipe's
own id fills the rest; the counts are reported and kept on the ingester.
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box

from openplaces.io.ingester.table_ingester import TableIngester
from openplaces.timing import Timer


def _admin():
    return gpd.GeoSeries(
        [box(0, 0, 1, 1), box(1, 0, 2, 1)],
        index=pd.Index(['XX-AA-AA', 'XX-AA-BB'], name='admin3_id'),
        crs='EPSG:4326',
    )


def _records():
    # Text says AA for all three; the second point lies in BB and the
    # third record has no point at all.
    return gpd.GeoDataFrame(
        {'admin3_id': ['XX-AA-AA'] * 3, 'value': [1, 2, 3]},
        geometry=[Point(0.5, 0.5), Point(1.5, 0.5), None],
        crs='EPSG:4326',
        index=pd.Index(['a', 'b', 'c'], name='record_id'),
    )


def _ingester(fallback):
    spec = {'admin_level': 3}
    if fallback is not None:
        spec['fallback_to_existing'] = fallback
    ingester = TableIngester.__new__(TableIngester)
    ingester.recipe = {
        'entity': None,
        'dataset': 'records',
        'process_by': {'admin_level': 2},
        'overlay_admin_ids': spec,
    }
    ingester.download_partition = {'admin_geometries': _admin()}
    ingester.processing_chunk = {'admin_id_to_process': 'XX-AA'}
    ingester.timer = Timer('test')
    ingester.verbose = False
    return ingester


def test_location_moves_rows_and_text_keeps_unlocated_ones(capsys):
    ingester = _ingester(True)

    df = ingester._preprocess_recipe_data(_records())

    assert df['admin3_id'].tolist() == ['XX-AA-AA', 'XX-AA-BB', 'XX-AA-AA']
    assert ingester.located_admin_id_counts == {
        'located': 2,
        'fallback': 1,
        'disagree': 1,
        'unresolved': 0,
    }
    assert 'by location' in capsys.readouterr().out


def test_without_the_option_an_unlocated_row_loses_its_unit():
    df = _ingester(None)._preprocess_recipe_data(_records())

    assert df['admin3_id'].iloc[:2].tolist() == ['XX-AA-AA', 'XX-AA-BB']
    assert pd.isna(df['admin3_id'].iloc[2])
