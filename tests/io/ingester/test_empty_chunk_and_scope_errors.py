"""An empty chunk, and the errors around it.

A recipe `query` can empty a whole partition by design (New England has no
level-4 subdivisions, so `US_admin-census-2025_admin4` filters it out). Both the
crosswalk join and the admin overlay skip an empty frame, so the split-by-admin
save then found no admin id column, raised about the missing column, and raised
a second error from `sample(1)` while building the message.
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from openplaces.io.ingester.table_ingester import TableIngester
from openplaces.timing import Timer


def _ingester(tmp_path, monkeypatch, recipe):
    from openplaces.io.ingester import table_ingester as ti_module

    written = []
    monkeypatch.setattr(
        ti_module,
        'get_output_path',
        lambda _r, admin_id, partition_id=None: tmp_path / f'{admin_id}.parquet',
    )
    monkeypatch.setattr(
        ti_module, 'save_parquet', lambda gdf, path: written.append(path)
    )

    ingester = TableIngester.__new__(TableIngester)
    ingester.recipe = recipe
    ingester.download_partition = {'partition_id_to_download': None}
    ingester.processing_chunk = {'admin_id_to_process': 'XX-AA'}
    ingester.timer = Timer('test')
    ingester.verbose = False
    ingester.admin_ids_to_save = ['XX-AA-AA']
    return ingester, written


def _recipe():
    return {
        'entity': None,
        'dataset': 'thing',
        'process_by': {'admin_level': 2},
        'save_to': {'admin_level': 3},
    }


def test_an_empty_chunk_writes_nothing_and_does_not_raise(tmp_path, monkeypatch):
    ingester, written = _ingester(tmp_path, monkeypatch, _recipe())
    empty = gpd.GeoDataFrame({'value': []}, geometry=[], crs='EPSG:4326')

    ingester._save_recipe_data(empty)

    assert written == []


def test_a_populated_chunk_missing_the_column_still_raises(tmp_path, monkeypatch):
    ingester, _written = _ingester(tmp_path, monkeypatch, _recipe())
    populated = gpd.GeoDataFrame(
        {'value': [1]}, geometry=[box(0, 0, 1, 1)], crs='EPSG:4326'
    )

    with pytest.raises(ValueError, match="'admin3_id' does not exist"):
        ingester._save_recipe_data(populated)


def test_an_uncovered_unit_names_the_crosswalk(tmp_path):
    ingester = TableIngester.__new__(TableIngester)
    ingester.recipe = {'process_by': {'file_pattern': '{admin3_id_admin2}_*.csv'}}
    ingester.download_partition = {
        'admin_id_crosswalk_reverse': pd.Series(
            ['011'],
            index=pd.Index(['XX-AA-AA'], name='admin3_id'),
            name='admin3_id_admin2',
        ),
        'partition_id_to_download': '2008',
    }
    ingester.processing_chunk = {'admin_id_to_process': 'XX-AA-ZZ'}
    ingester.recipe_heap_dir = tmp_path
    ingester.timer = Timer('test')
    ingester.verbose = False

    with pytest.raises(KeyError, match='admin_id_crosswalk'):
        ingester._resolve_file_pattern_path('{admin3_id_admin2}_*.csv')
