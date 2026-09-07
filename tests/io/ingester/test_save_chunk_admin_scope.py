"""A processing chunk saves only its own units, not a sibling's.

`_save_recipe_data` splits a chunk into per-unit files when `save_to` is
finer than `process_by`. It selected the units to write with a raw string
prefix test, which is not containment once leaf codes mix two and three
characters: 'US-NC-WA' is Wake County's pre-2026 id and 'US-NC-WAR' is
Warren County's current one, so Warren's towns were written as Wake's.
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from openplaces.core.schema import Entity
from openplaces.io.ingester import table_ingester
from openplaces.io.ingester.table_ingester import TableIngester

WAKE_TOWN = 'US-NC-WAK-CA'
WARREN_TOWN = 'US-NC-WAR-FO'


def _gdf():
    return gpd.GeoDataFrame(
        {'admin4_id': [WAKE_TOWN, WARREN_TOWN]},
        geometry=[box(i, i, i + 1, i + 1) for i in range(2)],
        crs='EPSG:4326',
    )


def _ingester(chunk_admin_id):
    ingester = TableIngester.__new__(TableIngester)
    ingester.recipe = {
        'entity': Entity('parcel'),
        'process_by': {'admin_level': 3},
        'save_to': {'admin_level': 4},
    }
    ingester.processing_chunk = {'admin_id_to_process': chunk_admin_id}
    ingester.download_partition = {}
    ingester.admin_ids_to_save = [WAKE_TOWN, WARREN_TOWN]
    return ingester


def _run(monkeypatch, chunk_admin_id, tmp_path):
    saved = {}

    def fake_get_output_path(recipe, admin_id, partition_id=None, **kwargs):
        return tmp_path / f'{admin_id}.parquet'

    def fake_save_parquet(gdf, output_path):
        saved[output_path.stem] = gdf

    monkeypatch.setattr(table_ingester, 'get_output_path', fake_get_output_path)
    monkeypatch.setattr(table_ingester, 'save_parquet', fake_save_parquet)
    _ingester(chunk_admin_id)._save_recipe_data(_gdf())
    return saved


def test_a_shorter_sibling_id_claims_no_units(monkeypatch, tmp_path):
    saved = _run(monkeypatch, 'US-NC-WA', tmp_path)
    assert saved == {}


def test_a_chunk_saves_its_own_units(monkeypatch, tmp_path):
    saved = _run(monkeypatch, 'US-NC-WAR', tmp_path)
    assert list(saved) == [WARREN_TOWN]
    assert isinstance(saved[WARREN_TOWN], pd.DataFrame)
