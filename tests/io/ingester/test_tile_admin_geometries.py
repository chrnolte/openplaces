"""A tile covering units from two different admin files loads both.

`get_admin` resolves one output file per call, keyed on the deepest id asked
for, so a tile straddling a state line came back with only one state's units.
The other state's counties then got null geometry, the left join gave their
buildings no admin id, and nothing was written for them.
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from openplaces.io.ingester import table_ingester as ti_module
from openplaces.io.ingester.table_ingester import TableIngester
from openplaces.timing import Timer

STATE_UNITS = {
    'XA': ['XA-AA', 'XA-AB'],
    'XB': ['XB-BA'],
}


def _one_file_get_admin(admin_id, level, **kwargs):
    """Stand-in for `get_admin`: answers from one state's file per call."""
    requested = admin_id if isinstance(admin_id, list) else [admin_id]
    state = str(requested[-1]).split('-')[0]
    units = STATE_UNITS[state]
    return gpd.GeoDataFrame(
        geometry=[box(i, 0, i + 1, 1) for i in range(len(units))],
        index=pd.Index(units, name='admin2_id'),
        crs='EPSG:4326',
    )


@pytest.fixture
def tile_ingester(monkeypatch, tmp_path):
    monkeypatch.setattr(ti_module, 'get_admin', _one_file_get_admin)
    monkeypatch.setattr(ti_module, 'get_crs', lambda *a, **k: 'EPSG:4326')

    ingester = TableIngester.__new__(TableIngester)
    ingester.recipe = {'overlay_admin_ids': {'admin_level': 2}}
    ingester.download_partition = {
        'admin_id_to_download': None,
        'data_path': tmp_path / 'tile.parquet',
        'admin_ids_in_tile': ['XA-AA', 'XB-BA'],
    }
    ingester.processing_chunk = {}
    ingester.timer = Timer('test')
    ingester.verbose = False
    return ingester


def test_both_states_units_are_loaded(tile_ingester):
    tile_ingester._load_admin_geometries()

    loaded = tile_ingester.download_partition['admin_geometries']
    assert sorted(loaded.index) == ['XA-AA', 'XB-BA']


def test_an_unresolvable_unit_warns_and_stops(tile_ingester):
    tile_ingester.download_partition['admin_ids_in_tile'] = ['XA-AA', 'XA-ZZ']

    with pytest.warns(UserWarning, match='no geometry at level 2'):
        tile_ingester._load_admin_geometries()

    loaded = tile_ingester.download_partition['admin_geometries']
    assert list(loaded.index) == ['XA-AA']
