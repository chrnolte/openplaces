"""`process_by.layer_pattern` reads one layer per admin unit.

MetroGIS ships its seven county parcel layers in one file geodatabase.
Each unit reads the layer its crosswalk code names; a name the file
lacks raises rather than falling back to the first layer, which would
save one county's rows under another county's id. The fixture is
fabricated: two layers of made-up ids.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from openplaces.io.ingester.table_ingester import TableIngester


class _Timer:
    def mark(self, *args, **kwargs):
        return None


class _Stop(Exception):
    pass


def _two_layer_gpkg(path: Path) -> Path:
    for name, ids in (('ParcelsAlpha', ['A1', 'A2']), ('ParcelsBeta', ['B1'])):
        gpd.GeoDataFrame(
            {'pin': ids},
            geometry=[box(i, 0, i + 1, 1) for i in range(len(ids))],
            crs='EPSG:26915',
        ).to_file(path, layer=name, driver='GPKG')
    return path


def _ingester(data_path: Path, admin_id: str) -> TableIngester:
    ti = TableIngester.__new__(TableIngester)
    ti.recipe = {
        'process_by': {'admin_level': 3, 'layer_pattern': 'Parcels{co_name_file}'},
    }
    ti.download_partition = {
        'data_path': data_path,
        'partition_id_to_download': None,
        'admin_id_crosswalk_reverse': pd.Series(
            ['Alpha', 'Beta', 'Gamma'],
            index=pd.Index(['XX-AA-AL', 'XX-AA-BE', 'XX-AA-GA'], name='admin3_id'),
            name='co_name_file',
        ),
    }
    ti.processing_chunk = {'admin_id_to_process': admin_id}
    ti.timer = _Timer()
    ti.verbose = False
    return ti


@pytest.mark.parametrize(
    ('admin_id', 'layer', 'pins'),
    [('XX-AA-AL', 'ParcelsAlpha', ['A1', 'A2']), ('XX-AA-BE', 'ParcelsBeta', ['B1'])],
)
def test_each_unit_reads_its_own_layer(tmp_path, admin_id, layer, pins):
    gpkg = _two_layer_gpkg(tmp_path / 'parcels.gpkg')
    ti = _ingester(gpkg, admin_id)

    resolved = ti._resolve_layer_pattern('Parcels{co_name_file}', gpkg)
    gdf = ti._read_recipe_data(layer_override=resolved)

    assert resolved == layer
    assert sorted(gdf['pin']) == pins


def test_process_passes_the_unit_layer_to_the_read(tmp_path, monkeypatch):
    gpkg = _two_layer_gpkg(tmp_path / 'parcels.gpkg')
    ti = _ingester(gpkg, 'XX-AA-BE')
    seen = {}

    def fake_read(**kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(ti, '_read_recipe_data', fake_read)
    with pytest.raises(_Stop):
        ti.process(process_in_chunks=True)

    assert seen['layer_override'] == 'ParcelsBeta'
    assert 'fids' not in seen


def test_a_missing_layer_raises_and_lists_the_layers(tmp_path):
    gpkg = _two_layer_gpkg(tmp_path / 'parcels.gpkg')
    ti = _ingester(gpkg, 'XX-AA-GA')

    with pytest.raises(ValueError, match='ParcelsGamma') as err:
        ti._resolve_layer_pattern('Parcels{co_name_file}', gpkg)

    assert 'ParcelsAlpha' in str(err.value)
    assert 'ParcelsBeta' in str(err.value)


def test_a_unit_outside_the_crosswalk_raises(tmp_path):
    gpkg = _two_layer_gpkg(tmp_path / 'parcels.gpkg')
    ti = _ingester(gpkg, 'XX-AA-ZZ')

    with pytest.raises(KeyError, match='layer name'):
        ti._resolve_layer_pattern('Parcels{co_name_file}', gpkg)
