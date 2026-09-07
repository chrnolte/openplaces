"""A compressed source `geopandas` cannot read in place is extracted and read
from the heap, but the extraction must not repoint the download partition's
shared `data_path`.

`data_path` is shared by every table (layer) in a download partition and is the
sentinel both the "already unzipped" skip and the heap cleanup key on. Writing
the extracted file back over it made every later table in the same partition
read whichever file this one happened to extract.
"""

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box

from openplaces.io.ingester import table_ingester as ti_module
from openplaces.io.ingester.table_ingester import TableIngester
from openplaces.timing import Timer


def _frame(value):
    return gpd.GeoDataFrame(
        {'value': [value]}, geometry=[box(0, 0, 1, 1)], crs='EPSG:4326'
    )


@pytest.fixture
def unreadable_archive(tmp_path, monkeypatch):
    """A `.zip` no reader accepts, plus the file extraction produces."""
    archive = tmp_path / 'source.zip'
    archive.write_bytes(b'not really a zip')
    heap = tmp_path / 'heap'
    heap.mkdir()
    extracted = heap / 'extracted.geojson'
    extracted.write_text('{}', encoding='utf-8')

    def fake_read_file(path, **kwargs):
        if Path(path) == archive:
            raise ValueError('cannot open archive in place')
        return _frame(str(Path(path).name))

    monkeypatch.setattr(ti_module.gpd, 'read_file', fake_read_file)
    monkeypatch.setattr(ti_module, 'unzip', lambda *a, **k: None)
    monkeypatch.setattr(ti_module, 'find_latest_file_or_gdb', lambda _dir: extracted)
    return archive, heap, extracted


def _ingester(archive, heap, partition):
    ingester = TableIngester.__new__(TableIngester)
    ingester.recipe = {}
    ingester.download_partition = partition
    ingester.processing_chunk = {}
    ingester.recipe_heap_dir = heap
    ingester.timer = Timer('test')
    ingester.verbose = False
    return ingester


def test_extraction_does_not_repoint_the_partition_data_path(unreadable_archive):
    archive, heap, extracted = unreadable_archive
    partition = {'data_path': archive}
    gdf = _ingester(archive, heap, partition)._read_recipe_data()

    assert gdf['value'].tolist() == ['extracted.geojson']
    assert partition['data_path'] == archive


def test_a_later_table_still_starts_from_the_archive(unreadable_archive):
    archive, heap, extracted = unreadable_archive
    partition = {'data_path': archive}
    first = _ingester(archive, heap, partition)
    first.recipe = {'layer': 'parcels'}
    first._read_recipe_data()

    second = _ingester(archive, heap, partition)
    second.recipe = {'layer': 'buildings'}
    gdf = second._read_recipe_data()

    assert partition['data_path'] == archive
    assert gdf['value'].tolist() == ['extracted.geojson']


def test_missing_extraction_names_the_archive_in_the_error(
    tmp_path, monkeypatch, unreadable_archive
):
    archive, heap, _extracted = unreadable_archive
    monkeypatch.setattr(ti_module, 'find_latest_file_or_gdb', lambda _dir: None)
    partition = {'data_path': archive}

    with pytest.raises(OSError, match='source.zip'):
        _ingester(archive, heap, partition)._read_recipe_data()
