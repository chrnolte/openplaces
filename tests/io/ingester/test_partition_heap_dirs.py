"""A partition reads the member of its own archive, never another's.

A recipe partitioned by year with one archive per year (`SDF_{year}.zip`)
whose member has the same name in every archive (`SALEDISC.txt`) used to
extract every year to one heap path. A member left there by an earlier
partition, or by a run that stopped before its cleanup, then passed the
"already unzipped" skip: the partition read another year's rows, never
opened its own archive, and the yearly files merged into one table full
of duplicates. Every value below is fabricated.
"""

import zipfile

import pytest

import openplaces.io.ingester.download as ingester_download
from openplaces.core.schema import AdminId
from openplaces.io.ingester import Ingester
from openplaces.timing import Timer

MEMBER = 'SALES.txt'


class _Source:
    download_url = 'https://example.invalid/SALES_{year}.zip'
    download_url_source = None
    download_url_scraper = None
    usage_requirement = None
    verify_ssl = True


class _Entity:
    source = _Source()


def _recipe():
    return {
        'admin_id': AdminId('US'),
        'entity': _Entity(),
        'download_by': {'partition': 'year', 'first': 2021, 'last': 2022},
        'compressed_file_name': 'SALES_{year}.zip',
        'uncompressed_file_name': MEMBER,
    }


def _ingester(recipe, year, tmp_path, monkeypatch):
    monkeypatch.setattr(
        ingester_download, 'heap_dir', lambda *a, **k: tmp_path / 'heap'
    )
    monkeypatch.setattr(
        ingester_download, 'external_dir', lambda *a, **k: tmp_path / 'external'
    )
    ingester = Ingester.__new__(Ingester)
    ingester.recipe = recipe
    ingester.timer = Timer('test')
    ingester.verbose = False
    ingester._owns_timer = False
    ingester.download_partition = {
        'admin_id_to_download': None,
        'partition_id_to_download': year,
        'download_url': f'https://example.invalid/SALES_{year}.zip',
    }
    return ingester


def _write_archive(tmp_path, year, text):
    external = tmp_path / 'external'
    external.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(external / f'SALES_{year}.zip', 'w') as zf:
        zf.writestr(MEMBER, text)
        zf.writestr('CONTACTS.txt', 'id\n1\n')


def test_each_year_has_its_own_heap_path(tmp_path, monkeypatch):
    first = _ingester(_recipe(), 2021, tmp_path, monkeypatch)
    first._resolve_downloaded_and_data_paths()
    second = _ingester(_recipe(), 2022, tmp_path, monkeypatch)
    second._resolve_downloaded_and_data_paths()

    assert (
        first.download_partition['data_path'] != second.download_partition['data_path']
    )
    assert first.download_partition['data_path'].name == MEMBER


def test_a_leftover_member_of_another_year_is_not_read(tmp_path, monkeypatch):
    _write_archive(tmp_path, 2021, 'sale_id\nA-2021\n')
    # What a run that failed on 2022 before its cleanup left behind,
    # in the heap layout before this fix and in 2022's own folder.
    for leftover in (tmp_path / 'heap', tmp_path / 'heap' / '2022'):
        leftover.mkdir(parents=True, exist_ok=True)
        (leftover / MEMBER).write_text('sale_id\nB-2022\n')

    ingester = _ingester(_recipe(), 2021, tmp_path, monkeypatch)
    ingester._resolve_downloaded_and_data_paths()
    ingester._download_and_unzip_recipe_data()

    data_path = ingester.download_partition['data_path']
    assert data_path.read_text() == 'sale_id\nA-2021\n'


def test_a_year_without_its_archive_fails_naming_it(tmp_path, monkeypatch):
    _write_archive(tmp_path, 2021, 'sale_id\nA-2021\n')
    done = _ingester(_recipe(), 2021, tmp_path, monkeypatch)
    done._resolve_downloaded_and_data_paths()
    done._download_and_unzip_recipe_data()

    def _no_file(url, target, **kwargs):
        raise FileNotFoundError(f'nothing published at {url}')

    monkeypatch.setattr(ingester_download, 'download', _no_file)
    missing = _ingester(_recipe(), 2022, tmp_path, monkeypatch)
    missing._resolve_downloaded_and_data_paths()

    with pytest.raises(FileNotFoundError, match='SALES_2022.zip'):
        missing._download_and_unzip_recipe_data()


def test_one_archive_with_a_member_per_year_keeps_one_folder(tmp_path, monkeypatch):
    recipe = _recipe()
    recipe['compressed_file_name'] = 'SALES.zip'
    recipe['uncompressed_file_name'] = 'SALES_{year}.txt'
    ingester = _ingester(recipe, 2021, tmp_path, monkeypatch)
    ingester._resolve_downloaded_and_data_paths()

    assert ingester.recipe_heap_dir == tmp_path / 'heap'
    assert ingester.download_partition['data_path'].name == 'SALES_2021.txt'


def test_an_unpartitioned_recipe_keeps_one_folder(tmp_path, monkeypatch):
    recipe = _recipe()
    del recipe['download_by']
    recipe['compressed_file_name'] = 'SALES.zip'
    ingester = _ingester(recipe, None, tmp_path, monkeypatch)
    ingester._resolve_downloaded_and_data_paths()

    assert ingester.recipe_heap_dir == tmp_path / 'heap'
