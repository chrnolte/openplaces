"""Extraction must be driven by the file type, not by `redownload`.

`_download_and_unzip_recipe_data` once read `if redownload or _is_archive`,
so re-running any flat-file recipe (a .geojson/.csv fetched straight from
`download_url`) handed a non-archive to `unzip` and raised BadZipFile --
after paying for the download. Archives must still extract on every run,
with or without the flag, because `unzip` has no skip-if-extracted branch.
"""

from __future__ import annotations

import openplaces.io.ingester as ingester_module
from openplaces.io.ingester import Ingester


class _Timer:
    def mark(self, *args, **kwargs):
        pass


class _Source:
    download_url = 'http://example.invalid/file'
    download_url_source = None
    download_url_scraper = None
    verify_ssl = True


class _Entity:
    source = _Source()


def _make_ingester(tmp_path, *, downloaded_name, data_name, monkeypatch):
    """Build an Ingester stub wired to a single already-present download."""
    external = tmp_path / 'external'
    heap = tmp_path / 'heap'
    external.mkdir()
    heap.mkdir()

    downloaded = external / downloaded_name
    downloaded.write_text('not really an archive')
    data = (external if data_name == downloaded_name else heap) / data_name

    ing = object.__new__(Ingester)
    ing.recipe = {'entity': _Entity()}
    ing.verbose = False
    ing.timer = _Timer()
    ing.recipe_external_dir = external
    ing.recipe_heap_dir = heap
    ing.download_partition = {
        'download_url': 'http://example.invalid/file',
        'downloaded_path': downloaded,
        'data_path': data,
        'admin_id_to_download': None,
        'partition_id_to_download': None,
    }

    calls = []

    def _fake_unzip(*args, **kwargs):
        calls.append(args)
        # Stand in for the real extraction, so the caller's
        # post-extraction existence check sees what it expects.
        data.write_text('extracted')

    monkeypatch.setattr(ingester_module, 'unzip', _fake_unzip)
    monkeypatch.setattr(
        ingester_module, 'download', lambda url, target, **k: downloaded
    )
    return ing, calls


def test_redownload_does_not_unzip_a_flat_file(tmp_path, monkeypatch):
    """The regression: flat file + redownload must not reach unzip."""
    ing, calls = _make_ingester(
        tmp_path,
        downloaded_name='parcels.geojson',
        data_name='parcels.geojson',
        monkeypatch=monkeypatch,
    )

    ing._download_and_unzip_recipe_data(redownload=True)

    assert calls == []


def test_archive_still_unzips_with_redownload(tmp_path, monkeypatch):
    ing, calls = _make_ingester(
        tmp_path,
        downloaded_name='parcels.zip',
        data_name='parcels.shp',
        monkeypatch=monkeypatch,
    )

    ing._download_and_unzip_recipe_data(redownload=True)

    assert len(calls) == 1


def test_archive_still_unzips_without_redownload(tmp_path, monkeypatch):
    """Guards against 'fixing' this by inverting the condition.

    Note the extracted file must be absent here: the caller returns early
    when `data_path` already exists and `redownload` is False, so this
    only exercises the extraction guard when there is nothing to reuse.
    """
    ing, calls = _make_ingester(
        tmp_path,
        downloaded_name='parcels.zip',
        data_name='parcels.shp',
        monkeypatch=monkeypatch,
    )

    ing._download_and_unzip_recipe_data(redownload=False)

    assert len(calls) == 1
