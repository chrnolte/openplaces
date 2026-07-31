"""Tests for the generic Google Drive folder scraper (network-free, gdown
mocked)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import openplaces.io.scrapers.google_drive_scraper as gd


class _FakeGdown:
    def __init__(
        self,
        index: dict,
        downloads: list,
        zip_member: str | None = None,
        fail_times: int = 0,
    ):
        self._index = index
        self.downloads = downloads
        self._zip_member = zip_member
        self._fail_times = fail_times
        self.listing_calls = 0

    def download_folder(self, url, skip_download=True, quiet=True):
        self.listing_calls += 1
        if self.listing_calls <= self._fail_times:
            raise TimeoutError('simulated stalled connection')
        return [SimpleNamespace(path=p, id=i) for p, i in self._index.items()]

    def download(self, id, output, quiet=True):
        self.downloads.append((id, output))
        if self._zip_member:
            with zipfile.ZipFile(output, 'w') as zf:
                zf.writestr(self._zip_member, 'member contents')
        else:
            Path(output).write_bytes(b'fake file contents')


def _patch_gdown(monkeypatch, index, zip_member=None):
    downloads = []
    fake = _FakeGdown(index, downloads, zip_member=zip_member)
    monkeypatch.setattr(gd, '_import_gdown', lambda: fake)
    return fake


def test_resolve_placeholders_admin_levels_and_partition_id():
    resolved = gd._resolve_placeholders(
        'parcels/{admin2}/{admin1}/{partition_id}.zip',
        admin_id_to_download='US-MA-MI',
        admin_key_column=None,
        partition_id='boundaries',
    )
    assert resolved == 'parcels/MA/US/boundaries.zip'


def test_resolve_placeholders_admin_key_uses_get_admin(monkeypatch):
    calls = []

    def fake_get_admin(admin_id, level, columns=None):
        calls.append((str(admin_id), level, columns))
        return pd.DataFrame({columns: ['25017']})

    monkeypatch.setattr(gd, 'get_admin', fake_get_admin)

    resolved = gd._resolve_placeholders(
        '{admin_key}_parcel_{partition_id}_parquet.zip',
        admin_id_to_download='US-MA-MI',
        admin_key_column='admin3_id_admin1',
        partition_id='predictions',
    )

    assert resolved == '25017_parcel_predictions_parquet.zip'
    assert calls == [('US-MA-MI', 3, 'admin3_id_admin1')]


def test_resolve_placeholders_admin_key_requires_column():
    with pytest.raises(ValueError, match='admin_key_column'):
        gd._resolve_placeholders(
            '{admin_key}.zip',
            admin_id_to_download='US-MA-MI',
            admin_key_column=None,
            partition_id='boundaries',
        )


def test_fetch_downloads_atomically_and_caches_folder_index(tmp_path, monkeypatch):
    url = 'https://drive.google.com/drive/folders/ABC123'
    index = {'parcels/MA/boundaries_parquet.zip': 'fileid-1'}
    fake = _patch_gdown(monkeypatch, index)

    target_path = tmp_path / 'boundaries_parquet.zip'
    result = gd.fetch(
        'boundaries',
        target_path,
        portal_url=url,
        admin_id_to_download='US-MA-MI',
        file_path='parcels/{admin2}/{partition_id}_parquet.zip',
        verbose=False,
    )

    assert result == target_path
    assert target_path.exists()
    assert target_path.read_bytes() == b'fake file contents'
    # Downloaded via a sibling .part path, not straight to the final name.
    assert fake.downloads[0][1].endswith('.part')
    assert not target_path.with_name(target_path.name + '.part').exists()

    cache_path = tmp_path / 'gdrive_folder_index.json'
    assert cache_path.exists()
    assert json.loads(cache_path.read_text()) == index


def test_fetch_skips_download_when_target_already_exists(tmp_path, monkeypatch):
    url = 'https://drive.google.com/drive/folders/ABC123'
    index = {'parcels/MA/boundaries_parquet.zip': 'fileid-1'}
    fake = _patch_gdown(monkeypatch, index)

    target_path = tmp_path / 'boundaries_parquet.zip'
    target_path.write_bytes(b'already here')

    result = gd.fetch(
        'boundaries',
        target_path,
        portal_url=url,
        admin_id_to_download='US-MA-MI',
        file_path='parcels/{admin2}/{partition_id}_parquet.zip',
        verbose=False,
    )

    assert result == target_path
    assert target_path.read_bytes() == b'already here'
    assert fake.downloads == []


def test_fetch_redownload_refetches_despite_existing_target(tmp_path, monkeypatch):
    url = 'https://drive.google.com/drive/folders/ABC123'
    index = {'parcels/MA/boundaries_parquet.zip': 'fileid-1'}
    fake = _patch_gdown(monkeypatch, index)

    target_path = tmp_path / 'boundaries_parquet.zip'
    target_path.write_bytes(b'stale contents')

    result = gd.fetch(
        'boundaries',
        target_path,
        portal_url=url,
        admin_id_to_download='US-MA-MI',
        file_path='parcels/{admin2}/{partition_id}_parquet.zip',
        redownload=True,
        verbose=False,
    )

    assert result == target_path
    assert target_path.read_bytes() == b'fake file contents'
    assert len(fake.downloads) == 1


def test_fetch_returns_none_when_not_yet_published(tmp_path, monkeypatch):
    url = 'https://drive.google.com/drive/folders/ABC123'
    index = {}  # boundaries not published for this county
    _patch_gdown(monkeypatch, index)

    target_path = tmp_path / '25099_parcel_boundaries_parquet.zip'
    result = gd.fetch(
        'boundaries',
        target_path,
        portal_url=url,
        admin_id_to_download='US-MA-XX',
        file_path='parcels/{admin2}/{partition_id}_parquet.zip',
        verbose=False,
    )

    assert result is None
    assert not target_path.exists()


def test_fetch_requires_file_path_and_portal_url(tmp_path):
    out_path = tmp_path / 'out.zip'
    with pytest.raises(ValueError, match='file_path'):
        gd.fetch('boundaries', out_path, portal_url='https://x', file_path=None)
    with pytest.raises(ValueError, match='portal_url'):
        gd.fetch('boundaries', out_path, portal_url=None, file_path='x.zip')


def test_fetch_extracts_single_member_zip_when_target_differs(tmp_path, monkeypatch):
    # Mirrors the real source: the zip's own name embeds "_parquet" before
    # the extension, but its single member doesn't (verified against a real
    # download -- see the recipe's uncompressed_file_name comment).
    url = 'https://drive.google.com/drive/folders/ABC123'
    index = {'parcels/MA/25017_parcel_boundaries_parquet.zip': 'fileid-1'}
    fake = _patch_gdown(
        monkeypatch, index, zip_member='25017_parcel_boundaries.parquet'
    )

    target_path = tmp_path / '25017_parcel_boundaries.parquet'
    result = gd.fetch(
        'boundaries',
        target_path,
        portal_url=url,
        admin_id_to_download='US-MA-MI',
        file_path='parcels/{admin2}/25017_parcel_{partition_id}_parquet.zip',
        verbose=False,
    )

    assert result == target_path
    assert target_path.read_text() == 'member contents'
    # The raw zip persists too, flat, under its own real Drive basename.
    raw_path = tmp_path / '25017_parcel_boundaries_parquet.zip'
    assert raw_path.exists()
    assert len(fake.downloads) == 1

    # Re-running skips both the download and the re-extraction.
    fake.downloads.clear()
    result2 = gd.fetch(
        'boundaries',
        target_path,
        portal_url=url,
        admin_id_to_download='US-MA-MI',
        file_path='parcels/{admin2}/25017_parcel_{partition_id}_parquet.zip',
        verbose=False,
    )
    assert result2 == target_path
    assert fake.downloads == []

    # redownload=True re-fetches the raw zip and re-extracts the member,
    # instead of returning early on the "already downloaded/extracted" checks.
    fake.downloads.clear()
    result3 = gd.fetch(
        'boundaries',
        target_path,
        portal_url=url,
        admin_id_to_download='US-MA-MI',
        file_path='parcels/{admin2}/25017_parcel_{partition_id}_parquet.zip',
        redownload=True,
        verbose=False,
    )
    assert result3 == target_path
    assert target_path.read_text() == 'member contents'
    assert len(fake.downloads) == 1


def test_fetch_retries_folder_listing_after_transient_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(gd.time, 'sleep', lambda seconds: None)
    url = 'https://drive.google.com/drive/folders/ABC123'
    index = {'parcels/MA/boundaries_parquet.zip': 'fileid-1'}
    fake = _patch_gdown(monkeypatch, index)
    fake._fail_times = 2  # fails twice, succeeds on the 3rd (default retries=3)

    target_path = tmp_path / 'boundaries_parquet.zip'
    result = gd.fetch(
        'boundaries',
        target_path,
        portal_url=url,
        admin_id_to_download='US-MA-MI',
        file_path='parcels/{admin2}/{partition_id}_parquet.zip',
        verbose=False,
    )

    assert result == target_path
    assert fake.listing_calls == 3


def test_fetch_gives_up_after_exhausting_retries(tmp_path, monkeypatch):
    monkeypatch.setattr(gd.time, 'sleep', lambda seconds: None)
    url = 'https://drive.google.com/drive/folders/ABC123'
    index = {'parcels/MA/boundaries_parquet.zip': 'fileid-1'}
    fake = _patch_gdown(monkeypatch, index)
    fake._fail_times = 5  # always fails within the default retries=3

    target_path = tmp_path / 'boundaries_parquet.zip'
    with pytest.raises(TimeoutError):
        gd.fetch(
            'boundaries',
            target_path,
            portal_url=url,
            admin_id_to_download='US-MA-MI',
            file_path='parcels/{admin2}/{partition_id}_parquet.zip',
            verbose=False,
        )
    assert fake.listing_calls == 3


def test_fetch_redownload_bypasses_stale_cached_folder_index(tmp_path, monkeypatch):
    url = 'https://drive.google.com/drive/folders/ABC123'
    target_path = tmp_path / '25099_parcel_boundaries_parquet.zip'

    # A cached index from an earlier (incomplete) listing that never saw the
    # file we now want -- simulates the "not published, skipping" false
    # positive from a stale on-disk cache.
    stale_cache = tmp_path / 'gdrive_folder_index.json'
    stale_cache.write_text(json.dumps({}))

    index = {'parcels/MA/25099_parcel_boundaries_parquet.zip': 'fileid-1'}
    _patch_gdown(monkeypatch, index)

    # Without redownload, the stale cache is trusted and the file is
    # reported missing even though gdown would now find it.
    result = gd.fetch(
        'boundaries',
        target_path,
        portal_url=url,
        admin_id_to_download='US-MA-MI',
        file_path='parcels/{admin2}/25099_parcel_{partition_id}_parquet.zip',
        verbose=False,
    )
    assert result is None

    # redownload=True bypasses the stale cache with a fresh live listing.
    result = gd.fetch(
        'boundaries',
        target_path,
        portal_url=url,
        admin_id_to_download='US-MA-MI',
        file_path='parcels/{admin2}/25099_parcel_{partition_id}_parquet.zip',
        redownload=True,
        verbose=False,
    )
    assert result == target_path
    assert json.loads(stale_cache.read_text()) == index


def test_fetch_lists_folder_at_most_once_per_process_despite_redownload(
    tmp_path, monkeypatch
):
    # Mirrors a real ingest run: many admin units/partitions, all sharing the
    # same Drive folder and all passing --redownload, so the disk-cache
    # bypass (`cache_path.exists() and not redownload`) would otherwise fire
    # -- and re-list the whole folder -- on every single call.
    url = 'https://drive.google.com/drive/folders/ABC123'
    index = {
        'parcels/25001_parcel_boundaries_parquet.zip': 'fileid-1',
        'parcels/25005_parcel_boundaries_parquet.zip': 'fileid-2',
    }
    fake = _patch_gdown(monkeypatch, index)

    for key in ['25001', '25005']:
        target_path = tmp_path / f'{key}_parcel_boundaries_parquet.zip'
        result = gd.fetch(
            'boundaries',
            target_path,
            portal_url=url,
            admin_id_to_download='US-MA-MI',
            file_path=f'parcels/{key}_parcel_{{partition_id}}_parquet.zip',
            redownload=True,
            verbose=False,
        )
        assert result == target_path

    assert fake.listing_calls == 1


def test_fetch_raises_when_raw_file_not_zip_but_target_differs(tmp_path, monkeypatch):
    url = 'https://drive.google.com/drive/folders/ABC123'
    index = {'parcels/MA/25017_boundaries.csv': 'fileid-1'}
    _patch_gdown(monkeypatch, index)  # writes plain (non-zip) bytes

    target_path = tmp_path / 'renamed.csv'
    with pytest.raises(ValueError, match='not a zip'):
        gd.fetch(
            'boundaries',
            target_path,
            portal_url=url,
            admin_id_to_download='US-MA-MI',
            file_path='parcels/{admin2}/25017_boundaries.csv',
            verbose=False,
        )
