"""`download_by.pinned_download_urls` holds one download unit to a
named file while every other unit keeps the scraped one (FEMA USA
Structures' 2026-09-13 release dropped most heights in some states).
"""

import io
import os

from openplaces.core.schema import AdminId, Entity, Source
from openplaces.io.ingester import Ingester
from openplaces.io.ingester import download as ingester_download
from openplaces.timing import Timer

PINNED = 'https://example.invalid/files/Old%20Name/Release2025XX.zip'


def _ingester(unit):
    source = Source(
        source_id='fabricated',
        download_url_source='https://example.invalid/list.json',
        download_url_source_regex='(https://example.invalid/files/[^"]+[.]zip)',
    )
    ingester = Ingester.__new__(Ingester)
    ingester.recipe = {
        'admin_id': AdminId('US'),
        'entity': Entity('footprint', source, '2023'),
        'download_by': {
            'admin_level': 2,
            'pinned_download_urls': {'US-XX': PINNED},
        },
        'compressed_file_name': '*.zip',
        'uncompressed_file_name': 'XX_Structures.gdb',
    }
    ingester.admin_ids = [AdminId('US')]
    ingester.partition_ids = None
    ingester.timer = Timer('test')
    ingester.verbose = False
    ingester._owns_timer = False
    ingester.download_partition = {
        'admin_id_to_download': AdminId(unit),
        'partition_id_to_download': None,
    }
    return ingester


class _Page(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_a_pinned_unit_takes_its_url_without_scraping(monkeypatch):
    def _no_scrape(*args, **kwargs):
        raise AssertionError('a pinned unit must not scrape the list')

    monkeypatch.setattr(ingester_download.urllib.request, 'urlopen', _no_scrape)
    ingester = _ingester('US-XX')

    ingester._resolve_download_url()

    assert ingester.download_partition['download_url'] == PINNED


def test_an_unpinned_unit_still_takes_the_scraped_url(monkeypatch):
    page = b'{"url": "https://example.invalid/files/New/Release2026XY.zip"}'
    monkeypatch.setattr(
        ingester_download.urllib.request, 'urlopen', lambda *a, **k: _Page(page)
    )
    ingester = _ingester('US-XY')

    ingester._resolve_download_url()

    assert ingester.download_partition['download_url'] == (
        'https://example.invalid/files/New/Release2026XY.zip'
    )


def test_a_pinned_unit_reads_the_pinned_file_not_a_newer_match(tmp_path, monkeypatch):
    external = tmp_path / 'external'
    external.mkdir()
    old = external / 'Release2025XX.zip'
    new = external / 'Release2026XX.zip'
    old.write_bytes(b'old')
    new.write_bytes(b'new')
    # The newer file is the most recent match of the recipe's
    # wildcard, which the glob would otherwise pick.
    os.utime(old, (1_000_000_000, 1_000_000_000))
    monkeypatch.setattr(
        ingester_download, 'heap_dir', lambda *a, **k: tmp_path / 'heap'
    )
    monkeypatch.setattr(ingester_download, 'external_dir', lambda *a, **k: external)
    ingester = _ingester('US-XX')
    ingester.download_partition['download_url'] = PINNED

    ingester._resolve_downloaded_and_data_paths()

    assert ingester.download_partition['downloaded_path'] == old
