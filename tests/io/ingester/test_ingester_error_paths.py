"""Ingester paths that were meant to report a problem but raised a different
error instead, or crashed a run that should have proceeded.

Each of these is reachable from a shipped recipe shape: a verbose image
ingest, a level-0 recipe whose admin resolution went wrong, a source with a
`download_url_source` but no `download_by`, and a source with no download URL
at all (a manually placed file).
"""

import pytest

from openplaces.core.schema import AdminId
from openplaces.io.ingester import Ingester
from openplaces.timing import Timer


def _bare(recipe):
    ingester = Ingester.__new__(Ingester)
    ingester.recipe = recipe
    ingester.admin_ids = [AdminId(None)]
    ingester.partition_ids = None
    ingester.timer = Timer('test')
    ingester.verbose = True
    ingester._owns_timer = False
    return ingester


def test_verbose_image_ingest_reports_the_recipe_id(capsys):
    from openplaces.core.schema import Entity

    ingester = _bare(
        {
            'admin_id': AdminId('US'),
            'entity': Entity('building', 'nsi', '2022'),
            'image_scraper': 'google_satellite',
        }
    )
    ingester.ingest()

    assert 'imagery is fetched on the fly' in capsys.readouterr().out


def test_level_zero_process_mismatch_names_the_ids():
    ingester = _bare({'admin_id': AdminId(None), 'save_to': {}})
    ingester.admin_ids_to_save = ['US-NC']

    with pytest.raises(ValueError, match='admin_ids_to_save is'):
        ingester._resolve_admin_ids_to_process()


def test_level_zero_download_mismatch_names_the_ids():
    ingester = _bare({'admin_id': AdminId(None), 'save_to': {}})
    ingester.admin_ids_to_process = ['US-NC']

    with pytest.raises(ValueError, match='admin_ids_to_process is'):
        ingester._resolve_admin_ids_to_download()


def test_download_url_source_without_download_by_explains_itself():
    from openplaces.core.schema import Entity, Source

    source = Source(
        source_id='fabricated',
        download_url_source='https://example.invalid/list.html',
        download_url_source_regex='href="(.*?[.]zip)"',
    )
    entity = Entity('parcel', source, '2026')
    ingester = _bare({'admin_id': AdminId('US'), 'entity': entity})
    ingester.download_partition = {}

    with pytest.raises(ValueError, match='`download_by` is not defined'):
        ingester._resolve_download_url()


def test_no_download_url_does_not_crash_path_resolution(tmp_path, monkeypatch):
    from openplaces.core.schema import Entity, Source
    from openplaces.io import ingester as ingester_module

    source = Source(source_id='fabricated', portal_url='https://example.invalid')
    entity = Entity('parcel', source, '2026')
    monkeypatch.setattr(ingester_module, 'heap_dir', lambda *a, **k: tmp_path / 'heap')
    monkeypatch.setattr(
        ingester_module, 'external_dir', lambda *a, **k: tmp_path / 'external'
    )
    ingester = _bare({'admin_id': AdminId('US'), 'entity': entity})
    ingester.download_partition = {
        'admin_id_to_download': None,
        'partition_id_to_download': None,
        'download_url': None,
    }

    ingester._resolve_downloaded_and_data_paths()

    assert ingester.download_partition['downloaded_path'] is None
