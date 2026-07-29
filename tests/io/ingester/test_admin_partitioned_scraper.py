"""_run_download_scraper forwards the current admin unit to admin-level-
partitioned scrapers (recipes with download_by.admin_level, no partition key,
where partition_id_to_download is always None)."""

from pathlib import Path

import openplaces.io.ingester as ingester_module


class _FakeTimer:
    def mark(self, *args, **kwargs):
        pass


def test_run_download_scraper_passes_admin_id_to_download(monkeypatch, tmp_path):
    calls = []

    def fake_fetch(**kwargs):
        calls.append(kwargs)
        Path(kwargs['target_path']).touch()
        return kwargs['target_path']

    monkeypatch.setattr(
        ingester_module.Ingester,
        '_load_scraper_fetch',
        staticmethod(lambda name: fake_fetch),
    )

    ing = ingester_module.Ingester.__new__(ingester_module.Ingester)
    ing.verbose = False
    ing.timer = _FakeTimer()
    ing.recipe = {'entity': None}
    ing.download_partition = {
        'data_path': None,
        'downloaded_path': tmp_path / 'out.parquet',
        'partition_id_to_download': None,
        'admin_id_to_download': 'US-MA-MI',
    }

    ing._run_download_scraper('fake_scraper')

    assert calls[0]['admin_id_to_download'] == 'US-MA-MI'
    assert calls[0]['partition_id'] is None
    assert Path(ing.download_partition['downloaded_path']).exists()
