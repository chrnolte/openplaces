"""`Ingester._load_scraper_fetch` loads a scraper module once per process.

Scraper modules are loaded dynamically by file path (their filenames may
contain hyphens), bypassing the `sys.modules` caching a normal `import`
provides for free. Without its own cache, `_load_scraper_fetch` would
re-execute the module from scratch on every call -- silently resetting any
module-level state the scraper keeps for itself (e.g. `google_drive_scraper`'s
per-process folder-listing memo) before every single admin unit/partition.
"""

from __future__ import annotations

import openplaces.io.ingester as ingester_module


def test_load_scraper_fetch_reuses_same_module_across_calls(monkeypatch):
    monkeypatch.setattr(ingester_module, '_SCRAPER_MODULE_CACHE', {})

    fetch_1 = ingester_module.Ingester._load_scraper_fetch('google_drive_scraper')
    fetch_2 = ingester_module.Ingester._load_scraper_fetch('google_drive_scraper')

    assert fetch_1.__module__ == fetch_2.__module__
    assert fetch_1.__globals__ is fetch_2.__globals__


def test_load_scraper_fetch_preserves_module_level_state_across_calls(monkeypatch):
    monkeypatch.setattr(ingester_module, '_SCRAPER_MODULE_CACHE', {})

    fetch = ingester_module.Ingester._load_scraper_fetch('google_drive_scraper')
    fetch.__globals__['_FOLDER_INDEX_MEMO']['sentinel'] = ('marker', True)

    fetch_again = ingester_module.Ingester._load_scraper_fetch('google_drive_scraper')

    assert fetch_again.__globals__['_FOLDER_INDEX_MEMO']['sentinel'] == (
        'marker',
        True,
    )
