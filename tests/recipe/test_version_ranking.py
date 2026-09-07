"""Tests for version ranking and the ingest auto-discovery scan.

Versions were compared as raw strings, which puts 'v2' above '2026' and
'v2' above 'v10'; and the dependency scan emitted every version of a
recipe where the harmonizer reads only the newest, so an older file
became a declared input of a job that never opens it.
"""

import pandas as pd
import pytest

from openplaces import recipe as recipe_module
from openplaces.recipe import _scan_ingest_recipe_ids, version_sort_key


@pytest.mark.parametrize(
    ('lower', 'higher'),
    [
        ('v2', '2026'),
        ('v2', 'v10'),
        ('2025', '2026'),
        ('2026', '2026pc'),
        ('', 'v0'),
    ],
)
def test_version_order(lower, higher):
    assert version_sort_key(lower) < version_sort_key(higher)


def _row(admin_id, source_id, version, filename_suffix=''):
    recipe_id = f'{admin_id}_parcel-{source_id}-{version}'
    if filename_suffix:
        recipe_id += f'_{filename_suffix}'
    return {
        'admin_id': admin_id,
        'source_id': source_id,
        'version': version,
        'exclude_from_auto_discover': False,
        'recipe_id': recipe_id,
        'filename_suffix': filename_suffix,
    }


def test_scan_keeps_only_the_newest_version_per_source(monkeypatch):
    rows = pd.DataFrame(
        [
            _row('US-FL', 'floridagio', '2025'),
            _row('US-FL', 'floridagio', '2026'),
            _row('US-FL', 'floridagio', '2026', 'improvement-detail'),
        ]
    )
    import openplaces.diagnostics as diagnostics

    monkeypatch.setattr(diagnostics, 'find_recipes', lambda *a, **k: rows)
    recipe_module._scan_ingest_recipe_ids.cache_clear()
    try:
        found = {s['recipe_id'] for s in _scan_ingest_recipe_ids('parcel')}
    finally:
        recipe_module._scan_ingest_recipe_ids.cache_clear()

    # The suffixed sibling is a different table and survives; the older
    # version of the same table does not.
    assert found == {
        'US-FL_parcel-floridagio-2026',
        'US-FL_parcel-floridagio-2026_improvement-detail',
    }


def test_entity_lookup_parses_each_recipe_file_once():
    """The resolver re-read and re-parsed the tree on every call."""
    from openplaces.recipe import _recipe_yaml, find_entity_recipe_id

    _recipe_yaml.cache_clear()
    find_entity_recipe_id('US-NC-BRU', 'footprint', stage='ingest', silent=True)
    after_first = _recipe_yaml.cache_info()
    find_entity_recipe_id('US-NC-BRU', 'footprint', stage='ingest', silent=True)
    after_second = _recipe_yaml.cache_info()

    assert after_first.misses > 0
    assert after_second.misses == after_first.misses
    assert after_second.hits > after_first.hits
