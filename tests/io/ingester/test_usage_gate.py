"""A resolved usage decline soft-skips the partition; it never downloads.

The usage gate sits in `_download_and_unzip_recipe_data`, the one point
every download mechanism passes through. A source whose requirement was
resolved as a decline reuses the unavailable-partition short-circuit --
the same soft skip a partition with no download URL gets -- so a batch
run moves on rather than dying. Only the unattended, unresolved case
raises, and that happens inside `require_usage_compatible` itself
(covered in tests/io/test_usage_profile.py).
"""

from __future__ import annotations

import pytest

import openplaces.io.ingester as ingester_module
from openplaces.core.schema import AdminId, Source
from openplaces.io.ingester import Ingester


class _Timer:
    def mark(self, *args, **kwargs):
        pass


class _Entity:
    def __init__(self, source):
        self.source = source


def _make_ingester(tmp_path, source):
    external = tmp_path / 'external'
    external.mkdir()

    ing = object.__new__(Ingester)
    ing.recipe = {
        'entity': _Entity(source),
        'admin_id': AdminId('US', 'CA'),
        'recipe_id': 'US-CA_parcel-gated-2026',
    }
    ing.verbose = False
    ing.timer = _Timer()
    ing.recipe_external_dir = external
    ing.recipe_heap_dir = external
    ing.download_partition = {
        'download_url': 'http://example.invalid/file',
        'downloaded_path': None,
        'data_path': None,
        'admin_id_to_download': None,
        'partition_id_to_download': None,
    }
    return ing


@pytest.fixture
def _no_network(monkeypatch):
    monkeypatch.setattr(
        ingester_module,
        'download',
        lambda *a, **k: pytest.fail('download() ran for a gated source'),
    )


def test_a_declined_requirement_soft_skips_the_partition(
    tmp_path, monkeypatch, _no_network
):
    source = Source(
        source_id='gated',
        download_url='http://example.invalid/file',
        usage_requirement={'non_commercial': True},
    )
    ing = _make_ingester(tmp_path, source)
    seen = {}

    def _decline(src, recipe_id=None, admin_id=None, verbose=False):
        seen.update({'source': src, 'recipe_id': recipe_id, 'admin_id': admin_id})
        return False

    monkeypatch.setattr(ingester_module, 'require_usage_compatible', _decline)

    ing._download_and_unzip_recipe_data()

    assert ing.download_partition['unavailable'] is True
    assert seen['source'] is source
    assert seen['recipe_id'] == 'US-CA_parcel-gated-2026'
    assert seen['admin_id'] == 'US-CA'


def test_a_source_without_a_requirement_never_consults_the_gate(tmp_path, monkeypatch):
    source = Source(source_id='open', download_url='http://example.invalid/file')
    ing = _make_ingester(tmp_path, source)
    # Let the run stop at the download itself; reaching it proves the
    # gate did not divert an unrestricted source.
    monkeypatch.setattr(
        ingester_module,
        'require_usage_compatible',
        lambda *a, **k: pytest.fail('gate consulted without a requirement'),
    )
    monkeypatch.setattr(
        ingester_module, 'download', lambda *a, **k: (_ for _ in ()).throw(_Stop())
    )

    with pytest.raises(_Stop):
        ing._download_and_unzip_recipe_data()

    assert 'unavailable' not in ing.download_partition


class _Stop(Exception):
    pass
