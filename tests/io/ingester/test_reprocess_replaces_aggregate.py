"""A reprocess of every partition replaces the combined file.

The default union merge keeps rows the new batch does not repeat
exactly, so a reprocess that changed a value appended a second copy of
every row: Orange County FL's sales file doubled to 3.1M rows on
2026-09-12, and the curate dedup then kept the stale first copies.
"""

import pytest

import openplaces.io.aggregate as aggregate_module
from openplaces.io.ingester import Ingester


def _ingester(partition_ids=None, **aggregate_by):
    ingester = Ingester.__new__(Ingester)
    ingester.recipe = {
        'recipe_id': 'US-XX_transaction-test-2026',
        'aggregate_by': {'single_file': True, **aggregate_by},
    }
    ingester.partition_ids = partition_ids
    ingester.verbose = False
    return ingester


@pytest.fixture
def captured(monkeypatch):
    calls = []
    monkeypatch.setattr(
        aggregate_module, 'aggregate_partitions', lambda *a, **k: calls.append(k)
    )
    return calls


def test_a_plain_run_unions(captured):
    _ingester()._aggregate_partitions(reprocess=False)
    assert captured[0]['how'] == 'union'


def test_a_full_reprocess_replaces(captured):
    _ingester()._aggregate_partitions(reprocess=True)
    assert captured[0]['how'] == 'replace'


def test_a_partial_reprocess_into_a_union_file_is_refused(captured):
    with pytest.raises(ValueError, match='every partition'):
        _ingester(partition_ids=['2021'])._aggregate_partitions(reprocess=True)
    assert captured == []


def test_a_partial_reprocess_with_a_partition_column_replaces_those_years(
    captured,
):
    """Named years are replaced by their column, not refused."""
    ingester = _ingester(partition_ids=['2021'], partition_column='tax_year')
    ingester._aggregate_partitions(reprocess=True)
    assert captured[0]['how'] == 'union'
    assert captured[0]['partition_column'] == 'tax_year'
