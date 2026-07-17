import pytest

import openplaces.io.ingester as ingester_module
import openplaces.io.ingester.registry_ingester as registry_module
from openplaces.recipe import get_recipe_by_id


class FakeIngester:
    calls = []

    def __init__(self, recipe, **kwargs):
        self.calls.append(('init', recipe, kwargs))

    def ingest(self, **kwargs):
        self.calls.append(('ingest', kwargs))


class FakeRegistryIngester:
    calls = []

    def __init__(self, recipe, **kwargs):
        self.calls.append(('init', recipe, kwargs))

    def ingest(self):
        self.calls.append(('ingest',))


def test_ingest_converts_years_to_standard_year_month_partitions(monkeypatch):
    FakeIngester.calls = []
    monkeypatch.setattr(ingester_module, 'Ingester', FakeIngester)
    recipe = {
        'download_by': {
            'partition': 'year_month',
            'first': 202501,
            'last': 202601,
        }
    }

    ingester_module.ingest(recipe, years=[2025])

    init_kwargs = FakeIngester.calls[0][2]
    assert init_kwargs['partition_ids'] == [
        '202501',
        '202502',
        '202503',
        '202504',
        '202505',
        '202506',
        '202507',
        '202508',
        '202509',
        '202510',
        '202511',
        '202512',
    ]


def test_wi_transaction_year_uses_standard_ingester(monkeypatch):
    FakeIngester.calls = []
    monkeypatch.setattr(ingester_module, 'Ingester', FakeIngester)
    recipe = get_recipe_by_id('US-WI_transaction-widor-2026')

    ingester_module.ingest(recipe, years=[2026])

    init_kwargs = FakeIngester.calls[0][2]
    assert init_kwargs['partition_ids'] == ['202601']


def test_ingest_forwards_years_to_registry_ingester(monkeypatch):
    FakeRegistryIngester.calls = []
    monkeypatch.setattr(
        registry_module,
        'RegistryIngester',
        FakeRegistryIngester,
    )
    recipe = {'scraper': {'ingester': 'registry'}}

    ingester_module.ingest(
        recipe,
        years=[2024],
    )

    init_kwargs = FakeRegistryIngester.calls[0][2]
    assert init_kwargs['years'] == [2024]
    assert init_kwargs['partition_ids'] is None


def test_ingest_forwards_month_partitions_to_registry_ingester(monkeypatch):
    FakeRegistryIngester.calls = []
    monkeypatch.setattr(
        registry_module,
        'RegistryIngester',
        FakeRegistryIngester,
    )
    recipe = {'scraper': {'ingester': 'registry'}}

    ingester_module.ingest(recipe, partition_ids=['202501'])

    init_kwargs = FakeRegistryIngester.calls[0][2]
    assert init_kwargs['partition_ids'] == ['202501']
    assert init_kwargs['years'] is None


def test_ingest_rejects_year_month_under_years():
    recipe = {
        'download_by': {
            'partition': 'year_month',
            'first': 202501,
            'last': 202501,
        }
    }

    with pytest.raises(ValueError, match='use partition_ids'):
        ingester_module.ingest(recipe, years=[202501])


def test_ingest_rejects_partition_ids_with_years():
    recipe = {
        'download_by': {
            'partition': 'year_month',
            'first': 202501,
            'last': 202501,
        }
    }

    with pytest.raises(ValueError, match='either partition_ids or years'):
        ingester_module.ingest(
            recipe,
            partition_ids=['202501'],
            years=[2025],
        )


def test_registry_ingester_uses_year_month_partition_ids():
    ingester = registry_module.RegistryIngester.__new__(
        registry_module.RegistryIngester
    )
    ingester.partition_ids = ['202501']
    ingester.years = None
    ingester.recipe = {'scraper': {}}

    assert ingester._date_partitions() == [('2025-01-01', '2025-01-31')]


def test_registry_ingester_rejects_year_month_under_years():
    ingester = registry_module.RegistryIngester.__new__(
        registry_module.RegistryIngester
    )
    ingester.partition_ids = None
    ingester.years = [202501]
    ingester.recipe = {'scraper': {}}

    with pytest.raises(ValueError, match='use partition_ids'):
        ingester._date_partitions()
