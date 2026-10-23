import pytest

import openplaces.io.ingester as ingester_module
from openplaces.recipe import get_recipe_by_id


class FakeIngester:
    calls = []

    def __init__(self, recipe, **kwargs):
        self.calls.append(('init', recipe, kwargs))

    def ingest(self, **kwargs):
        self.calls.append(('ingest', kwargs))


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


def test_ingest_refuses_a_scraper_recipe(monkeypatch):
    """A recipe that asks for a crawler is refused before any download.

    The registry crawler was removed because the site it served
    prohibits crawling; a recipe naming one must not fall through to
    the standard ingester either.
    """
    FakeIngester.calls = []
    monkeypatch.setattr(ingester_module, 'Ingester', FakeIngester)

    with pytest.raises(ValueError, match='does not crawl'):
        ingester_module.ingest({'scraper': {'ingester': 'registry'}})
    assert FakeIngester.calls == []


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
