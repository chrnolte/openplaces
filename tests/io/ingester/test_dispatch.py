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


def test_crosswalk_carries_platform_and_defaults_to_drivable(tmp_path, monkeypatch):
    """A crosswalk without a platform column reads as all-drivable.

    The column was added once Massachusetts turned out to spread its 21
    registries over at least three applications; a crosswalk written
    before that must keep working.
    """
    import pandas as pd

    csv_path = tmp_path / 'towns.csv'
    csv_path.write_text(
        'town_name,base_url,platform\n'
        'Alpha,https://example.invalid/Alpha/,avenu_i2\n'
        'Beta,https://beta.invalid/ALIS/WW400R.HTM,alis\n'
        'Gamma,https://example.invalid/Gamma/,\n',
        encoding='utf-8',
    )
    monkeypatch.setattr(
        registry_module,
        'get_admin',
        lambda *a, **k: pd.DataFrame(
            {'name': ['Alpha', 'Beta', 'Gamma'], 'type': ['town'] * 3},
            index=['US-XX-ALP', 'US-XX-BET', 'US-XX-GAM'],
        ),
    )

    cw = registry_module.RegistryIngester._load_crosswalk(csv_path, 'US-XX', level=3)

    assert cw['US-XX-ALP']['platform'] == registry_module.DRIVABLE_PLATFORM
    assert cw['US-XX-BET']['platform'] == 'alis'
    # An empty cell means "not stated", which is the pre-column meaning.
    assert cw['US-XX-GAM']['platform'] == registry_module.DRIVABLE_PLATFORM


def test_every_massachusetts_town_names_a_reachable_registry():
    """No town may point at a URL that 404s, which is how Brookline read.

    The shipped crosswalk pointed all 352 towns at masslandrecords with
    slugs that were never checked; ten districts are on their own sites
    and three more are spelled differently there. This pins the two
    facts that fix catches: every row states a platform, and only the
    drivable one uses a masslandrecords URL.
    """
    import csv as _csv

    from openplaces.path import recipe_path

    recipe = get_recipe_by_id('US-MA_transaction-masslandrecords-v1')
    path = recipe_path(
        recipe['admin_id'], recipe['entity'], filename='town_to_registry.csv'
    )
    rows = list(_csv.DictReader(path.open(encoding='utf-8')))
    assert rows, 'crosswalk is empty'
    assert 'platform' in rows[0], 'crosswalk states no platform'

    for r in rows:
        assert r['platform'], f'{r["town_name"]} states no platform'
        on_mlr = 'masslandrecords.com' in r['base_url']
        if on_mlr:
            assert r['platform'] == registry_module.DRIVABLE_PLATFORM, (
                f'{r["town_name"]} claims masslandrecords but platform '
                f'{r["platform"]!r}'
            )
        # The Berkshire districts are the ones we had spelled wrong.
        assert 'Berkshire' not in r['base_url'], (
            f'{r["town_name"]} uses the invented Berkshire* slug, '
            'which 404s; the site spells it BerkMiddle/North/South'
        )
