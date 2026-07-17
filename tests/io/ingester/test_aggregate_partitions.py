"""Tests for partition rollup: footer coverage, union/replace merge, skip."""

import json

import pandas as pd
import pytest

from openplaces import cfg
from openplaces.io import read_parquet, save_parquet
from openplaces.io.aggregate import (
    _aggregate_to_file,
    _legacy_upgrader,
    aggregate_partitions,
    read_file_metadata,
    read_partition_coverage,
)
from openplaces.recipe import get_output_path, get_recipe_by_id

WI_RECIPE_ID = 'US-WI_transaction-widor-2026'
MA_RECIPE_ID = 'US-MA_transaction-masslandrecords-v1'


def _month_df(month: str, n: int = 3) -> pd.DataFrame:
    """Small distinct frame standing in for one month's transactions."""
    return pd.DataFrame(
        {
            'instrument_number': [f'{month}-{i}' for i in range(n)],
            'price_raw': [f'${i}.00' for i in range(n)],
        }
    )


def _write_month(path, month, n=3):
    save_parquet(_month_df(month, n), path)
    return path


def test_footer_metadata_roundtrip(tmp_path):
    """save_parquet(file_metadata=...) writes a footer read back without rows."""
    path = tmp_path / 'x.parquet'
    save_parquet(
        _month_df('202101'),
        path,
        file_metadata={'openplaces:partitions': json.dumps(['202101', '202102'])},
    )
    assert read_partition_coverage(path) == {'202101', '202102'}
    # Data still reads normally and carries no stray metadata column.
    df = read_parquet(path)
    assert 'openplaces:partitions' not in df.columns
    assert len(df) == 3


def test_partition_aggregation_migrates_legacy_transaction_price(tmp_path):
    source = tmp_path / 'legacy.parquet'
    final = tmp_path / 'all.parquet'
    save_parquet(pd.DataFrame({'consideration_raw': ['$1,250.00']}), source)

    _aggregate_to_file(
        final,
        [('202601', source)],
        how='union',
        keep_original=True,
        reset_index=True,
        transform=_legacy_upgrader(get_recipe_by_id(WI_RECIPE_ID)),
    )

    result = read_parquet(final)
    assert 'consideration_raw' not in result
    assert result.loc[0, 'price_raw'] == '$1,250.00'
    assert result.loc[0, 'price'] == 1250.0


def test_read_coverage_missing_or_unmarked(tmp_path):
    """Empty set for a missing file or a file without the coverage key."""
    assert read_partition_coverage(tmp_path / 'nope.parquet') == set()
    plain = tmp_path / 'plain.parquet'
    save_parquet(_month_df('202101'), plain)
    assert read_partition_coverage(plain) == set()


def test_aggregate_to_file_union_append_and_idempotent(tmp_path):
    """union integrates new rows and is a no-op when re-adding a partition."""
    final = tmp_path / 'all.parquet'
    p1 = _write_month(tmp_path / 'm1.parquet', '202101')
    _aggregate_to_file(
        final,
        [('202101', p1)],
        how='union',
        reset_index=True,
        keep_original=True,
        file_metadata={'openplaces:partitions': json.dumps(['202101'])},
    )
    assert len(read_parquet(final)) == 3

    # Append a new month.
    p2 = _write_month(tmp_path / 'm2.parquet', '202102')
    _aggregate_to_file(
        final,
        [('202102', p2)],
        how='union',
        reset_index=True,
        keep_original=True,
        file_metadata={'openplaces:partitions': json.dumps(['202101', '202102'])},
    )
    assert len(read_parquet(final)) == 6

    # Re-add the same month: union de-dupes against the existing file (no-op).
    _aggregate_to_file(
        final,
        [('202102', p2)],
        how='union',
        reset_index=True,
        keep_original=True,
        file_metadata={'openplaces:partitions': json.dumps(['202101', '202102'])},
    )
    assert len(read_parquet(final)) == 6


def test_aggregate_to_file_rejects_internal_duplicates(tmp_path):
    """A new batch with internal full-row duplicates raises before merge."""
    final = tmp_path / 'all.parquet'
    dup = tmp_path / 'dup.parquet'
    save_parquet(pd.concat([_month_df('202101', 1)] * 2), dup)
    with pytest.raises(ValueError, match='duplicate'):
        _aggregate_to_file(
            final,
            [('202101', dup)],
            how='union',
            reset_index=True,
            keep_original=True,
        )


def test_aggregate_to_file_replace_overwrites(tmp_path):
    """replace overwrites the file with only the current inputs."""
    final = tmp_path / 'all.parquet'
    p1 = _write_month(tmp_path / 'm1.parquet', '202101')
    _aggregate_to_file(
        final, [('202101', p1)], how='union', reset_index=True, keep_original=True
    )
    assert len(read_parquet(final)) == 3

    p2 = _write_month(tmp_path / 'm2.parquet', '202102', n=2)
    _aggregate_to_file(
        final,
        [('202102', p2)],
        how='replace',
        reset_index=True,
        keep_original=True,
        file_metadata={'openplaces:partitions': json.dumps(['202102'])},
    )
    df = read_parquet(final)
    assert len(df) == 2
    assert set(df['instrument_number']) == {'202102-0', '202102-1'}
    assert read_partition_coverage(final) == {'202102'}


@pytest.fixture
def cache_to_tmp(tmp_path, monkeypatch):
    """Redirect the 'cache' data dir so tests never touch real outputs."""
    monkeypatch.setitem(cfg.config['directories'], 'cache', tmp_path)
    return tmp_path


def test_aggregate_partitions_single_file_only(cache_to_tmp):
    """single_file leaves only _all (no month/year files) with footer coverage."""
    recipe = get_recipe_by_id(WI_RECIPE_ID)
    months = ['202101', '202102', '202201']
    for m in months:
        _write_month(get_output_path(recipe, 'US-WI', partition_id=m), m)

    aggregate_partitions(
        recipe, single_file=True, admin_ids='US-WI', partition_ids=months
    )

    all_path = get_output_path(recipe, 'US-WI', partition_id='all')
    assert all_path.exists()
    assert read_partition_coverage(all_path) == set(months)
    assert len(read_parquet(all_path)) == 9
    # Month files deleted; no per-year files created.
    out_dir = all_path.parent
    leftovers = sorted(p.name for p in out_dir.glob('*.parquet'))
    assert leftovers == [all_path.name]


def test_ingester_skip_uses_footer_coverage(cache_to_tmp):
    """reprocess=False skips months recorded in the _all footer."""
    from openplaces.io.ingester import Ingester

    recipe = get_recipe_by_id(WI_RECIPE_ID)
    months = ['202101', '202102']
    for m in months:
        _write_month(get_output_path(recipe, 'US-WI', partition_id=m), m)
    aggregate_partitions(
        recipe, single_file=True, admin_ids='US-WI', partition_ids=months
    )

    ing = Ingester(WI_RECIPE_ID, partition_ids=['202101', '202102', '202103'])
    ing._resolve_admin_ids(False)
    ing._resolve_partition_ids(False)
    # 202101/202102 are in the footer; only the uncovered 202103 remains.
    assert ing.partition_ids_to_download == ['202103']

    ing2 = Ingester(WI_RECIPE_ID, partition_ids=['202101', '202102', '202103'])
    ing2._resolve_admin_ids(True)
    ing2._resolve_partition_ids(True)
    assert set(ing2.partition_ids_to_download) == {'202101', '202102', '202103'}


def _keyed_df(rows: dict[str, str]) -> pd.DataFrame:
    """Frame with a meaningful (non-default) string index."""
    df = pd.DataFrame({'value': list(rows.values())}, index=list(rows.keys()))
    df.index.name = 'record_id'
    return df


def test_duplicate_identity_includes_meaningful_index(tmp_path):
    """Equal values with distinct index keys are not duplicates."""
    final = tmp_path / 'all.parquet'
    p1 = tmp_path / 'p1.parquet'
    save_parquet(_keyed_df({'a': 'x', 'b': 'x'}), p1)
    _aggregate_to_file(final, [('p1', p1)], how='union', keep_original=True)
    df = read_parquet(final)
    assert len(df) == 2

    # Re-running the same input is a no-op (index + values identical).
    _aggregate_to_file(final, [('p1', p1)], how='union', keep_original=True)
    df = read_parquet(final)
    assert len(df) == 2
    assert df.index.name == 'record_id'
    assert set(df.index) == {'a', 'b'}

    # Identical index AND values within the new batch -> internal duplicate.
    p2 = tmp_path / 'p2.parquet'
    save_parquet(pd.concat([_keyed_df({'c': 'x'})] * 2), p2)
    with pytest.raises(ValueError, match='duplicate'):
        _aggregate_to_file(final, [('p2', p2)], how='union', keep_original=True)


@pytest.fixture
def external_to_tmp(tmp_path, monkeypatch):
    """Redirect the 'external' data dir so MA-style tests stay isolated."""
    monkeypatch.setitem(cfg.config['directories'], 'external', tmp_path)
    return tmp_path


TOWN = 'US-MA-MI-SO'
CKS = ['2021-01_Deed', '2021-02_Deed']


def test_aggregate_partitions_discovers_checkpoint_files(external_to_tmp):
    """No download_by: partition ids are discovered from files on disk."""
    recipe = get_recipe_by_id(MA_RECIPE_ID)
    for ck in CKS:
        _write_month(get_output_path(recipe, TOWN, partition_id=ck), ck)

    aggregate_partitions(recipe, single_file=True, admin_ids=TOWN, keep_original=True)

    all_path = get_output_path(recipe, TOWN, partition_id='all')
    assert all_path.exists()
    assert read_partition_coverage(all_path) == set(CKS)
    assert len(read_parquet(all_path)) == 6
    # keep_original retains the monthly scrape files next to _all.
    for ck in CKS:
        assert get_output_path(recipe, TOWN, partition_id=ck).exists()

    # New rows in a retained checkpoint file get integrated on re-run (union).
    ck = CKS[0]
    _write_month(get_output_path(recipe, TOWN, partition_id=ck), ck, n=5)
    aggregate_partitions(recipe, single_file=True, admin_ids=TOWN, keep_original=True)
    assert len(read_parquet(all_path)) == 8


def _registry_ingester(recipe):
    from openplaces.io.ingester.registry_ingester import RegistryIngester

    ing = RegistryIngester.__new__(RegistryIngester)
    ing.recipe = recipe
    ing.verbose = False
    ing._coverage_cache = {}
    return ing


def test_registry_save_stamps_scrape_time(external_to_tmp):
    """_save records the scrape timestamp in the parquet footer."""
    recipe = get_recipe_by_id(MA_RECIPE_ID)
    ing = _registry_ingester(recipe)
    ing._save([{'book': '1', 'page': '2', 'grantor': 'A'}], TOWN, partition_id=CKS[0])
    path = get_output_path(recipe, TOWN, partition_id=CKS[0])
    scraped_at = read_file_metadata(path).get('openplaces:scraped_at')
    assert scraped_at is not None and scraped_at[:2] == '20'


def test_registry_is_done_completeness(external_to_tmp):
    """Done iff scraped after month end; legacy files and footer coverage count."""
    recipe = get_recipe_by_id(MA_RECIPE_ID)
    ing = _registry_ingester(recipe)
    past, current = '2021-01_Deed', '2099-01_Deed'

    # Missing entirely.
    assert not ing._is_done(TOWN, past)

    # Scraped after the month closed (real _save stamps today's date).
    ing._save([{'book': '1', 'page': '2'}], TOWN, partition_id=past)
    assert ing._is_done(TOWN, past)

    # Scraped while the month was still running (month end in the future).
    ing._save([{'book': '1', 'page': '2'}], TOWN, partition_id=current)
    assert not ing._is_done(TOWN, current)

    # Legacy file without the timestamp counts as done.
    legacy = '2021-03_Deed'
    save_parquet(
        _month_df('202103'), get_output_path(recipe, TOWN, partition_id=legacy)
    )
    assert ing._is_done(TOWN, legacy)

    # Missing file whose checkpoint is covered by the _all footer. Fresh
    # instance: the coverage cache is only invalidated by _aggregate_town,
    # and the earlier _is_done calls cached the then-missing _all as empty.
    covered = '2021-04_Deed'
    save_parquet(
        _month_df('202104'),
        get_output_path(recipe, TOWN, partition_id='all'),
        file_metadata={'openplaces:partitions': json.dumps([covered])},
    )
    ing = _registry_ingester(recipe)
    assert ing._is_done(TOWN, covered)
    assert not ing._is_done(TOWN, '2021-05_Deed')
