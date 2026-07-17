"""Tests for io/cleanup.py on a synthetic data root."""

import json
import os
import time

import pandas as pd
import pytest

import openplaces.diagnostics as diagnostics
import openplaces.io as opio
from openplaces.config import cfg
from openplaces.io import cleanup as cl
from openplaces.recipe import get_output_path, get_recipe_by_id

NSI = 'US_building-nsi-2022'
FOOTPRINT_SPINE = 'US_footprint-spine-2026'
PARCEL_SPINE = 'US_parcel-spine-2026'
COUNTY = 'US-NC-BS'


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """Point every configured data directory at a synthetic root."""
    dirs = dict(cfg.config['directories'])
    dirs['data_root'] = tmp_path
    dirs['core'] = tmp_path / 'data/core'
    dirs['external'] = tmp_path / 'data/external'
    dirs['raw'] = tmp_path / 'data/raw'
    dirs['cache'] = tmp_path / 'data/cache'
    dirs['heap'] = tmp_path / 'data/cache/_heap'
    dirs['logs'] = tmp_path / 'data/cache/_logs'
    dirs['out'] = tmp_path / 'data/out'
    dirs['share'] = tmp_path / 'data/share'
    monkeypatch.setitem(cfg.config, 'directories', dirs)
    monkeypatch.delenv('SNAKEMAKE', raising=False)
    monkeypatch.delenv('OPENPLACES_ORCHESTRATED', raising=False)
    return tmp_path


def _write_parquet(path, df=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if df is None:
        df = pd.DataFrame({'geo_id': ['a', 'b'], 'value': [1.0, 2.0]})
    df.to_parquet(path)
    return path


def _nsi_path(admin=COUNTY):
    return get_output_path(get_recipe_by_id(NSI), admin_id=admin)


def _spine_paths(admin=COUNTY):
    return (
        get_output_path(get_recipe_by_id(FOOTPRINT_SPINE), admin_id=admin),
        get_output_path(get_recipe_by_id(PARCEL_SPINE), admin_id=admin),
    )


# Receipts


def test_receipt_round_trip(data_root):
    out = _nsi_path()
    receipt = {
        'recipe_id': NSI,
        'admin_id': COUNTY,
        'consumers_verified': [
            {'recipe_id': FOOTPRINT_SPINE, 'admin_id': COUNTY, 'path': 'x'}
        ],
    }
    rp = cl.write_receipt(out, receipt)
    assert rp.name == f'{COUNTY}_building-nsi-2022.consumed.json'
    loaded = cl.read_receipt(out)
    assert loaded['format'] == 1
    assert loaded['recipe_id'] == NSI
    cl.discard_receipt(out)
    assert cl.read_receipt(out) is None
    cl.discard_receipt(out)  # tolerant of absence


def test_receipt_paths_are_relative_forward_slash(data_root):
    path = data_root / 'data' / 'core' / 'US' / 'file.parquet'
    rel = cl._relative_posix(path)
    assert '\\' not in rel
    assert not rel.startswith(str(data_root))
    assert rel == 'data/core/US/file.parquet'


# Completeness


def test_is_output_complete_missing_and_valid(data_root):
    assert not cl.is_output_complete(NSI, COUNTY)
    _write_parquet(_nsi_path())
    assert cl.is_output_complete(NSI, COUNTY)


def test_is_output_complete_rejects_bad_registry_dtype(data_root):
    # improvement_value is a registry float; a string column must fail
    df = pd.DataFrame({'improvement_value': ['not', 'numeric']})
    _write_parquet(_nsi_path(), df)
    assert not cl.is_output_complete(NSI, COUNTY)


def test_is_output_complete_rejects_truncated_parquet(data_root):
    path = _nsi_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'PAR1 this is not a parquet footer')
    assert not cl.is_output_complete(NSI, COUNTY)


# Receipt-justified skip (design section 4.3)


def _receipt_for_nsi(consumers):
    verified = []
    for recipe_id in consumers:
        path = get_output_path(get_recipe_by_id(recipe_id), admin_id=COUNTY)
        verified.append(
            {
                'recipe_id': recipe_id,
                'admin_id': COUNTY,
                'path': cl._relative_posix(path),
            }
        )
    return {
        'recipe_id': NSI,
        'admin_id': COUNTY,
        'consumers_verified': verified,
    }


def test_receipt_skip_requires_all_recorded_consumers(data_root):
    fp_path, pc_path = _spine_paths()
    _write_parquet(fp_path)
    _write_parquet(pc_path)
    cl.write_receipt(_nsi_path(), _receipt_for_nsi([FOOTPRINT_SPINE, PARCEL_SPINE]))
    assert cl.receipt_justifies_skip(NSI, COUNTY)

    # A recorded consumer disappearing voids the skip
    fp_path.unlink()
    assert not cl.receipt_justifies_skip(NSI, COUNTY)


def test_receipt_skip_consumer_cascade(data_root):
    # A consumer replaced by its own receipt still counts (conceptual
    # existence), so upstream receipts stay valid
    fp_path, pc_path = _spine_paths()
    _write_parquet(pc_path)
    cl.write_receipt(
        fp_path,
        {
            'recipe_id': FOOTPRINT_SPINE,
            'admin_id': COUNTY,
            'consumers_verified': [
                {'recipe_id': 'US_footprint-cheer-2026', 'path': 'gone'}
            ],
        },
    )
    cl.write_receipt(_nsi_path(), _receipt_for_nsi([FOOTPRINT_SPINE, PARCEL_SPINE]))
    assert cl.receipt_justifies_skip(NSI, COUNTY)


def test_receipt_skip_voided_by_unrecorded_consumer(data_root):
    # Receipt recording only ONE of the two tree consumers must not skip
    fp_path, pc_path = _spine_paths()
    _write_parquet(fp_path)
    _write_parquet(pc_path)
    cl.write_receipt(_nsi_path(), _receipt_for_nsi([FOOTPRINT_SPINE]))
    assert not cl.receipt_justifies_skip(NSI, COUNTY)


def test_receipt_skip_voided_when_orchestrated(data_root, monkeypatch):
    fp_path, pc_path = _spine_paths()
    _write_parquet(fp_path)
    _write_parquet(pc_path)
    cl.write_receipt(_nsi_path(), _receipt_for_nsi([FOOTPRINT_SPINE, PARCEL_SPINE]))
    monkeypatch.setenv('OPENPLACES_ORCHESTRATED', '1')
    assert not cl.receipt_justifies_skip(NSI, COUNTY)


def test_receipt_skip_disabled_by_config(data_root, monkeypatch):
    fp_path, pc_path = _spine_paths()
    _write_parquet(fp_path)
    _write_parquet(pc_path)
    cl.write_receipt(_nsi_path(), _receipt_for_nsi([FOOTPRINT_SPINE, PARCEL_SPINE]))
    retention = {'cleanup': {'honor_receipts': False}, 'recipes': {}}
    monkeypatch.setitem(cfg.config, 'retention', retention)
    assert not cl.receipt_justifies_skip(NSI, COUNTY)


# Locks


def test_data_lock_exclusive(data_root):
    with cl.DataLock(COUNTY, timeout_s=0.1):
        with pytest.raises(TimeoutError):
            with cl.DataLock(COUNTY, timeout_s=0.1):
                pass
    # Released: can reacquire
    with cl.DataLock(COUNTY, timeout_s=0.1):
        pass


# cleanup()


def test_cleanup_dry_run_defaults_and_blocked(data_root):
    _write_parquet(_nsi_path())
    report = cl.cleanup('US_footprint-cheer-2026', admin_ids=[COUNTY], verbose=False)
    nsi_rows = report[report['recipe_id'] == NSI]
    assert not nsi_rows.empty
    # Spines missing: NSI is blocked, not deletable
    assert (nsi_rows['action'] == 'blocked').all()
    assert _nsi_path().exists()


def test_cleanup_deletes_consumed_input(data_root):
    _write_parquet(_nsi_path())
    fp_path, pc_path = _spine_paths()
    _write_parquet(fp_path)
    _write_parquet(pc_path)

    report = cl.cleanup('US_footprint-cheer-2026', admin_ids=[COUNTY], verbose=False)
    nsi_rows = report[report['recipe_id'] == NSI]
    assert (nsi_rows['action'] == 'would_delete').all()
    assert _nsi_path().exists()  # dry run never deletes

    report = cl.cleanup(
        'US_footprint-cheer-2026',
        admin_ids=[COUNTY],
        dry_run=False,
        verbose=False,
    )
    nsi_rows = report[report['recipe_id'] == NSI]
    assert (nsi_rows['action'] == 'deleted').all()
    assert not _nsi_path().exists()
    receipt = cl.read_receipt(_nsi_path())
    assert receipt is not None
    recorded = {c['recipe_id'] for c in receipt['consumers_verified']}
    assert {FOOTPRINT_SPINE, PARCEL_SPINE} <= recorded


def test_cleanup_stage_filter(data_root):
    _write_parquet(_nsi_path())
    fp_path, pc_path = _spine_paths()
    _write_parquet(fp_path)
    _write_parquet(pc_path)
    report = cl.cleanup(
        'US_footprint-cheer-2026',
        admin_ids=[COUNTY],
        stages=('harmonize',),
        verbose=False,
    )
    assert (report['recipe_id'] != NSI).all()


def _image_cache_frame(rows):
    return pd.DataFrame(
        rows,
        columns=['admin_id', 'source', 'version', 'n_files', 'size_mb', 'path'],
    )


def test_delete_image_caches_dry_run_filters_without_deleting(
    monkeypatch, tmp_path, capsys
):
    selected = tmp_path / 'selected'
    other_version = tmp_path / 'other-version'
    other_admin = tmp_path / 'other-admin'
    for path in (selected, other_version, other_admin):
        path.mkdir()

    caches = _image_cache_frame(
        [
            ['US-NC-BS-SH', 'googlesatellite', 'z20', 10, 12.5, selected],
            ['US-NC-BS-SM', 'googlesatellite', 'z19', 5, 6.0, other_version],
            ['US-MA-MI', 'googlesatellite', 'z20', 3, 2.0, other_admin],
        ]
    )
    monkeypatch.setattr(diagnostics, 'list_image_caches', lambda: caches)

    result = opio.delete_image_caches(
        'US-NC-BS', source='googlesatellite', version='z20'
    )

    assert result['admin_id'].tolist() == ['US-NC-BS-SH']
    assert selected.exists()
    output = capsys.readouterr().out
    assert 'Dry run: would delete 1 image cache(s), 12.5 MB total.' in output


def test_delete_image_caches_removes_matching_directories(monkeypatch, tmp_path):
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    for path in (first, second):
        path.mkdir()
        (path / 'image.jpg').write_bytes(b'image')

    caches = _image_cache_frame(
        [
            ['US-NC-BS-SH', 'googlesatellite', 'z20', 1, 1.0, first],
            ['US-NC-BS-SM', 'googlesatellite', 'z20', 1, 2.0, second],
        ]
    )
    monkeypatch.setattr(diagnostics, 'list_image_caches', lambda: caches)

    result = opio.delete_image_caches('US-NC-BS', dry_run=False)

    assert len(result) == 2
    assert not first.exists()
    assert not second.exists()


def test_delete_image_caches_handles_empty_inventory(monkeypatch, capsys):
    caches = _image_cache_frame([])
    monkeypatch.setattr(diagnostics, 'list_image_caches', lambda: caches)

    result = opio.delete_image_caches(dry_run=False)

    assert result.empty
    assert capsys.readouterr().out == 'No image caches found.\n'
    assert not hasattr(diagnostics, 'delete_image_caches')


# compact()


def test_compact_classification(data_root):
    nsi_path = _write_parquet(_nsi_path())
    heap_file = cfg.get_dir('heap') / 'leftover.tmp'
    heap_file.parent.mkdir(parents=True, exist_ok=True)
    heap_file.write_text('x')
    orphan = cfg.get_dir('cache') / 'US' / 'NC' / '_all' / 'US-NC_bogus-x-1.parquet'
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b'junk')
    old = time.time() - 30 * 86400
    os.utime(orphan, (old, old))

    report = cl.compact()
    by_path = report.set_index('path')

    assert by_path.loc[cl._relative_posix(nsi_path), 'class'] in (
        'intermediate/needed',
        'intermediate/consumed',
    )
    assert by_path.loc[cl._relative_posix(heap_file), 'class'] == 'heap'
    assert by_path.loc[cl._relative_posix(orphan), 'class'] == 'orphan'


def test_compact_recent_files_are_not_orphans(data_root):
    fresh = cfg.get_dir('cache') / 'US' / 'NC' / '_all' / 'US-NC_bogus-x-1.parquet'
    fresh.parent.mkdir(parents=True, exist_ok=True)
    fresh.write_bytes(b'junk')
    report = cl.compact()
    row = report[report['path'] == cl._relative_posix(fresh)]
    assert (row['class'] == 'recent').all()


def test_compact_requires_two_acts_to_delete(data_root):
    orphan = cfg.get_dir('cache') / 'US' / 'NC' / '_all' / 'US-NC_bogus-x-1.parquet'
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b'junk')
    old = time.time() - 30 * 86400
    os.utime(orphan, (old, old))

    # delete selection alone (dry_run defaults True) does not delete
    report = cl.compact(delete=('orphans', 'heap'))
    assert orphan.exists()
    row = report[report['path'] == cl._relative_posix(orphan)]
    assert (row['action'] == 'would_delete').all()

    # dry_run=False alone (no delete selection) does not delete either
    cl.compact(dry_run=False)
    assert orphan.exists()

    cl.compact(delete=('orphans',), dry_run=False)
    assert not orphan.exists()


def test_compact_orphan_gc_blocked_in_shared_buckets(data_root):
    orphan = cfg.get_dir('external') / 'US' / '_all' / 'US_bogus-y-2.parquet'
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b'junk')
    old = time.time() - 30 * 86400
    os.utime(orphan, (old, old))

    cl.compact(delete=('orphans',), dry_run=False)
    assert orphan.exists()

    cl.compact(delete=('orphans',), dry_run=False, include_shared=True)
    assert not orphan.exists()


def test_compact_exclude_patterns(data_root, monkeypatch):
    retention = {
        'cleanup': {'exclude_patterns': ['**/backups/**']},
        'recipes': {},
    }
    monkeypatch.setitem(cfg.config, 'retention', retention)
    keeper = cfg.get_dir('cache') / 'backups' / 'US-NC_bogus-x-1.parquet'
    keeper.parent.mkdir(parents=True, exist_ok=True)
    keeper.write_bytes(b'junk')
    old = time.time() - 30 * 86400
    os.utime(keeper, (old, old))

    report = cl.compact(delete=('orphans',), dry_run=False)
    assert keeper.exists()
    assert cl._relative_posix(keeper) not in set(report['path'])


def test_compact_prunes_stale_receipts(data_root):
    out = _nsi_path()
    cl.write_receipt(
        out,
        {
            'recipe_id': NSI,
            'admin_id': COUNTY,
            'consumers_verified': [
                {'recipe_id': FOOTPRINT_SPINE, 'path': 'data/gone.parquet'}
            ],
        },
    )
    report = cl.compact(delete=('heap',), dry_run=False)
    stale = report[report['class'] == 'receipt/stale']
    assert len(stale) == 1
    assert cl.read_receipt(out) is None


def test_compact_destructive_aborts_on_recipe_parse_error(data_root, monkeypatch):
    broken = cl._DependencyIndex.__new__(cl._DependencyIndex)
    broken.errors = [('US_broken-recipe-1', ValueError('bad yaml'))]
    broken.recipes = {}
    broken._literal = {}
    broken._auto_consumers = []
    broken._auto_cache = {}
    monkeypatch.setattr(cl, '_dependency_index', lambda: broken)
    with pytest.raises(RuntimeError, match='failed to load'):
        cl.compact(delete=('orphans',), dry_run=False)


def test_compact_min_recipe_guard(data_root, monkeypatch):
    tiny = cl._DependencyIndex(recipe_ids=[NSI])
    monkeypatch.setattr(cl, '_dependency_index', lambda: tiny)
    orphan = cfg.get_dir('cache') / 'US' / 'NC' / '_all' / 'US-NC_bogus-x-1.parquet'
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b'junk')
    old = time.time() - 30 * 86400
    os.utime(orphan, (old, old))

    report = cl.compact(delete=('orphans',), dry_run=False)
    assert orphan.exists()
    row = report[report['path'] == cl._relative_posix(orphan)]
    assert (row['action'] == 'blocked').all()


def test_receipt_written_atomically(data_root):
    out = _nsi_path()
    rp = cl.write_receipt(out, {'recipe_id': NSI, 'consumers_verified': []})
    # No temp files left behind
    leftovers = [p for p in rp.parent.iterdir() if '.tmp' in p.name]
    assert leftovers == []
    assert json.loads(rp.read_text(encoding='utf-8'))['recipe_id'] == NSI
