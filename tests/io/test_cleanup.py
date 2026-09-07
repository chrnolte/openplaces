"""Tests for io/cleanup.py on a synthetic data root."""

import json
import os
import time

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

import openplaces.diagnostics as diagnostics
import openplaces.io as opio
from openplaces.config import cfg
from openplaces.io import cleanup as cl
from openplaces.recipe import get_output_path, get_recipe_by_id

NSI = 'US_building-nsi-2026'
FOOTPRINT_SPINE = 'US_footprint-spine-2026'
PARCEL_SPINE = 'US_parcel-spine-2026'
FOOTPRINT_GEOSPINE = 'US_footprint-geospine-2026'
PARCEL_GEOSPINE = 'US_parcel-geospine-2026'
COUNTY = 'US-NC-BRU'
# Every recipe in the cheer tree that consumes NSI directly: under the
# geometry/attribute split, the geospine halves link it and the attribute
# halves attribute it, so a valid NSI receipt must record all four.
NSI_CONSUMERS = [FOOTPRINT_SPINE, PARCEL_SPINE, FOOTPRINT_GEOSPINE, PARCEL_GEOSPINE]


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
        get_output_path(get_recipe_by_id(FOOTPRINT_GEOSPINE), admin_id=admin),
        get_output_path(get_recipe_by_id(PARCEL_GEOSPINE), admin_id=admin),
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
    assert rp.name == f'{COUNTY}_building-nsi-2026.consumed.json'
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


def test_is_output_complete_rejects_bad_suffixed_registry_dtype(data_root):
    # improvement_value_parcel resolves to the improvement_value registry
    # float, so a string column must fail despite the provenance suffix
    df = pd.DataFrame({'improvement_value_parcel': ['not', 'numeric']})
    _write_parquet(_nsi_path(), df)
    assert not cl.is_output_complete(NSI, COUNTY)


def test_is_output_complete_requires_the_geometry_sidecar(data_root):
    """A write killed between the two files is not complete.

    save_parquet writes the attribute table first and the '_geo' sidecar
    second; the attribute half passes every check on its own, so a
    consumer killed in between counted as complete and its inputs were
    reclaimed with receipts.
    """
    path = _nsi_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = gpd.GeoDataFrame(
        {'value': [1.0, 2.0]},
        geometry=[Point(0, 0), Point(1, 1)],
        crs='EPSG:4326',
    )
    opio.save_parquet(frame, path)
    geo_path = path.with_name(path.stem + '_geo' + path.suffix)
    assert geo_path.exists()
    assert cl.is_output_complete(NSI, COUNTY)

    geo_path.unlink()
    assert not cl.is_output_complete(NSI, COUNTY)


def test_is_output_complete_rejects_truncated_geometry_sidecar(data_root):
    path = _nsi_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = gpd.GeoDataFrame({'value': [1.0]}, geometry=[Point(0, 0)], crs='EPSG:4326')
    opio.save_parquet(frame, path)
    geo_path = path.with_name(path.stem + '_geo' + path.suffix)
    geo_path.write_bytes(b'PAR1 truncated')
    assert not cl.is_output_complete(NSI, COUNTY)


def test_is_output_complete_rejects_truncated_parquet(data_root):
    path = _nsi_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'PAR1 this is not a parquet footer')
    assert not cl.is_output_complete(NSI, COUNTY)


# Coverage of a coarser consumer


STATE = 'US-NC'
OTHER_COUNTY = 'US-NC-CAB'


def _write_parquet_with_coverage(path, partitions):
    path.parent.mkdir(parents=True, exist_ok=True)
    opio.save_parquet(
        pd.DataFrame({'geo_id': ['a'], 'value': [1.0]}),
        path,
        file_metadata={'openplaces:partitions': json.dumps(list(partitions))},
    )
    return path


def _coarser_consumer_index(consumer_id, node_id):
    """Index with one consumer that saves a state-level aggregate."""
    recipe = dict(get_recipe_by_id(consumer_id))
    save_to = dict(recipe.get('save_to') or {})
    save_to['admin_level'] = 2
    recipe['save_to'] = save_to
    index = cl._DependencyIndex.__new__(cl._DependencyIndex)
    index.errors = []
    index.recipes = {consumer_id: recipe}
    index._literal = {node_id: {consumer_id}}
    index._auto_consumers = []
    index._auto_cache = {}
    return index


def test_coarser_consumer_must_have_consumed_the_county(data_root):
    """A state aggregate that has not read this county does not free it.

    The coverage requirement used to be ORed with a plain existence
    check, which re-ran the same test without it, so the requirement
    never bound and the county was deleted with a receipt that then made
    ingest skip regenerating it.
    """
    index = _coarser_consumer_index(FOOTPRINT_SPINE, NSI)
    consumer_path = get_output_path(index.recipes[FOOTPRINT_SPINE], admin_id=STATE)
    _write_parquet_with_coverage(consumer_path, [OTHER_COUNTY])
    deletable, blocked_by, _ = cl._consumers_complete(NSI, COUNTY, index)
    assert not deletable
    assert blocked_by == [FOOTPRINT_SPINE]

    _write_parquet_with_coverage(consumer_path, [OTHER_COUNTY, COUNTY])
    deletable, _, verified = cl._consumers_complete(NSI, COUNTY, index)
    assert deletable
    assert [c['recipe_id'] for c in verified] == [FOOTPRINT_SPINE]


def test_cascaded_receipt_must_record_the_county_it_consumed(data_root):
    """The receipt cascade carries the coverage requirement with it."""
    index = _coarser_consumer_index(FOOTPRINT_SPINE, NSI)
    consumer_path = get_output_path(index.recipes[FOOTPRINT_SPINE], admin_id=STATE)
    cl.write_receipt(
        consumer_path,
        {
            'recipe_id': FOOTPRINT_SPINE,
            'admin_id': STATE,
            'partitions': [OTHER_COUNTY],
            'consumers_verified': [],
        },
    )
    deletable, _, _ = cl._consumers_complete(NSI, COUNTY, index)
    assert not deletable

    cl.write_receipt(
        consumer_path,
        {
            'recipe_id': FOOTPRINT_SPINE,
            'admin_id': STATE,
            'partitions': [OTHER_COUNTY, COUNTY],
            'consumers_verified': [],
        },
    )
    deletable, _, _ = cl._consumers_complete(NSI, COUNTY, index)
    assert deletable


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
    fp_path = _spine_paths()[0]
    for spine_path in _spine_paths():
        _write_parquet(spine_path)
    cl.write_receipt(_nsi_path(), _receipt_for_nsi(NSI_CONSUMERS))
    assert cl.receipt_justifies_skip(NSI, COUNTY)

    # A recorded consumer disappearing voids the skip
    fp_path.unlink()
    assert not cl.receipt_justifies_skip(NSI, COUNTY)


def test_receipt_skip_consumer_cascade(data_root):
    # A consumer replaced by its own receipt still counts (conceptual
    # existence), so upstream receipts stay valid
    fp_path, pc_path, fp_geo_path, pc_geo_path = _spine_paths()
    _write_parquet(pc_path)
    _write_parquet(fp_geo_path)
    _write_parquet(pc_geo_path)
    cl.write_receipt(
        fp_path,
        {
            'recipe_id': FOOTPRINT_SPINE,
            'admin_id': COUNTY,
            'consumers_verified': [
                {'recipe_id': 'US_footprint-openplaces-2026', 'path': 'gone'}
            ],
        },
    )
    cl.write_receipt(_nsi_path(), _receipt_for_nsi(NSI_CONSUMERS))
    assert cl.receipt_justifies_skip(NSI, COUNTY)


def test_receipt_skip_voided_by_unrecorded_consumer(data_root):
    # Receipt recording only ONE of the two tree consumers must not skip
    for spine_path in _spine_paths():
        _write_parquet(spine_path)
    cl.write_receipt(_nsi_path(), _receipt_for_nsi([FOOTPRINT_SPINE]))
    assert not cl.receipt_justifies_skip(NSI, COUNTY)


def test_receipt_skip_voided_when_orchestrated(data_root, monkeypatch):
    for spine_path in _spine_paths():
        _write_parquet(spine_path)
    cl.write_receipt(_nsi_path(), _receipt_for_nsi(NSI_CONSUMERS))
    monkeypatch.setenv('OPENPLACES_ORCHESTRATED', '1')
    assert not cl.receipt_justifies_skip(NSI, COUNTY)


def test_receipt_skip_disabled_by_config(data_root, monkeypatch):
    for spine_path in _spine_paths():
        _write_parquet(spine_path)
    cl.write_receipt(_nsi_path(), _receipt_for_nsi(NSI_CONSUMERS))
    retention = {'cleanup': {'honor_receipts': False}, 'recipes': {}}
    monkeypatch.setitem(cfg.config, 'retention', retention)
    assert not cl.receipt_justifies_skip(NSI, COUNTY)


# Deletion


def test_delete_with_receipt_survives_a_truncated_footer(data_root):
    """A corrupt output is deleted, not raised on.

    The coverage footer was read without a guard, so an output truncated
    by a killed job raised ArrowInvalid mid-batch and, through the
    stages' cleanup='consumed' hook, crashed the caller after the unit's
    own output had been written.
    """
    out = _nsi_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b'PAR1 this is not a parquet footer')

    action, _ = cl._delete_output_with_receipt(out, NSI, COUNTY, [])

    assert action == 'deleted'
    assert not out.exists()
    assert cl.read_receipt(out)['partitions'] == []


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
    report = cl.cleanup(
        'US_footprint-openplaces-2026', admin_ids=[COUNTY], verbose=False
    )
    nsi_rows = report[report['recipe_id'] == NSI]
    assert not nsi_rows.empty
    # Spines missing: NSI is blocked, not deletable
    assert (nsi_rows['action'] == 'blocked').all()
    assert _nsi_path().exists()


def test_cleanup_deletes_consumed_input(data_root):
    _write_parquet(_nsi_path())
    for spine_path in _spine_paths():
        _write_parquet(spine_path)

    report = cl.cleanup(
        'US_footprint-openplaces-2026', admin_ids=[COUNTY], verbose=False
    )
    nsi_rows = report[report['recipe_id'] == NSI]
    assert (nsi_rows['action'] == 'would_delete').all()
    assert _nsi_path().exists()  # dry run never deletes

    report = cl.cleanup(
        'US_footprint-openplaces-2026',
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
    for spine_path in _spine_paths():
        _write_parquet(spine_path)
    report = cl.cleanup(
        'US_footprint-openplaces-2026',
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
            ['US-NC-BRU-SH', 'googlesatellite', 'z20', 10, 12.5, selected],
            ['US-NC-BRU-SM', 'googlesatellite', 'z19', 5, 6.0, other_version],
            ['US-MA-MI', 'googlesatellite', 'z20', 3, 2.0, other_admin],
        ]
    )
    monkeypatch.setattr(diagnostics, 'list_image_caches', lambda: caches)

    result = opio.delete_image_caches(
        'US-NC-BRU', source='googlesatellite', version='z20'
    )

    assert result['admin_id'].tolist() == ['US-NC-BRU-SH']
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
            ['US-NC-BRU-SH', 'googlesatellite', 'z20', 1, 1.0, first],
            ['US-NC-BRU-SM', 'googlesatellite', 'z20', 1, 2.0, second],
        ]
    )
    monkeypatch.setattr(diagnostics, 'list_image_caches', lambda: caches)

    result = opio.delete_image_caches('US-NC-BRU', dry_run=False)

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


# Matching a file to the recipe that writes it


MASSGIS = 'US-MA_parcel-massgis-2025'
MASSGIS_TOWN = 'US-MA-MI'
STORIES_ENRICH = 'US_footprint_built-n-stories-brails-2026'
VICTORIA = 'US-TX-VIC_property-victoriacad-2026'


def test_additional_layer_output_matches_its_host_recipe():
    """A secondary entity's output is not an orphan.

    Its token appears in no recipe ID, so indexing recipe IDs alone left
    the MassGIS property table matching nothing; orphan GC would unlink a
    table the recipe still produces and the harmonizer reads.
    """
    recipe_id, admin = cl._match_recipe_for_file(
        f'{MASSGIS_TOWN}_property-massgis-2025'
    )
    assert (recipe_id, admin) == (MASSGIS, MASSGIS_TOWN)


def test_additional_layer_output_survives_orphan_gc(data_root):
    layer_path = get_output_path(
        get_recipe_by_id(MASSGIS), admin_id=MASSGIS_TOWN, layer='property'
    )
    _write_parquet(layer_path)
    old = time.time() - 30 * 86400
    os.utime(layer_path, (old, old))

    report = cl.compact(delete=('orphans',), dry_run=False)
    row = report[report['path'] == cl._relative_posix(layer_path)]
    assert layer_path.exists()
    assert (row['class'] != 'orphan').all()
    assert (row['recipe_id'] == MASSGIS).all()


def test_enrich_evidence_matches_its_own_recipe():
    """Evidence is named after the spine, so it matched the spine.

    The trailing dataset is what identifies the enrich recipe; without
    it the evidence file was judged by the spine's retention class and
    the spine's consumer set, and `compact(recipes=[enrich_id])` listed
    nothing at all.
    """
    stem = f'{COUNTY}_footprint-spine-2026_built-n-stories-brails-2026'
    recipe_id, admin = cl._match_recipe_for_file(stem)
    assert (recipe_id, admin) == (STORIES_ENRICH, COUNTY)


def test_suffixed_recipe_wins_over_its_unsuffixed_sibling():
    recipe_id, _ = cl._match_recipe_for_file(
        'US-TX-VIC_property-victoriacad-2026_improvement-detail'
    )
    assert recipe_id == f'{VICTORIA}_improvement-detail'
    recipe_id, _ = cl._match_recipe_for_file(VICTORIA)
    assert recipe_id == VICTORIA


def test_geometry_sidecar_matches_its_own_output():
    recipe_id, _ = cl._match_recipe_for_file(f'{COUNTY}_footprint-spine-2026_geo')
    assert recipe_id == FOOTPRINT_SPINE


# compact()


def test_aggressive_keeps_explicit_retention(data_root):
    """Aggressive mode must not demote a recipe declaring its own retention.

    The geospine outputs (and their link sidecars) are exactly what
    `--reprocess attributes` reuses: an aggressive sweep deleting them
    would silently turn the next attribute-only rerun into a full
    geometry rerun. Their `save_to: retention: keep` wins over the
    aggressive core-bucket demotion; the plain spines (no explicit
    retention) are still demoted to until_consumed as before.
    """
    for spine_path in _spine_paths():
        _write_parquet(spine_path)
    # The curated consumer exists, so an until_consumed spine is deletable.
    curated = get_output_path(
        get_recipe_by_id('US_footprint-openplaces-2026'), admin_id=COUNTY
    )
    _write_parquet(curated)
    parcel_curated = get_output_path(
        get_recipe_by_id('US_parcel-openplaces-2026'), admin_id=COUNTY
    )
    _write_parquet(parcel_curated)

    report = cl.cleanup(
        'US_footprint-openplaces-2026',
        admin_ids=[COUNTY],
        aggressive=True,
        verbose=False,
    )
    geospine_rows = report[
        report['recipe_id'].isin([FOOTPRINT_GEOSPINE, PARCEL_GEOSPINE])
    ]
    assert not geospine_rows.empty
    assert (geospine_rows['class'] == 'keep').all()
    assert (geospine_rows['action'] == 'kept').all()
    for path in _spine_paths()[2:]:
        assert path.exists()


def test_aggressive_keeps_config_protected_recipe(data_root, monkeypatch):
    """A retention.recipes override survives the aggressive demotion.

    It is the documented way to protect a recipe whose YAML declares no
    retention of its own, so an aggressive sweep that reads only the
    YAML deletes exactly what the user pinned.
    """
    monkeypatch.setitem(
        cfg.config,
        'retention',
        {'cleanup': {}, 'recipes': {FOOTPRINT_SPINE: 'keep'}},
    )
    for spine_path in _spine_paths():
        _write_parquet(spine_path)
    for curated_id in ('US_footprint-openplaces-2026', 'US_parcel-openplaces-2026'):
        _write_parquet(get_output_path(get_recipe_by_id(curated_id), admin_id=COUNTY))

    report = cl.cleanup(
        'US_footprint-openplaces-2026',
        admin_ids=[COUNTY],
        aggressive=True,
        dry_run=False,
        verbose=False,
    )
    rows = report[report['recipe_id'] == FOOTPRINT_SPINE]
    assert not rows.empty
    assert (rows['class'] == 'keep').all()
    assert (rows['action'] == 'kept').all()
    assert _spine_paths()[0].exists()


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
