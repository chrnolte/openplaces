"""Versions of a delivered bundle: manifest, archive, index, withdrawal.

A bundle keeps its fixed path; a re-export first moves the bundle it
replaces to the archive under `versions/<version>/`, and refuses to
write when that fails. `versions/index.json` beside the bundle lists
every version (maintainer's rulings, 2026-11-15 to 2026-11-17). All
tables here are fabricated.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from openplaces.config import cfg
from openplaces.core.schema import AdminId, Entity
from openplaces.io import readers, save_parquet
from openplaces.io.delivery import (
    ArchiveError,
    export_delivery,
    versions,
    withdraw_version,
)
from openplaces.io.delivery.backfill import main, plan_backfill, write_plan
from openplaces.recipe import get_output_path

COUNTIES = ['US-NC-AL', 'US-NC-BB']


def _recipe(**delivery):
    block = {'admin_level': 2, **delivery}
    if 'regions' not in block:
        block.setdefault('admin_ids', COUNTIES)
    return {
        'recipe_id': 'US_transaction-test-2026',
        'admin_id': AdminId('US'),
        'stage': 'curate',
        'entity': Entity('transaction', 'test', '2026'),
        'process_by': {'admin_level': 3},
        'save_to': {'data_dir': 'core'},
        'share': {
            'columns': ['admin3_id', 'price'],
            'geometry': False,
            'delivery': block,
        },
    }


def _curate(price=100_000.0):
    for n, county in enumerate(COUNTIES):
        frame = pd.DataFrame(
            {
                'admin3_id': [county, county],
                'price': [price + n, price + n + 1],
                'price_source': ['recorder', 'recorder'],
            },
            index=pd.Index([f'{county}-a', f'{county}-b'], name='transaction_id'),
        )
        save_parquet(frame, get_output_path(_recipe(), county))


@pytest.fixture
def curated(mock_data_root):
    _curate()
    return mock_data_root


@pytest.fixture
def registry(monkeypatch):
    """A fabricated region registry with one two-county region."""
    frame = pd.DataFrame(
        [('test-pair', 'test-pair', 'US-NC', c) for c in COUNTIES],
        columns=['region_id', 'name', 'region_admin_id', 'admin_id'],
    )

    def fake(region_id=None):
        if region_id is None:
            return frame
        found = frame[frame['region_id'] == region_id]
        if found.empty:
            raise KeyError(f'Unknown region {region_id!r}')
        return found

    monkeypatch.setattr(readers, 'get_regions', fake)
    return frame


def _on_day(monkeypatch, day):
    """Make the next export's build time fall on *day*."""
    original = versions.timestamp

    def fixed(seconds=None):
        if seconds is None:
            return f'{day}T10:00:00-05:00'
        return original(seconds)

    monkeypatch.setattr(versions, 'timestamp', fixed)


def _export(recipe=None, **kwargs):
    return export_delivery(recipe or _recipe(), allow_stale=True, **kwargs)


def _index(paths):
    return versions.read_index(versions.versions_dir(paths['canonical']))


def _archive_root():
    return Path(cfg.get_dir('data_root')) / 'archive'


def _bytes(paths):
    return {role: path.read_bytes() for role, path in paths.items()}


def test_the_first_export_writes_a_manifest_and_an_index(curated, monkeypatch):
    _on_day(monkeypatch, '2026-11-16')
    paths = _export()

    manifest = versions.read_manifest(paths['canonical'].parent)
    assert manifest['version'] == '2026-11-16'
    assert manifest['recipes'] == ['US_transaction-test-2026']
    assert manifest['admin_id'] == 'US-NC'
    assert manifest['audience'] == 'public'
    assert sorted(f['name'] for f in manifest['files']) == sorted(
        p.name for p in paths.values()
    )
    canonical = next(f for f in manifest['files'] if f['name'].endswith('6.parquet'))
    assert canonical['rows'] == 4

    index = _index(paths)
    assert [(r['version'], r['status'], r['location']) for r in index['versions']] == [
        ('2026-11-16', 'current', 'share')
    ]
    assert index['bundle'] == 'US-NC_transaction-test-2026'
    markdown = versions.versions_dir(paths['canonical']) / versions.VERSIONS_MD
    assert '| 2026-11-16 | current | share |' in markdown.read_text(encoding='utf-8')


def test_the_current_files_match_their_manifest(curated):
    paths = _export()

    manifest = versions.read_manifest(paths['canonical'].parent)
    for entry in manifest['files']:
        path = paths['canonical'].parent / entry['name']
        assert versions.sha256_of(path) == entry['sha256']
        assert path.stat().st_size == entry['bytes']


def test_the_manifest_lists_files_not_values(curated):
    """It is written into the share folder, and carries no table value."""
    paths = _export()
    text = (paths['canonical'].parent / versions.MANIFEST_NAME).read_text()

    assert 'recorder' not in text
    assert 'US-NC-AL-a' not in text
    assert set(json.loads(text)) == set(versions.MANIFEST_FIELDS)


def test_a_second_export_archives_the_first(curated, monkeypatch):
    _on_day(monkeypatch, '2026-11-16')
    paths = _export()
    first = _bytes(paths)
    _curate(price=200_000.0)
    _on_day(monkeypatch, '2026-11-17')
    _export()

    index = _index(paths)
    rows = [(r['version'], r['status'], r['location']) for r in index['versions']]
    assert rows == [
        ('2026-11-16', 'superseded', 'archive'),
        ('2026-11-17', 'current', 'share'),
    ]
    share = Path(cfg.get_dir('share'))
    relative = paths['canonical'].parent.relative_to(share)
    target = _archive_root() / 'share' / relative / 'versions' / '2026-11-16'
    assert (
        index['versions'][0]['path'] == target.relative_to(_archive_root()).as_posix()
    )
    for role, path in paths.items():
        assert (target / path.name).read_bytes() == first[role]
    assert paths['canonical'].read_bytes() != first['canonical']
    assert versions.read_manifest(target)['version'] == '2026-11-16'
    assert versions.read_manifest(paths['canonical'].parent)['version'] == '2026-11-17'


def test_a_second_export_on_one_day_gets_a_letter(curated, monkeypatch):
    _on_day(monkeypatch, '2026-11-17')
    paths = _export()
    _export()
    _export()

    assert [r['version'] for r in _index(paths)['versions']] == [
        '2026-11-17',
        '2026-11-17b',
        '2026-11-17c',
    ]
    archived = _archive_root().rglob(versions.MANIFEST_NAME)
    assert sorted(p.parent.name for p in archived) == ['2026-11-17', '2026-11-17b']


def test_a_failed_archive_refuses_and_leaves_the_bundle(curated, monkeypatch):
    paths = _export()
    before = _bytes(paths)
    manifest = versions.read_manifest(paths['canonical'].parent)
    index = _index(paths)
    calls = []
    original = versions._move

    def flaky(source, target):
        calls.append(source)
        if len(calls) == 2:
            raise OSError('the file is open in another program')
        original(source, target)

    monkeypatch.setattr(versions, '_move', flaky)
    _curate(price=300_000.0)
    with pytest.raises(ArchiveError, match='left in place'):
        _export()

    assert _bytes(paths) == before
    assert versions.read_manifest(paths['canonical'].parent) == manifest
    assert _index(paths) == index
    folder = paths['canonical'].parent
    assert not list(folder.glob('*.partial'))
    assert not list(_archive_root().rglob('*.parquet'))


def test_without_an_archive_root_a_reexport_is_refused(curated):
    paths = _export()
    before = _bytes(paths)
    _archive_root().rmdir()

    with pytest.raises(ArchiveError, match='no archive root'):
        _export()
    assert _bytes(paths) == before

    _export(replace_unarchived=True)
    rows = _index(paths)['versions']
    assert rows[0]['location'] is None
    assert 'without an archive copy' in rows[0]['note']
    assert rows[1]['status'] == 'current'


def test_a_configured_archive_root_wins(curated, monkeypatch, tmp_path):
    elsewhere = tmp_path / 'elsewhere'
    directories = dict(cfg.config['directories'], archive=elsewhere)
    monkeypatch.setitem(cfg.config, 'directories', directories)
    paths = _export()
    _export()

    row = _index(paths)['versions'][0]
    assert (elsewhere / row['path'] / paths['canonical'].name).is_file()


def test_a_team_bundle_keeps_its_own_versions(curated, registry, monkeypatch):
    recipe = _recipe(regions=['test-pair'], team_regions=['test-pair'])
    _on_day(monkeypatch, '2026-11-16')
    public = _export(recipe, region='test-pair')
    team = _export(recipe, region='test-pair.team')
    _on_day(monkeypatch, '2026-11-17')
    _export(recipe, region='test-pair.team')

    assert team['canonical'].parent == public['canonical'].parent / 'team'
    assert [r['version'] for r in _index(public)['versions']] == ['2026-11-16']
    team_index = _index(team)
    assert team_index['region'] == 'test-pair.team'
    assert [(r['version'], r['location']) for r in team_index['versions']] == [
        ('2026-11-16', 'archive'),
        ('2026-11-17', 'share'),
    ]
    archived = _archive_root() / team_index['versions'][0]['path']
    assert archived.parent.parent.name == 'team'
    manifest = versions.read_manifest(team['canonical'].parent)
    assert manifest['audience'] == 'team'
    assert manifest['region'] == 'test-pair.team'


def test_a_withdrawn_version_stays_listed_and_withdrawn(curated, monkeypatch):
    _on_day(monkeypatch, '2026-11-16')
    paths = _export()
    row = withdraw_version(_recipe(), '2026-11-16', note='a later fix withholds')
    assert row['status'] == 'withdrawn'

    _on_day(monkeypatch, '2026-11-17')
    _export()
    rows = {r['version']: r for r in _index(paths)['versions']}
    assert rows['2026-11-16']['status'] == 'withdrawn'
    assert rows['2026-11-16']['location'] == 'archive'
    assert 'a later fix withholds' in rows['2026-11-16']['note']
    markdown = (versions.versions_dir(paths['canonical']) / 'VERSIONS.md').read_text()
    assert '| 2026-11-16 | withdrawn | archive |' in markdown

    with pytest.raises(ValueError, match='cannot be set by hand'):
        versions.set_status(
            versions.versions_dir(paths['canonical']), '2026-11-16', 'current'
        )
    with pytest.raises(ValueError, match='no version'):
        withdraw_version(_recipe(), '2020-01-01')


def test_a_cache_export_writes_a_manifest_but_no_versions(curated):
    paths = _export(output_dir='cache')
    _export(output_dir='cache')

    assert versions.read_manifest(paths['canonical'].parent)['version'].endswith('b')
    assert not versions.versions_dir(paths['canonical']).exists()
    assert not list(_archive_root().rglob('*'))


def test_the_version_names(monkeypatch):
    day = versions.parse_version('2026-11-17')[0]
    assert versions.next_version(day, []) == '2026-11-17'
    assert versions.next_version(day, ['2026-11-17']) == '2026-11-17b'
    assert versions.next_version(day, ['2026-11-17', '2026-11-17b']) == '2026-11-17c'
    assert sorted(
        ['2026-11-17b', '2026-11-18', '2026-11-17'], key=versions.version_key
    ) == [
        '2026-11-17',
        '2026-11-17b',
        '2026-11-18',
    ]
    with pytest.raises(ValueError):
        versions.parse_version('0.7')


def _fake_bundle(folder: Path, stem: str, text: str = 'x') -> None:
    folder.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({'price': [1.0, 2.0]}, index=pd.Index(['a', 'b'], name='id'))
    frame.to_parquet(folder / f'{stem}.parquet')
    frame.to_parquet(folder / f'{stem}_evidence.parquet')
    (folder / f'{stem}_LICENSE.txt').write_text(
        f"Cells read 'building-imputation-v0.8+open+imputed'. {text}", encoding='utf-8'
    )
    (folder / f'{stem}_map.qgz').write_text('a sidecar', encoding='utf-8')


def _fake_tree(root: Path) -> tuple[Path, Path]:
    share = root / 'data' / 'share'
    archive = root / 'archive'
    stem = 'US-NC_footprint-test-2026'
    region = share / 'US/NC/_all/footprint/test/2026/test-pair.imputed'
    _fake_bundle(region, stem)
    _fake_bundle(region / 'team', stem)
    mirror = archive / 'share/US/NC/_all/footprint/test/2026/test-pair.imputed'
    _fake_bundle(mirror / 'v0.6', stem, 'old')
    _fake_bundle(mirror / 'v0.6' / 'team', stem, 'old team')
    _fake_bundle(mirror / 'pre-2026-11-03', stem, 'older')
    # A flat copy from before region folders: no bundle above it.
    _fake_bundle(archive / 'share/US/NC/_all/footprint/test/2026', stem, 'flat')
    # Not a bundle.
    (mirror / 'map-2026-10-25').mkdir(parents=True)
    (mirror / 'map-2026-10-25' / f'{stem}_map.qgz').write_text('m', encoding='utf-8')
    return share, archive


def test_the_backfill_enters_share_and_archive_copies(mock_data_root):
    share, archive = _fake_tree(mock_data_root / 'tree')
    plan = plan_backfill(share, archive)

    assert len(plan.indexes) == 2
    by_folder = {f.parent.name: index for f, (index, _) in plan.indexes.items()}
    public = by_folder['test-pair.imputed']['versions']
    assert sorted((r['location'], r['note']) for r in public) == [
        ('archive', 'pre-2026-11-03'),
        ('archive', 'v0.6'),
        ('share', 'v0.8'),
    ]
    assert [r['status'] for r in public if r['location'] == 'share'] == ['current']
    team = by_folder['team']
    assert sorted(r['note'] for r in team['versions']) == ['v0.6', 'v0.8']
    assert team['region'] == 'test-pair.imputed.team'
    # All built today here, so the letters follow build order.
    assert len({r['version'] for r in public}) == 3
    share_row = next(r for r in public if r['location'] == 'share')
    assert all(f['sha256'] for f in share_row['files'])
    assert sorted(f['name'] for f in share_row['files']) == [
        'US-NC_footprint-test-2026.parquet',
        'US-NC_footprint-test-2026_LICENSE.txt',
        'US-NC_footprint-test-2026_evidence.parquet',
    ]
    # An archived copy is described without being opened.
    archived = next(r for r in public if r['location'] == 'archive')
    assert all(f['sha256'] is None and f['rows'] is None for f in archived['files'])
    assert any('no bundle of this stem' in u for u in plan.unclassified)
    assert any('map-2026-10-25' in u for u in plan.unclassified)


def test_the_backfill_dry_run_writes_nothing(mock_data_root, capsys, monkeypatch):
    share, archive = _fake_tree(mock_data_root / 'tree')
    directories = dict(cfg.config['directories'], share=share, archive=archive)
    monkeypatch.setitem(cfg.config, 'directories', directories)
    before = sorted(p for p in (mock_data_root / 'tree').rglob('*'))

    assert main(['--dry-run']) == 0
    assert sorted(p for p in (mock_data_root / 'tree').rglob('*')) == before
    out = capsys.readouterr().out
    assert 'Dry run: nothing written.' in out
    assert '2 index(es); rows: 2 current in share, 3 superseded in the archive' in out


def test_the_backfill_writes_once_and_a_reexport_archives_the_entered_bundle(
    curated, monkeypatch
):
    _on_day(monkeypatch, '2026-11-16')
    paths = _export()
    # A bundle exported before manifests: no manifest, no index.
    folder = paths['canonical'].parent
    (folder / versions.MANIFEST_NAME).chmod(0o644)
    (folder / versions.MANIFEST_NAME).unlink()
    for name in (versions.INDEX_NAME, versions.VERSIONS_MD):
        (versions.versions_dir(paths['canonical']) / name).unlink()

    share = Path(cfg.get_dir('share'))
    assert write_plan(plan_backfill(share, _archive_root())) == 1
    assert write_plan(plan_backfill(share, _archive_root())) == 0
    entered = _index(paths)['versions']
    assert [(r['status'], r['location']) for r in entered] == [('current', 'share')]

    # Later than any file time the backfill can have read.
    _on_day(monkeypatch, '2030-01-01')
    _export()
    rows = _index(paths)['versions']
    assert [(r['version'], r['location']) for r in rows] == [
        (entered[0]['version'], 'archive'),
        ('2030-01-01', 'share'),
    ]
    target = _archive_root() / rows[0]['path']
    assert (target / paths['canonical'].name).is_file()


def _state(paths):
    """Everything a failed export must leave as it found it."""
    vdir = versions.versions_dir(paths['canonical'])
    folder = paths['canonical'].parent
    return {
        'bundle': _bytes(paths),
        'manifest': (folder / versions.MANIFEST_NAME).read_bytes(),
        'index': (vdir / versions.INDEX_NAME).read_bytes(),
        'markdown': (vdir / versions.VERSIONS_MD).read_bytes(),
        'archived': sorted(p for p in _archive_root().rglob('*') if p.is_file()),
        'partials': sorted(folder.rglob('*.partial')),
    }


class _Interrupted(BaseException):
    """Stands in for a full disk or Ctrl+C at an awkward moment."""


@pytest.mark.parametrize(
    'failing, error',
    [
        # The second bundle file, after the canonical one moved in.
        ('_evidence.parquet.partial', OSError),
        # The archived copy's manifest, inside the archive move.
        ('manifest.json.partial', _Interrupted),
        # After the share folder's new manifest went in.
        ('VERSIONS.md.partial', _Interrupted),
        # The very last rename of the sequence.
        ('index.json.partial', OSError),
    ],
)
def test_a_failure_after_archiving_restores_everything(
    curated, monkeypatch, failing, error
):
    _on_day(monkeypatch, '2026-11-16')
    paths = _export()
    before = _state(paths)
    replace = versions.os.replace

    def flaky(source, target):
        if str(source).endswith(failing):
            raise error('the disk is full')
        return replace(source, target)

    _curate(price=400_000.0)
    _on_day(monkeypatch, '2026-11-17')
    monkeypatch.setattr(versions.os, 'replace', flaky)
    with pytest.raises(error):
        _export()
    monkeypatch.setattr(versions.os, 'replace', replace)

    after = _state(paths)
    assert after == before
    assert after['archived'] == []
    rows = _index(paths)['versions']
    assert [(r['version'], r['status'], r['location']) for r in rows] == [
        ('2026-11-16', 'current', 'share')
    ]

    # The rerun archives the old bundle, not a half-written one, under
    # its own name, and takes the day's name with no letter.
    _export()
    rows = _index(paths)['versions']
    assert [(r['version'], r['status'], r['location']) for r in rows] == [
        ('2026-11-16', 'superseded', 'archive'),
        ('2026-11-17', 'current', 'share'),
    ]
    archived = _archive_root() / rows[0]['path']
    for role, path in paths.items():
        assert (archived / path.name).read_bytes() == before['bundle'][role]


def test_a_failure_before_archiving_touches_nothing(curated, monkeypatch):
    """Pooling, a refused column, anything before the swap."""
    paths = _export()
    before = _state(paths)
    recipe = _recipe()
    recipe['share']['columns'] = [*recipe['share']['columns'], 'owner_name']
    with pytest.raises(Exception, match='owner_name'):
        _export(recipe)
    assert _state(paths) == before


def test_a_version_restored_by_snakemake_is_archived_once(curated, monkeypatch):
    """Killed after archiving, then restored from Snakemake's backup.

    The old version then sits both in the share folder and in the
    archive, and the index still calls it current. The next export
    finds the archived copy identical and keeps it.
    """
    _on_day(monkeypatch, '2026-11-16')
    paths = _export()
    share = Path(cfg.get_dir('share'))
    target = (
        _archive_root()
        / 'share'
        / paths['canonical'].parent.relative_to(share)
        / 'versions'
        / '2026-11-16'
    )
    target.mkdir(parents=True)
    for path in [*paths.values(), paths['canonical'].parent / 'manifest.json']:
        (target / path.name).write_bytes(path.read_bytes())

    _on_day(monkeypatch, '2026-11-17')
    _export()
    rows = _index(paths)['versions']
    assert [(r['version'], r['status'], r['location']) for r in rows] == [
        ('2026-11-16', 'superseded', 'archive'),
        ('2026-11-17', 'current', 'share'),
    ]
    assert len(list(target.iterdir())) == len(paths) + 1


def test_a_current_version_that_vanished_is_not_listed_twice(curated, monkeypatch):
    """Killed between moving the old bundle out and the new one in."""
    _on_day(monkeypatch, '2026-11-16')
    paths = _export()
    share = Path(cfg.get_dir('share'))
    folder = paths['canonical'].parent
    target = (
        _archive_root()
        / 'share'
        / folder.relative_to(share)
        / 'versions'
        / '2026-11-16'
    )
    target.mkdir(parents=True)
    for path in paths.values():
        path.chmod(0o644)
        path.replace(target / path.name)
    (target / 'manifest.json').write_bytes((folder / 'manifest.json').read_bytes())

    _on_day(monkeypatch, '2026-11-17')
    _export()
    rows = _index(paths)['versions']
    assert [(r['version'], r['status'], r['location']) for r in rows] == [
        ('2026-11-16', 'superseded', 'archive'),
        ('2026-11-17', 'current', 'share'),
    ]
