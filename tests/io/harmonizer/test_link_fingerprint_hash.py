"""The fingerprint hash fallback: a sync touch is not a data change.

Dropbox re-hydration bumped the mtime of every cache parquet on
2026-08-24 with byte-identical content, which read as region-wide
geospine staleness under raw-equality comparison. These tests pin the
replacement contract: an mtime move alone revalidates through the
stored content hash, anything else still fails closed.
"""

import pytest

from openplaces.config import cfg
from openplaces.io.harmonizer.links import (
    _fingerprints_match,
    _hash_file,
    _with_source_hashes,
)


@pytest.fixture
def source_file(tmp_path, monkeypatch):
    """A fake ingest input inside a temporary data root."""
    dirs = dict(cfg.config['directories'])
    dirs['data_root'] = tmp_path
    monkeypatch.setitem(cfg.config, 'directories', dirs)
    path = tmp_path / 'input.parquet'
    path.write_bytes(b'identical content')
    return path


def _fingerprint(path, mtime):
    return {
        'format': 2,
        'step_config': {'join': 'spatial_point'},
        'prior_geometry_steps': [],
        'sources': [
            {
                'path': path.name,
                'size': path.stat().st_size,
                'mtime': mtime,
            }
        ],
    }


def test_exact_equality_still_matches(source_file):
    fp = _fingerprint(source_file, 1000.0)
    assert _fingerprints_match(fp, fp)


def test_mtime_move_revalidates_through_hash(source_file):
    stored = _with_source_hashes(_fingerprint(source_file, 1000.0))
    fresh = _fingerprint(source_file, 2000.0)
    assert stored['sources'][0]['sha256'] == _hash_file(source_file)
    assert _fingerprints_match(stored, fresh)


def test_mtime_move_without_stored_hash_stays_stale(source_file):
    stored = _fingerprint(source_file, 1000.0)
    fresh = _fingerprint(source_file, 2000.0)
    assert not _fingerprints_match(stored, fresh)


def test_content_change_fails_despite_hash(source_file):
    stored = _with_source_hashes(_fingerprint(source_file, 1000.0))
    source_file.write_bytes(b'different content!')
    fresh = _fingerprint(source_file, 2000.0)
    assert not _fingerprints_match(stored, fresh)


def test_size_change_fails_without_hashing(source_file):
    stored = _with_source_hashes(_fingerprint(source_file, 1000.0))
    fresh = _fingerprint(source_file, 2000.0)
    fresh['sources'][0]['size'] += 1
    assert not _fingerprints_match(stored, fresh)


def test_step_config_change_fails(source_file):
    stored = _with_source_hashes(_fingerprint(source_file, 1000.0))
    fresh = _fingerprint(source_file, 1000.0)
    fresh['step_config'] = {'join': 'spatial_overlay'}
    assert not _fingerprints_match(stored, fresh)
