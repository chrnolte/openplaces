"""Extraction into a shared heap is atomic per entry.

Two ingest jobs of recipes that read one archive extracted it into the
same folder at the same time. One read a geodatabase the other was
still writing, another deleted it under a reader. Each test builds a
small fabricated archive (no real data) and checks one guarantee of
`unzip` and `delete_extracted`.
"""

import os
import socket
import threading
import time
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

import openplaces.io.archives as archives  # unzip looks its helpers up here
from openplaces.io.archives import (
    acquire_read_lease,
    delete_extracted,
    extracted_path_ready,
    release_read_lease,
    unzip,
)

_GDB_FILES = {
    'parcels.gdb/a00000001.gdbtable': b'T' * 4096,
    'parcels.gdb/a00000001.gdbtablx': b'X' * 512,
    'parcels.gdb/gdb': b'G' * 8,
}


def _archive(path, files=None, readme=b'fabricated'):
    files = dict(_GDB_FILES if files is None else files)
    files['readme.txt'] = readme
    with ZipFile(path, 'w', ZIP_DEFLATED) as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
    return path


def _internal(out_dir, prefix):
    return [p for p in out_dir.iterdir() if p.name.startswith(prefix)]


def _assert_complete(out_dir):
    for name, payload in _GDB_FILES.items():
        assert (out_dir / name).read_bytes() == payload
    assert (out_dir / 'readme.txt').read_bytes() == b'fabricated'


def _pause_first_extraction(monkeypatch):
    """Make the first extraction stop after writing, before publishing."""
    real = archives._extract_archive
    paused = threading.Event()
    release = threading.Event()
    calls = []

    def _extract(in_path, out_dir, members, verbose):
        calls.append(out_dir)
        written = real(in_path, out_dir, members, verbose)
        if len(calls) == 1:
            paused.set()
            release.wait(timeout=10)
        return written

    monkeypatch.setattr(archives, '_extract_archive', _extract)
    return paused, release, calls


def test_a_second_extraction_during_the_first_leaves_one_complete_folder(
    tmp_path, monkeypatch, capsys
):
    zip_path = _archive(tmp_path / 'cadastre.zip')
    out_dir = tmp_path / 'heap'
    paused, release, _ = _pause_first_extraction(monkeypatch)
    errors = []

    def _first():
        try:
            unzip(zip_path, out_dir, verbose=False)
        except Exception as error:  # reported by the assertion below
            errors.append(error)

    first = threading.Thread(target=_first)
    first.start()
    assert paused.wait(timeout=10)

    # The first job's files are all written, in its staging folder.
    assert not (out_dir / 'parcels.gdb').exists()
    assert not extracted_path_ready(out_dir / 'parcels.gdb', out_dir)

    unzip(zip_path, out_dir, verbose=False)
    _assert_complete(out_dir)
    assert extracted_path_ready(out_dir / 'parcels.gdb', out_dir)

    release.set()
    first.join(timeout=10)
    assert errors == []
    _assert_complete(out_dir)
    assert _internal(out_dir, archives._STAGING_PREFIX) == []
    assert len(_internal(out_dir, archives._MARKER_PREFIX)) == 1
    log = capsys.readouterr().out
    assert 'Another job is extracting into' in log
    assert 'already extracted from cadastre.zip' in log


def test_a_reader_never_sees_a_partial_entry(tmp_path, monkeypatch):
    zip_path = _archive(tmp_path / 'cadastre.zip')
    out_dir = tmp_path / 'heap'
    real = archives._extract_archive
    seen = []

    def _extract(in_path, staging, members, verbose):
        written = real(in_path, staging, members, verbose)
        # Everything is on disk, but only in the staging folder.
        seen.append(
            (
                (out_dir / 'parcels.gdb').exists(),
                (out_dir / 'readme.txt').exists(),
                staging.parent == out_dir,
            )
        )
        return written

    monkeypatch.setattr(archives, '_extract_archive', _extract)
    unzip(zip_path, out_dir, verbose=False)

    assert seen == [(False, False, True)]
    _assert_complete(out_dir)


def test_an_entry_beside_a_live_staging_folder_is_not_ready(tmp_path):
    # A shapefile's .shp renamed in before its .dbf: until the marker
    # names it, an entry next to a running extraction is not trusted.
    out_dir = tmp_path / 'heap'
    out_dir.mkdir()
    (out_dir / 'parcels.shp').write_bytes(b'shp')
    assert extracted_path_ready(out_dir / 'parcels.shp', out_dir)

    staging = out_dir / f'{archives._STAGING_PREFIX}{archives._owner_tag()}'
    staging.mkdir()
    assert not extracted_path_ready(out_dir / 'parcels.shp', out_dir)
    assert not extracted_path_ready(out_dir / 'missing.shp', out_dir)
    assert extracted_path_ready(tmp_path / 'heap', None)


def test_an_abandoned_staging_folder_of_a_dead_process_is_removed(
    tmp_path, monkeypatch, capsys
):
    zip_path = _archive(tmp_path / 'cadastre.zip')
    out_dir = tmp_path / 'heap'
    out_dir.mkdir()
    host = archives._owner_tag().rsplit('-', 2)[0]
    dead = out_dir / f'{archives._STAGING_PREFIX}{host}-999999-deadbeef'
    (dead / 'parcels.gdb').mkdir(parents=True)
    (dead / 'parcels.gdb' / 'gdb').write_bytes(b'half')
    monkeypatch.setattr(archives, '_pid_alive', lambda pid: pid != 999999)

    unzip(zip_path, out_dir, verbose=False)

    assert not dead.exists()
    _assert_complete(out_dir)
    assert 'Another job' not in capsys.readouterr().out


def test_an_old_staging_folder_from_another_host_is_removed(tmp_path):
    zip_path = _archive(tmp_path / 'cadastre.zip')
    out_dir = tmp_path / 'heap'
    out_dir.mkdir()
    other = out_dir / f'{archives._STAGING_PREFIX}otherhost-12-cafe0001'
    other.mkdir()
    old = time.time() - archives._abandoned_after_seconds() - 60
    os.utime(other, (old, old))

    unzip(zip_path, out_dir, verbose=False)

    assert not other.exists()
    _assert_complete(out_dir)


def test_a_fresh_staging_folder_from_another_host_is_left_alone(tmp_path):
    # Its job may still be running: no process can be checked on
    # another host, so only age makes it abandoned.
    zip_path = _archive(tmp_path / 'cadastre.zip')
    out_dir = tmp_path / 'heap'
    out_dir.mkdir()
    other = out_dir / f'{archives._STAGING_PREFIX}otherhost-12-cafe0001'
    other.mkdir()

    unzip(zip_path, out_dir, verbose=False)

    assert other.exists()
    _assert_complete(out_dir)


def test_a_complete_extraction_is_reused(tmp_path, monkeypatch):
    zip_path = _archive(tmp_path / 'cadastre.zip')
    out_dir = tmp_path / 'heap'
    unzip(zip_path, out_dir, verbose=False)

    def _refuse(*args, **kwargs):
        raise AssertionError('extracted again')

    monkeypatch.setattr(archives, '_extract_archive', _refuse)
    unzip(zip_path, out_dir, verbose=False)
    _assert_complete(out_dir)


def test_a_deleted_entry_is_extracted_again(tmp_path):
    zip_path = _archive(tmp_path / 'cadastre.zip')
    out_dir = tmp_path / 'heap'
    unzip(zip_path, out_dir, verbose=False)
    (out_dir / 'readme.txt').unlink()

    unzip(zip_path, out_dir, verbose=False)
    _assert_complete(out_dir)


def test_a_new_archive_replaces_the_old_extraction(tmp_path):
    zip_path = _archive(tmp_path / 'cadastre.zip')
    out_dir = tmp_path / 'heap'
    unzip(zip_path, out_dir, verbose=False)

    newer = {'parcels.gdb/gdb': b'N' * 8}
    _archive(zip_path, files=newer, readme=b'newer')
    later = time.time() + 5
    os.utime(zip_path, (later, later))
    unzip(zip_path, out_dir, verbose=False)

    assert sorted(p.name for p in (out_dir / 'parcels.gdb').iterdir()) == ['gdb']
    assert (out_dir / 'parcels.gdb' / 'gdb').read_bytes() == b'N' * 8
    assert (out_dir / 'readme.txt').read_bytes() == b'newer'
    assert _internal(out_dir, archives._ASIDE_PREFIX) == []


def test_an_extraction_from_before_markers_is_replaced_not_reused(tmp_path):
    # A heap written by the old, non-atomic code has no marker and may
    # be half written; an extraction that runs replaces it.
    zip_path = _archive(tmp_path / 'cadastre.zip')
    out_dir = tmp_path / 'heap'
    (out_dir / 'parcels.gdb').mkdir(parents=True)
    (out_dir / 'parcels.gdb' / 'gdb').write_bytes(b'half')

    unzip(zip_path, out_dir, verbose=False)
    _assert_complete(out_dir)


def test_cleanup_keeps_an_extraction_another_job_reads(tmp_path, capsys):
    zip_path = _archive(tmp_path / 'cadastre.zip')
    out_dir = tmp_path / 'heap'
    unzip(zip_path, out_dir, verbose=False)
    own = acquire_read_lease(out_dir)
    other = acquire_read_lease(out_dir)

    assert delete_extracted([out_dir / 'parcels.gdb'], out_dir, own) is False
    _assert_complete(out_dir)
    assert 'another job is reading it' in capsys.readouterr().out

    release_read_lease(other)
    assert delete_extracted([out_dir / 'parcels.gdb'], out_dir, own) is True
    assert not (out_dir / 'parcels.gdb').exists()
    assert _internal(out_dir, archives._ASIDE_PREFIX) == []
    # The marker named the deleted entry, so the next call re-extracts.
    assert _internal(out_dir, archives._MARKER_PREFIX) == []
    release_read_lease(own, stop_at=tmp_path)


def test_a_lease_taken_during_the_delete_puts_everything_back(tmp_path, monkeypatch):
    zip_path = _archive(tmp_path / 'cadastre.zip')
    out_dir = tmp_path / 'heap'
    unzip(zip_path, out_dir, verbose=False)
    answers = iter([[], [out_dir / 'late-lease']])
    monkeypatch.setattr(archives, 'other_readers', lambda *a: next(answers))

    assert delete_extracted([out_dir / 'parcels.gdb'], out_dir) is False
    _assert_complete(out_dir)
    assert _internal(out_dir, archives._ASIDE_PREFIX) == []


def test_an_abandoned_lease_does_not_block_the_cleanup(tmp_path, monkeypatch):
    zip_path = _archive(tmp_path / 'cadastre.zip')
    out_dir = tmp_path / 'heap'
    unzip(zip_path, out_dir, verbose=False)
    host = archives._owner_tag().rsplit('-', 2)[0]
    (out_dir / f'{archives._LEASE_PREFIX}{host}-999999-deadbeef').touch()
    monkeypatch.setattr(archives, '_pid_alive', lambda pid: pid != 999999)

    assert delete_extracted([out_dir / 'parcels.gdb'], out_dir) is True
    assert not (out_dir / 'parcels.gdb').exists()


def test_release_removes_the_folders_it_empties(tmp_path):
    out_dir = tmp_path / 'heap' / 'CO' / 'igac'
    lease = acquire_read_lease(out_dir)
    release_read_lease(lease, stop_at=tmp_path / 'heap')
    assert not (tmp_path / 'heap' / 'CO').exists()
    assert (tmp_path / 'heap').exists()


def test_owner_tag_round_trips():
    tag = archives._owner_tag()
    prefix = archives._STAGING_PREFIX
    host, pid = archives._owner(prefix + tag, prefix)
    assert pid == os.getpid()
    assert host and host in tag
    assert '.' not in tag
    assert socket.gethostname()


@pytest.mark.skipif(os.name != 'nt', reason='An open file blocks a rename on Windows.')
def test_a_stale_entry_held_open_on_windows_raises_and_stays_intact(tmp_path):
    zip_path = _archive(tmp_path / 'cadastre.zip')
    out_dir = tmp_path / 'heap'
    out_dir.mkdir()
    (out_dir / 'readme.txt').write_bytes(b'old')

    with (out_dir / 'readme.txt').open('rb'):
        with pytest.raises(archives.ExtractionTargetInUseError, match='in use'):
            unzip(zip_path, out_dir, verbose=False)
    assert (out_dir / 'readme.txt').read_bytes() == b'old'
    assert _internal(out_dir, archives._STAGING_PREFIX) == []
