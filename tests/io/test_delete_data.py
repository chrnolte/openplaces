"""Tests for delete_data interrupting on partial deletions.

A file sync app (e.g. Dropbox) locking files mid-removal can leave a partially
deleted geodatabase that a later ingest would silently reuse. delete_data must
raise (not warn) and name the leftover path with a clickable file link.
"""

import pytest

import openplaces.io.deletion as opio  # delete_data looks its retry pause up here
from openplaces.io.deletion import DataDeletionError, _deletion_interrupted_error


@pytest.fixture(autouse=True)
def no_retry_wait(monkeypatch):
    # The retry is real; waiting for it is not what these tests measure.
    monkeypatch.setattr(opio, '_GDB_DELETE_RETRY_SECONDS', 0)


def test_deletion_error_message_has_path_and_clickable_link(tmp_path):
    target = tmp_path / 'NC_Parcels_all.gdb'
    target.mkdir()

    error = _deletion_interrupted_error(target, is_dir=True)

    assert isinstance(error, DataDeletionError)
    message = str(error)
    assert str(target.resolve()) in message
    # Clickable file URI that opens in the OS file browser.
    assert target.resolve().as_uri() in message
    assert message.startswith('\n\n') and 'file://' in message


def test_delete_gdb_success_removes_directory(tmp_path):
    gdb = tmp_path / 'parcels.gdb'
    gdb.mkdir()
    (gdb / 'a00000001.gdbtable').write_bytes(b'data')

    opio.delete_data(gdb, delete_empty_parent_dirs=False)

    assert not gdb.exists()


def test_delete_gdb_permission_error_raises(monkeypatch, tmp_path):
    gdb = tmp_path / 'parcels.gdb'
    gdb.mkdir()

    def _locked(*args, **kwargs):
        raise PermissionError('file locked by sync app')

    monkeypatch.setattr(opio.shutil, 'rmtree', _locked)

    with pytest.raises(DataDeletionError, match='Interrupted while deleting'):
        opio.delete_data(gdb, delete_empty_parent_dirs=False)


def test_delete_gdb_retries_while_a_sync_app_holds_the_handle(monkeypatch, tmp_path):
    # Dropbox indexing a fresh extraction releases its handle within
    # seconds; the first attempts fail, a later one succeeds, and the
    # caller never sees the transient lock.
    gdb = tmp_path / 'parcels.gdb'
    gdb.mkdir()
    (gdb / 'a00000001.gdbtable').write_bytes(b'data')
    real_rmtree = opio.shutil.rmtree
    calls = []

    def _locked_twice(path, *args, **kwargs):
        calls.append(path)
        if len(calls) <= 2:
            raise PermissionError('file locked by sync app')
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(opio.shutil, 'rmtree', _locked_twice)

    opio.delete_data(gdb, delete_empty_parent_dirs=False)

    assert not gdb.exists()
    assert len(calls) == 3


def test_delete_gdb_gives_up_after_the_last_attempt(monkeypatch, tmp_path):
    gdb = tmp_path / 'parcels.gdb'
    gdb.mkdir()
    calls = []

    def _locked(*args, **kwargs):
        calls.append(1)
        raise PermissionError('file locked by sync app')

    monkeypatch.setattr(opio.shutil, 'rmtree', _locked)

    with pytest.raises(DataDeletionError):
        opio.delete_data(gdb, delete_empty_parent_dirs=False)
    assert len(calls) == opio._GDB_DELETE_ATTEMPTS


def test_delete_gdb_partial_leftover_raises(monkeypatch, tmp_path):
    # rmtree returns without error but leaves the directory behind (Windows can
    # stop partway when a lock releases mid-walk).
    gdb = tmp_path / 'parcels.gdb'
    gdb.mkdir()
    (gdb / 'leftover').write_bytes(b'x')

    monkeypatch.setattr(opio.shutil, 'rmtree', lambda *a, **k: None)

    with pytest.raises(DataDeletionError):
        opio.delete_data(gdb, delete_empty_parent_dirs=False)


def test_delete_plain_file_permission_error_raises(monkeypatch, tmp_path):
    from pathlib import Path

    data = tmp_path / 'data.parquet'
    data.write_bytes(b'x')

    def _locked(self, *args, **kwargs):
        raise PermissionError('file locked by sync app')

    monkeypatch.setattr(Path, 'unlink', _locked)

    with pytest.raises(DataDeletionError):
        opio.delete_data(data, delete_empty_parent_dirs=False)
