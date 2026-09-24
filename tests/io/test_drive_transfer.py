"""A failed Drive upload must be loud, and must never lose the local data.

`share` compresses, uploads, and then deletes. Before the fix `to_drive`
ignored rclone's exit code, so an expired token or a missing remote left
`share` printing success while it unlinked the ZIP, with the original
already removed by `compress`. The data then existed nowhere.
"""

import subprocess

import pandas as pd
import pytest

from openplaces.io import DriveTransferError, share, to_drive
from openplaces.io import transfer as op_io  # share looks to_drive up here


class _Completed:
    def __init__(self, returncode):
        self.returncode = returncode


def test_to_drive_raises_on_nonzero_exit(tmp_path, monkeypatch):
    filepath = tmp_path / 'out.zip'
    filepath.write_text('x')
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: _Completed(1))
    with pytest.raises(DriveTransferError, match='exit code 1'):
        to_drive(filepath, 'somewhere', verbose=False)


def test_to_drive_raises_when_rclone_is_missing(tmp_path, monkeypatch):
    filepath = tmp_path / 'out.zip'
    filepath.write_text('x')

    def _missing(*args, **kwargs):
        raise FileNotFoundError('rclone')

    monkeypatch.setattr(subprocess, 'run', _missing)
    with pytest.raises(DriveTransferError, match='Could not run'):
        to_drive(filepath, 'somewhere', verbose=False)


def test_to_drive_succeeds_on_zero_exit(tmp_path, monkeypatch):
    filepath = tmp_path / 'out.zip'
    filepath.write_text('x')
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: _Completed(0))
    to_drive(filepath, 'somewhere', verbose=False)


def test_share_keeps_every_local_copy_when_the_upload_fails(tmp_path, monkeypatch):
    filepath = tmp_path / 'table.csv'
    zip_path = tmp_path / 'table_csv.zip'
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: _Completed(1))

    with pytest.raises(DriveTransferError):
        share(
            pd.DataFrame({'a': [1]}),
            filepath,
            drive_dir='somewhere',
            verbose=False,
        )

    assert filepath.exists()
    assert zip_path.exists()


def test_share_deletes_local_copies_after_a_successful_upload(tmp_path, monkeypatch):
    filepath = tmp_path / 'table.csv'
    zip_path = tmp_path / 'table_csv.zip'
    uploaded = []
    monkeypatch.setattr(
        op_io,
        'to_drive',
        lambda path, directory, **kwargs: uploaded.append(path),
    )

    share(
        pd.DataFrame({'a': [1]}),
        filepath,
        drive_dir='somewhere',
        verbose=False,
    )

    assert uploaded == [zip_path]
    assert not filepath.exists()
    assert not zip_path.exists()
