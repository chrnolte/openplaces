"""`needs_rerun` picks the units a batched ingest declares.

A batch of every unit, rerun for one of them, rewrote all of them and
reran every consumer of every unit; the Snakefile now asks this of each
unit and declares only those that answer yes.
"""

from __future__ import annotations

import os

from openplaces.flow.dag import needs_rerun


def _write(path, mtime):
    path.write_text('x')
    os.utime(path, (mtime, mtime))
    return path


def test_up_to_date_pair_needs_no_rerun(tmp_path):
    admin = _write(tmp_path / 'admin.parquet', 100)
    out = _write(tmp_path / 'u.parquet', 200)
    geo = _write(tmp_path / 'u_geo.parquet', 200)

    assert needs_rerun([out, geo], [admin]) is False


def test_missing_attribute_file_beside_its_sidecar_needs_a_rerun(tmp_path):
    """The state the 2026-10-27 batch failure left 685 units in."""
    admin = _write(tmp_path / 'admin.parquet', 100)
    geo = _write(tmp_path / 'u_geo.parquet', 200)

    assert needs_rerun([tmp_path / 'u.parquet', geo], [admin]) is True


def test_input_newer_than_the_oldest_output_needs_a_rerun(tmp_path):
    admin = _write(tmp_path / 'admin.parquet', 150)
    out = _write(tmp_path / 'u.parquet', 200)
    geo = _write(tmp_path / 'u_geo.parquet', 100)

    assert needs_rerun([out, geo], [admin]) is True


def test_missing_input_needs_a_rerun(tmp_path):
    """Another job of the run will write it, so this unit follows."""
    out = _write(tmp_path / 'u.parquet', 200)

    assert needs_rerun([out], [tmp_path / 'admin.parquet']) is True
