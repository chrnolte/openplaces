"""A .json sidecar never beats a data file in find_latest_file_or_gdb."""

import os
import time

from openplaces.io import find_latest_file_or_gdb


def _touch(path, mtime):
    path.write_text('x')
    os.utime(path, (mtime, mtime))


def test_newer_json_sidecar_does_not_beat_data_file(tmp_path):
    now = time.time()
    _touch(tmp_path / 'parcels.csv', now - 100)
    _touch(tmp_path / 'metadata.json', now)
    assert find_latest_file_or_gdb(tmp_path) == tmp_path / 'parcels.csv'


def test_json_is_chosen_when_it_is_the_only_candidate(tmp_path):
    _touch(tmp_path / 'municipalities.json', time.time())
    assert find_latest_file_or_gdb(tmp_path) == tmp_path / 'municipalities.json'


def test_newest_wins_across_formats(tmp_path):
    # The heap directory is shared across reruns, so a freshly extracted
    # table must beat a stale geospatial file left by an earlier run.
    now = time.time()
    _touch(tmp_path / 'stale.gpkg', now - 100)
    _touch(tmp_path / 'fresh.csv', now)
    assert find_latest_file_or_gdb(tmp_path) == tmp_path / 'fresh.csv'


def test_gdb_directory_is_a_candidate(tmp_path):
    now = time.time()
    _touch(tmp_path / 'old.csv', now - 100)
    gdb = tmp_path / 'parcels.gdb'
    gdb.mkdir()
    os.utime(gdb, (now, now))
    assert find_latest_file_or_gdb(tmp_path) == gdb


def test_uppercase_data_file_beats_a_newer_json_sidecar(tmp_path):
    # County and state shapefile archives routinely ship uppercase
    # extensions. Matching by raw case left the data file unrecognized
    # and handed the caller the JSON the demotion rule exists to avoid.
    now = time.time()
    _touch(tmp_path / 'PARCELS.SHP', now - 100)
    _touch(tmp_path / 'meta.json', now)
    assert find_latest_file_or_gdb(tmp_path) == tmp_path / 'PARCELS.SHP'


def test_uppercase_json_sidecar_is_still_demoted(tmp_path):
    now = time.time()
    _touch(tmp_path / 'parcels.csv', now - 100)
    _touch(tmp_path / 'META.JSON', now)
    assert find_latest_file_or_gdb(tmp_path) == tmp_path / 'parcels.csv'


def test_uppercase_gdb_directory_is_a_candidate(tmp_path):
    now = time.time()
    _touch(tmp_path / 'old.csv', now - 100)
    gdb = tmp_path / 'PARCELS.GDB'
    gdb.mkdir()
    os.utime(gdb, (now, now))
    assert find_latest_file_or_gdb(tmp_path) == gdb
