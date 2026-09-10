"""Orchestrator bookkeeping is marked so a sync client skips it.

A full run writes tens of thousands of Snakemake metadata files and
stage timers; inside a synced folder every one of them is uploaded.
"""

import sys

import pytest

from openplaces.flow.sync import (
    exclude_bookkeeping_from_sync,
    exclude_from_sync,
    is_excluded_from_sync,
)

supports_marker = pytest.mark.skipif(
    sys.platform not in ('win32', 'linux', 'darwin'),
    reason='no attribute or stream support on this platform',
)


@supports_marker
def test_marker_is_set_created_and_idempotent(tmp_path):
    folder = tmp_path / 'sync' / '.snakemake'
    assert not folder.exists()
    ok = exclude_from_sync(folder)
    if not ok:
        pytest.skip('this filesystem cannot carry the marker')
    assert folder.is_dir()
    assert is_excluded_from_sync(folder)
    assert exclude_from_sync(folder)
    # Files inside are untouched; the marker sits on the folder.
    (folder / 'metadata').mkdir()
    assert not is_excluded_from_sync(folder / 'metadata')


def test_a_missing_path_is_not_marked_unless_created(tmp_path):
    assert not is_excluded_from_sync(tmp_path / 'nowhere')
    assert not exclude_from_sync(tmp_path / 'nowhere', create=False)
    assert not (tmp_path / 'nowhere').exists()


@supports_marker
def test_bookkeeping_folders_of_a_run(tmp_path):
    marked = exclude_bookkeeping_from_sync(tmp_path / 'repo', tmp_path / 'cache')
    if not marked:
        pytest.skip('this filesystem cannot carry the marker')
    assert {p.name for p in marked} == {'.snakemake', '_logs'}
