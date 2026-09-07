"""
Tests for the Google Drive scraper's retry timeout and zip staging.

No network: `_call_with_retry` is driven with plain local callables, and
the extraction test works on a zip built in the test's own tmp_path.
"""

import errno
import threading
import time
import zipfile
from pathlib import Path

import pytest

from openplaces.io.scrapers import google_drive_scraper as scraper


def test_timeout_returns_control_while_the_worker_is_still_stuck():
    """
    A timed-out call must return to the caller, not join the hung thread.

    The executor used to be a context manager inside the retry loop, so
    its __exit__ ran shutdown(wait=True) and waited out the whole stall
    the timeout existed to bound.
    """
    release = threading.Event()
    started = threading.Event()

    def stuck():
        started.set()
        release.wait(30)

    start = time.monotonic()
    try:
        with pytest.raises(Exception):
            scraper._call_with_retry(
                stuck, timeout=0.2, retries=1, verbose=False, label='test'
            )
        elapsed = time.monotonic() - start
        assert started.is_set()
        assert elapsed < 5.0
    finally:
        release.set()


def test_a_retry_is_not_queued_behind_the_hung_attempt():
    """The second attempt must run even while the first is still stuck."""
    release = threading.Event()
    calls = []

    def stuck_then_fine():
        calls.append(len(calls))
        if len(calls) == 1:
            release.wait(30)
            return 'late'
        return 'fresh'

    start = time.monotonic()
    try:
        result = scraper._call_with_retry(
            stuck_then_fine, timeout=0.2, retries=2, verbose=False, label='test'
        )
        assert result == 'fresh'
        # 0.2s timeout plus one 2s backoff; anything near the 30s stall
        # means the first attempt was waited out rather than abandoned.
        assert time.monotonic() - start < 10.0
    finally:
        release.set()


def test_zero_retries_raises_the_real_error():
    """`retries=0` used to raise None instead of the underlying failure."""

    def boom():
        raise ConnectionError('drive said no')

    with pytest.raises(ConnectionError, match='drive said no'):
        scraper._call_with_retry(
            boom, timeout=5, retries=0, verbose=False, label='test'
        )


def test_zip_member_is_staged_on_the_targets_own_filesystem(tmp_path, monkeypatch):
    """
    Extraction must not stage in the system temp directory.

    `Path.replace` is `os.replace`, which fails with EXDEV whenever the
    data root sits on another drive or share. The failure is simulated
    here by refusing any move that starts in the system temp directory.
    """
    zip_path = tmp_path / 'source.zip'
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('member.csv', 'a,b\n1,2\n')

    real_replace = Path.replace

    def guarded_replace(self, target):
        # Stands in for two filesystems: a move only succeeds when it
        # starts inside the destination's own directory.
        source = Path(self).resolve()
        destination = Path(target).resolve()
        if destination.parent not in source.parents:
            raise OSError(
                errno.EXDEV, 'simulated cross-device move', str(self), str(target)
            )
        return real_replace(self, target)

    monkeypatch.setattr(Path, 'replace', guarded_replace)

    target = tmp_path / 'out' / 'data.csv'
    scraper._extract_single_member(zip_path, target)

    assert target.read_text(encoding='utf-8') == 'a,b\n1,2\n'
    assert not list(tmp_path.glob('out/tmp*'))
