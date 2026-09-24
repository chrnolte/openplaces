"""
The exclusive data-root lock for destructive operations and the
cluster guard that refuses to run beside a live job.
"""

import getpass
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

from openplaces.config import cfg
from openplaces.io.cleanup.receipts import _utc_now_iso


class DataLock:
    """Exclusive lock file under the data root for destructive operations.

    A county-scoped lock (`.openplaces.<admin_id>.lock`) lets cleanups of
    different counties run concurrently; the global lock
    (`.openplaces.lock`) serializes data-root-wide operations like
    compact(). Stale locks (older than `stale_after_s`) are taken over.

    The lock file carries a per-instance owner token, and three rules
    follow from it. A takeover claims the file by atomic replace and then
    reads it back, so of two jobs racing on the same stale lock only the
    one whose token survived proceeds. Release unlinks the file only
    while it still holds that token, so a job that lost the race, or one
    that overran, cannot delete its successor's lock. And `touch`
    refreshes the mtime during a long operation, so exclusion is not lost
    part way through a compact that runs longer than `stale_after_s`.

    Parameters
    ----------
    admin_id : str or AdminId, optional
        Admin unit to scope the lock to; None takes the global lock.
    timeout_s : float
        How long to wait for a held lock before raising TimeoutError.
    stale_after_s : float
        Age past which a lock file is treated as abandoned.
    """

    def __init__(self, admin_id=None, timeout_s=10.0, stale_after_s=3600.0):
        name = f'.openplaces.{admin_id}.lock' if admin_id else '.openplaces.lock'
        self.path = Path(cfg.data_root) / name
        self.timeout_s = timeout_s
        self.stale_after_s = stale_after_s
        self.token = f'{os.getpid()}@{socket.gethostname()}:{os.urandom(6).hex()}'
        self._held = False
        self._touched_at = 0.0

    @property
    def _payload(self) -> str:
        return f'{self.token} {_utc_now_iso()}'

    def _holds_lock(self) -> bool:
        try:
            return self.token in self.path.read_text(encoding='utf-8')
        except OSError:
            return False

    def _take_over_if_stale(self) -> bool:
        """Claim an abandoned lock; True when this instance won it."""
        try:
            age = time.time() - self.path.stat().st_mtime
        except OSError:
            return False  # vanished: the plain create will retry
        if age <= self.stale_after_s:
            return False
        tmp = self.path.with_name(f'{self.path.name}.take{os.getpid()}')
        try:
            tmp.write_text(self._payload, encoding='utf-8')
            os.replace(tmp, self.path)
        except OSError:
            tmp.unlink(missing_ok=True)
            return False
        # Two jobs can replace the same stale lock; the last write wins
        # and only its owner may proceed
        return self._holds_lock()

    def __enter__(self):
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(fd, self._payload.encode())
                finally:
                    os.close(fd)
                self._held = True
                self._touched_at = time.monotonic()
                return self
            except FileExistsError:
                if self._take_over_if_stale():
                    self._held = True
                    self._touched_at = time.monotonic()
                    return self
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f'Could not acquire lock {self.path}. Another '
                        'cleanup may be running; remove the lock file if '
                        'it is stale.'
                    ) from None
                time.sleep(0.5)

    def touch(self) -> None:
        """Refresh the lock's mtime so a long operation stays exclusive.

        Rate-limited to a quarter of `stale_after_s`, so callers may call
        it as often as they like (e.g. once per scanned directory).
        """
        if not self._held:
            return
        now = time.monotonic()
        if now - self._touched_at < self.stale_after_s / 4:
            return
        try:
            os.utime(self.path, None)
        except OSError:
            return
        self._touched_at = now

    def __exit__(self, *exc):
        if self._held and self._holds_lock():
            self.path.unlink(missing_ok=True)
        self._held = False
        return False


def _touch_lock(lock) -> None:
    """Refresh a lock's mtime, tolerating the dry-run nullcontext."""
    if isinstance(lock, DataLock):
        lock.touch()


def _cluster_busy() -> bool:
    """True when a cluster queue has pending or running jobs for the user."""
    if shutil.which('qstat') is None:
        return False
    try:
        result = subprocess.run(
            ['qstat', '-u', getpass.getuser()],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        return False
    return bool(result.stdout.strip())
