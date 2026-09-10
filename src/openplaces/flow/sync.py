"""Keep orchestrator bookkeeping out of a file-sync client's way.

A full run writes tens of thousands of small files that no one needs on
another machine: Snakemake's per-job metadata under ``.snakemake``, its
benchmarks, and the per-unit stage timers under the cache's ``_logs``
tree. When the repository or the data root lives inside a synced folder
(Dropbox, on this project's machines), every one of them is uploaded, and
the client can fall hours behind the run it is watching.

Dropbox honors a per-item marker: an NTFS alternate data stream named
``com.dropbox.ignored`` holding ``1`` on Windows, an extended attribute of
the same name on macOS and Linux. The item stays on disk and stops
syncing; deleting the marker resumes it. Nothing here depends on Dropbox
being installed: a marker on an unsynced folder is inert.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

IGNORE_MARKER = 'com.dropbox.ignored'


def is_excluded_from_sync(path: str | Path) -> bool:
    """True when *path* carries the sync-ignore marker.

    Parameters
    ----------
    path : str or Path
        A file or directory.
    """
    path = Path(path)
    if not path.exists():
        return False
    try:
        if sys.platform == 'win32':
            with open(f'{path}:{IGNORE_MARKER}', encoding='utf-8') as stream:
                return stream.read().strip() == '1'
        value = os.getxattr(path, f'user.{IGNORE_MARKER}')  # type: ignore[attr-defined]
        return value.strip() == b'1'
    except OSError:
        return False


def exclude_from_sync(path: str | Path, create: bool = True) -> bool:
    """Mark *path* so a sync client skips it, creating the directory if asked.

    Idempotent, and never raises: a filesystem that cannot carry the
    marker (a network share, a FAT volume) leaves the folder synced as
    before, which is the pre-existing behavior rather than a failure of
    the run that called this.

    Parameters
    ----------
    path : str or Path
        Directory (or file) to mark.
    create : bool
        Create *path* as a directory when it does not exist yet, so the
        marker is in place before the first file lands in it.

    Returns
    -------
    bool
        True when the marker is present afterwards.
    """
    path = Path(path)
    if not path.exists():
        if not create:
            return False
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            return False
    if is_excluded_from_sync(path):
        return True
    try:
        if sys.platform == 'win32':
            with open(f'{path}:{IGNORE_MARKER}', 'w', encoding='utf-8') as stream:
                stream.write('1')
        else:
            os.setxattr(path, f'user.{IGNORE_MARKER}', b'1')  # type: ignore[attr-defined]
    except OSError:
        return False
    return is_excluded_from_sync(path)


def exclude_bookkeeping_from_sync(workdir: str | Path, cache_dir: str | Path) -> list:
    """Mark the orchestrator's bookkeeping folders for one run.

    Parameters
    ----------
    workdir : str or Path
        The Snakemake working directory, which holds ``.snakemake``.
    cache_dir : str or Path
        The openplaces cache directory, whose ``_logs`` tree holds the
        stage timers and Snakemake benchmarks.

    Returns
    -------
    list of Path
        The folders now carrying the marker.
    """
    marked = []
    for folder in (Path(workdir) / '.snakemake', Path(cache_dir) / '_logs'):
        if exclude_from_sync(folder):
            marked.append(folder)
    return marked
