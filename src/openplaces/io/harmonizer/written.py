"""The files a harmonize run writes besides its own output.

A step that persists a side table (a link sidecar, a link table, a
re-keyed copy) notes its path here with `note_written`, and the run
reads the journal back to stamp the attribute checkpoint with what the
steps before it wrote. A restored checkpoint skips those steps, so a
side file deleted since (Snakemake deletes a forced job's declared
outputs when the job starts) would never be rewritten; the stamps let
the restore see that and run the full pipeline instead
(`_read_checkpoint`).

A context variable rather than a field on `HarmonizeState`, because
the writers are shared helpers several steps call, and most of them are
never handed the state. Outside `journal_writes` nothing is recorded.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_JOURNAL: ContextVar[list[Path] | None] = ContextVar('harmonize_writes', default=None)


@contextmanager
def journal_writes() -> Iterator[list[Path]]:
    """Record every `note_written` path for the duration of the block.

    Yields the list the paths are appended to, in write order.
    """
    written: list[Path] = []
    token = _JOURNAL.set(written)
    try:
        yield written
    finally:
        _JOURNAL.reset(token)


def note_written(path) -> None:
    """Record *path* in the open journal, if there is one."""
    written = _JOURNAL.get()
    if written is not None:
        written.append(Path(path))


def file_stamps(paths) -> list[dict]:
    """Path, size and mtime of each distinct file in *paths*.

    A path that does not exist (a writer that wrote nothing after all)
    is stamped with size and mtime None, and `stamps_hold` then
    requires it to be absent still.
    """
    stamps = []
    for path in dict.fromkeys(Path(p) for p in paths):
        entry = {'path': str(path), 'size': None, 'mtime': None}
        if path.exists():
            stat = path.stat()
            entry['size'] = stat.st_size
            entry['mtime'] = round(stat.st_mtime, 3)
        stamps.append(entry)
    return stamps


def stamps_hold(stamps: list[dict]) -> bool:
    """Whether every stamped file is still as `file_stamps` found it."""
    return file_stamps(s['path'] for s in stamps) == stamps
