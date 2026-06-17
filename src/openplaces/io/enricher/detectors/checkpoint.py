"""Incremental prediction checkpoints for interruptible detector runs."""

from __future__ import annotations

import json
import os
import time
import warnings
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from platformdirs import user_cache_path


def local_checkpoint_path(source_path: Path | str) -> Path:
    """Return a stable, unsynced cache path for an output-side checkpoint."""
    source_path = Path(source_path)
    path_key = str(source_path.resolve()).casefold().encode()
    digest = sha256(path_key).hexdigest()[:16]
    return (
        user_cache_path('openplaces', 'placeslab')
        / 'prediction_checkpoints'
        / digest
        / source_path.name
    )


def prune_local_checkpoints(
    root: Path | str | None = None,
    max_age_days: int = 14,
    max_total_mb: int = 1024,
    keep_latest: int = 8,
) -> list[Path]:
    """Remove old local checkpoint files so the cache cannot grow forever."""
    root = (
        Path(root)
        if root is not None
        else user_cache_path('openplaces', 'placeslab') / 'prediction_checkpoints'
    )
    if not root.exists():
        return []

    now = time.time()
    candidates: list[tuple[float, Path]] = []
    for path in root.rglob('*.parquet'):
        try:
            stat = path.stat()
        except OSError:
            continue
        age_days = (now - stat.st_mtime) / 86400
        if age_days <= max_age_days:
            continue
        candidates.append((stat.st_mtime, path))

    removed: list[Path] = []
    for _, path in sorted(candidates)[:-keep_latest] if keep_latest else candidates:
        try:
            path.unlink()
            removed.append(path)
        except OSError:
            continue

    remaining = sorted(
        (
            (path.stat().st_mtime, path)
            for path in root.rglob('*.parquet')
            if path.exists()
        ),
        reverse=True,
    )
    total_mb = sum(path.stat().st_size for _, path in remaining) / 2**20
    while total_mb > max_total_mb and len(remaining) > keep_latest:
        _, path = remaining.pop()
        try:
            total_mb -= path.stat().st_size / 2**20
            path.unlink()
            removed.append(path)
        except OSError:
            continue

    return removed


class PredictionCheckpoint:
    """Persist detector predictions incrementally so interrupted runs resume.

    A transient single-column parquet file stored in the local application
    cache. Keeping checkpoints outside synced data trees avoids file-lock
    conflicts with Dropbox and similar clients. Detectors load it on start
    to skip already-predicted keys, add each new prediction, and flush
    periodically; the Enricher deletes it once the evidence file is saved.

    Parameters
    ----------
    path : Path or str
        Checkpoint parquet path.
    save_every : int
        Flush to disk after this many added predictions.
    legacy_paths : iterable of Path or str, optional
        Former checkpoint locations to migrate on first load.
    """

    def __init__(
        self,
        path: Path | str,
        save_every: int = 500,
        legacy_paths=(),
    ) -> None:
        self.path = Path(path)
        self.save_every = save_every
        self.legacy_paths = [Path(path) for path in legacy_paths]
        self._predictions: dict[Any, Any] = {}
        self._n_unsaved = 0
        self._pruned = False

    def load(self) -> dict[Any, Any]:
        """Return previously checkpointed predictions ({} when none).

        None results (e.g. missing image files) count as done and are
        not recomputed on resume.
        """
        candidates = [path for path in [self.path, *self.legacy_paths] if path.exists()]
        if not candidates:
            return dict(self._predictions)

        loaded = []
        for path in candidates:
            try:
                series = pd.read_parquet(path)['value']
            except Exception as exc:
                warnings.warn(
                    f'Ignoring unreadable prediction checkpoint {path}: {exc}',
                    stacklevel=2,
                )
                continue
            loaded.append((len(series), path.stat().st_mtime, path, series))
        if not loaded:
            return dict(self._predictions)

        _, _, loaded_path, series = max(loaded, key=lambda item: item[:2])
        self._predictions = {key: json.loads(value) for key, value in series.items()}
        if loaded_path != self.path:
            self._n_unsaved = 1
            self.flush()
        return dict(self._predictions)

    def add(self, key: Any, value: Any) -> None:
        """Record one prediction; flush every save_every additions."""
        self._predictions[key] = value
        self._n_unsaved += 1
        if self._n_unsaved >= self.save_every:
            self.flush()

    def flush(self) -> None:
        """Write all recorded predictions to disk atomically.

        Values are JSON-encoded per cell so mixed prediction types
        (class labels, counts, None) round-trip faithfully.
        """
        if not self._n_unsaved:
            return
        encoded = {key: json.dumps(value) for key, value in self._predictions.items()}
        frame = pd.DataFrame({'value': pd.Series(encoded, dtype=object)})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_name(f'.{self.path.name}.{uuid4().hex}.tmp')
        try:
            frame.to_parquet(temp_path)
            self._replace_with_retry(temp_path, self.path)
        finally:
            temp_path.unlink(missing_ok=True)
        self._n_unsaved = 0
        self._delete_legacy_paths()
        self.prune_cache()

    def delete(self) -> None:
        """Remove the checkpoint file (call after evidence is saved)."""
        self.path.unlink(missing_ok=True)
        self._delete_legacy_paths()
        self._predictions = {}
        self._n_unsaved = 0
        self.prune_cache()

    @staticmethod
    def _replace_with_retry(
        source: Path,
        target: Path,
        attempts: int = 8,
    ) -> None:
        """Atomically replace *target*, retrying transient Windows locks."""
        for attempt in range(attempts):
            try:
                os.replace(source, target)
                return
            except PermissionError as exc:
                if attempt == attempts - 1:
                    raise PermissionError(
                        f'Could not update prediction checkpoint after '
                        f'{attempts} attempts: {target}'
                    ) from exc
                time.sleep(0.1 * (attempt + 1))

    def _delete_legacy_paths(self) -> None:
        """Best-effort cleanup of former synced checkpoint sidecars."""
        for path in self.legacy_paths:
            try:
                path.unlink(missing_ok=True)
            except PermissionError:
                warnings.warn(
                    f'Could not remove legacy checkpoint locked by sync: {path}',
                    stacklevel=2,
                )

    def prune_cache(self) -> list[Path]:
        """Prune old local checkpoints once per checkpoint instance."""
        if self._pruned:
            return []
        self._pruned = True
        return prune_local_checkpoints(self.path.parent.parent)
