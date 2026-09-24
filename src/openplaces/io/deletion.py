"""
Delete data-root files and directories safely: a geodatabase held
open by another process, an interrupted delete, and the image
caches that must never persist.
"""

import shutil
import time
import warnings
from pathlib import Path

import pandas as pd

from openplaces.config import cfg
from openplaces.core.constants import (
    SHAPEFILE_EXTENSIONS,
)
from openplaces.core.schema import AdminId


class DataDeletionError(OSError):
    """Raised when an unzipped dataset could not be fully deleted.

    A partial deletion, typically caused by a file sync app (e.g. Dropbox)
    locking files mid-removal, leaves a corrupt copy on disk that a later
    ingest would silently reuse. Deletion failures therefore interrupt the run
    rather than warn.
    """


# How long delete_data keeps trying to remove a geodatabase a sync app is
# still holding: five attempts, three seconds apart.
_GDB_DELETE_ATTEMPTS = 5
_GDB_DELETE_RETRY_SECONDS = 3.0


def _deletion_interrupted_error(path: Path, *, is_dir: bool) -> DataDeletionError:
    """Build a DataDeletionError naming *path* with a clickable file link.

    The ``file://`` URI is rendered as a clickable link by most terminals and
    Jupyter, opening the location in the OS file browser (e.g. Explorer) so the
    leftover, partially deleted dataset can be removed by hand.

    Parameters
    ----------
    path : Path
        Path whose deletion was interrupted.
    is_dir : bool
        Whether the path is a directory (e.g. a geodatabase) or a single file.
    """
    resolved = Path(path).resolve()
    target = 'directory' if is_dir else 'file'
    return DataDeletionError(
        f'\n\nInterrupted while deleting an unzipped {target}.\n\n'
        'A file sync app (e.g., Dropbox) is most likely locking files here,\n'
        'leaving a partially deleted, corrupt copy that the next ingest would '
        'silently reuse.\n\n'
        f'  {resolved}\n'
        f'  {resolved.as_uri()}\n\n'
        'Pause or quit the sync app, delete the path above, then re-run.\n\n'
    )


def delete_data(data_path, delete_empty_parent_dirs=True):
    """Delete dataset from openplaces filesystem

    Handles geodatabases and shapefiles with compantion files

    Parameters
    ----------
    data_path : Path
        Path to file to be deleted. Extension determines how deletion
        occurs (e.g. '.shp' files and '.gdb' folders are handled)
    delete_empty_parent_dirs : bool
        Deletes any parent directories that are now empty.
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f'File to delete not found: {data_path.relative_to(cfg.data_root)}'
        )

    if data_path.suffix == '.gdb':
        # A sync app indexing a freshly extracted geodatabase holds a
        # handle on it for a few seconds, and rmtree then fails on the
        # directory itself after removing its contents. Try again a few
        # times before giving up: the handle is released on its own, and
        # an orchestrator job has no one to pause the app and re-run.
        for attempt in range(_GDB_DELETE_ATTEMPTS):
            try:
                shutil.rmtree(data_path)
            except OSError as error:
                if attempt == _GDB_DELETE_ATTEMPTS - 1:
                    raise _deletion_interrupted_error(data_path, is_dir=True) from error
                time.sleep(_GDB_DELETE_RETRY_SECONDS)
                continue
            # rmtree can stop partway when a lock is released mid-walk;
            # confirm the directory is actually gone so a partial
            # geodatabase is never reused.
            if not data_path.exists():
                break
            if attempt == _GDB_DELETE_ATTEMPTS - 1:
                raise _deletion_interrupted_error(data_path, is_dir=True)
            time.sleep(_GDB_DELETE_RETRY_SECONDS)

    elif data_path.suffix == '.shp':
        for shapefile_extension in SHAPEFILE_EXTENSIONS:
            data_path.with_suffix(shapefile_extension).unlink(missing_ok=True)
    else:
        try:
            data_path.unlink()
        except OSError as error:
            raise _deletion_interrupted_error(data_path, is_dir=False) from error

    if delete_empty_parent_dirs:
        current_dir = data_path.parent
        while True:
            if current_dir == cfg.data_root:
                break

            # Check if directory is empty
            if current_dir.exists() and not any(current_dir.iterdir()):
                try:
                    current_dir.rmdir()
                except PermissionError:
                    warnings.warn(
                        '\n\nUnable to delete empty directory due to permission error:'
                        + f'\n\n{current_dir}\n\n'
                        'Is a file sync app running (e.g., Dropbox)? '
                        'If so, quit and retry, or remove the directory manually.\n\n'
                    )
                current_dir = current_dir.parent
            else:
                break


def delete_image_caches(
    admin_ids: str | list | None = None,
    source: str | None = None,
    version: str | None = None,
    dry_run: bool = True,
) -> pd.DataFrame:
    """Delete location-specific image caches from the external directory.

    Parameters
    ----------
    admin_ids : str, AdminId, list, or None
        Admin units whose caches to delete; a coarser unit (e.g. a county)
        matches all caches of its children. None matches all locations.
    source : str or None
        Restrict to one image source (e.g. 'googlesatellite').
    version : str or None
        Restrict to one recipe version (e.g. 'z20').
    dry_run : bool
        If True (default), only report what would be deleted. If False,
        remove each matched cache directory, including images and the
        image metadata parquet.

    Returns
    -------
    pd.DataFrame
        The matched caches: admin_id, source, version, n_files, size_mb,
        path.
    """
    from openplaces.diagnostics import list_image_caches

    caches = list_image_caches()
    if caches.empty:
        print('No image caches found.')
        return caches

    if admin_ids is not None:
        if isinstance(admin_ids, str | AdminId):
            admin_ids = [admin_ids]
        selectors = [AdminId(str(a)) for a in admin_ids]
        caches = caches[
            [
                any(
                    str(sel) == str(aid) or sel.is_parent_of(aid)
                    for sel in selectors
                    for aid in [AdminId(cache_admin_id)]
                )
                for cache_admin_id in caches['admin_id']
            ]
        ]
    if source is not None:
        caches = caches[caches['source'] == source]
    if version is not None:
        caches = caches[caches['version'] == str(version)]
    caches = caches.reset_index(drop=True)

    total_mb = caches['size_mb'].sum()
    if dry_run:
        print(
            f'Dry run: would delete {len(caches)} image cache(s), '
            f'{total_mb:,.1f} MB total. Pass dry_run=False to delete.'
        )
        return caches

    for cache_path in caches['path']:
        shutil.rmtree(cache_path, ignore_errors=True)
    print(f'Deleted {len(caches)} image cache(s), {total_mb:,.1f} MB total.')
    return caches
