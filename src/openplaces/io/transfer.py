"""
Move files out of the data root: compress them and copy them to a
shared drive with rclone, and the one-call share() that does both.
"""

import subprocess
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from openplaces.config import cfg
from openplaces.core.constants import (
    SHAPEFILE_EXTENSIONS,
)
from openplaces.io.tables import save


def compress(
    filepaths: str | Path | list[str] | set[str],
    zip_filepath: str | None = None,
    delete_original: bool = False,
) -> list[Path]:
    """Compress one or more files.

    Parameters
    ----------
    filepaths : str or list of str
        Single filepath or list of filepaths
    zip_filepath : str, optional
        Output ZIP filepath. If None, derived from the first entry in filepaths.
    delete_original : bool
        If True, deletes the original file(s) after compression.

    Returns
    -------
    list of Path
        The files actually written into the archive, with a shapefile
        expanded to its sibling parts. Empty when none existed. Callers
        that defer deletion until a later step succeeds (see `share`)
        delete exactly these.
    """
    if isinstance(filepaths, str | Path):
        filepaths = [filepaths]
    elif not isinstance(filepaths, list | set):
        raise ValueError(f'`filepaths` argument not understood: {filepaths}')

    paths = [Path(fp) for fp in filepaths]

    if zip_filepath is None:
        zip_path = paths[0].parent / f'{paths[0].stem}_{paths[0].suffix[1:]}.zip'
    else:
        zip_path = Path(zip_filepath)
        if zip_path.suffix != '.zip':
            zip_path = zip_path.parent / f'{zip_path.stem.replace(".", "_")}.zip'

    paths_to_compress: list[Path] = []
    for path in paths:
        if not path.exists():
            print(f'Warning: file does not exist: {path}')
            continue
        if path.suffix == '.shp':
            paths_to_compress.extend(
                path.with_suffix(ext)
                for ext in SHAPEFILE_EXTENSIONS
                if path.with_suffix(ext).exists()
            )
        else:
            paths_to_compress.append(path)

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, 'w', ZIP_DEFLATED) as z:
        for p in paths_to_compress:
            z.write(p, arcname=p.name)

    if delete_original:
        for p in paths_to_compress:
            p.unlink()

    return paths_to_compress


class DriveTransferError(RuntimeError):
    """Raised when an `rclone` transfer did not complete successfully."""


def to_drive(filepath, directory, remote='budrive', verbose=True):
    """Copy file to Google Drive

    Uses `rclone`. Remote 'drive' must exist: https://rclone.org/drive/

    Parameters
    ----------
    filepath : str
        Path of file to copy
    directory : str
        Drive folder to copy to
    remote : str
        Name of the `rclone` remote to copy to
    verbose : bool
        If True, print progress

    Raises
    ------
    DriveTransferError
        If `rclone` cannot be run, or exits non-zero (an expired token,
        a missing remote, an exhausted quota). Callers must not delete
        any local copy until this returns without raising.
    """

    cmd = ['rclone', 'copy', str(filepath), f'{remote}:{directory}']
    if verbose and sys.stdout.isatty():
        cmd += ['--progress']

    try:
        completed = subprocess.run(cmd, check=False)
    except OSError as error:
        raise DriveTransferError(
            f'Could not run `rclone` to upload {filepath}: {error}'
        ) from error

    if completed.returncode != 0:
        raise DriveTransferError(
            f'`rclone copy` failed with exit code {completed.returncode} '
            f'uploading {filepath} to {remote}:{directory}. No local file '
            'was deleted. Check that the remote exists and its token is '
            f'still valid: `rclone listremotes`, `rclone about {remote}:`.'
        )


def share(df, filepath, drive_dir=None, delete_original=True, verbose=True):
    """Shortcut for saving, compressing, and uploading to Drive

    File format is deduced from filepath extension.

    Drive folder is deduced from filepath and assumed to be in the
    `share` data directory (openplaces.config.cfg.share_dir)

    Parameters
    ----------
    df : DataFrame or GeoDataFrame
        Dataset to be saved
    filepath : pathlib.Path
        Filepath used for saving (and for the compressed ZIP file).
    delete_original : bool
        If True, deletes the unzipped file and the ZIP once the upload
        has succeeded.
    verbose : bool
        If True, prints statements ('Saving', 'compressing', etc.)

    Raises
    ------
    DriveTransferError
        If the upload fails. Every local copy is left in place, so a
        failed transfer never leaves the data existing nowhere.
    """

    if verbose:
        print('Saving...', end='')
    save(df, filepath)

    if verbose:
        print(' compressing...', end='')
    zip_path = filepath.parent / f'{filepath.stem}_{filepath.suffix[1:]}.zip'
    # The uncompressed file is kept until the upload has succeeded: if
    # it were deleted here and `to_drive` then failed, the ZIP would be
    # unlinked below and the data would exist nowhere.
    compressed = compress(filepath, zip_path, delete_original=False)

    if drive_dir is None:
        drive_dir = filepath.parent.relative_to(cfg.share_dir)
    if verbose:
        print(f' uploading to: {drive_dir} ...', end='')
    to_drive(zip_path, drive_dir, verbose=verbose)
    if verbose:
        print(' done!')

    if delete_original:
        for path in compressed:
            path.unlink(missing_ok=True)
        zip_path.unlink()
