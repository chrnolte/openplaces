"""
Generic downloader for a file published in a single public Google Drive
folder. Not tied to any recipe or dataset: the folder (via the source's
own `portal_url`) and the per-admin-unit/per-partition relative path
within it (via `scraper_options.file_path`, a template) are both supplied
by the recipe. Downloads exactly one file per call -- multi-file datasets
(e.g. several tables per admin unit) are handled by combining this
scraper with `download_by: {partition: table, table_names: [...]}`, one
`fetch()` call per (admin unit, table), not by this module.

`download_url_scraper`-based recipes get no unzip step from the Ingester
(that machinery -- `Ingester._download_and_unzip_recipe_data` -- is
specific to the plain `download_url` path; a scraper is documented to
hand back a directly readable file). So when the resolved Drive file is a
single-member zip, this scraper extracts it itself: the raw zip is kept,
flat and under its own real Drive basename, alongside the extracted
member written to `target_path` (the recipe's own `uncompressed_file_name`
-- both real filenames, no renaming beyond what the recipe already
declares).
"""

from __future__ import annotations

import json
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path

import requests

from openplaces.core.schema import AdminId
from openplaces.io.readers import get_admin

DEFAULT_TIMEOUT = 60.0
DEFAULT_RETRIES = 3


def _log(message: str) -> None:
    print(f'  {message}')


def _import_gdown():
    try:
        import gdown
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            'google_drive_scraper needs gdown, which is not installed.\n\n'
            'Install it into the openplaces environment:\n\n'
            '    pip install gdown\n'
        ) from exc
    return gdown


def _call_with_retry(fn, *, timeout, retries, verbose, label):
    """Run *fn* (a zero-arg callable) with a timeout and retry-with-backoff.

    Neither `gdown.download_folder()` nor `gdown.download()` accept a
    `timeout=` -- they build their own `requests.Session` with none set, so
    a slow/stalled connection can hang indefinitely instead of raising. This
    runs *fn* in a helper thread and bounds how long we wait on it; a
    timed-out call can't be cancelled mid-flight (the thread keeps running
    in the background), but the caller no longer hangs or has to guess why
    nothing came back. `gdown.exceptions.DownloadError`/`FileURLRetrievalError`
    (genuine permission/parsing failures) and `requests` network errors are
    retried the same as a timeout, since on a flaky connection either can be
    transient.
    """
    from gdown.exceptions import DownloadError, FileURLRetrievalError

    last_exc = None
    for attempt in range(1, retries + 1):
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(fn)
            try:
                return future.result(timeout=timeout)
            except (
                FutureTimeoutError,
                requests.exceptions.RequestException,
                DownloadError,
                FileURLRetrievalError,
            ) as exc:
                last_exc = exc
                if verbose:
                    kind = (
                        'timed out' if isinstance(exc, FutureTimeoutError) else str(exc)
                    )
                    _log(
                        f'{label}: attempt {attempt}/{retries} {kind} '
                        f'({timeout:.0f}s timeout), retrying...'
                        if attempt < retries
                        else f'{label}: attempt {attempt}/{retries} {kind}, giving up'
                    )
                if attempt < retries:
                    time.sleep(2**attempt)
    raise last_exc


# {(external_dir, drive_folder_url): (index, is_live)} -- per-process
# single-flight cache in front of the on-disk cache below. `is_live` marks
# whether the entry came from an actual gdown listing this process (vs. a
# disk-cache read that might still be stale); see `_folder_index`.
_FOLDER_INDEX_MEMO: dict[tuple[str, str], tuple[dict, bool]] = {}


def _folder_index(
    external_dir: Path,
    drive_folder_url: str,
    verbose: bool,
    redownload: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> dict:
    """Return {relative_path: file_id} for every file in the shared folder.

    Cached to *external_dir*/``gdrive_folder_index.json`` (a plain,
    human-readable name rather than the Drive folder id -- *external_dir* is
    already a recipe- and version-specific directory, so it only ever holds
    one Drive folder's index) so the (slow, rate-limited) folder listing
    survives across process runs. On top of that, a per-process memo
    (`_FOLDER_INDEX_MEMO`) guarantees the folder is never listed more than
    once within a single process, no matter how many admin units/partitions
    call this -- including when *redownload* is `True` on every call (e.g. a
    whole ingest run driven with `--redownload`), which would otherwise
    bypass the disk cache on every single call.

    gdown's folder listing has no pagination/completeness signal of its own,
    so a listing cut short by a network hiccup is still a "successful" call
    as far as this function can tell -- it gets cached and trusted forever,
    silently misreporting genuinely-published files as missing on every
    later run. *redownload* is the escape hatch: the first call in a process
    that requests it bypasses/overwrites a cached index with one fresh live
    listing instead of trusting one that might be stale or incomplete; every
    later call in that same process (regardless of its own *redownload*
    value) reuses that fresh result rather than listing again.
    """
    memo_key = (str(external_dir), drive_folder_url)
    memo_entry = _FOLDER_INDEX_MEMO.get(memo_key)
    if memo_entry is not None and (memo_entry[1] or not redownload):
        return memo_entry[0]

    cache_path = external_dir / 'gdrive_folder_index.json'
    if cache_path.exists() and not redownload:
        index = json.loads(cache_path.read_text())
        _FOLDER_INDEX_MEMO[memo_key] = (index, False)
        return index

    if verbose:
        _log(f'Listing Drive folder {drive_folder_url}...')
    files = _call_with_retry(
        lambda: _import_gdown().download_folder(
            drive_folder_url, skip_download=True, quiet=True
        ),
        timeout=timeout,
        retries=retries,
        verbose=verbose,
        label='folder listing',
    )
    index = {str(Path(f.path).as_posix()): f.id for f in files}
    external_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(index))
    _FOLDER_INDEX_MEMO[memo_key] = (index, True)
    if verbose:
        _log(f'Indexed {len(index)} files in Drive folder:')
        for rel in sorted(index):
            _log(f'  {rel}')
    return index


def _resolve_placeholders(
    file_path: str,
    admin_id_to_download: str | None,
    admin_key_column: str | None,
    partition_id,
) -> str:
    placeholders = {}
    if admin_id_to_download:
        for i, level in enumerate(str(admin_id_to_download).split('-'), start=1):
            placeholders[f'admin{i}'] = level
    if '{admin_key}' in file_path:
        if not (admin_id_to_download and admin_key_column):
            raise ValueError(
                "'{admin_key}' in file_path requires both admin_id_to_download "
                'and scraper_options.admin_key_column.'
            )
        admin_id = AdminId(admin_id_to_download)
        placeholders['admin_key'] = get_admin(
            admin_id, admin_id.get_level(), columns=admin_key_column
        ).iloc[0, 0]
    if partition_id is not None:
        placeholders['partition_id'] = str(partition_id)
    return file_path.format(**placeholders)


def _download_atomic(
    file_id: str,
    out_path: Path,
    verbose: bool,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> None:
    part_path = out_path.with_name(out_path.name + '.part')
    part_path.unlink(missing_ok=True)
    _call_with_retry(
        lambda: _import_gdown().download(
            id=file_id, output=str(part_path), quiet=not verbose
        ),
        timeout=timeout,
        retries=retries,
        verbose=verbose,
        label=f'download {out_path.name}',
    )
    part_path.replace(out_path)


def _extract_single_member(zip_path: Path, target_path: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        if len(members) != 1:
            raise ValueError(
                f'{zip_path.name} expected exactly one member, found '
                f'{len(members)}: {members}'
            )
        with tempfile.TemporaryDirectory() as tmp:
            zf.extract(members[0], tmp)
            Path(tmp, members[0]).replace(target_path)


def fetch(
    partition_id,
    target_path,
    portal_url=None,
    *,
    admin_id_to_download=None,
    file_path=None,
    admin_key_column=None,
    label=None,
    redownload=False,
    verbose=False,
    timeout=DEFAULT_TIMEOUT,
    retries=DEFAULT_RETRIES,
) -> Path | None:
    """Download one file from a public Google Drive folder.

    Parameters
    ----------
    partition_id : str or None
        Current partition id (e.g. a table name for `download_by:
        {partition: table}`); resolves `{partition_id}` in `file_path`.
    target_path : str or pathlib.Path
        Where the recipe expects to read the data from afterward (its own
        `uncompressed_file_name`, or `compressed_file_name` if the file
        needs no extraction). If the resolved Drive file is a single-member
        zip and `target_path` differs from that zip's own real basename,
        the zip is extracted into `target_path`; otherwise the download
        lands directly at `target_path`.
    portal_url : str
        The public Drive folder URL (resolved by the Ingester from the
        recipe's `source.portal_url` unless overridden via
        `scraper_options.portal_url`).
    admin_id_to_download : str, optional
        The current admin unit (e.g. `'US-MA-MI'`), used to resolve
        `{admin1}`/`{admin2}`/... and `{admin_key}` placeholders in
        `file_path`.
    file_path : str
        Relative-path template within the Drive folder, resolved with:
        `{admin1}`, `{admin2}`, `{admin3}`, ... -- the `-`-split levels of
        `admin_id_to_download`; `{admin_key}` -- looked up via
        `get_admin(admin_id_to_download, level, columns=admin_key_column)`
        (only when `{admin_key}` is used and `admin_key_column` is set --
        the same lookup `Ingester._get_admin_partition_key` itself uses,
        reused rather than reimplemented, for admin-derived keys that
        aren't literally one of `admin_id_to_download`'s own `-`-split
        levels, e.g. a county FIPS code); `{partition_id}` -- the raw
        partition id, passed straight through.
    admin_key_column : str, optional
        Column name to look up for `{admin_key}` (e.g. `'admin3_id_admin1'`
        for a county FIPS code -- see `Ingester._get_admin_partition_key`,
        `io/ingester/__init__.py:1190-1258`, for how that column name is
        derived from an `{adminN_id_adminM}`-style filename placeholder).
    label : str, optional
        Human-readable prefix for progress messages.
    redownload : bool, default False
        Delete and re-fetch the raw Drive file, and re-extract *target_path*
        from it, even if both already exist on disk. Also bypasses/refreshes
        the cached folder listing (see `_folder_index`) instead of trusting
        one that might be stale or incomplete from an earlier flaky run.
    verbose : bool, default False
        Print progress messages.
    timeout : float, default 60.0
        Seconds to wait on each folder-listing/download attempt before
        treating it as failed and retrying (gdown sets no timeout of its
        own, so a stalled connection would otherwise hang indefinitely).
    retries : int, default 3
        Attempts per folder-listing/download call before giving up and
        letting the underlying error propagate.

    Returns
    -------
    pathlib.Path or None
        Path to the readable file (equal to *target_path*), or `None` if
        no file in the Drive folder matches the resolved relative path
        (treated as "not yet published" -- signals the Ingester to skip
        this partition without aborting the rest of the run).
    """
    if not file_path:
        raise ValueError(
            "google_drive_scraper requires 'file_path' (recipe scraper_options)."
        )
    if not portal_url:
        raise ValueError(
            "google_drive_scraper requires a Drive folder URL, via the source's "
            "'portal_url'."
        )
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    external_dir = target_path.parent

    rel_path = _resolve_placeholders(
        file_path, admin_id_to_download, admin_key_column, partition_id
    )

    if verbose:
        prefix = label or 'google_drive_scraper'
        _log(f'{prefix}: fetching {rel_path}')

    index = _folder_index(
        external_dir,
        portal_url,
        verbose,
        redownload=redownload,
        timeout=timeout,
        retries=retries,
    )
    file_id = index.get(rel_path)
    if file_id is None:
        if verbose:
            _log(f'{rel_path}: not published, skipping')
        return None

    # The zip (or other raw file) keeps its own real Drive basename and
    # persists as a flat external artifact, independent of what the
    # recipe's own target_path is named.
    raw_path = external_dir / Path(rel_path).name
    if redownload and raw_path.exists():
        raw_path.unlink()
    if raw_path.exists():
        if verbose:
            _log(f'already downloaded ({raw_path.name})')
    else:
        _download_atomic(file_id, raw_path, verbose, timeout=timeout, retries=retries)
        if verbose:
            _log(f'{rel_path}: downloaded -> {raw_path.name}')

    if raw_path == target_path:
        return target_path
    if redownload and target_path.exists():
        target_path.unlink()
    if target_path.exists():
        if verbose:
            _log(f'already extracted ({target_path.name})')
        return target_path
    if raw_path.suffix.lower() != '.zip':
        raise ValueError(
            f'{raw_path.name} is not a zip, but target_path ({target_path.name}) '
            "differs from it -- nothing to extract. Set the recipe's "
            'uncompressed_file_name to match the real Drive filename.'
        )
    _extract_single_member(raw_path, target_path)
    if verbose:
        _log(f'{raw_path.name}: extracted -> {target_path.name}')
    return target_path
