"""
Unpack what was downloaded: zip, 7z, tar and nested archives, with
member selection and a common-prefix strip, and find the newest
usable file or geodatabase among what came out.

An extraction folder (the heap) is shared: two ingest jobs whose
recipes read one archive extract it into the same folder, often at
the same time on a cluster. Every extraction is therefore staged in a
folder of its own beside the target and renamed into place entry by
entry once complete, so a reader sees an entry either whole or not at
all; a marker written last records that the set is complete; and a
job reading the folder holds a lease file in it, which the cleanup
after an ingest honors before deleting anything.

Why rename and not locks: the cluster's data root is a network
filesystem on which a rename within one directory is atomic while
advisory locks (fcntl, lock files that rely on them) are not reliable
across hosts. Every bookkeeping entry is created, renamed or deleted,
never locked, and is named with the host, process id and a random
token, so two jobs never write the same name.
"""

import bz2
import fnmatch
import hashlib
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from zipfile import ZIP_BZIP2, ZIP_DEFLATED, ZIP_LZMA, ZIP_STORED, BadZipFile, ZipFile

from tqdm import tqdm

from openplaces.config import cfg
from openplaces.core.constants import (
    GEOPANDAS_EXTENSIONS,
    PANDAS_EXTENSIONS,
)

_NESTED_ARCHIVE_SUFFIXES = {'.zip'}

# A zip nested deeper than this is treated as malformed (or as a
# self-reproducing zip) rather than unpacked without end.
_MAX_ARCHIVE_NESTING = 5

# Every name this module writes into an extraction folder starts with
# INTERNAL_PREFIX, and every reader of the folder skips such names.
INTERNAL_PREFIX = '.openplaces-'
_STAGING_PREFIX = INTERNAL_PREFIX + 'extract-'
_MARKER_PREFIX = INTERNAL_PREFIX + 'extracted-'
_LEASE_PREFIX = INTERNAL_PREFIX + 'reading-'
_ASIDE_PREFIX = INTERNAL_PREFIX + 'deleting-'

# A staging folder or lease left by a job on another host cannot be
# checked for a live process, so it counts as abandoned once it is
# older than this. The bound has to exceed the longest extraction or
# read of one partition; a day does with a wide margin, and the cost of
# a stale entry is only a redundant extraction or a heap kept a day
# longer. Overridden by `extraction_abandoned_after_hours` in config.
_ABANDONED_AFTER_HOURS_DEFAULT = 24


class ExtractionTargetInUseError(PermissionError):
    """An extracted entry from an earlier archive could not be replaced.

    Raised on Windows when another process holds the old entry open, so
    the new extraction cannot be renamed into place. The old entry is
    left as it was; nothing half written is left behind.
    """


def unzip(in_path, out_dir=None, members=None, verbose=True):
    """Extract files from an archive, including archives nested in it.

    Supports standard ZIP (deflate) and Deflate64 ZIP files, tar.gz,
    tar.bz2 and bare bz2. Deflate64 extraction requires 7z to be
    installed (see dev.py ensure_7zip()).

    A .zip member is extracted in turn, into the directory it was
    extracted to, and then deleted, down to any depth up to five levels.
    Sources ship zips of zips (a county roll with one zip per table; a
    file host wrapping a download in a zip of its own), and a reader
    cannot open the inner archive, so after this call the output
    directory holds the files a recipe names, as if the source had
    shipped one archive.

    The extraction is atomic per top-level entry. It runs in a staging
    folder inside `out_dir`, and each entry the archive produced (a
    file, or a folder such as a geodatabase) is renamed into `out_dir`
    only once the whole archive has been extracted, so no reader ever
    sees a half-written entry. A marker written after the last rename
    records the archive's size and modification time; a later call for
    the same archive and `members` finds it, sees every entry still in
    place, and returns without extracting again.

    Two calls extracting the same archive at once each extract their
    own copy. The first to finish renames its entries into place; the
    second keeps those and discards its copy. An entry left by an
    earlier, different archive (a new download of the source) is
    replaced.

    Parameters
    ----------
    in_path : str or Path
        Path to input archive.
    out_dir : str or Path, optional
        Output directory. If None, extracts to directory named after
        the zip file (without extension) in the same location.
        Example: 'data.zip' -> 'data/'
    members : list of str, optional
        Names or glob patterns (case-insensitive, matched against a
        member's full path or its file name) of the files to extract.
        If None, extracts all files. Nested .zip members are always
        opened, and the patterns then select among their members.
        Ignored when falling back to 7z and for tar archives.
    verbose : bool, default True
        If True, might print warnings, e.g. when switching to 7z.

    Returns
    -------
    Path
        Path to output directory

    Raises
    ------
    ValueError
        If two archives write the same file (a nested archive
        extracted beside its siblings would otherwise overwrite one
        table with another), or if archives nest more than five deep.
    ExtractionTargetInUseError
        If an entry left by a different archive is held open by another
        process and cannot be replaced (Windows only).

    Examples
    --------
    >>> unzip('data/raw/parcels.zip')  # -> data/raw/parcels/
    >>> unzip('data.zip', 'data/heap')  # -> data/heap/
    >>> unzip('data.zip', members=['file1.txt', '*_INFO.TXT'])
    """
    in_path = Path(in_path)
    if not in_path.exists():
        raise FileNotFoundError(f'Zip file not found: {in_path}')
    if out_dir is None:
        out_dir = in_path.parent / in_path.stem
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    identity = _archive_identity(in_path, members)
    done = _completed_entries(out_dir, identity)
    if done is not None:
        # Touched so that find_latest_file_or_gdb, which picks the
        # newest entry in a shared folder, still picks this one.
        for entry in done:
            _touch(entry)
        if verbose:
            print(f'{in_path.name} is already extracted in {out_dir}; reusing it.')
        return out_dir

    _sweep_abandoned(out_dir)
    peers = _live_entries(out_dir, _STAGING_PREFIX)
    if peers:
        # Printed whatever `verbose` says: two jobs paying for one
        # extraction is worth seeing in a cluster log.
        print(
            f'Another job is extracting into {out_dir} ({peers[0].name}). '
            f'Extracting a separate copy of {in_path.name}; whichever '
            'finishes first is kept.'
        )
    before = {p.name for p in out_dir.iterdir() if not is_internal(p, out_dir)}
    staging = out_dir / f'{_STAGING_PREFIX}{_owner_tag()}'
    staging.mkdir()
    try:
        written = _extract_archive(in_path, staging, members, verbose)
        _extract_nested_archives(in_path, written, members, verbose)
        entries = _publish(staging, out_dir, identity, before)
        _write_marker(out_dir, identity, entries)
    finally:
        # Holds only the copies another job had already published.
        shutil.rmtree(staging, ignore_errors=True)
    return out_dir


def is_internal(path, root):
    """Return True if `path` is bookkeeping written by this module.

    Parameters
    ----------
    path : Path
        A path at or below `root`.
    root : Path
        The extraction folder.

    Returns
    -------
    bool
        True when any component of `path` below `root` is a staging
        folder, marker, lease or folder set aside for deletion.
    """
    try:
        parts = Path(path).relative_to(root).parts
    except ValueError:
        return False
    return any(part.startswith(INTERNAL_PREFIX) for part in parts)


def extracted_path_ready(path, out_dir):
    """Return True if `path` exists and may be read as complete.

    An entry an extraction renamed into place is complete as soon as it
    exists. Until the marker naming it is written, though, entries of
    the same archive may still be on their way in (a shapefile's .shp
    before its .dbf), so an entry no marker names is trusted only when
    no live staging folder sits beside it. Folders extracted before
    markers existed have neither, and are trusted as before: requiring
    a marker would re-extract every existing heap once, and those were
    written when only one job ran per archive.

    Parameters
    ----------
    path : Path
        The file or folder a recipe reads.
    out_dir : Path or None
        The extraction folder. A `path` outside it is ready when it
        exists.

    Returns
    -------
    bool
    """
    path = Path(path)
    if not path.exists():
        return False
    if out_dir is None or not path.is_relative_to(out_dir):
        return True
    top = path.relative_to(out_dir).parts[0]
    if any(top in marker.get('entries', []) for marker in _read_markers(out_dir)):
        return True
    return not _live_entries(out_dir, _STAGING_PREFIX)


def acquire_read_lease(out_dir):
    """Record that this process reads from `out_dir`.

    The lease is an empty file named with the host, process id and a
    random token. `delete_extracted` in another job leaves the folder
    alone while it exists.

    Parameters
    ----------
    out_dir : Path
        The extraction folder; created if missing.

    Returns
    -------
    Path
        The lease, to pass to `release_read_lease`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lease = out_dir / f'{_LEASE_PREFIX}{_owner_tag()}'
    lease.touch(exist_ok=False)
    return lease


def release_read_lease(lease, stop_at=None):
    """Remove a lease, and the folders it leaves empty.

    Parameters
    ----------
    lease : Path or None
        The lease `acquire_read_lease` returned. None does nothing.
    stop_at : Path, optional
        Empty parent folders are removed up to, not including, this
        one. None removes none.
    """
    if lease is None:
        return
    lease = Path(lease)
    lease.unlink(missing_ok=True)
    if stop_at is None:
        return
    folder = lease.parent
    while folder != Path(stop_at) and folder.is_relative_to(stop_at):
        try:
            folder.rmdir()
        except OSError:
            # Not empty (another job's entries or lease), or held open.
            break
        folder = folder.parent


def other_readers(out_dir, own_lease=None):
    """Return the live leases of other jobs in `out_dir`.

    Parameters
    ----------
    out_dir : Path
        The extraction folder.
    own_lease : Path, optional
        This job's lease, left out of the result.

    Returns
    -------
    list of Path
    """
    own = Path(own_lease).name if own_lease is not None else None
    return [p for p in _live_entries(out_dir, _LEASE_PREFIX) if p.name != own]


def delete_extracted(paths, out_dir, own_lease=None, verbose=True):
    """Delete extracted entries unless another job is reading the folder.

    Each path is first renamed into a folder set aside inside `out_dir`,
    so a reader sees it whole or not at all, never half deleted. The
    leases are checked before and again after the renames; a lease
    that appeared in between moves the entries back. A set-aside folder
    that cannot be removed (a sync app or a reader holding a file) is
    left for a later call: no reader opens it, so it cannot be mistaken
    for data.

    Parameters
    ----------
    paths : iterable of Path
        Files or folders inside `out_dir`. A shapefile takes its
        sidecar files with it. Missing paths are skipped.
    out_dir : Path
        The extraction folder.
    own_lease : Path, optional
        This job's lease, which does not count as another reader.
    verbose : bool, default True
        Print what was deleted.

    Returns
    -------
    bool
        False if another job's lease kept everything in place.
    """
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return True
    readers = other_readers(out_dir, own_lease)
    if readers:
        print(
            f'Kept the extracted data in {out_dir}: another job is reading it '
            f'({readers[0].name}).'
        )
        return False
    targets = []
    for path in paths:
        path = Path(path)
        is_shapefile = path.suffix.lower() == '.shp'
        targets.extend(_with_sidecars(path) if is_shapefile else [path])
    targets = [p for p in dict.fromkeys(targets) if os.path.lexists(p)]
    if not targets:
        return True

    aside = out_dir / f'{_ASIDE_PREFIX}{_owner_tag()}'
    aside.mkdir()
    moved = []
    try:
        for index, path in enumerate(targets):
            destination = aside / f'{index}-{path.name}'
            os.rename(path, destination)
            moved.append((path, destination))
    except OSError as error:
        _restore(moved)
        shutil.rmtree(aside, ignore_errors=True)
        print(f'Kept the extracted data in {out_dir}: {error}')
        return False
    readers = other_readers(out_dir, own_lease)
    if readers:
        _restore(moved)
        shutil.rmtree(aside, ignore_errors=True)
        print(
            f'Kept the extracted data in {out_dir}: another job started '
            f'reading it ({readers[0].name}).'
        )
        return False

    # The markers name what is gone now; a later call re-extracts.
    tops = {p.relative_to(out_dir).parts[0] for p in targets}
    for marker_path in out_dir.glob(f'{_MARKER_PREFIX}*'):
        marker = _read_json(marker_path)
        if marker is None or tops & set(marker.get('entries', [])):
            marker_path.unlink(missing_ok=True)
    if verbose:
        print('Deleting unzipped data.')
    shutil.rmtree(aside, ignore_errors=True)
    for path in targets:
        _remove_empty_parents(path.parent, out_dir)
    return True


def _owner_tag():
    """Host, process id and a random token, safe in a file name."""
    host = re.sub(r'[^A-Za-z0-9]', '_', socket.gethostname())[:40] or 'host'
    return f'{host}-{os.getpid()}-{secrets.token_hex(4)}'


def _owner(name, prefix):
    """Return (host, pid) from a name `_owner_tag` built, or None."""
    parts = name.removeprefix(prefix).rsplit('-', 2)
    if len(parts) != 3 or not parts[1].isdigit():
        return None
    return parts[0], int(parts[1])


def _pid_alive(pid):
    try:
        import psutil
    except ImportError:
        return True
    return psutil.pid_exists(pid)


def _abandoned_after_seconds():
    try:
        hours = cfg.extraction_abandoned_after_hours
    except AttributeError:
        hours = _ABANDONED_AFTER_HOURS_DEFAULT
    return float(hours) * 3600


def _is_abandoned(path, prefix):
    """True if the job that named `path` is gone, or it is too old."""
    owner = _owner(path.name, prefix)
    this_host = re.sub(r'[^A-Za-z0-9]', '_', socket.gethostname())[:40]
    if owner is not None and owner[0] == this_host and not _pid_alive(owner[1]):
        return True
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return True
    return age > _abandoned_after_seconds()


def _live_entries(out_dir, prefix):
    """Entries named with `prefix` whose owner is still running."""
    try:
        found = sorted(Path(out_dir).glob(f'{prefix}*'))
    except OSError:
        return []
    return [p for p in found if not _is_abandoned(p, prefix)]


def _sweep_abandoned(out_dir):
    """Remove staging folders and leases whose job is gone.

    Set-aside folders are removed too when no other job holds a lease,
    since nothing reads them by name. Failures are ignored: a sweep is
    housekeeping, and a later call tries again.
    """
    for prefix in (_STAGING_PREFIX, _LEASE_PREFIX):
        for path in Path(out_dir).glob(f'{prefix}*'):
            if _is_abandoned(path, prefix):
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
    if not _live_entries(out_dir, _LEASE_PREFIX):
        for path in Path(out_dir).glob(f'{_ASIDE_PREFIX}*'):
            shutil.rmtree(path, ignore_errors=True)


def _archive_identity(in_path, members):
    """What a marker records: the archive's name, size, mtime, members."""
    stat = in_path.stat()
    if isinstance(members, str):
        members = [members]
    return {
        'archive': in_path.name,
        'size': stat.st_size,
        'mtime_ns': stat.st_mtime_ns,
        'members': sorted(members) if members else None,
    }


def _marker_path(out_dir, identity):
    key = json.dumps([identity['archive'], identity['members']])
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    return out_dir / f'{_MARKER_PREFIX}{digest}'


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def _read_markers(out_dir):
    markers = (_read_json(p) for p in Path(out_dir).glob(f'{_MARKER_PREFIX}*'))
    return [m for m in markers if isinstance(m, dict)]


def _same_archive(marker, identity):
    return all(
        marker.get(key) == identity[key] for key in ('archive', 'size', 'mtime_ns')
    )


def _completed_entries(out_dir, identity):
    """The entries of a complete earlier extraction, or None."""
    marker = _read_json(_marker_path(out_dir, identity))
    if not isinstance(marker, dict) or not _same_archive(marker, identity):
        return None
    if marker.get('members') != identity['members']:
        return None
    entries = [out_dir / name for name in marker.get('entries', [])]
    if not entries or not all(os.path.lexists(e) for e in entries):
        return None
    return entries


def _write_marker(out_dir, identity, entries):
    """Write the marker last, by rename, so it is never half written."""
    fd, temp = tempfile.mkstemp(prefix=f'{INTERNAL_PREFIX}marker-', dir=out_dir)
    with os.fdopen(fd, 'w', encoding='utf-8') as handle:
        json.dump({**identity, 'entries': sorted(entries)}, handle)
    os.replace(temp, _marker_path(out_dir, identity))


def _publish(staging, out_dir, identity, before):
    """Rename each entry of `staging` into `out_dir`; return their names.

    An entry already in `out_dir` is kept when it came from this same
    archive: a marker of the same archive names it, or it was not there
    when this extraction started, so another job extracting the same
    archive put it there meanwhile. Anything else is left over from a
    different archive, and is replaced.
    """
    same = set()
    for marker in _read_markers(out_dir):
        if _same_archive(marker, identity):
            same.update(marker.get('entries', []))
    names = []
    for entry in sorted(staging.iterdir()):
        target = out_dir / entry.name
        names.append(entry.name)
        if not os.path.lexists(target):
            try:
                # A rename, not a copy: atomic on one filesystem,
                # which is why the staging folder sits in out_dir.
                os.rename(entry, target)
                continue
            except OSError:
                if not os.path.lexists(target):
                    raise
        if entry.name in same or entry.name not in before:
            print(f'Kept {target}: already extracted from {identity["archive"]}.')
            continue
        _replace(entry, target, out_dir)
    return names


def _replace(entry, target, out_dir):
    """Swap a stale `target` for `entry` without a half-written moment.

    os.replace cannot put a folder over a non-empty one on any platform,
    so a folder is swapped in two renames: the old one aside, the new
    one in. Between them the target is absent, never partial.
    """
    in_use = ExtractionTargetInUseError(
        f'Could not replace {target} with a new extraction: it is in use, '
        'most likely held open by another process (another ingest of the '
        'same source, or a sync app). The existing entry is unchanged. '
        'Re-run once that process has finished.'
    )
    if not entry.is_dir() or entry.is_symlink():
        try:
            os.replace(entry, target)
        except PermissionError as error:
            raise in_use from error
        return
    aside = out_dir / f'{_ASIDE_PREFIX}{_owner_tag()}'
    aside.mkdir()
    try:
        os.rename(target, aside / target.name)
    except PermissionError as error:
        aside.rmdir()
        raise in_use from error
    try:
        os.rename(entry, target)
    except OSError:
        _restore([(target, aside / target.name)])
        raise
    # Deleted now unless another job reads this folder; otherwise the
    # next sweep without readers removes it.
    if not _live_entries(out_dir, _LEASE_PREFIX):
        shutil.rmtree(aside, ignore_errors=True)


def _touch(path):
    try:
        os.utime(path)
    except OSError:
        pass


def _with_sidecars(shp_path):
    """A shapefile and every file beside it sharing its stem."""
    stem = shp_path.stem.lower()
    try:
        return [p for p in shp_path.parent.iterdir() if p.stem.lower() == stem]
    except OSError:
        return [shp_path]


def _restore(moved):
    for original, destination in reversed(moved):
        if not os.path.lexists(original):
            try:
                os.rename(destination, original)
            except OSError:
                pass


def _remove_empty_parents(folder, stop_at):
    while folder != stop_at and folder.is_relative_to(stop_at):
        try:
            folder.rmdir()
        except OSError:
            break
        folder = folder.parent


def _extract_archive(in_path, out_dir, members, verbose):
    """Extract one archive level and return the files it wrote."""
    suffixes = [s.lower() for s in in_path.suffixes]
    _is_tar_gz = in_path.suffix.lower() == '.tgz' or (
        len(suffixes) >= 2 and suffixes[-2] == '.tar' and suffixes[-1] == '.gz'
    )
    if _is_tar_gz:
        with tarfile.open(in_path, 'r:gz') as tar:
            tar.extractall(out_dir)
            return [out_dir / m.name for m in tar.getmembers() if m.isfile()]

    if in_path.suffix.lower() in {'.bz2', '.tbz2'}:
        # Casefolded like the tar.gz test above: uppercase extensions
        # are routine in county and state archives.
        if in_path.stem.lower().endswith('.tar') or in_path.suffix.lower() == '.tbz2':
            with tarfile.open(in_path, 'r:bz2') as tar:
                tar.extractall(out_dir)
                return [out_dir / m.name for m in tar.getmembers() if m.isfile()]
        # Handle bare .bz2 (single compressed file)
        out_file = out_dir / in_path.stem
        with bz2.open(in_path, 'rb') as src, out_file.open('wb') as dst:
            shutil.copyfileobj(src, dst)
        return [out_file]

    if _needs_7z(in_path):
        if members and verbose:
            print(f'Extracting every member of {in_path.name}: 7z ignores `members`.')
        return _unzip_with_7z(in_path, out_dir, verbose)
    return _unzip_standard(in_path, out_dir, members)


def _is_nested_archive(path):
    return path.suffix.lower() in _NESTED_ARCHIVE_SUFFIXES and path.is_file()


def _extract_nested_archives(in_path, written, members, verbose):
    """Extract every .zip among `written`, recursively, then delete it.

    Each inner archive is extracted into its own directory rather than
    into a subdirectory named after it, so a recipe names the file it
    reads exactly as it would in a single-level archive. The price is
    that sibling archives share a directory; a file written twice is
    therefore an error, not a silent overwrite.
    """
    origin = {path: in_path.name for path in written}
    pending = [(path, 1) for path in written if _is_nested_archive(path)]
    while pending:
        archive, depth = pending.pop(0)
        if depth > _MAX_ARCHIVE_NESTING:
            raise ValueError(
                f'{archive.name} is nested more than {_MAX_ARCHIVE_NESTING} '
                f'archives deep inside {in_path.name}.'
            )
        if verbose:
            print(f'Extracting nested archive {archive.name}...')
        inner = _extract_archive(archive, archive.parent, members, verbose)
        for path in inner:
            if origin.get(path, archive.name) != archive.name:
                raise ValueError(
                    f'{path.name} is in both {origin[path]} and {archive.name} '
                    f'(inside {in_path.name}); extracting both into '
                    f'{archive.parent} would overwrite one with the other.'
                )
            origin[path] = archive.name
        # The inner archive is only a copy of bytes still inside the
        # source archive; keeping it would double the heap's size.
        archive.unlink()
        pending.extend((path, depth + 1) for path in inner if _is_nested_archive(path))


def _member_selected(name, patterns):
    """Return True if a zip member name matches one of `patterns`."""
    if not patterns:
        return True
    lowered = name.lower()
    base = lowered.rsplit('/', 1)[-1]
    for pattern in patterns:
        pattern = pattern.lower()
        if fnmatch.fnmatchcase(lowered, pattern) or fnmatch.fnmatchcase(base, pattern):
            return True
    return False


def _needs_7z(zip_path):
    """Return True if Python's zipfile cannot extract the archive.

    Either because zipfile cannot open the container at all (a 7z, rar
    or bare gz file), or because an entry inside a real zip uses a
    compression type zipfile cannot deflate. Probing the container
    first would leave the whole 7z fallback unreachable for the first
    case, which then failed with a misleading 'not a zip file'.
    """
    try:
        with ZipFile(zip_path, 'r') as z:
            return any(
                info.compress_type
                not in {
                    ZIP_STORED,
                    ZIP_DEFLATED,
                    ZIP_BZIP2,
                    ZIP_LZMA,
                }
                for info in z.infolist()
            )
    except BadZipFile:
        return True


def _unzip_standard(in_path, out_dir, members):
    """Extract using Python's zipfile (deflate and store).

    Returns the paths of the files written. `members` holds names or
    glob patterns; nested .zip members are kept whatever it says, so
    that the patterns can select among their members in turn.
    """
    written = []
    with ZipFile(in_path, 'r') as z:
        # Some archives made on Windows store '\' as the separator
        # (Florida DOR's 2002 roll: '2002F\NAL_2002_11Alachua_F.csv').
        # Windows reads it as a folder; Linux and macOS would write one
        # file with a backslash in its name. Paths use '/' throughout;
        # the stored name is kept only to read the member.
        names = {m: m.replace('\\', '/') for m in z.namelist()}
        all_members = list(names.values())
        member_list = [
            m
            for m in names
            if names[m].endswith('/')
            or Path(names[m]).suffix.lower() in _NESTED_ARCHIVE_SUFFIXES
            or _member_selected(names[m], members)
        ]

        # Computed over every member, so a pattern selecting a single
        # file does not change the directory it lands in.
        strip_prefix = _get_strip_prefix(all_members)

        total_size = sum(z.getinfo(m).file_size for m in member_list)
        with tqdm(
            total=total_size, unit='B', unit_scale=True, desc='Extracting'
        ) as pbar:
            for stored in member_list:
                member = names[stored]
                if strip_prefix and member == strip_prefix.rstrip('/'):
                    continue
                if strip_prefix and member.startswith(strip_prefix):
                    relative_path = member.removeprefix(strip_prefix)
                    if not relative_path:
                        continue
                    target = out_dir / relative_path
                else:
                    target = out_dir / member
                if member.endswith('/'):
                    if not members:
                        target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(stored) as src, open(target, 'wb') as dst:
                        while chunk := src.read(8192):
                            dst.write(chunk)
                            pbar.update(len(chunk))
                    written.append(target)
    return written


def _find_7z():
    """Return path to 7z executable, checking known install locations."""
    path = shutil.which('7z')
    if path:
        return path
    if sys.platform == 'win32':
        default = r'C:\Program Files\7-Zip\7z.exe'
        if Path(default).exists():
            return default
    return None


def _unzip_with_7z(in_path, out_dir, verbose=True):
    """Extract using 7z for compression types unsupported by zipfile.

    Returns the paths of the files written, as far as they can be
    listed: a zip container zipfile opens (a Deflate64 entry) is listed
    by name, while a 7z or rar container is not, so archives nested in
    one are left unextracted.
    """
    sz = _find_7z()
    if not sz:
        raise RuntimeError(
            f'Archive {in_path.name} cannot be extracted by `zipfile` (an '
            'unsupported compression type, or not a zip container at all). '
            'Install 7z: brew install sevenzip (macOS), '
            'winget install 7zip.7zip (Windows), or sudo apt install 7zip (Linux).'
        )
    if verbose:
        print('Extracting with 7z...')
    result = subprocess.run(
        [sz, 'x', str(in_path), f'-o{out_dir}', '-y'],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise BadZipFile(f'7z failed to extract {in_path.name}:\n{result.stderr}')
    try:
        with ZipFile(in_path, 'r') as z:
            return [out_dir / m for m in z.namelist() if not m.endswith('/')]
    except BadZipFile:
        return []


def _get_strip_prefix(member_list):
    """Return common top-level directory prefix to strip, or None."""
    top_level_items = set()
    for m in member_list:
        if m:
            top_level_items.add(m.split('/')[0])
    if len(top_level_items) == 1:
        common_prefix = next(iter(top_level_items))
        has_nested = any('/' in m for m in member_list)
        if has_nested and not common_prefix.endswith('.gdb'):
            return f'{common_prefix}/'
    return None


def find_latest_file_or_gdb(
    directory: str, extensions: set[str] = GEOPANDAS_EXTENSIONS | PANDAS_EXTENSIONS
) -> Path | None:
    """
    Find the most recently modified file or .gdb directory in a directory.

    The newest candidate wins, with one exception: a .json file is chosen
    only when nothing else qualifies. An archive that ships a data file
    often ships a metadata or schema JSON beside it, and the newer of the
    two is not the data. Newest-wins is otherwise kept deliberately: the
    heap directory is shared across partitions and reruns, and ingest
    never clears it before extracting, so the file just extracted must
    beat a stale one of any other format.

    Suffixes are compared case-insensitively. County and state shapefile
    archives routinely ship uppercase extensions, and matching them by
    raw case would leave the data file unrecognized and hand the caller
    the JSON sidecar the rule above exists to demote.

    Parameters
    ----------
    directory : str
        Path to the directory to search
    extensions : set[str]
        Accepted file extensions (e.g., {'.csv', '.txt', '.json'})

    Returns
    -------
    Path | None
        Path to the most recent file or .gdb directory, or None if no matches found
    """
    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        raise ValueError(f'Directory does not exist: {directory}')

    # Normalize extensions to include a leading dot, and casefold so an
    # uppercase suffix in the archive still matches
    normalized_exts = {
        (ext if ext.startswith('.') else f'.{ext}').lower() for ext in extensions
    }

    # Staging folders, markers and leases are never data
    children = [p for p in dir_path.iterdir() if not p.name.startswith(INTERNAL_PREFIX)]

    # Find all files with matching extensions
    matching_files = [
        f for f in children if f.is_file() and f.suffix.lower() in normalized_exts
    ]

    # Find all .gdb directories
    gdb_dirs = [d for d in children if d.is_dir() and d.suffix.lower() == '.gdb']

    # Combine files and .gdb directories
    all_matches = matching_files + gdb_dirs

    if not all_matches:
        return None

    # Newest wins, except that a .json sidecar never beats a data file
    return max(
        all_matches, key=lambda f: (f.suffix.lower() != '.json', f.stat().st_mtime)
    )
