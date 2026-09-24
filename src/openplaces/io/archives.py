"""
Unpack what was downloaded: zip, 7z, tar and nested archives, with
member selection and a common-prefix strip, and find the newest
usable file or geodatabase among what came out.
"""

import bz2
import fnmatch
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from zipfile import ZIP_BZIP2, ZIP_DEFLATED, ZIP_LZMA, ZIP_STORED, BadZipFile, ZipFile

from tqdm import tqdm

from openplaces.core.constants import (
    GEOPANDAS_EXTENSIONS,
    PANDAS_EXTENSIONS,
)

_NESTED_ARCHIVE_SUFFIXES = {'.zip'}

# A zip nested deeper than this is treated as malformed (or as a
# self-reproducing zip) rather than unpacked without end.
_MAX_ARCHIVE_NESTING = 5


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

    written = _extract_archive(in_path, out_dir, members, verbose)
    _extract_nested_archives(in_path, written, members, verbose)
    return out_dir


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
        all_members = z.namelist()
        member_list = [
            m
            for m in all_members
            if m.endswith('/')
            or Path(m).suffix.lower() in _NESTED_ARCHIVE_SUFFIXES
            or _member_selected(m, members)
        ]

        # Computed over every member, so a pattern selecting a single
        # file does not change the directory it lands in.
        strip_prefix = _get_strip_prefix(all_members)

        total_size = sum(z.getinfo(m).file_size for m in member_list)
        with tqdm(
            total=total_size, unit='B', unit_scale=True, desc='Extracting'
        ) as pbar:
            for member in member_list:
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
                    with z.open(member) as src, open(target, 'wb') as dst:
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

    # Find all files with matching extensions
    matching_files = [
        f
        for f in dir_path.iterdir()
        if f.is_file() and f.suffix.lower() in normalized_exts
    ]

    # Find all .gdb directories
    gdb_dirs = [
        d for d in dir_path.iterdir() if d.is_dir() and d.suffix.lower() == '.gdb'
    ]

    # Combine files and .gdb directories
    all_matches = matching_files + gdb_dirs

    if not all_matches:
        return None

    # Newest wins, except that a .json sidecar never beats a data file
    return max(
        all_matches, key=lambda f: (f.suffix.lower() != '.json', f.stat().st_mtime)
    )
