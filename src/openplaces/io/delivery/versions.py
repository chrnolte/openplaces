"""
Versions of a delivered bundle: a manifest, an archive and an index.

A bundle keeps its fixed path in the share bucket (`delivery_paths`),
so every reader and the orchestrator find the current one where they
always did. What changes is what happens to the bundle a re-export
replaces, and what a bundle says about itself:

- The current bundle carries a `manifest.json` beside its files: its
  version, when and from which commit it was built, the recipe, the
  imputation model token if any, and each file's name, size, row count
  and SHA-256.
- Before `export_delivery` replaces a bundle in the share bucket, it
  moves the old files to the archive, under `versions/<version>/` at
  the bundle's own path mirrored below the archive root, and refuses
  to write the new bundle if that move fails.
- `versions/index.json` beside the current bundle lists every version
  of it (the manifest's fields plus `status`, `location` and `path`),
  and `versions/VERSIONS.md` is the same list for a person to read.

A version is named by the day the export ran (`2026-11-17`); a second
export that day is `2026-11-17b`, then `c`. A dated name sorts in
build order and says when it was built, which a counter (`0.7`, `0.8`)
does not (maintainer, 2026-11-15). The imputation model token keeps
naming the model, since a filled cell's provenance has to name the
model that filled it, not the bundle it shipped in.

The share bucket takes redacted bundles only (`io.tables`,
`UnredactedShareWriteError`). The manifest and the index are part of a
bundle: they hold file names, sizes, row counts and checksums, never a
value of the table, so no personal or restricted column can appear in
them. They are written as JSON and Markdown by this module, which the
table guard does not cover and does not need to.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import shutil
import stat
from collections.abc import Iterable
from pathlib import Path

MANIFEST_NAME = 'manifest.json'
VERSIONS_DIR = 'versions'
INDEX_NAME = 'index.json'
VERSIONS_MD = 'VERSIONS.md'

# The user config key naming the archive root, and the folder under
# the data root that serves as one when the key is unset. The archive
# mirrors the data root by bucket name (`<archive>/share/...`), as the
# copies made by hand with scripts/storage/snapshot.py already do.
ARCHIVE_KEY = 'archive'
ARCHIVE_DIR = 'archive'

STATUSES = ('current', 'superseded', 'withdrawn')
LOCATIONS = ('share', 'archive', 'drive')
INDEX_SCHEMA = 1

# Fields a manifest holds. An index row adds status, location and path,
# which change after the build and so are not part of the manifest.
MANIFEST_FIELDS = (
    'version',
    'built',
    'commit',
    'dirty',
    'package_version',
    'recipes',
    'imputation',
    'region',
    'admin_id',
    'audience',
    'files',
    'note',
)
ROW_FIELDS = (*MANIFEST_FIELDS, 'status', 'location', 'path')

_VERSION = re.compile(r'^(\d{4}-\d{2}-\d{2})([b-z]?)$')
_CHUNK = 8 * 1024 * 1024


class ArchiveError(RuntimeError):
    """Raised when the bundle a re-export would replace cannot be archived."""


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, 'rb', buffering=0) as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def parquet_rows(path: Path) -> int | None:
    """Return a parquet file's row count from its footer, or None."""
    import pyarrow.parquet as pq

    try:
        return pq.read_metadata(path).num_rows
    except Exception:
        return None


def file_entry(path: Path, name: str | None = None, checksum: bool = True) -> dict:
    """Describe one bundle file: name, bytes, rows and SHA-256.

    Parameters
    ----------
    path : pathlib.Path
        The file to describe.
    name : str, optional
        The name to record, when the file sits under a temporary name
        (a staged `.partial` file). Defaults to the file's own name.
    checksum : bool, optional
        Compute the SHA-256. When False, `sha256` is None.

    Returns
    -------
    dict
        Keys `name`, `bytes`, `rows` (None for a file that is not a
        table) and `sha256`.
    """
    path = Path(path)
    name = name or path.name
    return {
        'name': name,
        'bytes': path.stat().st_size,
        'rows': parquet_rows(path) if name.endswith('.parquet') else None,
        'sha256': sha256_of(path) if checksum else None,
    }


def parse_version(version: str) -> tuple[dt.date, str]:
    """Split a version name into its date and its letter ('' for none).

    Raises
    ------
    ValueError
        When *version* is not a date, optionally followed by one letter
        from b to z.
    """
    match = _VERSION.match(str(version))
    if not match:
        raise ValueError(
            f'{version!r} is not a version name: expected a build date '
            "('2026-11-17'), optionally with a letter ('2026-11-17b')."
        )
    return dt.date.fromisoformat(match.group(1)), match.group(2)


def next_version(day: dt.date, taken: Iterable[str]) -> str:
    """Return the version name for a build on *day*.

    The date alone when no version of that day exists yet, else the
    date with the next letter: `2026-11-17`, `2026-11-17b`, `...c`.

    Raises
    ------
    ValueError
        After `z`: 25 exports of one bundle on one day.
    """
    base = day.isoformat()
    taken = set(taken)
    if base not in taken:
        return base
    for code in range(ord('b'), ord('z') + 1):
        candidate = f'{base}{chr(code)}'
        if candidate not in taken:
            return candidate
    raise ValueError(f'Every version name of {base} is taken, up to {base}z.')


def build_manifest(
    *,
    version: str,
    built: str,
    files: list[dict],
    recipe_id: str | None = None,
    region: str | None = None,
    admin_id: str | None = None,
    audience: str | None = None,
    imputation: str | None = None,
    commit: str | None = None,
    dirty: bool | None = None,
    package_version: str | None = None,
    note: str | None = None,
) -> dict:
    """Assemble a manifest from its fields, in the documented order."""
    parse_version(version)
    return {
        'version': version,
        'built': built,
        'commit': commit,
        'dirty': dirty,
        'package_version': package_version,
        'recipes': [recipe_id] if recipe_id else [],
        'imputation': imputation,
        'region': region,
        'admin_id': str(admin_id) if admin_id is not None else None,
        'audience': audience,
        'files': files,
        'note': note,
    }


def code_identity() -> dict:
    """The package version and commit of the code running the export.

    The commit comes from the checkout (`io.delivery._code_revision`,
    which also reads a cluster job's stated revision), else from the
    bundled recipe root's record, which a parquet footer carries too.
    """
    from openplaces.io.delivery import _code_revision
    from openplaces.path import recipe_root_records

    record = recipe_root_records()[0]
    revision = _code_revision() or {}
    return {
        'commit': revision.get('commit') or record.get('commit'),
        'dirty': revision.get('dirty'),
        'package_version': record.get('version'),
    }


def footer_identity(path: Path) -> dict:
    """The package version and commit a parquet footer records, if any."""
    import pyarrow.parquet as pq

    from openplaces.path import RECIPE_ROOTS_METADATA_KEY

    try:
        metadata = pq.read_metadata(path).metadata or {}
        records = json.loads(metadata[RECIPE_ROOTS_METADATA_KEY.encode()])
    except Exception:
        return {}
    for record in records:
        if record.get('provider') == 'bundled':
            return {
                'commit': record.get('commit'),
                'package_version': record.get('version'),
            }
    return {}


def timestamp(seconds: float | None = None) -> str:
    """An ISO timestamp with the local offset, to the second."""
    moment = (
        dt.datetime.now().astimezone()
        if seconds is None
        else dt.datetime.fromtimestamp(seconds).astimezone()
    )
    return moment.isoformat(timespec='seconds')


def _write_json(path: Path, payload) -> None:
    """Write JSON beside its destination, then move it into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.partial')
    partial.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    _writable(path)
    os.replace(partial, path)


def _writable(path: Path) -> None:
    if path.exists():
        path.chmod(path.stat().st_mode | stat.S_IWUSR)


def _read_only(path: Path) -> None:
    if path.exists():
        path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def read_manifest(folder: Path) -> dict | None:
    """Return the manifest in *folder*, or None when there is none."""
    path = Path(folder) / MANIFEST_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def write_manifest(folder: Path, manifest: dict) -> Path:
    """Write *manifest* into *folder*, read-only like the bundle files."""
    path = Path(folder) / MANIFEST_NAME
    _write_json(path, {key: manifest.get(key) for key in MANIFEST_FIELDS})
    _read_only(path)
    return path


def versions_dir(canonical: Path) -> Path:
    """The `versions/` folder beside a bundle whose canonical file is given."""
    return Path(canonical).parent / VERSIONS_DIR


def empty_index(**meta) -> dict:
    """An index with no versions, carrying the bundle's identity."""
    return {
        'schema': INDEX_SCHEMA,
        'bundle': meta.get('bundle'),
        'recipe_id': meta.get('recipe_id'),
        'region': meta.get('region'),
        'admin_id': str(meta['admin_id']) if meta.get('admin_id') else None,
        'versions': [],
    }


def read_index(folder: Path) -> dict | None:
    """Return the index in a `versions/` folder, or None when there is none."""
    path = Path(folder) / INDEX_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def version_key(version: str) -> tuple:
    """Sort key putting versions in build order; unparseable names last."""
    try:
        day, letter = parse_version(version)
    except ValueError:
        return (dt.date.max, str(version))
    return (day, letter)


def _partial(path: Path) -> Path:
    return path.with_name(path.name + '.partial')


def stage_index(folder: Path, index: dict) -> dict:
    """Write an index and its `VERSIONS.md` beside their files, unrenamed.

    Returns
    -------
    dict
        Each final path mapped to the `.partial` file holding its text;
        `install` moves them into place.
    """
    folder = Path(folder)
    index['versions'] = sorted(
        index['versions'], key=lambda row: version_key(row['version'])
    )
    for row in index['versions']:
        _check_row(row)
    folder.mkdir(parents=True, exist_ok=True)
    texts = {
        folder / VERSIONS_MD: render_versions_md(index),
        # Last, so the index changes only once its Markdown has.
        folder / INDEX_NAME: json.dumps(index, indent=2) + '\n',
    }
    staged = {}
    for path, text in texts.items():
        _partial(path).write_text(text, encoding='utf-8')
        staged[path] = _partial(path)
    return staged


def install(staged: dict) -> None:
    """Move staged texts onto their final paths, in order."""
    for path, partial in staged.items():
        _writable(path)
        os.replace(partial, path)


def write_index(folder: Path, index: dict) -> Path:
    """Write `index.json` and the `VERSIONS.md` generated from it."""
    install(stage_index(folder, index))
    return Path(folder) / INDEX_NAME


def _check_row(row: dict) -> None:
    if row.get('status') not in STATUSES:
        raise ValueError(
            f'Version {row.get("version")!r}: status {row.get("status")!r}'
        )
    location = row.get('location')
    if location is not None and location not in LOCATIONS:
        raise ValueError(f'Version {row.get("version")!r}: location {location!r}')


def _size(row: dict) -> str:
    total = sum(entry.get('bytes') or 0 for entry in row.get('files') or [])
    return f'{total / 1e6:,.1f} MB'


def render_versions_md(index: dict) -> str:
    """Render an index as a Markdown table, newest version first."""
    title = index.get('bundle') or 'bundle'
    if index.get('region'):
        title = f'{title} ({index["region"]})'
    lines = [
        f'# Versions of {title}',
        '',
        'Generated from index.json by openplaces; edit that file, not this one.',
        'A withdrawn version must not be handed out again.',
        '',
        '| version | status | location | built | commit | files | size | note |',
        '|---|---|---|---|---|---|---|---|',
    ]
    for row in reversed(index['versions']):
        commit = row.get('commit') or ''
        if row.get('dirty'):
            commit = f'{commit} (uncommitted changes)'
        note = str(row.get('note') or '').replace('|', '/').replace('\n', ' ')
        lines.append(
            f'| {row["version"]} | {row["status"]} | {row.get("location") or "none"} '
            f'| {row.get("built") or ""} | {commit} | {len(row.get("files") or [])} '
            f'| {_size(row)} | {note} |'
        )
    return '\n'.join(lines) + '\n'


def archive_root() -> Path | None:
    """The configured archive root, or None when this user has none.

    Read from the user config's `directories: archive:` key. Unset, the
    `archive` folder under the data root serves when it exists, which is
    where copies were archived by hand before this module
    (`Dropbox/Data/earth/archive`). Nothing creates that folder: a user
    without one has no archive until they configure or create it.
    """
    from openplaces.config import cfg

    directories = cfg.config.get('directories') or {}
    configured = directories.get(ARCHIVE_KEY)
    if configured:
        return Path(configured)
    try:
        default = Path(cfg.data_root) / ARCHIVE_DIR
    except KeyError:
        return None
    return default if default.is_dir() else None


def archive_folder(
    canonical: Path, version: str, bucket_root: Path, root: Path, bucket: str
) -> Path:
    """Where a version of a bundle is archived.

    The bundle's folder, relative to its bucket, mirrored under the
    archive root by bucket name, then `versions/<version>/`:
    `<archive>/share/US/NC/.../cheer-eastern-nc/versions/2026-11-03/`.

    Raises
    ------
    ArchiveError
        When the bundle does not lie inside its bucket.
    """
    folder = Path(canonical).parent.resolve()
    try:
        relative = folder.relative_to(Path(bucket_root).resolve())
    except ValueError as error:
        raise ArchiveError(
            f'{folder} is not inside the {bucket} bucket ({bucket_root}), so it '
            'has no mirrored place in the archive.'
        ) from error
    return Path(root) / bucket / relative / VERSIONS_DIR / version


def _relative(path: Path, root: Path) -> str:
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def _move(source: Path, target: Path) -> None:
    """Move one file, across volumes too, and check its bytes arrived.

    A rename where both lie on one volume. Otherwise a copy to a
    `.partial` name, compared by SHA-256 with the source, renamed, and
    only then the source removed, so a failed copy leaves the source.
    """
    try:
        os.replace(source, target)
        return
    except OSError:
        if not source.exists() or target.exists():
            raise
    partial = target.with_name(target.name + '.partial')
    shutil.copy2(source, partial)
    if sha256_of(partial) != sha256_of(source):
        partial.unlink(missing_ok=True)
        raise OSError(f'Copy of {source} to {target} does not match the source.')
    os.replace(partial, target)
    _writable(source)
    source.unlink()


def current_row(index: dict) -> dict | None:
    """The index row of the bundle now in the share folder, if listed."""
    rows = [row for row in index['versions'] if row.get('location') == 'share']
    return rows[-1] if rows else None


def describe_present(paths: dict, index: dict) -> dict | None:
    """The manifest of the bundle now at *paths*, or None if none is there.

    Its own `manifest.json` when it has one; else its row in the index
    (a bundle the backfill entered); else a manifest built from its
    files, named by the date of the canonical file's modification time.
    """
    present = [path for path in paths.values() if path.exists()]
    if not present:
        return None
    folder = Path(paths['canonical']).parent
    manifest = read_manifest(folder)
    if manifest is not None:
        return manifest
    row = current_row(index)
    if row is not None:
        return {key: row.get(key) for key in MANIFEST_FIELDS}
    reference = paths['canonical'] if paths['canonical'].exists() else present[0]
    seconds = reference.stat().st_mtime
    version = next_version(
        dt.date.fromtimestamp(seconds), [row['version'] for row in index['versions']]
    )
    identity = footer_identity(reference)
    return build_manifest(
        version=version,
        built=timestamp(seconds),
        files=[file_entry(path) for path in present],
        recipe_id=index.get('recipe_id'),
        region=index.get('region'),
        admin_id=index.get('admin_id'),
        commit=identity.get('commit'),
        package_version=identity.get('package_version'),
        note='exported before bundles carried a manifest; named by file time',
    )


def _upsert(index: dict, row: dict) -> None:
    for position, existing in enumerate(index['versions']):
        if existing['version'] == row['version']:
            index['versions'][position] = row
            return
    index['versions'].append(row)


def _already_archived(files: list[Path], target: Path, manifest: dict) -> bool:
    """True when *target* holds this version, file for file, byte for byte.

    An export killed after archiving the old bundle, under Snakemake,
    has the old files restored into the share folder from Snakemake's
    backup while the archive keeps its copy. The next export then finds
    the version archived already.
    """
    archived = read_manifest(target)
    if not archived or archived.get('version') != manifest.get('version'):
        return False
    for source in files:
        twin = target / source.name
        if not twin.is_file() or twin.stat().st_size != source.stat().st_size:
            return False
        if sha256_of(twin) != sha256_of(source):
            return False
    return True


def _archive(files: list[Path], target: Path, manifest: dict) -> list[tuple]:
    """Move *files* into *target* with their manifest; undo on failure.

    Returns (source, archived copy, kind) per file: kind 'move' for a
    file moved, 'kept' for one whose identical copy was archived
    already (the source stays, and is restored by copying back).
    """
    if target.exists() and any(target.iterdir()):
        if _already_archived(files, target, manifest):
            return [(source, target / source.name, 'kept') for source in files]
        raise ArchiveError(
            f'{target} already holds files that differ from the bundle; a '
            'version is archived once, never over another.'
        )
    moved = []
    try:
        target.mkdir(parents=True, exist_ok=True)
        for source in files:
            destination = target / source.name
            _move(source, destination)
            moved.append((source, destination, 'move'))
        write_manifest(target, manifest)
    except BaseException as error:
        _restore(moved)
        for leftover in (target / MANIFEST_NAME, _partial(target / MANIFEST_NAME)):
            _writable(leftover)
            leftover.unlink(missing_ok=True)
        try:
            target.rmdir()
        except OSError:
            pass
        if not isinstance(error, OSError):
            raise
        raise ArchiveError(
            f'Could not archive the bundle to {target} ({error}); it was left '
            'in place and the new bundle was not written.'
        ) from error
    for _, destination, _ in moved:
        _read_only(destination)
    return moved


def _restore(moved: list[tuple]) -> None:
    """Put archived files back at their share paths."""
    for source, destination, kind in reversed(moved):
        _writable(source)
        if kind == 'kept':
            shutil.copy2(destination, source)
        else:
            _writable(destination)
            os.replace(destination, source)


def refuse_unarchivable(paths: dict, root: Path | None, replace_unarchived: bool):
    """Raise before any work when a bundle is present and cannot be archived.

    Raises
    ------
    ArchiveError
        When a bundle exists at *paths*, no archive root is configured
        and *replace_unarchived* is False.
    """
    if root is not None or replace_unarchived:
        return
    if any(path.exists() for path in paths.values()):
        raise ArchiveError(
            f'{paths["canonical"].parent} holds a bundle, and a re-export moves '
            'it to the archive first, but no archive root is configured. Set '
            "'directories: archive:' in your openplaces config (or create an "
            "'archive' folder under the data root), or pass "
            'replace_unarchived=True to replace it without a copy.'
        )


def replace_bundle(
    paths: dict,
    staged: dict,
    manifest: dict,
    *,
    bucket: str,
    bucket_root: Path,
    root: Path | None,
    replace_unarchived: bool = False,
    meta: dict | None = None,
    verbose: bool = False,
) -> dict:
    """Archive the bundle at *paths*, move the staged one in, index both.

    The step `export_delivery` ends with in the share bucket. The new
    bundle's version is chosen here, once the old one's is known: the
    build date, with a letter when that date is taken.

    Parameters
    ----------
    paths : dict of str to pathlib.Path
        The bundle's files, by role (`delivery_paths`).
    staged : dict of str to pathlib.Path
        The new files, written beside them under temporary names.
    manifest : dict
        The new bundle's manifest; its `version` is replaced by the one
        chosen here.
    bucket : str
        The bucket name the archive mirrors (`share`).
    bucket_root : pathlib.Path
        That bucket's folder.
    root : pathlib.Path or None
        The archive root (`archive_root`).
    replace_unarchived : bool, optional
        Replace an existing bundle without an archive copy when there is
        no archive root. Its index row stays, with no location.
    meta : dict, optional
        The bundle's identity for a new index (`bundle`, `recipe_id`,
        `region`, `admin_id`).
    verbose : bool, optional
        Print where the old bundle went.

    Returns
    -------
    dict
        The new bundle's index row.

    Raises
    ------
    ArchiveError
        When the old bundle cannot be archived. It is left in place and
        the staged files are removed.

    Notes
    -----
    Everything is decided and written beside its destination before
    anything moves: the new version name, the index rows, the new
    manifest, the new index and its Markdown. Then the old bundle moves
    to the archive, the staged files move in, the manifest, Markdown
    and index are renamed into place, the index last. Any exception in
    that sequence (a failed move, a full disk, an interrupt) undoes it:
    the old files come back from the archive, the old manifest and
    Markdown are put back, the archive folder is removed, and the index
    was never changed. So the index never calls a version current that
    the share folder does not hold, and a rerun finds the old bundle,
    its manifest and its version name exactly as before.
    """
    canonical = Path(paths['canonical'])
    folder = canonical.parent
    vdir = versions_dir(canonical)
    index = read_index(vdir) or empty_index(**(meta or {}))
    _settle_vanished_current(index, paths, canonical, bucket_root, root, bucket)
    old = describe_present(paths, index)
    present = [path for path in paths.values() if path.exists()]

    old_row = None
    target = None
    try:
        if old is not None:
            old_row = {
                **{key: old.get(key) for key in MANIFEST_FIELDS},
                'status': 'superseded',
                'location': None,
                'path': None,
            }
            listed = next(
                (r for r in index['versions'] if r['version'] == old['version']),
                None,
            )
            # The index row may say more than the manifest: a
            # withdrawal, and the note given with it.
            if listed is not None:
                old_row['note'] = listed.get('note')
                if listed.get('status') == 'withdrawn':
                    old_row['status'] = 'withdrawn'
            if root is None:
                refuse_unarchivable(paths, root, replace_unarchived)
                old_row['note'] = append_note(
                    old_row.get('note'), 'replaced without an archive copy'
                )
            else:
                target = archive_folder(
                    canonical, old['version'], bucket_root, root, bucket
                )
                old_row['location'] = 'archive'
                old_row['path'] = _relative(target, root)
    except ArchiveError:
        for path in staged.values():
            path.unlink(missing_ok=True)
        raise

    taken = [row['version'] for row in index['versions']]
    if old is not None:
        taken.append(old['version'])
    day = dt.date.fromisoformat(manifest['built'][:10])
    manifest = {**manifest, 'version': next_version(day, taken)}
    if old_row is not None:
        _upsert(index, old_row)
    row = {
        **{key: manifest.get(key) for key in MANIFEST_FIELDS},
        'status': 'current',
        'location': 'share',
        'path': _relative(folder, bucket_root),
    }
    _upsert(index, row)
    for field, value in (meta or {}).items():
        if value is not None:
            index[field] = str(value) if field == 'admin_id' else value

    manifest_path = folder / MANIFEST_NAME
    texts = {}
    moved, swapped, installed = [], [], []
    previous = {
        path: path.read_bytes() if path.exists() else None
        for path in (manifest_path, vdir / VERSIONS_MD, vdir / INDEX_NAME)
    }
    try:
        partial = _partial(manifest_path)
        partial.write_text(
            json.dumps({k: manifest.get(k) for k in MANIFEST_FIELDS}, indent=2) + '\n',
            encoding='utf-8',
        )
        texts = {manifest_path: partial, **stage_index(vdir, index)}
        if target is not None:
            moved = _archive(present, target, old)
            if verbose:
                print(f'Archived version {old["version"]} to {target}')
        for role, path in paths.items():
            os.replace(staged[role], path)
            swapped.append(path)
        for path, text in texts.items():
            _writable(path)
            os.replace(text, path)
            installed.append(path)
        _read_only(manifest_path)
    except BaseException:
        _undo(moved, swapped, installed, previous, target, staged, texts)
        raise
    return row


def _settle_vanished_current(index, paths, canonical, bucket_root, root, bucket):
    """Correct a `current` row whose files are no longer in the share folder.

    Only an export killed between moving the old bundle out and moving
    the new one in (no exception, so nothing undone) leaves this. The
    row then names where its files are: the archive when they arrived
    there, else nowhere. Without this, the next export would add a
    second `current` row.
    """
    row = current_row(index)
    if row is None or any(path.exists() for path in paths.values()):
        return
    row['status'] = 'withdrawn' if row['status'] == 'withdrawn' else 'superseded'
    if root is not None:
        target = archive_folder(canonical, row['version'], bucket_root, root, bucket)
        if (target / MANIFEST_NAME).is_file():
            row['location'] = 'archive'
            row['path'] = _relative(target, root)
            return
    row['location'] = None
    row['path'] = None
    row['note'] = append_note(
        row.get('note'), 'not in the share folder when the next export ran'
    )


def _undo(moved, swapped, installed, previous, target, staged, texts) -> None:
    """Put the share folder and the archive back as they were."""
    # Old files back over the new ones; a new file with no old
    # counterpart is removed.
    restored = {source for source, _, _ in moved}
    _restore(moved)
    if moved:
        for path in swapped:
            if path not in restored:
                _writable(path)
                path.unlink(missing_ok=True)
    for path in installed:
        before = previous.get(path)
        _writable(path)
        if before is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(before)
            if path.name == MANIFEST_NAME:
                _read_only(path)
    # The archive folder goes only when this export created it.
    if target is not None and any(kind == 'move' for _, _, kind in moved):
        _writable(target / MANIFEST_NAME)
        (target / MANIFEST_NAME).unlink(missing_ok=True)
        try:
            target.rmdir()
        except OSError:
            pass
    for path in [*staged.values(), *texts.values()]:
        Path(path).unlink(missing_ok=True)


def append_note(note: str | None, addition: str) -> str:
    """Join a note and an addition with a semicolon."""
    return f'{note}; {addition}' if note else addition


def set_status(folder: Path, version: str, status: str, note: str | None = None):
    """Set the status of one version in a `versions/` folder's index.

    Only `withdrawn` and `superseded` can be set by hand: an export is
    what makes a version current. Withdrawing a version marks it as one
    that must not be handed out again; its row stays, and no file is
    moved or deleted.

    Parameters
    ----------
    folder : pathlib.Path
        The `versions/` folder holding `index.json`.
    version : str
        The version to change.
    status : str
        `withdrawn`, or `superseded` to undo a withdrawal.
    note : str, optional
        Why, appended to the row's note.

    Returns
    -------
    dict
        The changed row.

    Raises
    ------
    ValueError
        For another status, or a version the index does not list.
    FileNotFoundError
        When the folder holds no index.
    """
    if status not in ('withdrawn', 'superseded'):
        raise ValueError(
            f'Status {status!r} cannot be set by hand: withdrawn or superseded.'
        )
    index = read_index(folder)
    if index is None:
        raise FileNotFoundError(f'No {INDEX_NAME} in {folder}.')
    for row in index['versions']:
        if row['version'] == version:
            if status == 'superseded' and row.get('location') == 'share':
                status = 'current'
            row['status'] = status
            if note:
                row['note'] = append_note(row.get('note'), note)
            write_index(folder, index)
            return row
    listed = ', '.join(row['version'] for row in index['versions']) or 'none'
    raise ValueError(f'{folder} lists no version {version!r} (listed: {listed}).')
