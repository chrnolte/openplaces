"""
Enter bundles exported before versioning into their version indexes.

Bundles shipped before `export_delivery` kept versions sit in the share
bucket at their fixed paths, and earlier copies of some sit in the
archive under folders named by hand (`v0.6`, `pre-2026-11-03`,
`cheer-drive-prior/2026-10-19`). This command lists them in the
`versions/index.json` beside each bundle in the share bucket, without
moving, renaming or opening any archived file:

- A bundle in the share bucket is entered as `current`, under the date
  of its canonical file's modification time (the day it was exported),
  with its imputation model token as the note where its licence notice
  names one (`building-imputation-v0.8`).
- A copy in the archive is entered as `superseded` with `location:
  archive`, under its own file time, with the folder it was filed under
  as the note (the old label). It belongs to the share bundle of the
  same file stem whose folder its archive folder lies in, mirrored; a
  copy filed under a `team` folder belongs to that bundle's team twin.
- A copy that matches no bundle in the share bucket, or a folder that
  holds files but no bundle, is reported as unclassified and entered
  nowhere.

Archived files are described from their directory entries alone (size
and modification time): the archive is online-only, and reading a file
would download it. Their checksums and row counts are therefore left
empty. A bundle in the share bucket is read for its checksums, row
counts and footer commit unless `--no-checksums` is given.

Usage
-----
python -m openplaces.io.delivery.backfill --dry-run
python -m openplaces.io.delivery.backfill
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from openplaces.core.constants import ESCAPE_DIR
from openplaces.io.delivery import versions

# The files a bundle consists of, by suffix after its stem. Sidecars
# written beside a bundle by other tools (admin outlines, a QGIS
# project, a README) are not part of a version.
BUNDLE_SUFFIXES = ('.parquet', '_point.parquet', '_geo.parquet', '_evidence.parquet')
LICENSE_SUFFIX = '_LICENSE.txt'
TEAM_DIR = 'team'
_IMPUTATION_TOKEN = re.compile(r'building-imputation-v[0-9][0-9.]*[0-9]')


@dataclass
class Bundle:
    """One bundle found on disk: its folder, file stem and files."""

    root: Path
    folder: Path
    stem: str
    files: list[Path] = field(default_factory=list)

    @property
    def relative(self) -> Path:
        """The folder relative to the root it was found under."""
        return self.folder.relative_to(self.root)


def find_bundles(root: Path) -> tuple[list[Bundle], list[str]]:
    """Find every bundle under *root*, by its canonical file and notice.

    A bundle is a `<stem>.parquet` with a `<stem>_LICENSE.txt` beside
    it; its files are those two and the `_point`, `_geo` and
    `_evidence` files of the same stem. `versions/` folders are skipped.

    Returns
    -------
    tuple
        The bundles, and the folders (relative to *root*) that hold
        files but no bundle, with their file counts.
    """
    bundles, other = [], []
    root = Path(root)
    if not root.is_dir():
        return bundles, other
    for folder in sorted(p for p in [root, *root.rglob('*')] if p.is_dir()):
        relative = folder.relative_to(root)
        if versions.VERSIONS_DIR in relative.parts:
            continue
        names = {p.name for p in folder.iterdir() if p.is_file()}
        stems = sorted(
            name[: -len(LICENSE_SUFFIX)]
            for name in names
            if name.endswith(LICENSE_SUFFIX)
            and f'{name[: -len(LICENSE_SUFFIX)]}.parquet' in names
        )
        for stem in stems:
            files = [
                folder / f'{stem}{suffix}'
                for suffix in (*BUNDLE_SUFFIXES, LICENSE_SUFFIX)
                if f'{stem}{suffix}' in names
            ]
            bundles.append(Bundle(root, folder, stem, files))
        if not stems and names and relative.name != 'accuracies':
            other.append(f'{relative.as_posix()} ({len(names)} files, no bundle)')
    return bundles, other


def region_of(relative: Path) -> str | None:
    """The region id a bundle folder names, or None for a flat bundle.

    Below `_all/<entity>/<source>/<version>/`, a folder is the region's
    and a `team` folder in it the team twin's (`cheer-eastern-nc.team`).
    """
    parts = relative.parts
    if ESCAPE_DIR not in parts:
        return None
    below = parts[parts.index(ESCAPE_DIR) + 4 :]
    if not below:
        return None
    if below[-1] == TEAM_DIR and len(below) > 1:
        return f'{below[-2]}.team'
    return below[0]


def recipe_of(bundle: Bundle) -> str | None:
    """The curation recipe a bundle's path names, when one is installed.

    The stem names the unit (`US-NC_footprint-cheer-2026`) and the
    recipe its own scope (`US_footprint-cheer-2026`), so each shorter
    admin prefix is tried; a recipe renamed since (`openplaces` to
    `cheer`) resolves to none.
    """
    from openplaces.recipe import get_recipe_by_id

    admin, _, rest = bundle.stem.partition('_')
    parts = admin.split('-')
    for length in range(len(parts), 0, -1):
        candidate = f'{"-".join(parts[:length])}_{rest}'
        try:
            recipe = get_recipe_by_id(candidate)
        except Exception:
            continue
        if isinstance(recipe, dict) and recipe.get('share'):
            return candidate
    return None


def _modified(path: Path) -> float:
    return path.stat().st_mtime


def _reference(bundle: Bundle) -> Path:
    return bundle.folder / f'{bundle.stem}.parquet'


def share_row(bundle: Bundle, share_root: Path, checksums: bool) -> dict:
    """Describe a bundle in the share bucket as a `current` row."""
    reference = _reference(bundle)
    identity = versions.footer_identity(reference)
    notice = bundle.folder / f'{bundle.stem}{LICENSE_SUFFIX}'
    tokens = sorted(set(_IMPUTATION_TOKEN.findall(notice.read_text(encoding='utf-8'))))
    imputation = tokens[-1] if tokens else None
    return {
        'version': None,
        'built': versions.timestamp(_modified(reference)),
        'commit': identity.get('commit'),
        'dirty': None,
        'package_version': identity.get('package_version'),
        'recipes': [r for r in [recipe_of(bundle)] if r],
        'imputation': imputation,
        'region': region_of(bundle.relative),
        'admin_id': bundle.stem.partition('_')[0],
        'audience': 'team' if bundle.folder.name == TEAM_DIR else 'public',
        'files': [versions.file_entry(p, checksum=checksums) for p in bundle.files],
        'note': imputation.rpartition('-')[2] if imputation else None,
        'status': 'current',
        'location': 'share',
        'path': bundle.relative.as_posix(),
    }


def archive_row(bundle: Bundle, label: str, archive_root: Path) -> dict:
    """Describe an archived copy from its directory entries only."""
    reference = _reference(bundle)
    return {
        'version': None,
        'built': versions.timestamp(_modified(reference)),
        'commit': None,
        'dirty': None,
        'package_version': None,
        'recipes': [],
        'imputation': None,
        'region': None,
        'admin_id': bundle.stem.partition('_')[0],
        'audience': None,
        'files': [
            {'name': p.name, 'bytes': p.stat().st_size, 'rows': None, 'sha256': None}
            for p in bundle.files
        ],
        'note': label,
        'status': 'superseded',
        'location': 'archive',
        'path': bundle.folder.relative_to(archive_root).as_posix(),
    }


def owner_of(copy: Bundle, current: dict) -> tuple[Path | None, str]:
    """The share bundle folder an archived copy belongs to, and its label.

    *current* maps (relative folder, stem) of the share bundles. The
    deepest share folder of the same stem that the copy's folder lies
    in owns it; a `team` folder below that routes it to the team twin.
    Returns (None, reason) when no share bundle owns it.
    """
    relative = copy.relative
    owners = [
        folder
        for folder, stem in current
        if stem == copy.stem
        and folder != relative
        and folder.parts == relative.parts[: len(folder.parts)]
        and TEAM_DIR not in folder.parts[-1:]
    ]
    if not owners:
        return None, 'no bundle of this stem in the share folder above it'
    owner = max(owners, key=lambda folder: len(folder.parts))
    rest = relative.parts[len(owner.parts) :]
    if TEAM_DIR in rest:
        team = owner / TEAM_DIR
        if (team, copy.stem) not in current:
            return None, 'filed under team/, but the share folder has no team twin'
        owner = team
        rest = tuple(part for part in rest if part != TEAM_DIR)
    return owner, '/'.join(rest) or '.'


@dataclass
class Plan:
    """What a backfill would write: one index per share bundle."""

    indexes: dict = field(default_factory=dict)
    unclassified: list = field(default_factory=list)


def _same_files(row: dict, other: dict) -> bool:
    def key(entry):
        return entry['name'], entry['bytes']

    return row['built'] == other['built'] and sorted(map(key, row['files'])) == sorted(
        map(key, other['files'])
    )


def plan_backfill(
    share_root: Path,
    archive_root: Path | None,
    bucket: str = 'share',
    checksums: bool = True,
) -> Plan:
    """Work out every index row a backfill would write; write nothing.

    Parameters
    ----------
    share_root : pathlib.Path
        The share bucket's folder.
    archive_root : pathlib.Path or None
        The archive root; its `<bucket>/` folder is searched.
    bucket : str, optional
        The bucket name the archive mirrors.
    checksums : bool, optional
        Read each share bundle's files for their SHA-256.

    Returns
    -------
    Plan
        `indexes` maps each share bundle's `versions/` folder to the
        index it would hold; `unclassified` lists what was entered
        nowhere, with the reason.
    """
    plan = Plan()
    shared, other = find_bundles(share_root)
    plan.unclassified += [f'share: {entry}' for entry in other]
    current = {(b.relative, b.stem): b for b in shared}
    rows: dict = {}
    for bundle in shared:
        rows[(bundle.relative, bundle.stem)] = [
            share_row(bundle, share_root, checksums)
        ]
    if archive_root is not None:
        archived, other = find_bundles(Path(archive_root) / bucket)
        plan.unclassified += [f'archive: {entry}' for entry in other]
        for copy in archived:
            owner, label = owner_of(copy, current)
            if owner is None:
                plan.unclassified.append(
                    f'archive: {copy.relative.as_posix()} ({copy.stem}): {label}'
                )
                continue
            rows[(owner, copy.stem)].append(archive_row(copy, label, archive_root))

    for (relative, stem), entries in rows.items():
        bundle = current[(relative, stem)]
        folder = versions.versions_dir(_reference(bundle))
        share = entries[0]
        index = versions.read_index(folder) or versions.empty_index(
            bundle=stem,
            recipe_id=(share['recipes'] or [None])[0],
            region=share['region'],
            admin_id=share['admin_id'],
        )
        listed = {(r.get('location'), r.get('path')) for r in index['versions']}
        taken = [r['version'] for r in index['versions']]
        # Oldest first, so letters follow build order within a day.
        entries = sorted(
            entries, key=lambda r: (r['built'], r['location'] != 'archive')
        )
        added = []
        for row in entries:
            if (row['location'], row['path']) in listed:
                continue
            if row['location'] == 'share' and versions.current_row(index):
                continue
            day = dt.date.fromisoformat(row['built'][:10])
            row['version'] = versions.next_version(day, taken)
            taken.append(row['version'])
            twin = next((r for r in added if _same_files(row, r)), None)
            if twin is not None:
                row['note'] = versions.append_note(
                    row['note'], f'same files as {twin["version"]}'
                )
            added.append(row)
        index['versions'] += added
        plan.indexes[folder] = (index, added)
    return plan


def write_plan(plan: Plan) -> int:
    """Write every index of *plan*; return how many were written."""
    written = 0
    for folder, (index, added) in plan.indexes.items():
        if added:
            versions.write_index(folder, index)
            written += 1
    return written


def summarize(plan: Plan, share_root: Path) -> list[str]:
    """Lines describing a plan: per region and bundle, then unclassified."""
    lines = []
    by_location = {'share': 0, 'archive': 0}
    for folder, (index, added) in sorted(plan.indexes.items()):
        relative = folder.parent.relative_to(share_root).as_posix()
        lines.append(
            f'{relative} [{index["bundle"]}]: {len(added)} new row(s): '
            + ', '.join(
                f'{r["version"]} {r["status"]} {r["location"]}'
                + (f' ({r["note"]})' if r.get('note') else '')
                for r in added
            )
        )
        for row in added:
            by_location[row['location']] += 1
    lines.append(
        f'{len(plan.indexes)} index(es); rows: {by_location["share"]} current '
        f'in share, {by_location["archive"]} superseded in the archive'
    )
    lines.append(f'{len(plan.unclassified)} unclassified:')
    lines += [f'  {entry}' for entry in plan.unclassified]
    return lines


def main(argv=None) -> int:
    """Command line: plan the backfill, print it, write unless a dry run."""
    from openplaces.config import cfg

    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument(
        '--dry-run', action='store_true', help='print what would be written'
    )
    parser.add_argument(
        '--no-checksums',
        action='store_true',
        help="do not read the share bundles' files for their SHA-256",
    )
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    share_root = Path(cfg.get_dir('share'))
    archive = versions.archive_root()
    print(f'share: {share_root}')
    print(f'archive: {archive or "none configured"}')
    plan = plan_backfill(share_root, archive, checksums=not args.no_checksums)
    for line in summarize(plan, share_root):
        print(line)
    if args.dry_run:
        print('Dry run: nothing written.')
        return 0
    print(f'Wrote {write_plan(plan)} index(es).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
