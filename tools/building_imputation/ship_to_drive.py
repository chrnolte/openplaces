"""Copy imputed bundles to a shared drive, archiving what they replace.

Reads each region's `<region>.imputed` folder (publish_imputed.py). For
each ``region=remote-folder`` pair, the bundle files already in the
remote folder move to ``prior/<date>/`` there, the named old
subfolders (``--retire``) move to ``prior/``, the new files are copied,
and each copy's size is compared with the local file. Every move is
file by file (``rclone moveto``), so source and destination never
overlap, and the first failing rclone call stops the run. A team bundle
is refused unless ``--team`` is given: it may go only to a drive shared
with the team, never to an open one.

Usage
-----
    python tools/building_imputation/ship_to_drive.py \
        "cheer-eastern-nc=<remote>:<folder>" [...] [--retire imputed-v0] \
        [--team] [--dry-run]
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess

from publish_imputed import cache_files, imputed_dir


def rclone(args, dry_run):
    command = ['rclone', *args]
    if dry_run and args[0] in {'moveto', 'copyto'}:
        command.append('--dry-run')
    print(' '.join(command), flush=True)
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def listing(remote):
    """Entries of a remote folder by name; empty for a folder not there yet."""
    try:
        entries = json.loads(rclone(['lsjson', remote], dry_run=False))
    except subprocess.CalledProcessError as error:
        if 'directory not found' in (error.stderr or ''):
            return {}
        raise
    return {entry['Name']: entry for entry in entries}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pairs', nargs='+', help='region=remote:folder')
    parser.add_argument('--retire', nargs='*', default=[])
    parser.add_argument('--date', default=datetime.date.today().isoformat())
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument(
        '--team',
        action='store_true',
        help='allow team bundles; the remote must be shared with the team only',
    )
    args = parser.parse_args()
    for pair in args.pairs:
        region, remote = pair.split('=', 1)
        if region.endswith('.team') and not args.team:
            raise SystemExit(f'{region}: a team bundle needs --team.')
        folder = imputed_dir(region)
        files = [folder / path.name for path in cache_files(region)]
        missing = [path.name for path in files if not path.exists()]
        if missing:
            raise SystemExit(f'{region}: not in {folder}: {missing}')
        present = listing(remote)
        prior = f'{remote}/prior/{args.date}'
        # A second ship on the same date would overwrite that date's
        # archive, losing the vintage it holds; refuse instead.
        archived = listing(f'{remote}/prior') if 'prior' in present else {}
        if args.date in archived and any(path.name in present for path in files):
            raise SystemExit(
                f'{prior} already exists; pass another --date for this archive.'
            )
        for path in files:
            if path.name in present:
                rclone(
                    ['moveto', f'{remote}/{path.name}', f'{prior}/{path.name}'],
                    args.dry_run,
                )
        for old in args.retire:
            if old in present:
                rclone(
                    ['moveto', f'{remote}/{old}', f'{remote}/prior/{old}'], args.dry_run
                )
        for path in files:
            rclone(['copyto', str(path), f'{remote}/{path.name}'], args.dry_run)
        if args.dry_run:
            continue
        after = listing(remote)
        for path in files:
            size = path.stat().st_size
            got = after.get(path.name, {}).get('Size')
            status = 'ok' if got == size else 'MISMATCH'
            print(f'{region}: {path.name} local {size:,d} remote {got} {status}')


if __name__ == '__main__':
    main()
