"""Copy an imputed bundle from the cache to its `<region>.imputed` folder.

The imputation runs from a branch that main does not contain, so its
bundles must never land in a region's own share folder: main's exports
and the value pages read those, and two code versions writing one path
leave whichever ran last. The imputed bundle sits beside the region's
folder instead, under the region id plus '.imputed'
(share/US/NC/_all/footprint/openplaces/2026/cheer-eastern-nc.imputed/,
with a team bundle in team/ inside it, as team bundles sit inside
their region's folder). A bundle already there is replaced only with
--archived, once it was copied to the archive root (which mirrors the
data root) with scripts/storage/snapshot.py: the share bucket holds
one copy, never earlier vintages.

Export to the cache first (the export tool, with --output-dir cache).

Usage: pass the region ids (cheer-eastern-nc cheer-eastern-nc.team
cheer-coastal-tx).
"""

from __future__ import annotations

import argparse
import shutil
import stat
from pathlib import Path

from openplaces.io.delivery import delivery_paths

RECIPE = 'US_footprint-openplaces-2026'


def imputed_dir(region: str) -> Path:
    """The `<region>.imputed` folder a region's imputed bundle goes to."""
    base, _, audience = region.partition('.')
    share = delivery_paths(RECIPE, region=base, output_dir='share')['canonical']
    folder = share.parent if share.parent.name == base else share.parent / base
    target = folder.with_name(f'{base}.imputed')
    return target / audience if audience else target


def cache_files(region: str) -> list[Path]:
    """The cached bundle's files, in their delivery roles."""
    return list(delivery_paths(RECIPE, region=region, output_dir='cache').values())


def publish(region: str, archived: bool) -> list[Path]:
    target = imputed_dir(region)
    sources = cache_files(region)
    missing = [p for p in sources if not p.exists()]
    if missing:
        raise SystemExit(f'{region}: not in the cache: {[p.name for p in missing]}')
    target.mkdir(parents=True, exist_ok=True)
    old = [target / p.name for p in sources if (target / p.name).exists()]
    if old and not archived:
        raise SystemExit(
            f'{target} holds a bundle: archive it first (scripts/storage/'
            'snapshot.py copy, then verify, into the archive root mirroring '
            'share/), then pass --archived.'
        )
    for path in old:
        # The shipped copies are read-only.
        path.chmod(path.stat().st_mode | stat.S_IWRITE)
    written = []
    for source in sources:
        destination = target / source.name
        shutil.copy2(source, destination)
        # The cached bundle is read-only; the copy stays writable so a
        # later publish can archive it.
        destination.chmod(destination.stat().st_mode | stat.S_IWRITE)
        written.append(destination)
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('regions', nargs='+')
    parser.add_argument(
        '--archived',
        action='store_true',
        help='the bundle already in the target was archived; replace it',
    )
    args = parser.parse_args()
    for region in args.regions:
        for path in publish(region, args.archived):
            print(f'{region}: {path} ({path.stat().st_size:,d} bytes)')


if __name__ == '__main__':
    main()
