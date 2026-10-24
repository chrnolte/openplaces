"""Copy an imputed bundle from the cache to its `<region>.imputed` folder.

The imputation runs from a branch that main does not contain, so its
bundles must never land in a region's own share folder: main's exports
and the value pages read those, and two code versions writing one path
leave whichever ran last. The imputed bundle sits beside the region's
folder instead, under the region id plus '.imputed'
(share/US/NC/_all/footprint/openplaces/2026/cheer-eastern-nc.imputed/,
with a team bundle in team/ inside it, as team bundles sit inside
their region's folder). Files already there move to prior/<date>/
first.

Export to the cache first (`export_and_check.py --output-dir cache`).

Usage
-----
    python tools/building_imputation/publish_imputed.py \
        cheer-eastern-nc cheer-eastern-nc.team cheer-coastal-tx
"""

from __future__ import annotations

import argparse
import datetime
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


def publish(region: str, date: str) -> list[Path]:
    target = imputed_dir(region)
    sources = cache_files(region)
    missing = [p for p in sources if not p.exists()]
    if missing:
        raise SystemExit(f'{region}: not in the cache: {[p.name for p in missing]}')
    target.mkdir(parents=True, exist_ok=True)
    old = [target / p.name for p in sources if (target / p.name).exists()]
    if old:
        prior = target / 'prior' / date
        if prior.exists():
            raise SystemExit(f'{prior} exists; pass another --date.')
        prior.mkdir(parents=True)
        for path in old:
            path.rename(prior / path.name)
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
    parser.add_argument('--date', default=datetime.date.today().isoformat())
    args = parser.parse_args()
    for region in args.regions:
        for path in publish(region, args.date):
            print(f'{region}: {path} ({path.stat().st_size:,d} bytes)')


if __name__ == '__main__':
    main()
