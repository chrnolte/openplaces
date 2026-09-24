"""Report every measured claim in the source tree and whether its scope
has been rebuilt since the claim was measured.

A docstring may quote a statistic that can move, and should: the number
is the evidence for a method. It quotes it in one form (a fabricated
example, so this file does not report itself),

    Measured 2026-01-15 on XX-AA-BBB: 12.5% of rows carry the value
    the roll recorded.

so that this report can find it (the prefix), compare it (the date) and
say what it was measured on (the scope: an admin id, a region id, or a
recipe id). A claim whose scope has a newer build than the claim's date
is reported as stale. The report shows a claim's first line only, so
the number belongs on the same line as the prefix. It exits 0 whatever
it finds: a stale number is documentation debt, not a broken pipeline.

Run as ``python -m openplaces.flow.measured_claims [--root DIR]``.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

CLAIM = re.compile(
    r'Measured (?P<date>\d{4}-\d{2}-\d{2}) on (?P<scope>[A-Za-z0-9_.:-]+):\s*'
    r'(?P<claim>.*)'
)
ADMIN_ID = re.compile(r'^[A-Z]{2}(-[A-Z0-9]{1,4}){0,3}$')


@dataclass
class Claim:
    file: Path
    line: int
    measured: date
    scope: str
    text: str


def find_claims(root: Path) -> list[Claim]:
    """Every claim under *root*, in file order."""
    found = []
    for path in sorted(root.rglob('*.py')):
        # This module documents the form; it is not a claim. Matched by
        # name, not by identity, so a run from one checkout over another
        # checkout's tree skips that tree's copy too.
        if path.name == 'measured_claims.py' and path.parent.name == 'flow':
            continue
        try:
            text = path.read_text(encoding='utf8')
        except OSError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            match = CLAIM.search(line)
            if match:
                found.append(
                    Claim(
                        file=path,
                        line=number,
                        measured=date.fromisoformat(match['date']),
                        scope=match['scope'],
                        text=match['claim'].strip(),
                    )
                )
    return found


def verdict(measured: date, built: date | None) -> str:
    """'stale' when the scope was rebuilt after the claim, else 'current'."""
    if built is None:
        return 'no build'
    return 'stale' if built > measured else 'current'


def newest_build(scope: str) -> date | None:
    """The newest curated or delivered file for *scope*, or None.

    An admin id resolves to the curated parcel and footprint outputs
    for that unit; a region id to its delivery bundle; anything else is
    unresolved (None).
    """
    from openplaces.recipe import get_output_path

    stamps: list[float] = []
    if ADMIN_ID.match(scope):
        for recipe in ('US_parcel-openplaces-2026', 'US_footprint-openplaces-2026'):
            try:
                path = get_output_path(recipe, admin_id=scope)
            except Exception:
                continue
            if path.exists():
                stamps.append(path.stat().st_mtime)
    else:
        try:
            from openplaces.io.delivery import delivery_paths

            paths = delivery_paths('US_footprint-openplaces-2026', region=scope)
            canonical = Path(paths['canonical'])
            if canonical.exists():
                stamps.append(canonical.stat().st_mtime)
        except Exception:
            return None
    if not stamps:
        return None
    return datetime.fromtimestamp(max(stamps)).date()


def report(claims: list[Claim], resolve=newest_build) -> list[tuple[Claim, str]]:
    rows = []
    for claim in claims:
        try:
            built = resolve(claim.scope)
        except Exception:
            built = None
        rows.append((claim, verdict(claim.measured, built)))
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument(
        '--root',
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help='source tree to scan (default: the installed openplaces package)',
    )
    args = parser.parse_args(argv)
    rows = report(find_claims(args.root))
    if not rows:
        print(f'no measured claims under {args.root}')
        return 0
    width = max(len(r[1]) for r in rows)
    for claim, state in rows:
        where = f'{claim.file.relative_to(args.root)}:{claim.line}'
        print(f'{state:<{width}}  {claim.measured}  {claim.scope:<24} {where}')
        print(f'{"":<{width}}  {claim.text}')
    counts = {}
    for _, state in rows:
        counts[state] = counts.get(state, 0) + 1
    print(', '.join(f'{n} {state}' for state, n in sorted(counts.items())))
    return 0


if __name__ == '__main__':
    sys.exit(main())
