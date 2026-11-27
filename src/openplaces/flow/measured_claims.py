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
the number belongs on the same line as the prefix.

Three kinds of file carry claims: Python modules (anywhere in a line),
recipe YAML files (comment lines only) and README.md files. A line that
starts a claim but does not parse is listed as malformed with its
reason: no date, a scope that is not one token followed by a colon, a
claim wrapped onto the next line, or a scope that names no admin unit,
region or recipe.

It exits 0 whatever it finds: a stale number is documentation debt,
not a broken pipeline. With --strict it exits 1 when a claim is
malformed, so a check can hold the form without holding the numbers.

Run as `python -m openplaces.flow.measured_claims [--root DIR]
[--strict]`.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

CLAIM = re.compile(
    r'Measured (?P<date>\d{4}-\d{2}-\d{2}) on (?P<scope>[A-Za-z0-9_.:~-]+):\s*'
    r'(?P<claim>.*)'
)
# A line that starts a claim, parsed or not: the capitalized prefix
# followed by a date, by "on", or by the end of the line (a claim
# wrapped onto the next one). Prose that merely uses the word, such as
# "measured against" or a "Measured 3D" geometry type, is not one.
CANDIDATE = re.compile(r'\bMeasured\b(?=\s+(?:on\b|\d{4}-\d{2}-\d{2})|\s*$)')
ADMIN_ID = re.compile(r'^[A-Z]{2}(-[A-Z0-9]{1,4}){0,3}$')
DATE = re.compile(r'\d{4}-\d{2}-\d{2}')

#: File kind by how its claims are found.
KINDS = {'.py': 'python', '.yaml': 'recipe', '.yml': 'recipe'}
README = 'README.md'


@dataclass
class Claim:
    file: Path
    line: int
    measured: date
    scope: str
    text: str
    kind: str = 'python'


@dataclass
class Malformed:
    file: Path
    line: int
    kind: str
    text: str
    reason: str


def _kind(path: Path) -> str | None:
    if path.name == README:
        return 'readme'
    return KINDS.get(path.suffix)


def _lines(path: Path, kind: str):
    """(number, line) for every line of *path* that may carry a claim."""
    try:
        text = path.read_text(encoding='utf8')
    except (OSError, UnicodeDecodeError):
        return
    for number, line in enumerate(text.splitlines(), start=1):
        # A recipe's claims are comments; a key's value that happens to
        # contain the word is data, not documentation.
        if kind == 'recipe' and not line.lstrip().startswith('#'):
            continue
        yield number, line


def _why_malformed(line: str) -> str:
    """The reason a candidate line failed CLAIM."""
    start = CANDIDATE.search(line).start()
    rest = line[start + len('Measured') :].strip()
    if not rest:
        return 'claim wraps onto the next line'
    if not DATE.match(rest):
        return 'no date after the prefix'
    after = rest[10:].split()
    if not after or after[0] != 'on':
        return 'no scope ("on <scope>:") after the date'
    if len(after) <= 2 and not after[-1].endswith(':'):
        return 'claim wraps onto the next line'
    return 'scope is not one token followed by a colon'


def scan(root: Path) -> tuple[list[Claim], list[Malformed]]:
    """Every claim under *root* and every line that fails to parse as one.

    Parameters
    ----------
    root : pathlib.Path
        Tree to scan: Python modules, recipe YAML files (comment lines)
        and README.md files.

    Returns
    -------
    tuple of (list of Claim, list of Malformed)
        Both in file order. Scopes are not resolved here; see
        `resolve_scopes`.
    """
    claims: list[Claim] = []
    malformed: list[Malformed] = []
    for path in sorted(root.rglob('*')):
        kind = _kind(path)
        if kind is None or not path.is_file():
            continue
        # This module documents the form; it is not a claim. Matched by
        # name, not by identity, so a run from one checkout over another
        # checkout's tree skips that tree's copy too.
        if path.name == 'measured_claims.py' and path.parent.name == 'flow':
            continue
        for number, line in _lines(path, kind):
            match = CLAIM.search(line)
            if match:
                claims.append(
                    Claim(
                        file=path,
                        line=number,
                        measured=date.fromisoformat(match['date']),
                        scope=match['scope'],
                        text=match['claim'].strip(),
                        kind=kind,
                    )
                )
            elif CANDIDATE.search(line):
                malformed.append(
                    Malformed(
                        file=path,
                        line=number,
                        kind=kind,
                        text=line.strip(),
                        reason=_why_malformed(line),
                    )
                )
    return claims, malformed


def find_claims(root: Path) -> list[Claim]:
    """Every claim under *root*, in file order."""
    return scan(root)[0]


def known_scope(recipes_dir: Path | None = None) -> Callable[[str], bool]:
    """A test of whether a scope names an admin unit, region or recipe.

    An admin id is recognized by its form (the spine is data, not
    source); a region by the region registry, with or without its
    `.team` suffix; a recipe or dataset by a recipe file whose id is
    the scope, or starts with it, a version's dots written as tildes
    (`admin-gadm-4.1` names `admin-gadm-4~1_admin2`).
    """
    try:
        from openplaces.io.readers import get_regions

        regions = set(get_regions()['region_id'])
    except Exception:
        regions = set()
    if recipes_dir is None:
        recipes_dir = Path(__file__).resolve().parents[1] / 'recipes'
    stems = {p.stem for p in recipes_dir.rglob('*.yaml')}

    def known(scope: str) -> bool:
        if ADMIN_ID.match(scope):
            return True
        if scope.removesuffix('.team') in regions:
            return True
        name = scope.replace('.', '~')
        return any(s == name or s.startswith(name + '_') for s in stems)

    return known


def resolve_scopes(
    claims: list[Claim], known: Callable[[str], bool]
) -> tuple[list[Claim], list[Malformed]]:
    """Split *claims* into those whose scope is known and those it is not."""
    kept, unknown = [], []
    for claim in claims:
        if known(claim.scope):
            kept.append(claim)
        else:
            unknown.append(
                Malformed(
                    file=claim.file,
                    line=claim.line,
                    kind=claim.kind,
                    text=f'Measured {claim.measured} on {claim.scope}: {claim.text}',
                    reason=f'scope {claim.scope!r} names no unit, region or recipe',
                )
            )
    return kept, unknown


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
        for recipe in ('US_parcel-openplaces-2026', 'US_footprint-cheer-2026'):
            try:
                path = get_output_path(recipe, admin_id=scope)
            except Exception:
                continue
            if path.exists():
                stamps.append(path.stat().st_mtime)
    else:
        try:
            from openplaces.io.delivery import delivery_paths

            paths = delivery_paths('US_footprint-cheer-2026', region=scope)
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


def main(argv=None, known: Callable[[str], bool] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument(
        '--root',
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help='source tree to scan (default: the installed openplaces package)',
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='exit 1 when a claim is malformed',
    )
    args = parser.parse_args(argv)
    claims, malformed = scan(args.root)
    claims, unknown = resolve_scopes(claims, known or known_scope())
    malformed = sorted(malformed + unknown, key=lambda m: (str(m.file), m.line))
    rows = report(claims)
    if not rows and not malformed:
        print(f'no measured claims under {args.root}')
        return 0
    width = max([len(r[1]) for r in rows] + [len('malformed')])
    for claim, state in rows:
        where = f'{claim.file.relative_to(args.root)}:{claim.line}'
        print(
            f'{state:<{width}}  {claim.measured}  {claim.scope:<24} '
            f'{claim.kind:<7} {where}'
        )
        print(f'{"":<{width}}  {claim.text}')
    for bad in malformed:
        where = f'{bad.file.relative_to(args.root)}:{bad.line}'
        print(f'{"malformed":<{width}}  {bad.kind:<7} {where}  ({bad.reason})')
        print(f'{"":<{width}}  {bad.text}')
    counts: dict[str, int] = {}
    for claim, state in rows:
        counts[state] = counts.get(state, 0) + 1
    kinds: dict[str, int] = {}
    for claim, _ in rows:
        kinds[claim.kind] = kinds.get(claim.kind, 0) + 1
    if malformed:
        counts['malformed'] = len(malformed)
    print(', '.join(f'{n} {state}' for state, n in sorted(counts.items())))
    print(
        'claims by kind: '
        + ', '.join(f'{n} {kind}' for kind, n in sorted(kinds.items()))
    )
    if args.strict and malformed:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
