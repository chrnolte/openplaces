"""Emit an old-to-new identifier crosswalk for a re-mint.

Written for the hand-off after a global re-mint: a consumer that already
has outputs keyed on the previous identifiers should be able to rename
files and recategorize columns rather than reprocess anything.

The crosswalk is keyed on the **unit**, never on the identifier string.
A re-mint recycles identifiers, so `US-NC-CD` naming Camden before and
Caldwell after is normal, and a string-to-string mapping built by
comparing the two spines row-wise would silently pair the wrong
counties. Each unit is matched on the code its own source assigns it
(`admin3_id_admin1`, a federal FIPS or GEOID that openplaces does not
issue and cannot re-mint), falling back to the name within its parent
where a unit carries no such code.

Usage
-----
    python tools/remint_crosswalk.py --before <git-ref> [--scope US-NC US-TX]

The default scope is North Carolina and Texas, the two regions with
shipped inventories. Writes CSV to stdout.
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys

import pandas as pd

SPINE = 'src/openplaces/recipes/_all/admin/spine/2026/admin-spine-2026_admin{level}.csv'


def read_at(ref: str, path: str) -> pd.DataFrame:
    """Return a committed CSV as of one git ref."""
    blob = subprocess.run(
        ['git', 'show', f'{ref}:{path}'],
        capture_output=True,
        check=True,
    ).stdout.decode('utf-8-sig')
    return pd.read_csv(io.StringIO(blob), dtype=str, keep_default_na=False)


def read_now(path: str) -> pd.DataFrame:
    """Return a CSV as it stands in the working tree."""
    with open(path, encoding='utf-8-sig') as handle:
        text = handle.read()
    return pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)


def unit_key(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a per-row key that survives a re-mint.

    The source's own code where there is one, else the unit's name
    qualified by its parent identifier, and always qualified by country.
    All three are properties of the unit rather than of the identifier
    openplaces gave it.
    """
    code = frame.get(f'{column}_admin1', pd.Series('', index=frame.index))
    code = code.fillna('').astype(str).str.strip()
    parent = frame[column].str.rsplit('-', n=1).str[0]
    # A national code is national, not global: North Carolina's county
    # FIPS 37103 and a Philippine PSGC code are both five digits and
    # collide outright. Qualify every key by country, which no re-mint
    # changes, or the crosswalk pairs Jones County with Zambales.
    country = frame[column].str.split('-').str[0]
    fallback = parent + '|' + frame['name'].astype(str).str.strip().str.upper()
    return country + '|' + code.where(code != '', fallback)


def crosswalk(before: str, level: int, scope: list[str]) -> pd.DataFrame:
    path = SPINE.format(level=level)
    column = f'admin{level}_id'
    old, new = read_at(before, path), read_now(path)
    for frame in (old, new):
        frame['_key'] = unit_key(frame, column)

    in_scope = new[column].str.startswith(tuple(f'{s}-' for s in scope))
    merged = (
        new[in_scope]
        .merge(old, on='_key', how='left', suffixes=('_new', '_old'))
        .rename(
            columns={f'{column}_new': 'admin_id_new', f'{column}_old': 'admin_id_old'}
        )
    )
    out = merged[['admin_id_old', 'admin_id_new', 'name_new', '_key']].rename(
        columns={'name_new': 'name', '_key': 'source_code'}
    )
    out['level'] = level
    # A unit with no prior row is new to the spine, not renamed; say so
    # rather than emitting an empty cell a consumer might read as a
    # deletion.
    out['admin_id_old'] = out['admin_id_old'].fillna('')
    out['status'] = out.apply(
        lambda r: (
            'new'
            if not r['admin_id_old']
            else ('unchanged' if r['admin_id_old'] == r['admin_id_new'] else 'renamed')
        ),
        axis=1,
    )
    return out[
        ['level', 'status', 'admin_id_old', 'admin_id_new', 'name', 'source_code']
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--before', required=True, help='git ref holding the pre-mint spine'
    )
    parser.add_argument('--scope', nargs='+', default=['US-NC', 'US-TX'])
    parser.add_argument('--levels', nargs='+', type=int, default=[3, 4])
    parser.add_argument('--changed-only', action='store_true')
    args = parser.parse_args()

    frames = [crosswalk(args.before, level, args.scope) for level in args.levels]
    out = pd.concat(frames, ignore_index=True)
    if args.changed_only:
        out = out[out['status'] != 'unchanged']
    out.to_csv(sys.stdout, index=False, lineterminator='\n')

    changed = int((out['status'] == 'renamed').sum())
    print(
        f'# {changed:,} of {len(out):,} units in {", ".join(args.scope)} '
        f'change identifier',
        file=sys.stderr,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
