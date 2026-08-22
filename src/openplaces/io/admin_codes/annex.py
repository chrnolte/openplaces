"""Regenerate Annex A of the vote package from the current spine.

The annex says of itself: "Generated from the current spine. Regenerate
after any re-mint; do not edit by hand." No generator existed, and several
re-mints have happened since it was written, so 49 of the identifiers it
cited no longer exist. This is that generator.

Everything printed here is read from the committed spine or recomputed with
the production code path. Nothing is transcribed.
"""

import sys

import pandas as pd

from openplaces.io.admin_codes import spine_path
from openplaces.io.admin_codes.assign import assign_codes
from openplaces.io.admin_codes.candidates import generate_candidates
from openplaces.io.admin_codes.derive import (
    MIN_GROUP_FOR_OPACITY,
    OPAQUE_RULES,
)

sys.stdout.reconfigure(encoding='utf-8')

COUNTRIES = [
    ('US', 'United States', 'states', 'counties and New England towns'),
    ('DE', 'Germany', 'states', 'districts'),
    ('IR', 'Iran', 'provinces', 'counties'),
    ('CO', 'Colombia', 'departments', 'municipalities'),
    ('NG', 'Nigeria', 'states', 'LGAs'),
    ('TR', 'Turkey', 'provinces', 'districts'),
    ('KR', 'South Korea', 'provinces', 'districts'),
]

# What each rule name means, for the worked-example table.
MEANING = {
    'name': 'first letters of the name',
    'name.ends': 'first letters plus the final letter',
    'skeleton': 'consonant skeleton',
    'initials': 'initials of the significant words',
    'compound': 'first word plus the next initial',
    'compound.alt': 'first initial plus the second word',
    'conjunction': 'initials either side of the conjunction',
    'conjunction.alt': 'first initial plus the second term',
    'preposition': 'initials of all tokens including the preposition',
    'preposition.significant': 'initials of the significant words',
    'preposition.alt': 'first word plus the last significant initial',
    'qualifier': 'leading qualifier plus the head noun',
    'qualifier.spread': 'qualifier plus spread letters of the head',
    'qualifier.alt': 'two letters of the qualifier plus the head',
    'anchor': 'published or reviewed code, not derived',
    'any2': 'OPAQUE -- letters from the name, not in a guessable shape',
    'any3': 'OPAQUE -- letters from the name, not in a guessable shape',
    'letter_number': 'OPAQUE -- first letter plus digits from the name',
    'sequential': 'NO SIGNAL -- fallback, nothing from the name',
    'placeholder': 'NO SIGNAL -- the unit carries no name',
}

SETTINGS = [('looser', 0.33, 333), ('standard', 0.25, 250), ('tighter', 0.17, 168)]


def load(level):
    df = pd.read_csv(spine_path(level), dtype=str, keep_default_na=False)
    idc = f'admin{level}_id'
    df = df[df[idc].str.strip() != ''].copy()
    df['rule'] = df[f'{idc}_source']
    df['parent'] = df[idc].str.rsplit('-', n=1).str[0]
    return df.rename(columns={idc: 'admin_id'})


S2, S3 = load(2), load(3)


def opacity_two_char(names):
    """Recompute what two characters would have produced for a group."""
    cands = {
        f'{i}:{n}': generate_candidates(n, lengths=(2,)) for i, n in enumerate(names)
    }
    assigned = assign_codes(cands)
    rules = [r for _, r in assigned.values()]
    return sum(1 for r in rules if r in OPAQUE_RULES) / len(rules)


def widen_counts(country):
    """Parents that would widen at each of the three settings, level 3."""
    kids = S3[S3['admin_id'].str.startswith(f'{country}-')]
    out = {}
    for label, max_opaque, max_siblings in SETTINGS:
        widened = []
        for parent, grp in kids.groupby('parent'):
            n = len(grp)
            if n > max_siblings:
                widened.append(parent)
                continue
            if n < MIN_GROUP_FOR_OPACITY:
                continue
            try:
                if opacity_two_char(list(grp['name'])) > max_opaque:
                    widened.append(parent)
            except Exception:  # noqa: BLE001 - skip an unsolvable group
                continue
        out[label] = sorted(widened)
    return out


def sheet(country, label, l2_noun, l3_noun):
    l2 = S2[S2['admin_id'].str.startswith(f'{country}-')]
    l3 = S3[S3['admin_id'].str.startswith(f'{country}-')]
    lines = [f'### {label} (`{country}`)', '']

    sizes = l3['parent'].value_counts()
    biggest = sizes.index[0] if len(sizes) else None
    l2_op = opacity_two_char(list(l2['name'])) if len(l2) else 0.0

    lines += [
        f'| | {l2_noun} | {l3_noun} |',
        '|---|---:|---:|',
        f'| units | {len(l2):,} | {len(l3):,} |',
        f'| opacity at 2 characters | {l2_op:.0%} | see below |',
        f'| largest group | -- | {sizes.iloc[0]} (`{biggest}`) |'
        if biggest
        else '| largest group | -- | -- |',
        '',
    ]

    counts = widen_counts(country)
    summary = ' · '.join(
        f'{lab} {int(mo * 100)}%/{ms}: {len(counts[lab])}' for lab, mo, ms in SETTINGS
    )
    lines.append(f'**Three-character groups** -- {summary}')
    standard = counts['standard']
    lines.append(
        '  At the standard setting: '
        + (', '.join(f'`{p}`' for p in standard) if standard else 'none')
    )
    lines.append('')

    lines.append(f'**All {len(l2)} codes, {l2_noun}**')
    lines.append('')
    lines.append(
        ' · '.join(
            f'`{r.admin_id}` {r.name}'
            for r in l2.sort_values('admin_id').itertuples(index=False)
        )
    )
    lines.append('')

    if biggest is not None:
        grp = l3[l3['parent'] == biggest]
        parent_name = l2.loc[l2['admin_id'] == biggest, 'name']
        pretty = parent_name.squeeze() if len(parent_name) else biggest
        try:
            grp_op = opacity_two_char(list(grp['name']))
            op_text = f', {grp_op:.0%} opaque at two characters'
        except Exception:  # noqa: BLE001 - omit rather than guess
            op_text = ''
        lines += [
            f'**How the logic plays out -- `{biggest}` ({pretty}), '
            f'{len(grp)} {l3_noun}{op_text}**',
            '',
            '| code | unit | rule | what it means |',
            '|---|---|---|---|',
        ]
        # Up to two worked examples per rule that actually fires, with the
        # opaque rules last so the readable cases lead.
        seen = {}
        for row in grp.sort_values('admin_id').itertuples(index=False):
            seen.setdefault(row.rule, []).append(row)
        ordered = sorted(
            seen, key=lambda r: (r in OPAQUE_RULES or r == 'placeholder', r)
        )
        for rule in ordered:
            for row in seen[rule][:2]:
                lines.append(
                    f'| `{row.admin_id}` | {row.name} | {rule} | '
                    f'{MEANING.get(rule, rule)} |'
                )
        lines.append('')
    return '\n'.join(lines)


print('<!-- Annex A regenerated from the spine; do not edit by hand. -->')
print()
for country, label, n2, n3 in COUNTRIES:
    print(sheet(country, label, n2, n3))
