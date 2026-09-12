"""Checks an identifier set must pass.

:func:`audit_spine` states the invariants: format, hierarchy, uniqueness,
one code width per parent, and -- the one that matters most -- that
re-deriving every code from names alone reproduces the spine. That check
runs with pinning *off*: pinning reads the committed code back out of the
registry, which is the file under audit, so a pinned run reproduces the
spine by construction and would not move on a derivation regression
(measured at level 2: 3648 of 3654 pinned against 3337 derived). Fed the
population weights the mint used, the derived run is the same computation
as the re-mint and must reproduce every code; unweighted it is a floor.

Only the present spine is audited, because it is the only one kept.
There is no record of earlier identifiers to resolve a retired one
against: a file keyed on admin ids that outlives a re-mint is regenerated
from its source's own codes.
"""

import pandas as pd

from openplaces.io.admin_codes.candidates import CODE_PATTERN
from openplaces.io.admin_codes.frame import assign_admin_ids
from openplaces.io.admin_codes.registry import spine_path

LEVELS = (2, 3, 4)


def _read(level):
    return pd.read_csv(
        spine_path(level),
        dtype=str,
        keep_default_na=False,
        encoding='utf-8',
    )


def audit_spine(levels=LEVELS, reproduce=True, weights=None):
    """Check every invariant an identifier set must satisfy.

    Parameters
    ----------
    levels : iterable of int, optional
        Admin levels to check. Defaults to 2, 3 and 4.
    reproduce : bool, optional
        Also re-derive every code at each level from names alone, with
        the registry pinning switched off, and count the codes the
        derivation agrees with. Without `weights` the pass runs
        unweighted, so the audit needs nothing from the data tree, and
        the count is a floor rather than an invariant: the spine was
        minted with population deciding contested codes (measured
        2026-09-06: 3337 of 3654 at level 2, 40588 of 48696 at level 3,
        185306 of 218754 at level 4).
    weights : mapping of int to pandas.Series, optional
        Population per admin id, keyed by level, as read from
        `build.population_path`. With the weights the mint used, the
        derivation is the same computation as `build.remint_spine` and
        must reproduce every code (48696 of 48696 at level 3); a
        shortfall is a real regression in the generator or the weights.
        A unit missing from the series competes with weight zero.

    Returns
    -------
    pandas.DataFrame
        One row per level, with the count of violations found by each
        check. A clean spine is all zeros except `reproduced`, the number
        of codes the derivation agrees with: equal to `units` when
        weighted, a documented floor when not.

    Notes
    -----
    Pinning must stay off here. `assign_admin_ids` pins by default, and
    pinned it reads each committed code back out of the registry, which
    is the file being audited; the check then reports the spine
    reproduced whatever the generator would derive. That is how a
    resolver defect went unnoticed: the number never moved.
    """
    rows = []
    for level in levels:
        column = f'admin{level}_id'
        spine = _read(level)
        codes = spine[column].str.rsplit('-', n=1).str[1]
        parents = spine[column].str.rsplit('-', n=1).str[0]

        widths = pd.DataFrame({'parent': parents, 'width': codes.str.len()})
        mixed = int((widths.groupby('parent')['width'].nunique() > 1).sum())

        orphans = 0
        if level > min(levels):
            above = _read(level - 1)[f'admin{level - 1}_id']
            orphans = int((~parents.isin(set(above))).sum())

        reproduced = None
        if reproduce:
            work = spine.copy()
            work['_parent'] = parents
            weight_col = None
            if weights is not None and level in weights:
                weight_col = '_population'
                population = pd.Series(weights[level], dtype=float)
                work[weight_col] = spine[column].map(population).fillna(0.0).to_numpy()
            # pin_to_spine=False is the point of the check: a pinned run
            # copies the committed code out of the registry and proves
            # nothing about whether the names derive it.
            minted = assign_admin_ids(
                work,
                new_admin_id_col=column,
                parent_admin_id_col='_parent',
                weight_col=weight_col,
                pin_to_spine=False,
            )
            reproduced = int(
                (
                    minted.index.to_numpy(dtype=object)
                    == spine[column].to_numpy(dtype=object)
                ).sum()
            )

        rows.append(
            {
                'level': level,
                'units': len(spine),
                'bad_format': int((~codes.str.fullmatch(CODE_PATTERN.pattern)).sum()),
                'orphan_parents': orphans,
                'duplicate_ids': int(spine[column].duplicated().sum()),
                'mixed_width_parents': mixed,
                'reproduced': reproduced,
            }
        )
    return pd.DataFrame(rows).set_index('level')
