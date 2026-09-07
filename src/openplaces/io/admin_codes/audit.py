"""Checks an identifier set must pass, and the safe way to repair references.

Two jobs that were learned the hard way and are easy to get wrong again.

:func:`audit_spine` states the invariants: format, hierarchy, uniqueness,
one code width per parent, and -- the one that matters most -- that
re-deriving every code from names alone reproduces the spine. That check
runs with pinning *off*: pinning reads the committed code back out of the
registry, which is the file under audit, so a pinned run reproduces the
spine by construction and would not move on a derivation regression
(measured at level 2: 3648 of 3654 pinned against 3337 derived). Fed the
population weights the mint used, the derived run is the same computation
as the re-mint and must reproduce every code; unweighted it is a floor.

:func:`resolve_identifier` is the only safe way to point an old reference
at a live unit. Identifiers get *recycled*: US-NC-HD named Hyde and now
names Hertford. So an id may not be treated as a stable string, and an id
that looks live may still be the wrong unit. Resolution goes through the
name, scoped to the unit's own resolved parent, and refuses to guess when
the name is ambiguous.
"""

from functools import lru_cache

import pandas as pd

from openplaces.io.admin_codes.candidates import CODE_PATTERN
from openplaces.io.admin_codes.frame import assign_admin_ids
from openplaces.io.admin_codes.registry import spine_path

LEVELS = (2, 3, 4)


def _read(level, superseded=False):
    name = (
        f'admin-spine-2026_superseded-admin{level}.csv'
        if superseded
        else f'admin-spine-2026_admin{level}.csv'
    )
    return pd.read_csv(
        spine_path(level).with_name(name),
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


@lru_cache(maxsize=1)
def _resolution_tables():
    """Live units by (parent, name) and by (state, name), plus past names."""
    children, state_scope, past_names = {}, {}, {}
    for level in LEVELS:
        column = f'admin{level}_id'
        live = _read(level)
        for admin_id, name in zip(live[column], live['name']):
            name = name.strip()
            if not name:
                continue
            children.setdefault((admin_id.rsplit('-', 1)[0], name), []).append(admin_id)
            scope = '-'.join(admin_id.split('-')[:2])
            state_scope.setdefault((scope, name), []).append(admin_id)
        for admin_id, name in zip(
            *(lambda f: (f[column], f['name']))(_read(level, superseded=True))
        ):
            if name.strip():
                past_names.setdefault(admin_id, name.strip())
    return children, state_scope, past_names


def resolve_identifier(admin_id, past_names=None):
    """Point an identifier from an earlier vintage at the live unit.

    Parameters
    ----------
    admin_id : str
        The identifier as some committed file records it.
    past_names : mapping of str to str, optional
        Additional id-to-name knowledge, e.g. read from another branch's
        spine. Merged over the superseded spine shipped beside the live
        one.

    Returns
    -------
    str or None
        The live identifier for that unit, or None when the unit cannot be
        named or its name is ambiguous within its parent.

    Notes
    -----
    Deliberately has no "already live" shortcut. An identifier that exists
    in the current spine may name a *different* unit than it did before --
    that is what recycling means -- so every identifier is resolved
    through its name, and the parent is resolved first by the same rule.
    Returning None is correct behaviour: an ambiguous reference must be
    left alone, never guessed at.
    """
    children, state_scope, known = _resolution_tables()
    names = {**known, **(past_names or {})}

    def step(current):
        name = names.get(current)
        if not name:
            return None
        parent_before = current.rsplit('-', 1)[0]
        parent = parent_before if '-' not in parent_before else step(parent_before)
        if parent is not None:
            hits = children.get((parent, name), [])
            if len(hits) == 1:
                return hits[0]
        # Widen to the state when the parent does not settle it -- either
        # the parent's whole level is gone (New England's counties) or the
        # parent itself no longer resolves, which is the same situation
        # seen from one level down. The state prefix is readable straight
        # off the identifier, so this does not depend on the parent at
        # all; a unique match is still required.
        scope = '-'.join(current.split('-')[:2])
        hits = state_scope.get((scope, name), [])
        return hits[0] if len(hits) == 1 else None

    return step(admin_id)
