"""
Pipeline step assigning a finer admin unit from a name the source states:
  - assign_admin_id_from_name: give each row the id of the unit, inside
    the admin unit being harmonized, whose name (and type) its own
    record names

A non-spatial source often says where a record is in words only: a
transfer return names its municipality ("Saint Germain, Town of") and
carries no code for it. Entities with geometry get their town from a
spatial join; this is the equivalent for a table that has none.

It exists for one consumer so far, the town-scoped address key
(`derive_address_id_local`). A parcel's key is scoped by its town, so
that two streets of one name in two towns of a county stay apart. A
sale with no town id gets an empty scope and its key can match nothing:
measured 2026-09-21 on Vilas County WI, 0 of 12,203 keyed returns found
a parcel by address, and 0.759 did with the scope ignored, which is not
a safe way to match. With the town assigned the scope is the same on
both sides.

The match is exact and all-or-nothing, on purpose. A name is normalized
(case, punctuation, "Saint"/"St", and the type word the Census appends to
some names, as in "Adams city"), compared for equality with the names of
the units of this one admin unit, and used only if exactly one unit
answers. Nothing is scored, nothing close is accepted, and a row that
resolves to no unit or to two keeps a missing id. Measured on all
2,757,462 Wisconsin transfer returns of 2014 to 2026: 99.15% resolve,
none ambiguously; most of the rest name a municipality that changed type
since (a town that incorporated as a village).
"""

from __future__ import annotations

import re
import warnings

import pandas as pd

from openplaces.io.harmonizer import HarmonizeState, _register, normalize_admin_name
from openplaces.io.readers import get_admin

_TYPE_SUFFIX = re.compile(r'\s+(city|town|village|township|borough)$', re.I)


def _normalized_name(names: pd.Series) -> pd.Series:
    """Fold a unit name to the form two sources can agree on."""
    # The type word goes first: a municipality's is the Census suffix on
    # its name ("Adams city"), not part of what the source calls it.
    text = names.astype('string').str.strip()
    return normalize_admin_name(text.str.replace(_TYPE_SUFFIX, '', regex=True))


def match_admin_names(
    names: pd.Series,
    types: pd.Series | None,
    units: pd.DataFrame,
) -> pd.Series:
    """Match stated unit names to admin ids, exactly and uniquely.

    Parameters
    ----------
    names : pandas.Series
        Unit names as the source states them.
    types : pandas.Series or None
        Unit types as the source states them ("Town", "village"), or
        None when the source states none. Used only where given: two
        units of one name (a town and the village inside it) are told
        apart by it.
    units : pandas.DataFrame
        Candidate units, indexed by admin id, with a `name` column and
        optionally a `type` column.

    Returns
    -------
    pandas.Series
        The admin id per row, missing where no unit or more than one
        unit answers.
    """
    unit_key = _normalized_name(units['name'])
    use_type = types is not None and 'type' in units.columns
    if use_type:
        unit_key = unit_key + '|' + units['type'].astype('string').str.upper()
    candidates = pd.Series(units.index, index=unit_key.to_numpy())
    candidates = candidates[~candidates.index.duplicated(keep=False)]
    candidates = candidates[candidates.index.notna()]

    row_key = _normalized_name(names)
    if use_type:
        row_key = row_key + '|' + types.astype('string').str.strip().str.upper()
    return row_key.map(candidates).astype('string')


@_register('assign_admin_id_from_name')
def assign_admin_id_from_name(
    state: HarmonizeState,
    name_column: str,
    level: int,
    name_pattern: str | None = None,
    output_column: str | None = None,
) -> HarmonizeState:
    """Assign each row the admin unit its record names.

    Units are those of *level* inside the admin unit being harmonized. A
    value already present in *output_column* is kept. A no-op when the
    spine lacks *name_column*, or when the admin unit has no units at
    *level* (a New England state, where level 3 is already the town).

    Parameters
    ----------
    name_column : str
        Spine column holding the stated name.
    level : int
        Admin level of the units to assign.
    name_pattern : str, optional
        Regular expression, matched case-insensitively against
        *name_column*, with a named group `name` and optionally a named
        group `type`: how this source spells a unit
        (`'^(?P<name>.*?),?\\s+(?P<type>town|village|city)\\b'` reads
        "Saint Germain, Town of"). It belongs to the recipe because it
        describes one source's format. Without it the whole value is
        the name.
    output_column : str, optional
        Column written. Default `admin{level}_id`.
    """
    spine = state.spine
    if spine is None or name_column not in spine.columns:
        return state
    output_column = output_column or f'admin{level}_id'
    try:
        units = get_admin(state.admin_id, level=level)
    except ValueError:
        return state
    if units is None or len(units) == 0 or 'name' not in units.columns:
        return state

    stated = spine[name_column].astype('string')
    types = None
    if name_pattern:
        parts = stated.str.extract(name_pattern, flags=re.IGNORECASE)
        if 'name' not in parts.columns:
            raise ValueError(
                "assign_admin_id_from_name: name_pattern needs a named group 'name'."
            )
        stated = parts['name']
        types = parts['type'] if 'type' in parts.columns else None

    assigned = match_admin_names(stated, types, units)
    if output_column in spine.columns:
        existing = spine[output_column].astype('string')
        assigned = existing.where(existing.notna(), assigned)
    spine[output_column] = assigned
    state.spine = spine

    share = float(assigned.notna().mean()) if len(assigned) else 0.0
    if state.verbose:
        print(
            f'  assign_admin_id_from_name: {share:.1%} of {len(spine):,d} rows '
            f'assigned {output_column}'
        )
    if len(assigned) and share < 0.5 and stated.notna().mean() > 0.5:
        warnings.warn(
            f'assign_admin_id_from_name: only {share:.1%} of rows naming a '
            f'unit in {name_column!r} resolved for {state.admin_id}. The name '
            'pattern may not fit this source.'
        )
    return state
