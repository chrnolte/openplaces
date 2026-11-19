"""A property's values stay off the lot: what `link_by_id` notes when a
table of properties writes onto a parcel spine, and the step that takes
those writes back (`withhold_property_values`).

A roll account, or a unit of a bundled property layer, states one land
value, one improvement value, one total and one dwelling count. The
parcel geospine's link passes reduce such rows per lot key and write the
result onto every lot naming the key, which puts an account drawn as
several lots on each of them whole. Those copies are what the parcel
curate's `divide_account_values` replaces: it divides each account over
its lots from the property spine and the link table, and is the one
step that gives a lot a roll value (plans/account-values-one-deciding-
step.md, package V4; AGENTS.md, stage roles: an attribute lands once,
on the entity it describes).

Why the passes still write in memory, and the copies leave at the end
of the geospine instead of never being made. Three harmonize steps read
a lot's value to decide geometry or links, not to state it: the
shared-land groups and the common-area contiguity links of the parcel
geospine, and the building geospine's cut, which replays the passes in
memory to ask whether a lot records a structure. They keep reading what
they read before, the cut's pass configurations and its sidecar
fingerprint do not change (so building ids stay), and no link sidecar
of the parcel geospine goes stale, since the step runs after all of
them. What the geospine persists is the parcel layer's own statement.

A value a parcel layer states on its own rows (a statewide layer such
as Florida's, Texas's or Wisconsin's, DC's tax lots, NC OneMap joined
onto a county's geometry) is that layer's statement about its polygon,
not a copy from another entity, and stays: a cell is taken back only
where the last write to it came from a table of properties, and it
returns to what it held before the first such write, value and label.
A value the roll rule `not_on_roll` emptied stays empty: that write is
not a property's.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from openplaces.io.harmonizer import (
    _PROVENANCE_SUFFIX,
    HarmonizeState,
    _register,
)
from openplaces.io.harmonizer.links.combine import _blank_as_missing

#: `state.metadata` key holding, per column, the cells a property write
#: holds and what they held before it.
PROPERTY_WRITES_KEY = 'property_writes'

#: Suffix of the column a withheld value is kept in, where a later
#: harmonize step still needs it (AGENTS.md: a preliminary column is
#: labeled preliminary).
PRELIMINARY_SUFFIX = '_preliminary'


def _same(a: pd.Series, b: pd.Series) -> pd.Series:
    """Cell-wise equality with missing equal to missing."""
    both_missing = a.isna() & b.isna()
    try:
        equal = (a == b).fillna(False).astype(bool)
    except (TypeError, ValueError):
        equal = pd.Series(
            [x == y for x, y in zip(a.astype(object), b.astype(object), strict=True)],
            index=a.index,
            dtype=bool,
        )
    return equal | both_missing


def snapshot(spine: pd.DataFrame, name: str) -> tuple[pd.Series, pd.Series]:
    """A column's values and labels before a write (missing if absent)."""
    side = f'{name}{_PROVENANCE_SUFFIX}'
    values = (
        spine[name].copy()
        if name in spine.columns
        else pd.Series(np.nan, index=spine.index, dtype=object)
    )
    labels = (
        spine[side].astype(object).copy()
        if side in spine.columns
        else pd.Series(pd.NA, index=spine.index, dtype=object)
    )
    return values, labels


def note_write(
    state: HarmonizeState,
    spine: pd.DataFrame,
    name: str,
    before: tuple[pd.Series, pd.Series],
    from_properties: bool,
) -> None:
    """Record which cells of *name* a write changed, and by whom.

    A cell a table of properties changed (value or label) is held by a
    property from then on, and the first such write keeps what the cell
    held before it. A cell any other write changed (a parcel table, the
    roll rule emptying a coarser value) is no longer held: it states
    what that write put there.

    Parameters
    ----------
    state : HarmonizeState
        The run; the record lives in its metadata.
    spine : pandas.DataFrame
        The spine after the write.
    name : str
        The column written.
    before : tuple of pandas.Series
        `snapshot` of the column taken before the write.
    from_properties : bool
        Whether the write came from rows that are properties (a roll, a
        bundled property layer).
    """
    old_values, old_labels = before
    # A write turns an empty string into missing on every row it
    # touches, stated or not (`_blank_as_missing`): that is no one's
    # statement, so it neither marks a cell held nor is kept as the
    # value a cell returns to.
    old_values = _blank_as_missing(old_values)
    new_values, new_labels = snapshot(spine, name)
    changed = ~_same(old_values, new_values) | ~_same(old_labels, new_labels)
    if not changed.any():
        return
    records = state.metadata.setdefault(PROPERTY_WRITES_KEY, {})
    record = records.get(name)
    if record is None:
        record = pd.DataFrame(
            {
                'held': pd.Series(False, index=spine.index),
                'value': pd.Series(np.nan, index=spine.index, dtype=object),
                'label': pd.Series(pd.NA, index=spine.index, dtype=object),
            }
        )
        records[name] = record
    record = record.reindex(spine.index)
    record['held'] = record['held'].fillna(False).astype(bool)
    if from_properties:
        first = changed & ~record['held']
        record.loc[first, 'value'] = old_values[first].astype(object)
        record.loc[first, 'label'] = old_labels[first]
        record.loc[changed, 'held'] = True
    else:
        record.loc[changed, 'held'] = False
    records[name] = record


def replay_on_held(
    state: HarmonizeState,
    name: str,
    write: Callable[[pd.DataFrame], None],
) -> None:
    """Replay a write that is no property's on what held cells return to.

    A cell a property holds keeps the property's value in memory, so a
    later parcel table that only fills (a `fill_only` pass, or a lot
    another parcel layer draws) finds it full and writes nothing there.
    Without a replay, `withhold_property_values` would return the cell
    to what it held before the property wrote, and the parcel table's
    value would be lost. Run on those earlier values instead, the write
    decides what the cell holds once the property's copy is withdrawn,
    as it would have without the copy.

    Parameters
    ----------
    state : HarmonizeState
        The run; the record lives in its metadata.
    name : str
        The column written.
    write : callable
        The write just made on the spine, taking the frame to write
        into; called once on a frame of the held cells' earlier values
        and labels (the spine's own index).
    """
    record = (state.metadata.get(PROPERTY_WRITES_KEY) or {}).get(name)
    if record is None:
        return
    held = record['held'].fillna(False).astype(bool)
    if not held.any():
        return
    side = f'{name}{_PROVENANCE_SUFFIX}'
    earlier = pd.DataFrame(
        {name: record['value'].astype(object), side: record['label'].astype(object)},
        index=record.index,
    )
    write(earlier)
    record.loc[held, 'value'] = earlier.loc[held, name].astype(object)
    record.loc[held, 'label'] = earlier.loc[held, side].astype(object)


def _preliminary_columns(arguments) -> list[str]:
    return [f'{c}{PRELIMINARY_SUFFIX}' for c in arguments.get('preliminary') or () if c]


@_register(
    'withhold_property_values',
    phase='geometry',
    reads=['{columns}', '{columns}_source'],
    writes=['{columns}', '{columns}_source', _preliminary_columns],
)
def withhold_property_values(
    state: HarmonizeState,
    columns: list[str] = (),
    preliminary: list[str] = (),
) -> HarmonizeState:
    """Take back what tables of properties wrote onto the lots.

    Every cell of *columns* whose last write came from rows that are
    properties (a roll's pass, a bundled property layer such as
    MassGIS's L3_ASSESS or the stacked units every parcel ingest
    splits off) returns to the value and `_source` it held before the
    first such write: the parcel layer's own statement, or missing.
    Everything else stays: a parcel table's values, a value the roll
    rule `not_on_roll` emptied. The record is the one `link_by_id`
    keeps while the passes run (`note_write`), so the step reads
    nothing from disk. See the module docstring for why the copies are
    made in memory and taken back here, at the end of the geospine.

    Parameters
    ----------
    state : HarmonizeState
        The run; its spine is changed in place.
    columns : list of str
        Columns to take the property writes back from (the parcel
        geospine names the account values and the dwelling count).
    preliminary : list of str
        Columns of *columns* whose value before this step a later
        harmonize step still reads. Each is kept, for every lot, as
        `{column}_preliminary` (the parcel spine's
        `estimate_property_counts` reads `n_dwellings_preliminary`). A
        preliminary column is never a curated value: the parcel curate
        decides the column itself from the property spine.

    Returns
    -------
    HarmonizeState
        The state, its spine holding no property's value in *columns*.
    """
    spine = state.spine
    if spine is None or not len(columns):
        return state
    records = state.metadata.get(PROPERTY_WRITES_KEY) or {}
    counts = {}
    for name in preliminary or ():
        if name in spine.columns:
            spine[f'{name}{PRELIMINARY_SUFFIX}'] = spine[name].copy()
    for name in columns:
        record = records.get(name)
        if record is None or name not in spine.columns:
            continue
        record = record.reindex(spine.index)
        held = record['held'].fillna(False).astype(bool)
        if not held.any():
            continue
        # Each column is built anew with `where`, never assigned into:
        # under pandas copy-on-write an `astype` that changes nothing
        # can hand back a read-only block, and `.loc` assignment into
        # it raised on CI (2026-11-19).
        if pd.api.types.is_numeric_dtype(spine[name]):
            prior = pd.to_numeric(record['value'], errors='coerce')
            spine[name] = (
                spine[name].astype('float64').where(~held, prior.astype('float64'))
            )
        else:
            spine[name] = (
                spine[name].astype(object).where(~held, record['value'].astype(object))
            )
        side = f'{name}{_PROVENANCE_SUFFIX}'
        if side in spine.columns:
            was_category = isinstance(spine[side].dtype, pd.CategoricalDtype)
            # A cell left without a value names no source.
            restored = record['label'].astype(object).where(spine[name].notna(), pd.NA)
            labels = spine[side].astype(object).where(~held, restored)
            spine[side] = labels.astype('category') if was_category else labels
        counts[name] = int(held.sum())
    state.spine = spine
    if state.verbose:
        print(f'  withhold_property_values: cells returned to the layer {counts}')
    return state
