"""Split a parcel table's stacked ownership units into a property layer.

A parcel row is a unit of land as the cadastre draws it. Sixteen parcel
sources in the recipe tree stack several ownership records on one lot
polygon instead: every condominium unit of a building carries the
building's outline (Florida's statewide layer, 1.13 million such rows),
or every account on a lot repeats the lot (Texas, North Carolina). Those
rows are properties, and keeping them as parcels double-counts land and
value and multiplies the parcel count (AGENTS.md, "Entity model and
stage roles").

This module regroups such a table, by the source's own lot identity,
into one parcel row per lot and one property row per source record. It
is the ingest-time default for every parcel table (a recipe opts out
with `stacked_units: false`) and is keyed on `geo_id`, the hash of the
row's geometry, unless the recipe names a source lot id (`lot_key`).

Rationale for the mechanism, recorded for the patent check AGENTS.md
asks for on new parcel-record grouping features. The nearest claim read
is claim 1 of US9298740B2 (CoreLogic, 2013), which requires (b) a
repository search for mapping or addressing data associated with a
parcel identifier, (c) verifying whether it matches, (d) flagging the
parcels that do not verify, (e) grouping those non-verified parcels and
(f) normalizing the group. This step does none of (b), (c), (d) or
(f): it searches nothing, verifies nothing, flags nothing and
normalizes nothing (every source record survives unchanged as a
property row), and the grouping is of all rows by the source's own
geometry or lot id, never of a verification failure. Under the
all-elements rule a method that practices none of those elements does
not infringe that claim. Only claim 1 has been read; the patent's other
independent claims are to be checked before a public release (user
decision 2026-09-13).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from openplaces.core.attribute_registry import get_agg_func

#: Recipe key that turns the split off for one parcel table.
OPT_OUT_KEY = 'stacked_units'
#: Recipe key naming the column that identifies a lot; default `geo_id`.
LOT_KEY = 'lot_key'
#: The lot column used when a recipe names none.
DEFAULT_LOT_KEY = 'geo_id'
#: Column the property rows carry to reach their parcel.
LINK_KEY = 'parcel_id_local'
#: Columns that describe the row's identity or geometry, never an
#: attribute a stack's members could disagree on.
_IDENTITY_COLUMNS = ('geometry', 'geo_id')


@dataclass
class SplitResult:
    """What `split_stacked_units` produced for one table."""

    parcels: pd.DataFrame
    properties: pd.DataFrame | None
    n_lots: int
    n_stacks: int
    n_rows_to_properties: int
    n_exact_duplicates: int

    def summary(self) -> str:
        """One line for the ingest log."""
        return (
            f'{self.n_lots:,} lots; {self.n_stacks:,} stacks holding '
            f'{self.n_rows_to_properties:,} rows moved to properties; '
            f'{self.n_exact_duplicates:,} exact duplicates dropped'
        )


def is_enabled(recipe: dict) -> bool:
    """Whether the split applies to a recipe's primary parcel table."""
    entity = recipe.get('entity')
    if entity is None or str(entity.entity_type) != 'parcel':
        return False
    return recipe.get(OPT_OUT_KEY, True) is not False


def lot_key_of(recipe: dict) -> str:
    """The column that identifies a lot for this recipe."""
    return recipe.get(LOT_KEY) or DEFAULT_LOT_KEY


def split_stacked_units(
    df: pd.DataFrame, lot_key: str = DEFAULT_LOT_KEY, link_key: str = LINK_KEY
) -> SplitResult:
    """Regroup a parcel table into one row per lot and a property row per record.

    Rows are grouped by *lot_key*. Within a group, rows that repeat every
    attribute exactly collapse to one (a source that ships each parcel
    twice). A group left with one row is a lot and passes through
    unchanged. A group left with several distinct rows is a stack: the
    parcel row keeps the first member's index and geometry and only the
    attribute values every member agrees on (a varying value is left
    missing, so nothing is double-counted). A column the attribute
    registry aggregates by `sum` (floor areas, values, dwelling counts)
    is left missing on a stack's parcel row even where the members
    agree: one unit's value is not the lot's total, and the total is
    the property layer's to supply. Every member goes,
    unchanged, to the property table with *link_key* set to the parcel
    row's value so that the property spine and the curate aggregation
    reach it.

    Parameters
    ----------
    df : pandas.DataFrame or geopandas.GeoDataFrame
        The preprocessed parcel table, one row per source record, with a
        unique index.
    lot_key : str, optional
        Column identifying the lot (default `geo_id`, the geometry hash).
        Rows with a missing lot key are lots of their own.
    link_key : str, optional
        Column the property rows carry to reach their parcel (default
        `parcel_id_local`). Where a stack's members disagree on it, the
        parcel row takes the lot key's value and the members follow.

    Returns
    -------
    SplitResult
        The parcel table, the property table (None when nothing was
        stacked), and the counts.
    """
    if lot_key not in df.columns:
        raise KeyError(f'lot_key column {lot_key!r} is not in the table.')
    attribute_columns = [
        c for c in df.columns if c not in _IDENTITY_COLUMNS and c != lot_key
    ]
    lot = df[lot_key]
    keyed = lot.notna()
    # Exact duplicates: same lot, same every attribute. Hash the
    # attribute row once rather than comparing object columns pairwise.
    row_hash = pd.util.hash_pandas_object(df[attribute_columns], index=False)
    duplicate = keyed & pd.DataFrame({'lot': lot, 'row': row_hash}).duplicated()
    n_exact = int(duplicate.sum())
    df = df[~duplicate]
    lot = lot[~duplicate]
    keyed = keyed[~duplicate]

    group_size = lot.map(lot[keyed].value_counts()).fillna(1)
    stacked = keyed & (group_size > 1)
    n_lots = int((~stacked).sum() + lot[stacked].nunique())
    if not stacked.any():
        return SplitResult(df, None, n_lots, 0, 0, n_exact)

    members = df[stacked]
    grouped = members.groupby(lot[stacked], sort=False)
    first = grouped.head(1)
    # A value is kept on the parcel row only where every member agrees;
    # nunique counts distinct non-missing values, and a column that is
    # missing on some members but constant on the rest is treated as
    # agreed, since nothing contradicts it.
    agreed = grouped[attribute_columns].nunique(dropna=True).le(1)
    agreed.index = agreed.index.astype(object)
    # Agreement is not a total for an additive column: a building of
    # identical condo units agrees on one unit's living area, and
    # keeping it would stand in for the lot's sum. Curate's
    # aggregate_from_entities fills only empty cells, so a kept value
    # would also block the true sum from the property layer
    # (measured 2026-09-16, plans/floor-area-onto-footprints.md).
    additive = [c for c in attribute_columns if get_agg_func(c) == 'sum']
    if additive:
        agreed[additive] = False
    parcel_rows = first.copy()
    parcel_lots = lot.loc[parcel_rows.index]
    for column in attribute_columns:
        keep = parcel_lots.map(agreed[column]).fillna(False).to_numpy(dtype=bool)
        if not keep.all():
            parcel_rows.loc[~keep, column] = None
    if link_key in attribute_columns:
        missing_link = parcel_rows[link_key].isna()
        if missing_link.any():
            parcel_rows.loc[missing_link, link_key] = parcel_lots[missing_link].astype(
                str
            )
        link_of_lot = dict(zip(parcel_lots, parcel_rows[link_key], strict=True))
    else:
        link_of_lot = {value: str(value) for value in parcel_lots}
    properties = pd.DataFrame(
        members.drop(columns=[c for c in ('geometry',) if c in members.columns])
    )
    properties[link_key] = lot[stacked].map(link_of_lot).to_numpy()
    keep = (~stacked) | df.index.isin(parcel_rows.index)
    parcels = df[keep].copy()
    parcels.loc[parcel_rows.index, attribute_columns] = parcel_rows[attribute_columns]
    return SplitResult(
        parcels,
        properties,
        n_lots,
        int(parcel_rows.shape[0]),
        int(members.shape[0]),
        n_exact,
    )
