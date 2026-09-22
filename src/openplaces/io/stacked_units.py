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
not infringe that claim. The patent's other two independent claims were
read from its text on 2026-09-21 (an agent's reading, not legal advice).
Claim 15, the method claim, recites the same six steps as claim 1.
Claim 11, the system claim, requires identifying a parcel "that could
not be verified by mapping data and addressing data", determining
whether it lies in a wilderness area, grouping it with other parcels if
so, normalizing the parcel on that grouping and storing the result.
This step identifies no unverified parcel, asks nothing about
wilderness and normalizes nothing, so it practices none of claim 11's
elements either. All 19 claims depend on claims 1, 11 or 15.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import pandas as pd
import shapely

from openplaces.core.attribute_registry import get_agg_func

#: Recipe key that turns the split off for one parcel table.
OPT_OUT_KEY = 'stacked_units'
#: Recipe key naming the column that identifies a lot; default `geo_id`.
LOT_KEY = 'lot_key'
#: Recipe key naming the column that identifies a unit within its lot,
#: for the repeated-id count in the ingest log only.
UNIT_KEY = 'unit_key'
#: The lot column used when a recipe names none.
DEFAULT_LOT_KEY = 'geo_id'
#: The parcel table's matching key. A property row keeps its own value.
LINK_KEY = 'parcel_id_local'
#: Column in which a property row names its lot: the `parcel_id_local`
#: of the lot's parcel row. The unit's own `parcel_id_local` is left
#: untouched, because where units have account numbers of their own it
#: is the key their county's tax roll joins on, and overwriting it with
#: the lot's left nothing to map a roll row to the lot (measured on
#: Galveston County TX, 2026-09-20: 5,186 roll rows lost their parcel in
#: 119 of 184 stacks). The pair (own key, lot key) is what
#: `io.harmonizer.entity_links` turns into property-to-parcel link rows,
#: for the split's rows and for any other source's row keyed on a unit.
LOT_LINK_KEY = 'lot_id_local'
#: Appended to the source label of these rows in a property spine
#: (`txgio:units`), so that a `source` or `link_source` token says by
#: itself that the rows were split off a parcel table. A reader summing
#: over a lot ranks them below any roll describing the same lot.
STACKED_UNITS_LABEL_SUFFIX = ':units'
#: Columns that identify a unit within its lot, best first. Only
#: counted and reported, never used to drop a row.
UNIT_KEY_CANDIDATES = (
    'property_id_assessor',
    'property_id_admin2',
    'property_id_cama',
    'parcel_id_assessor',
)
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
    n_multipart_lots: int = 0
    n_parts_merged: int = 0
    n_lots_with_repeated_unit_id: int = 0
    n_rows_with_repeated_unit_id: int = 0
    unit_key: str | None = None

    def summary(self) -> str:
        """One line for the ingest log."""
        parts = [
            f'{self.n_lots:,} lots',
            f'{self.n_stacks:,} stacks holding '
            f'{self.n_rows_to_properties:,} rows moved to properties',
            f'{self.n_exact_duplicates:,} exact duplicates dropped',
        ]
        if self.n_multipart_lots:
            parts.append(
                f'{self.n_multipart_lots:,} lots drawn as several polygons '
                f'({self.n_parts_merged:,} repeated rows merged)'
            )
        elif self.n_parts_merged:
            # Under the geometry hash a second polygon in one group is a
            # collision of the quantized hash: merged as before, but no
            # longer counted as an exact duplicate, so it shows here.
            parts.append(
                f'{self.n_parts_merged:,} rows repeating a record on '
                f'another polygon merged'
            )
        if self.n_rows_with_repeated_unit_id:
            parts.append(
                f'{self.n_rows_with_repeated_unit_id:,} property rows in '
                f'{self.n_lots_with_repeated_unit_id:,} lots repeat their '
                f'{self.unit_key}'
            )
        return '; '.join(parts)


def is_enabled(recipe: dict) -> bool:
    """Whether the split applies to a recipe's primary parcel table."""
    entity = recipe.get('entity')
    if entity is None or str(entity.entity_type) != 'parcel':
        return False
    return recipe.get(OPT_OUT_KEY, True) is not False


def lot_key_of(recipe: dict) -> str:
    """The column that identifies a lot for this recipe."""
    return recipe.get(LOT_KEY) or DEFAULT_LOT_KEY


def unit_key_of(recipe: dict) -> str | None:
    """The column a recipe names as its unit id, if it names one."""
    return recipe.get(UNIT_KEY) or None


def _geometry_key(df: pd.DataFrame) -> pd.Series:
    """A comparable key per row's geometry, empty where there is none."""
    if isinstance(df, gpd.GeoDataFrame) and 'geometry' in df.columns:
        return df.geometry.to_wkb(hex=True).fillna('')
    return pd.Series('', index=df.index)


def _lot_unions(df, lot: pd.Series, lots) -> dict:
    """Union each named lot's polygons into the one outline of the lot."""
    subset = df.loc[lot.isin(set(lots))]
    grouped = subset.geometry.groupby(lot.loc[subset.index], sort=False)
    return grouped.agg(lambda s: shapely.union_all(s.to_numpy())).to_dict()


def _count_repeated_unit_ids(
    properties: pd.DataFrame, link_key: str, unit_key: str | None
) -> tuple[str | None, int, int]:
    """Count property rows whose unit id repeats inside their lot.

    Texas repeats a `Prop_ID` across the records of 19,327 stacks and
    Maine a `STATE_ID`; North Carolina's statewide layer gives a stack's
    accounts no id of their own at all, so every member carries the
    lot's. The rows are distinct records either way and all of them are
    kept: this reports only how far the source's unit id identifies a
    unit, so a recipe author can tell whether a `lot_key` override needs
    a unit id mapped beside it.
    """
    if unit_key is None:
        unit_key = next(
            (c for c in UNIT_KEY_CANDIDATES if c in properties.columns), None
        )
    if unit_key is None or unit_key not in properties.columns:
        return None, 0, 0
    repeated = properties.duplicated(subset=[link_key, unit_key], keep=False)
    repeated &= properties[unit_key].notna()
    return (
        unit_key,
        int(properties.loc[repeated, link_key].nunique()),
        int(repeated.sum()),
    )


def split_stacked_units(
    df: pd.DataFrame,
    lot_key: str = DEFAULT_LOT_KEY,
    link_key: str = LINK_KEY,
    unit_key: str | None = None,
) -> SplitResult:
    """Regroup a parcel table into one row per lot and a property row per record.

    Rows are grouped by *lot_key*. Within a group, rows that repeat every
    attribute and the same geometry collapse to one (a source that ships
    each parcel twice). Rows that repeat every attribute but are drawn as
    different polygons are parts of one lot rather than records of it,
    and they merge the same way, into a lot whose outline is the union of
    the parts. A group left with one row is a lot and passes through
    unchanged. A group left with several distinct rows is a stack: the
    parcel row keeps the first member's index and only the attribute
    values every member agrees on (a varying value is left missing, so
    nothing is double-counted). A column the attribute registry
    aggregates by `sum` (floor areas, values, dwelling counts) is left
    missing on a stack's parcel row even where the members agree: one
    unit's value is not the lot's total, and the total is the property
    layer's to supply. Every member goes, unchanged, to the property
    table, its own keys included, with one column added:
    `lot_id_local`, the parcel row's *link_key* value. That pair is the
    ingest-time knowledge of which unit sits on which lot; the
    property-to-parcel link table is built from it
    (`io.harmonizer.entity_links`), for these rows and for a tax roll's
    rows keyed on the same units.

    The lot's outline is unioned only when *lot_key* names a source lot
    id. Under the default `geo_id` the group is defined by the geometry
    itself, so two members drawn differently are a collision of the
    quantized shape hash, and unioning whatever collided would invent an
    outline no source drew; there the first member's geometry stands.

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
    unit_key : str, optional
        Column identifying a unit within its lot, for the repeated-id
        count only. Defaults to the first of `UNIT_KEY_CANDIDATES` the
        table carries.

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
    has_geometry = isinstance(df, gpd.GeoDataFrame) and 'geometry' in df.columns
    union_parts = has_geometry and lot_key != DEFAULT_LOT_KEY
    lot = df[lot_key]
    keyed = lot.notna()
    # Hash the attribute row once rather than comparing object columns
    # pairwise, and keep the geometry key beside it: a row the source
    # shipped twice and a second part of one lot are indistinguishable
    # without it, and collapsing the part loses the piece it drew.
    row_hash = pd.util.hash_pandas_object(df[attribute_columns], index=False)
    # Only a row sharing its lot with another can be a duplicate or a
    # part, and serializing geometry is the costly step here, so the
    # lots of one row (nearly all of any table) are left unkeyed.
    shares_lot = keyed & lot.duplicated(keep=False)
    geometry_key = pd.Series('', index=df.index)
    if shares_lot.any():
        geometry_key[shares_lot] = _geometry_key(df[shares_lot])
    exact = (
        keyed
        & pd.DataFrame(
            {'lot': lot, 'row': row_hash, 'geometry': geometry_key}
        ).duplicated()
    )
    n_exact = int(exact.sum())
    if n_exact:
        df = df[~exact]
        lot, keyed = lot[~exact], keyed[~exact]
        row_hash, geometry_key = row_hash[~exact], geometry_key[~exact]

    # A lot drawn as several polygons: one source lot id, one record,
    # several rows differing only in their part of the outline (Wilson
    # County NC 1,060 lot ids, Vermont 60, Hertford 7). The union is
    # taken before they merge, so no part is dropped silently.
    unions: dict = {}
    n_multipart = 0
    if union_parts:
        distinct = geometry_key[keyed].groupby(lot[keyed], sort=False).nunique()
        multipart = distinct.index[distinct > 1]
        n_multipart = int(len(multipart))
        if n_multipart:
            unions = _lot_unions(df, lot, multipart)
    part = keyed & pd.DataFrame({'lot': lot, 'row': row_hash}).duplicated()
    n_parts_merged = int(part.sum())
    if n_parts_merged:
        df = df[~part]
        lot, keyed = lot[~part], keyed[~part]

    group_size = lot.map(lot[keyed].value_counts()).fillna(1)
    stacked = keyed & (group_size > 1)
    n_lots = int((~stacked).sum() + lot[stacked].nunique())

    properties = None
    parcel_index = df.index[:0]
    if stacked.any():
        members = df[stacked]
        grouped = members.groupby(lot[stacked], sort=False)
        first = grouped.head(1)
        # A value is kept on the parcel row only where every member
        # agrees; nunique counts distinct non-missing values, and a
        # column that is missing on some members but constant on the
        # rest is treated as agreed, since nothing contradicts it.
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
                parcel_rows.loc[missing_link, link_key] = parcel_lots[
                    missing_link
                ].astype(str)
            link_of_lot = dict(zip(parcel_lots, parcel_rows[link_key], strict=True))
        else:
            link_of_lot = {value: str(value) for value in parcel_lots}
        properties = pd.DataFrame(
            members.drop(columns=[c for c in ('geometry',) if c in members.columns])
        )
        properties[LOT_LINK_KEY] = lot[stacked].map(link_of_lot).to_numpy()
        parcel_index = parcel_rows.index
        keep_rows = (~stacked) | df.index.isin(parcel_index)
        parcels = df[keep_rows].copy()
        parcels.loc[parcel_index, attribute_columns] = parcel_rows[attribute_columns]
    else:
        parcels = df.copy()

    if unions:
        parcels = _apply_lot_unions(parcels, lot_key, unions)

    if properties is None:
        unit_key, n_lots_repeated, n_rows_repeated = None, 0, 0
    else:
        unit_key, n_lots_repeated, n_rows_repeated = _count_repeated_unit_ids(
            properties, LOT_LINK_KEY, unit_key
        )
    return SplitResult(
        parcels,
        properties,
        n_lots,
        int(len(parcel_index)),
        0 if properties is None else int(properties.shape[0]),
        n_exact,
        n_multipart,
        n_parts_merged,
        n_lots_repeated,
        n_rows_repeated,
        unit_key,
    )


def _apply_lot_unions(parcels, lot_key: str, unions: dict):
    """Give each unioned lot its whole outline and a matching `geo_id`.

    `geo_id` is the hash of the row's geometry and the key the geometry
    sidecar is written under, so a lot that has just taken in its other
    parts needs a new one; leaving the first part's would label the whole
    outline with the hash of a piece of it, and a second lot whose own
    geometry hashed there would be served this one.
    """
    from openplaces.geo.ids import get_geo_ids

    mask = parcels[lot_key].isin(unions)
    if not mask.any():
        return parcels
    geometry = parcels.geometry.copy()
    geometry.loc[mask] = parcels.loc[mask, lot_key].map(unions).to_numpy()
    parcels = parcels.set_geometry(geometry)
    if 'geo_id' in parcels.columns:
        parcels.loc[mask, 'geo_id'] = get_geo_ids(
            parcels.loc[mask], handle_duplicates=False
        ).to_numpy()
    return parcels
