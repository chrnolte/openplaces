"""Registered curation steps that reduce one entity's rows onto another.

Curate assembles one dataset per entity type by aggregating across
entities: a parcel's year built or living area comes from the property
rows on it, reduced here, not from a copy the geometry phase made. The
harmonize stage keeps each entity's core table with minimal redundancy
(see AGENTS.md, "Entity model and stage roles"), so a curate recipe that
wants a property attribute on parcels asks for it here.
"""

from __future__ import annotations

import warnings

import pandas as pd

from openplaces.core.attribute_registry import get_agg_func
from openplaces.io.curator import CurateState, _register
from openplaces.io.curator.provenance import record_sources, source_column
from openplaces.io.readers import describe_recipe, get_entities
from openplaces.io.stacked_units import STACKED_UNITS_LABEL_SUFFIX
from openplaces.recipe import resolve_attribute_name, source_id_from_recipe_id
from openplaces.table import _agg_func_for, _has_agg_func

#: Reference column naming the source each row came from whole, as
#: `union_spine_sources` writes it on a property spine.
ROW_SOURCE_COLUMN = 'source'


def _row_source_tokens(labels: pd.Series) -> pd.Series:
    """A spine's per-row `source` labels as provenance tokens.

    A delivery withholds a restricted source's cell by finding its
    source id among the sidecar's tokens, and reads any other token as
    a different source that exonerates the cell. So each `+` part is
    reduced to the source id: the stacked-units label
    (`edgecombecounty:units`) loses its suffix, and an explicit source
    labeled with its recipe id becomes that recipe's source id. Copied
    verbatim, either would have cleared Edgecombe County's restricted
    values for shipping.
    """

    def normalize(label):
        parts = []
        for part in str(label).split('+'):
            part = part.removesuffix(STACKED_UNITS_LABEL_SUFFIX)
            if '_' in part:
                part = source_id_from_recipe_id(part)
            if part and part not in parts:
                parts.append(part)
        return '+'.join(parts) or None

    values = labels.astype('string')
    lookup = {v: normalize(v) for v in values.dropna().unique()}
    return values.map(lookup).astype(object)


def _linked_recipe_id(recipe_id: str) -> str:
    """The recipe whose rows a link table was written for.

    A link table is written in harmonize, between spines. A curate
    recipe keeps its spine's index (`US_property-openplaces-2026`'s
    rows are `US_property-spine-2026`'s property ids), so its rows are
    reached through the spine's link.
    """
    from openplaces.recipe import get_recipe_by_id

    seen = set()
    while recipe_id not in seen:
        seen.add(recipe_id)
        recipe = get_recipe_by_id(recipe_id)
        predecessor = recipe.get('entity_recipe')
        if recipe.get('stage') != 'curate' or not predecessor:
            return recipe_id
        recipe_id = predecessor
    return recipe_id


def _rows_by_link(state, recipe_id, reference, values, functions):
    """Arrange the reference's values by the link table, when there is one.

    A property reaches its parcel through a key column only where one
    key can say it. The link table (`io.harmonizer.entity_links`) also
    holds a unit on a stacked lot, whose key names no parcel row, and an
    account on several lots. Where a valid link exists for the unit
    being curated, the rows are grouped by it:

    - links found on a key shared by very many rows on both sides
      (`*_shared_key`) are left out, because summing over them is the
      harm that label records;
    - on a lot that any source other than the ingest-time split
      describes, the split's own rows (`<source>:units`) are left out,
      so a tax roll and the units split off the parcel layer are not
      summed twice. A row both describe is one merged row and stays;
    - a property on several lots contributes its `share` of every
      summed column to each.

    Returns None when there is no valid link or it shares no ids with
    the two tables, and the caller joins on the key column as before.
    """
    from openplaces.geo.link import get_entity_link_path, get_link_owner_recipe_id
    from openplaces.io.harmonizer.entity_links import (
        SHARED_KEY_SUFFIX,
        read_entity_link,
    )
    from openplaces.io.stacked_units import STACKED_UNITS_LABEL_SUFFIX

    try:
        owner = get_link_owner_recipe_id(state.recipe)
        path = get_entity_link_path(_linked_recipe_id(recipe_id), owner, state.admin_id)
        links = read_entity_link(path)
    except Exception:
        # No resolvable link for this recipe and unit (a curate recipe
        # with no geospine, a reference outside ENTITY_LINK_ORDER): the
        # key join below is the documented behavior.
        return None
    if links is None or links.empty:
        return None
    finer_id, coarser_id = links.columns[0], links.columns[1]
    # Which side the curated entity is: the property-to-parcel table
    # reduces finer rows (properties) onto the coarser curated parcels;
    # the transaction-to-parcel table reduces the coarser side (parcels)
    # onto the finer curated sales. The id columns are named for their
    # entity, and so is the curated index.
    curated_name = state.curated.index.name
    if curated_name == finer_id:
        own_id, other_id = finer_id, coarser_id
    else:
        own_id, other_id = coarser_id, finer_id
    other_ids = links[other_id].astype('string')
    own_ids = links[own_id].astype('string')
    links = links[
        other_ids.isin(pd.Index(reference.index).astype('string'))
        & own_ids.isin(pd.Index(state.curated.index).astype('string'))
    ]
    if links.empty:
        warnings.warn(
            f'aggregate_from_entities: the link {path.name} shares no ids with '
            f'the tables for {state.admin_id}; joining on the key column.',
            stacklevel=3,
        )
        return None
    links = links[~links['link_method'].str.endswith(SHARED_KEY_SUFFIX, na=False)]
    tokens = links['link_source'].fillna('').str.split('+')
    only_units = tokens.map(
        lambda parts: all(p.endswith(STACKED_UNITS_LABEL_SUFFIX) for p in parts)
    )
    described = set(links.loc[~only_units, own_id])
    links = links[~(only_units & links[own_id].isin(described))]

    reference_lookup = values.copy()
    reference_lookup.index = pd.Index(reference.index).astype('string')
    rows = reference_lookup.loc[links[other_id].astype('string')].reset_index(drop=True)
    share = links['share'].astype('Float64').reset_index(drop=True)
    for column, function in functions.items():
        if function == 'sum' and share.notna().any():
            scaled = pd.to_numeric(rows[column], errors='coerce') * share.astype(
                'float64'
            )
            rows[column] = scaled.where(share.notna(), rows[column])
    group_key = links[own_id].astype('string').reset_index(drop=True)
    index_as_string = pd.Index(state.curated.index).astype('string')
    keys = pd.Series(index_as_string, index=state.curated.index)
    if state.verbose:
        print(
            f'  aggregate_from_entities: grouped by {path.name} ({len(links):,d} links)'
        )
    return rows, group_key, keys


#: Reduction for a class a parcel can only state if its properties
#: agree on it (see `_reduce`).
AGREE = 'agree'


def _reduce(values: pd.DataFrame, group_key, functions: dict) -> pd.DataFrame:
    """Reduce *values* per group, one function per column.

    `agree` keeps a group's value only where every row that states one
    states the same one, and leaves the group missing otherwise. It is
    the rule the ingest-time stacked-units split applies to a lot's
    units, and it exists for a class such as a foundation type: a
    parcel with a slab house and a crawl-space house has no single
    foundation, and `first` would name whichever row came first.
    """
    plain = {c: f for c, f in functions.items() if f != AGREE}
    grouped = values[list(functions)].groupby(group_key)
    if plain:
        reduced = grouped.agg(plain)
        # pandas sums an all-missing group to 0; a parcel none of whose
        # properties states a floor area has no floor area, not a zero one.
        reduced = reduced.where(grouped.count()[list(plain)] > 0)
    else:
        reduced = pd.DataFrame(index=grouped.size().index)
    for column in functions:
        if functions[column] != AGREE:
            continue
        by_group = values[column].groupby(group_key)
        reduced[column] = by_group.first().where(by_group.nunique() == 1)
    return reduced[list(functions)]


def _union_tokens(values: pd.DataFrame, group_key, column: str, sidecar: str):
    """Per group, the provenance tokens of the rows that state *column*.

    Tokens are the `+`-separated parts of each row's sidecar, kept once
    each in order of first appearance, so a parcel whose properties
    came from two sources names both.
    """
    stating = values[column].notna() & values[sidecar].notna()
    tokens = values.loc[stating, sidecar].astype(str)

    def union(series):
        parts = dict.fromkeys(t for v in series for t in v.split('+') if t)
        return '+'.join(parts) or None

    return tokens.groupby(group_key[stating]).agg(union)


@_register('aggregate_from_entities', phase='gather')
def aggregate_from_entities(
    state: CurateState,
    recipe_id: str,
    columns: list | dict,
    key: str | list[str] | None = 'parcel_id_local',
    reference_key: str | list[str] | None = None,
    fill_only: bool = True,
    recover: list[str] | None = None,
) -> CurateState:
    """Reduce a reference entity's rows onto the curated rows they belong to.

    Reads the reference recipe for the unit being curated, groups its rows
    by *reference_key*, reduces each named column with the registry's
    aggregation function (or the override given for it), and writes the
    result onto the curated rows by *key*. A parcel's year built becomes
    the earliest of its properties', its living area the sum, a room count
    the sum, without the harmonize stage ever having copied them.

    Where a link table between the two entities exists for the unit
    (`io.harmonizer.entity_links`), the rows are grouped by it instead of
    by the key, whichever side of the link the curated entity is: curated
    parcels take their properties' rows, curated sales take their
    parcels' rows through the transaction-to-parcel table, each parcel's
    sums scaled by the sale's `share` of it. With *key* None the link is
    the only route, and a unit without one is skipped with a warning.

    A column named in *recover* is written under a stricter rule, made
    for a value a stacked lot lost at ingest: the lot row keeps only what
    its units agree on, so a condominium lot's improvement value is
    missing or zero while its units carry the figure. The reduction is
    written only where it is positive, onto a cell that is missing or
    not positive. Nothing is written where no reference row records a
    positive value: a lot whose roll does not value the building (a
    manufactured home on a personal-property title, an exemption,
    business personal property, agricultural use value) stays as it was,
    because filling it would be estimation, which is a declared
    imputation step or nothing (maintainer's decision, 2026-09-22).
    The recovered figure is recorded, not modeled, so it carries no
    imputed marker.
    Measured 2026-09-22 on cheer-eastern-nc: 179 of 503 such lots, $44.1M.
    Measured 2026-09-22 on cheer-coastal-tx: 447 of 15,555, $64.8M (Texas
    taxes business personal property and manufactured homes on their own
    rolls, so its gaps are mostly real).

    Parameters
    ----------
    recipe_id : str
        The entity recipe to reduce (`US_property-spine-2026`). Named here
        so the recipe graph sees the dependency.
    columns : list of str or dict
        Columns to reduce. A list uses the registry's aggregation for each
        column; a mapping gives a per-column override (`{year_built: min,
        n_stories: max}`), for a column whose registry rule (a mean for
        both) is not what the parcel needs. `agree` writes a value only
        where every reference row stating one states the same one, for
        a class such as `foundation_type`. Where the reference carries
        a `{column}_source` sidecar, the curated column gets one too,
        naming the tokens of the rows its value came from. A column
        with no sidecar takes its tokens from the reference's per-row
        `source` (a property spine's), reduced to source ids.
    key : str or list of str or None, optional
        Curated column the reference joins on (default `parcel_id_local`).
        A list is tried in order and the first column present on both
        sides is used, so one recipe serves a source keyed on the local
        parcel id and one keyed on a state-wide id (`parcel_id_admin2`).
        None: no key join at all; the link table alone places the rows.
    reference_key : str or list of str, optional
        Reference column to group by; defaults to *key*.
    fill_only : bool, optional
        Fill only curated cells that are missing (default). With False the
        reduced value replaces whatever the curated column held.
    recover : list of str, optional
        Columns written only where the reduction is positive, onto cells
        that are missing or not positive (see above). Takes precedence
        over *fill_only* for the columns it names.
    """
    curated = state.curated
    link_only = key is None
    keys_ = [] if link_only else [key] if isinstance(key, str) else list(key)
    reference_keys = (
        keys_
        if reference_key is None
        else [reference_key]
        if isinstance(reference_key, str)
        else list(reference_key)
    )
    spec = {c: None for c in columns} if isinstance(columns, list) else dict(columns)
    try:
        available = set(describe_recipe(recipe_id, state.admin_id).index)
    except (FileNotFoundError, OSError, ValueError) as exc:
        warnings.warn(
            f'aggregate_from_entities: could not read {recipe_id} for '
            f'{state.admin_id}: {exc}. Skipping.',
            stacklevel=2,
        )
        return state
    pair = next(
        (
            (k, r)
            for k, r in zip(keys_, reference_keys, strict=True)
            if k in curated.columns and r in available
        ),
        None,
    )
    if pair is None and not link_only:
        warnings.warn(
            f'aggregate_from_entities: no key of {keys_} is present on both '
            f'the curated table and {recipe_id} for {state.admin_id}; skipping.',
            stacklevel=2,
        )
        return state
    key, reference_key = pair if pair is not None else (None, None)
    # A source that states none of a column is not an error here: the
    # step fills what the unit's sources can say and leaves the rest.
    present = [c for c in spec if c in available]
    if not present:
        return state
    sidecars = {c: source_column(c) for c in present if source_column(c) in available}
    # A property spine records which source each row came from, not a
    # sidecar per value: there the row's source is every value's
    # source, and without it a curated value would carry no token at
    # all, so a delivery could withhold it only by the coarser scope
    # rules (Lake County FL sales: land_value_source on 0.1% of rows).
    row_source = (
        ROW_SOURCE_COLUMN
        if ROW_SOURCE_COLUMN in available and len(sidecars) < len(present)
        else None
    )
    reference = get_entities(
        recipe_id,
        state.admin_id,
        columns=[
            *([reference_key] if reference_key else []),
            *present,
            *sidecars.values(),
            *([row_source] if row_source else []),
        ],
    )
    if reference is None or len(reference) == 0:
        return state
    if row_source and row_source in reference.columns:
        tokens = _row_source_tokens(reference[row_source])
        for column in present:
            if column not in sidecars:
                sidecars[column] = source_column(column)
                reference[sidecars[column]] = tokens
    # Rows without the key stay: the link table may still place them (a
    # unit that names its lot), and the key join's groupby drops them.
    functions = {}
    for column in present:
        override = spec[column]
        if override is not None:
            functions[column] = override
        else:
            canonical = resolve_attribute_name(column)
            fname = get_agg_func(canonical)
            if not _has_agg_func(fname):
                warnings.warn(
                    f'aggregate_from_entities: {column!r} has no registry '
                    'aggregation and no override; skipped.',
                    stacklevel=2,
                )
                continue
            functions[column] = _agg_func_for(canonical, fname)
    if not functions:
        return state
    numeric = {}
    for column in functions:
        series = reference[column]
        numeric[column] = (
            pd.to_numeric(series, errors='coerce')
            if functions[column] in ('sum', 'mean', 'median', 'min', 'max')
            and not pd.api.types.is_numeric_dtype(series)
            else series
        )
    for column, sidecar in sidecars.items():
        if column in functions:
            numeric[sidecar] = reference[sidecar]
    values = pd.DataFrame(numeric)
    linked = _rows_by_link(state, recipe_id, reference, values, functions)
    if linked is not None:
        values, group_key, keys = linked
    elif link_only:
        warnings.warn(
            f'aggregate_from_entities: no link table between this entity and '
            f'{recipe_id} for {state.admin_id} and no key to join on; skipping.',
            stacklevel=2,
        )
        return state
    else:
        group_key, keys = reference[reference_key], curated[key]
    reduced = _reduce(values, group_key, functions)
    written = []
    recover = set(recover or ())
    for column in reduced.columns:
        incoming = keys.map(reduced[column])
        if column in recover:
            positive = pd.to_numeric(incoming, errors='coerce').gt(0).fillna(False)
            if column in curated.columns:
                existing = curated[column]
                stated = pd.to_numeric(existing, errors='coerce').gt(0).fillna(False)
                wrote = positive & ~stated
                curated[column] = existing.where(~wrote, incoming)
            else:
                wrote = positive
                curated[column] = incoming.where(positive)
        elif column in curated.columns and fill_only:
            existing = curated[column]
            wrote = existing.isna() & incoming.notna()
            curated[column] = existing.where(existing.notna(), incoming)
        else:
            wrote = incoming.notna()
            curated[column] = incoming
        if column in sidecars and wrote.any():
            tokens = _union_tokens(values, group_key, column, sidecars[column])
            record_sources(curated, column, keys.map(tokens).where(wrote))
        written.append(column)
    state.curated = curated
    if state.verbose:
        print(
            f'  aggregate_from_entities: {recipe_id} -> {", ".join(written)} '
            f'on {int(keys.isin(reduced.index).sum()):,} of {len(curated):,} rows'
        )
    return state
