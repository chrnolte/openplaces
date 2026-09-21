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
from openplaces.io.readers import describe_recipe, get_entities
from openplaces.recipe import resolve_attribute_name
from openplaces.table import _agg_func_for, _has_agg_func


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
        path = get_entity_link_path(recipe_id, owner, state.admin_id)
        links = read_entity_link(path)
    except Exception:
        # No resolvable link for this recipe and unit (a curate recipe
        # with no geospine, a reference outside ENTITY_LINK_ORDER): the
        # key join below is the documented behavior.
        return None
    if links is None or links.empty:
        return None
    finer_id, coarser_id = links.columns[0], links.columns[1]
    links = links[
        links[finer_id].isin(reference.index)
        & links[coarser_id].isin(state.curated.index)
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
    described = set(links.loc[~only_units, coarser_id])
    links = links[~(only_units & links[coarser_id].isin(described))]

    rows = values.loc[links[finer_id]].reset_index(drop=True)
    share = links['share'].astype('Float64').reset_index(drop=True)
    for column, function in functions.items():
        if function == 'sum' and share.notna().any():
            scaled = pd.to_numeric(rows[column], errors='coerce') * share.astype(
                'float64'
            )
            rows[column] = scaled.where(share.notna(), rows[column])
    group_key = links[coarser_id].reset_index(drop=True)
    keys = pd.Series(state.curated.index, index=state.curated.index)
    if state.verbose:
        print(
            f'  aggregate_from_entities: grouped by {path.name} ({len(links):,d} links)'
        )
    return rows, group_key, keys


@_register('aggregate_from_entities')
def aggregate_from_entities(
    state: CurateState,
    recipe_id: str,
    columns: list | dict,
    key: str | list[str] = 'parcel_id_local',
    reference_key: str | list[str] | None = None,
    fill_only: bool = True,
) -> CurateState:
    """Reduce a reference entity's rows onto the curated rows they belong to.

    Reads the reference recipe for the unit being curated, groups its rows
    by *reference_key*, reduces each named column with the registry's
    aggregation function (or the override given for it), and writes the
    result onto the curated rows by *key*. A parcel's year built becomes
    the earliest of its properties', its living area the sum, a room count
    the sum, without the harmonize stage ever having copied them.

    Parameters
    ----------
    recipe_id : str
        The entity recipe to reduce (`US_property-spine-2026`). Named here
        so the recipe graph sees the dependency.
    columns : list of str or dict
        Columns to reduce. A list uses the registry's aggregation for each
        column; a mapping gives a per-column override (`{year_built: min,
        n_stories: max}`), for a column whose registry rule (a mean for
        both) is not what the parcel needs.
    key : str or list of str, optional
        Curated column the reference joins on (default `parcel_id_local`).
        A list is tried in order and the first column present on both
        sides is used, so one recipe serves a source keyed on the local
        parcel id and one keyed on a state-wide id (`parcel_id_admin2`).
    reference_key : str or list of str, optional
        Reference column to group by; defaults to *key*.
    fill_only : bool, optional
        Fill only curated cells that are missing (default). With False the
        reduced value replaces whatever the curated column held.
    """
    curated = state.curated
    keys_ = [key] if isinstance(key, str) else list(key)
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
    if pair is None:
        warnings.warn(
            f'aggregate_from_entities: no key of {keys_} is present on both '
            f'the curated table and {recipe_id} for {state.admin_id}; skipping.',
            stacklevel=2,
        )
        return state
    key, reference_key = pair
    # A source that states none of a column is not an error here: the
    # step fills what the unit's sources can say and leaves the rest.
    present = [c for c in spec if c in available]
    if not present:
        return state
    reference = get_entities(
        recipe_id, state.admin_id, columns=[reference_key, *present]
    )
    if reference is None or len(reference) == 0:
        return state
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
    values = pd.DataFrame(numeric)
    linked = _rows_by_link(state, recipe_id, reference, values, functions)
    if linked is not None:
        values, group_key, keys = linked
    else:
        group_key, keys = reference[reference_key], curated[key]
    grouped = values.groupby(group_key)
    reduced = grouped.agg(functions)
    # pandas sums an all-missing group to 0; a parcel none of whose
    # properties states a floor area has no floor area, not a zero one.
    reduced = reduced.where(grouped.count() > 0)
    written = []
    for column in reduced.columns:
        incoming = keys.map(reduced[column])
        if column in curated.columns and fill_only:
            existing = curated[column]
            curated[column] = existing.where(existing.notna(), incoming)
        else:
            curated[column] = incoming
        written.append(column)
    state.curated = curated
    if state.verbose:
        print(
            f'  aggregate_from_entities: {recipe_id} -> {", ".join(written)} '
            f'on {int(keys.isin(reduced.index).sum()):,} of {len(curated):,} rows'
        )
    return state
