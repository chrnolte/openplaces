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
        area_sqft: sum}`), for a column whose registry rule (a mean for
        `year_built`) is not what the parcel needs.
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
    reference = reference.dropna(subset=[reference_key])
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
    grouped = values.groupby(reference[reference_key])
    reduced = grouped.agg(functions)
    # pandas sums an all-missing group to 0; a parcel none of whose
    # properties states a floor area has no floor area, not a zero one.
    reduced = reduced.where(grouped.count() > 0)
    keys = curated[key]
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
