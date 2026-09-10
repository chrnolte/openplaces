"""Registered curation steps that remove records from canonical datasets."""

from __future__ import annotations

import pandas as pd

from openplaces.io.curator import CurateState, _register


@_register('exclude_by_value')
def exclude_by_value(state: CurateState, column: str, values: list) -> CurateState:
    """Drop rows whose *column* value is in *values*.

    For records that don't belong in the canonical dataset at all (e.g. a
    water-body/right-of-way placeholder row masquerading as a parcel) --
    distinct from land-use classification, which assigns a class to every
    row that remains. No-op if *column* is absent.

    Parameters
    ----------
    column : str
        Column to test.
    values : list
        Values whose rows are dropped.
    """
    curated = state.curated
    if column not in curated.columns:
        return state
    mask = curated[column].astype(object).isin(values)
    if mask.any():
        curated = curated.loc[~mask].copy()
    state.curated = curated
    if state.verbose:
        print(
            f'  exclude_by_value: dropped {int(mask.sum()):,} rows '
            f'where {column!r} in {values!r}'
        )
    return state


@_register('keep_by_value')
def keep_by_value(state: CurateState, column: str, values: list) -> CurateState:
    """Keep only rows whose *column* value is in *values*.

    The "keep only" counterpart to :func:`exclude_by_value`, for a filter
    more naturally stated as an allow-list than a block-list (e.g. only
    the qualification-code labels that mean "arm's length"). No-op if
    *column* is absent.

    Parameters
    ----------
    column : str
        Column to test.
    values : list
        Values whose rows are kept; every other row is dropped.
    """
    curated = state.curated
    if column not in curated.columns:
        return state
    mask = curated[column].astype(object).isin(values)
    curated = curated.loc[mask].copy()
    state.curated = curated
    if state.verbose:
        print(
            f'  keep_by_value: kept {len(curated):,} rows '
            f'where {column!r} in {values!r}'
        )
    return state


@_register('filter_numeric_at_least')
def filter_numeric_at_least(state: CurateState, column: str, min: float) -> CurateState:
    """Keep rows where *column*, read as numeric, is at least *min*.

    Named after the ``numeric_at_least`` indicator type already used
    inside :func:`~openplaces.io.curator.reconcilers.resolve_by_vote`'s
    decision DSL. No-op if *column* is absent.

    Parameters
    ----------
    column : str
        Column to test.
    min : float
        Rows below this value are dropped; non-numeric values are dropped.
    """
    curated = state.curated
    if column not in curated.columns:
        return state
    values = pd.to_numeric(curated[column], errors='coerce')
    curated = curated.loc[values >= min].copy()
    state.curated = curated
    if state.verbose:
        print(
            f'  filter_numeric_at_least: kept {len(curated):,} rows '
            f'where {column!r} >= {min}'
        )
    return state


@_register('exclude_numeric_above')
def exclude_numeric_above(state: CurateState, column: str, max: float) -> CurateState:
    """Drop rows where *column*, read as numeric, is known to exceed *max*.

    Unlike :func:`filter_numeric_at_least`, a missing or non-numeric value
    is left in place rather than dropped: this excludes a *known*
    violation, not an uncertain one. No-op if *column* is absent.

    Parameters
    ----------
    column : str
        Column to test.
    max : float
        Rows confirmed above this value are dropped.
    """
    curated = state.curated
    if column not in curated.columns:
        return state
    values = pd.to_numeric(curated[column], errors='coerce')
    mask = values > max
    curated = curated.loc[~mask].copy()
    state.curated = curated
    if state.verbose:
        print(
            f'  exclude_numeric_above: dropped {int(mask.sum()):,} rows '
            f'where {column!r} > {max}'
        )
    return state


@_register('normalize_id_column')
def normalize_id_column(state: CurateState, column: str) -> CurateState:
    """Strip non-alphanumeric characters from *column*, in place.

    Some sources change an id's punctuation mid-series (FL DOR Lake
    County adds dashes to ``parcel_id_assessor`` starting 2022; the same
    physical parcel reads as a bare digit string in earlier years). A
    plain string comparison across that boundary misses every match
    unless both sides are normalized this way first. No-op if *column*
    is absent.

    Parameters
    ----------
    column : str
        Column to normalize.
    """
    curated = state.curated
    if column not in curated.columns:
        return state
    curated[column] = (
        curated[column].astype(str).str.replace(r'[^0-9A-Za-z]', '', regex=True)
    )
    state.curated = curated
    return state
