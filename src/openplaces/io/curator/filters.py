"""Registered curation steps that remove records from canonical datasets."""

from __future__ import annotations

import warnings

import pandas as pd

from openplaces.io.curator import CurateState, _register


@_register('exclude_by_value', phase='standardize')
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


@_register('keep_by_value', phase='standardize')
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


@_register('filter_numeric_at_least', phase='standardize')
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


@_register('exclude_numeric_above', phase='standardize')
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


_RANGE_KEYS = {'column', 'min', 'max', 'inclusive', 'keep_null'}


@_register('filter_by_range', phase='standardize')
def filter_by_range(state: CurateState, filters: list[dict]) -> CurateState:
    """Drop rows whose numeric value falls outside a declared range.

    The curate-stage home for sample definitions that are numeric
    cutoffs: a year window, a positive price, an area-ratio sanity band.
    Unlike :func:`filter_numeric_at_least` and
    :func:`exclude_numeric_above`, one step carries several cutoffs and
    reports a count per filter, so a training table's row count can be
    reconciled filter by filter against the analysis it reproduces.

    Parameters
    ----------
    filters : list of dict
        Each entry has ``column`` and at least one of ``min`` / ``max``.
        ``inclusive`` (bool, default True) decides whether a value equal
        to a bound is kept. ``keep_null`` (bool, default False) decides
        whether a missing value survives the filter; the default drops
        it, because a row whose year or price is unknown cannot be said
        to be inside the window. A filter naming an absent column is
        skipped with a warning rather than dropping every row.
    """
    curated = state.curated
    for spec in filters:
        unknown = set(spec) - _RANGE_KEYS
        if unknown:
            raise ValueError(
                f'filter_by_range: unknown key(s) {sorted(unknown)} in {spec!r}.'
            )
        column = spec['column']
        if 'min' not in spec and 'max' not in spec:
            raise ValueError(f'filter_by_range: {column!r} needs min and/or max.')
        if column not in curated.columns:
            warnings.warn(f'filter_by_range: column {column!r} absent; skipped.')
            continue
        values = pd.to_numeric(curated[column], errors='coerce')
        inclusive = spec.get('inclusive', True)
        keep = pd.Series(True, index=curated.index)
        if 'min' in spec:
            keep &= values.ge(spec['min']) if inclusive else values.gt(spec['min'])
        if 'max' in spec:
            keep &= values.le(spec['max']) if inclusive else values.lt(spec['max'])
        if spec.get('keep_null', False):
            keep |= values.isna()
        n_dropped = int((~keep).sum())
        if n_dropped:
            curated = curated.loc[keep].copy()
        if state.verbose:
            bounds = ', '.join(f'{k}={spec[k]}' for k in ('min', 'max') if k in spec)
            print(
                f'  filter_by_range: dropped {n_dropped:,} rows on {column!r} '
                f'({bounds}); {len(curated):,} remain'
            )
    state.curated = curated
    return state


@_register('normalize_id_column', phase='standardize')
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
