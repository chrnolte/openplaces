"""Registered curation steps that remove records from canonical datasets."""

from __future__ import annotations

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
