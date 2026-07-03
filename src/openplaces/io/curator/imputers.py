"""Registered curation steps that fill missing canonical values."""

from __future__ import annotations

import numpy as np
import pandas as pd

from openplaces.io.curator import CurateState, _register


@_register('fill_missing_numeric')
def fill_missing_numeric(
    state: CurateState,
    columns: list[str],
    fill_value: float = 0,
    dtype: str = 'int64',
) -> CurateState:
    """Fill missing values in numeric columns, then cast to *dtype*.

    For an evidence-derived column where a missing value carries different
    meaning than a confirmed value of *fill_value* (e.g. "no Overture address
    point matched this footprint at all" vs. "an Overture point matched and
    reported 0 dwellings"), place this step after anything that should see
    the true missing value (a priority reconciliation, a vote) — from that
    point on it only reshapes the column for output.

    Parameters
    ----------
    columns : list of str
        Columns to fill and cast. Missing columns are skipped.
    fill_value : float, optional
        Value used to fill missing entries (default 0).
    dtype : str, optional
        Target dtype after filling (default ``'int64'``).
    """
    curated = state.curated
    for col in columns:
        if col in curated.columns:
            curated[col] = (
                pd.to_numeric(curated[col], errors='coerce')
                .fillna(fill_value)
                .astype(dtype)
            )
    state.curated = curated
    return state


@_register('impute_n_dwellings')
def impute_n_dwellings(state: CurateState) -> CurateState:
    """Fill missing ``n_dwellings`` from an occupancy-class lookup.

    Rows still missing a dwelling-unit count after value reconciliation are
    filled from the occupancy class of the first available
    ``purpose_subgroup`` / ``occupancy_type`` column, using the
    occupancy-to-units mapping from Lochhead et al. (2026, Table 3).
    """
    from openplaces.io.harmonizer.attributes import _OCC_UNITS

    curated = state.curated
    if 'n_dwellings' not in curated.columns:
        curated['n_dwellings'] = np.nan

    null_mask = curated['n_dwellings'].isna()
    if null_mask.any():
        subgroup_col = next(
            (
                c
                for c in curated.columns
                if c.startswith('use_subgroup')
                or c.startswith('purpose_subgroup')
                or c.startswith('occupancy_type')
            ),
            None,
        )
        if subgroup_col is not None:
            inferred = curated.loc[null_mask, subgroup_col].map(_OCC_UNITS)
            curated.loc[null_mask, 'n_dwellings'] = inferred
            from openplaces.io.curator.provenance import record_source

            filled = null_mask & curated['n_dwellings'].notna()
            record_source(curated, 'n_dwellings', filled, 'imputed')

    state.curated = curated
    return state
