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


_GROUP_STATISTICS = {
    'mode': lambda s: s.mode().iloc[0] if not s.mode().empty else pd.NA,
    'mean': 'mean',
    'median': 'median',
    'min': 'min',
    'max': 'max',
}


@_register('impute_from_group_statistic')
def impute_from_group_statistic(
    state: CurateState,
    group_column: str,
    value_column: str,
    output: str,
    statistic: str = 'mode',
    overrides: str | None = None,
) -> CurateState:
    """Impute each row's output from a grouped statistic of another column.

    For every row, *output* is set to a statistic of *value_column* computed
    across all rows sharing the same *group_column* value (its cohort). The
    default *statistic* is the mode (most common value), which learns a
    group -> value mapping by majority vote; mean, median, min, and max are also
    supported for numeric columns.

    An optional *overrides* crosswalk corrects known-bad group mappings: a
    two-column lookup (group value -> corrected output) loaded by recipe id.
    Corrections win wherever the row's group is a key in the table — even when
    the correction itself is blank (explicit null), which suppresses the
    learned statistic for that group rather than falling back to it. Matching
    is exact after trimming surrounding whitespace only (no case-folding or
    punctuation normalization), so override CSV keys must match the group
    column's real values. The grouped statistic fills every other group.

    Generic over any pair of columns: holds no references to specific entities
    or sources, so it can be reused for any cross-linked categorical columns.

    Parameters
    ----------
    group_column : str
        Column whose value defines each row's cohort.
    value_column : str
        Column the statistic is computed over within each cohort.
    output : str
        Name of the column to write.
    statistic : str, optional
        Cohort statistic: mode (default), mean, median, min, or max.
    overrides : str, optional
        Recipe id of a two-column correction crosswalk
        (group value -> corrected output). Corrections take precedence over the
        computed statistic.
    """
    curated = state.curated
    if group_column not in curated or value_column not in curated:
        # Still declare the output: downstream steps and curated-reference
        # readers treat a missing declared column as a recipe error. A
        # cohort input this admin unit lacks (e.g. no NSI coverage) yields
        # an all-null output, the enricher's absent-coverage convention.
        if output not in curated:
            curated[output] = np.nan
        return state

    func = _GROUP_STATISTICS.get(statistic)
    if func is None:
        raise ValueError(
            f'Unknown statistic {statistic!r}; expected one of '
            f'{", ".join(_GROUP_STATISTICS)}.'
        )

    paired = curated[[group_column, value_column]].dropna()
    base = paired.groupby(group_column, observed=True)[value_column].agg(func)
    mapped = curated[group_column].map(base)

    if overrides:
        from openplaces.io.transform import get_crosswalk

        corrections = get_crosswalk({'recipe_id': overrides})
        keys = curated[group_column].astype('string').str.strip()
        corrections.index = corrections.index.astype('string').str.strip()
        has_override = keys.isin(corrections.index)
        mapped = mapped.where(~has_override, keys.map(corrections))

    curated[output] = mapped

    # Every value this step writes is a cohort statistic or a
    # hand-entered correction -- none of it read off the row's own
    # record -- so the whole column is marked derived. The override
    # token stays distinguishable from the learned one: they fail in
    # different ways, and a reader chasing a wrong group mapping needs
    # to know which produced it.
    from openplaces.io.curator.provenance import record_source

    record_source(curated, output, mapped.notna(), 'group_statistic', imputed=True)
    if overrides:
        record_source(
            curated, output, has_override & mapped.notna(), 'override', imputed=True
        )

    state.curated = curated

    if state.verbose:
        n = int(mapped.notna().sum())
        print(
            f'  impute_from_group_statistic: {output} set for {n:,} rows '
            f'(statistic={statistic}).'
        )
    return state
