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


# Columns impute_n_dwellings reads, in preference order, most specific
# first. Deliberately an explicit ordered list and not a prefix scan:
# several occupancy-shaped columns coexist on the footprint spine with
# different vocabularies (a coarsened class, a vote on Overture dwelling
# counts, a raw source code), only one of which keys the lookup, and a
# prefix scan silently took whichever DataFrame order happened to put
# first. That made the step's output depend on incidental column order,
# up to imputing nothing at all.
_DWELLING_CLASS_COLUMNS = (
    'occupancy_type_building_nsi',
    'occupancy_type',
    'purpose_subgroup',
    'use_subgroup',
)


@_register('impute_n_dwellings')
def impute_n_dwellings(state: CurateState, column: str | None = None) -> CurateState:
    """Fill missing ``n_dwellings`` from an occupancy-class lookup.

    Rows still missing a dwelling-unit count after value reconciliation are
    filled from an occupancy-class column, using the occupancy-to-units
    mapping from Lochhead et al. (2026, Table 3). That mapping is keyed on
    the NSI occupancy vocabulary, so the column has to carry it: a column
    holding a coarsened class or a raw source code matches no key and
    imputes nothing.

    Parameters
    ----------
    column : str, optional
        Occupancy-class column to read. Defaults to the first of
        ``occupancy_type_building_nsi``, ``occupancy_type``,
        ``purpose_subgroup``, ``use_subgroup`` present on the curated
        frame. A named column that is absent is skipped, like a missing
        default, rather than raising.
    """
    from openplaces.io.harmonizer.attributes import _OCC_UNITS

    curated = state.curated
    if 'n_dwellings' not in curated.columns:
        curated['n_dwellings'] = np.nan

    null_mask = curated['n_dwellings'].isna()
    if null_mask.any():
        candidates = (column,) if column else _DWELLING_CLASS_COLUMNS
        subgroup_col = next((c for c in candidates if c in curated.columns), None)
        if subgroup_col is None and state.verbose:
            print(
                '  impute_n_dwellings: none of '
                f'{", ".join(candidates)} present; skipping.'
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
        corrections.index = corrections.index.astype('string').str.strip()
        repeated = corrections.index[corrections.index.duplicated()].unique()
        if len(repeated):
            shown = ', '.join(repr(str(k)) for k in repeated[:5])
            more = f' (and {len(repeated) - 5} more)' if len(repeated) > 5 else ''
            # pandas would otherwise raise InvalidIndexError from the
            # .map() below, naming neither the crosswalk nor the key.
            raise ValueError(
                f'Override crosswalk {overrides!r} maps {shown}{more} more '
                f'than once, so {output!r} has no single correction for '
                'them. Keys are compared after trimming surrounding '
                'whitespace; each group value must appear exactly once.'
            )
        keys = curated[group_column].astype('string').str.strip()
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
