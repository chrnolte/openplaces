"""Registered curation steps that estimate a missing parcel land value.

Land-value imputation and inference is expected to grow into a substantial area
of this codebase, so it gets its own submodule rather than living alongside the
generic gap-fillers in :mod:`imputers`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from openplaces.io.curator import CurateState, _register

_DEFAULT_RESIDENTIAL_CLASSES = [
    'Single-Family',
    'Multi-Family',
    'Manufactured Home',
    'Manufactured Home Park',
    'Multiple Single-Family',
    'Townhome',
    'Condominium',
]

_RATE_STATISTICS = ('mean', 'median', 'min', 'max')


@_register('impute_land_value')
def impute_land_value(
    state: CurateState,
    land_value_column: str = 'land_value',
    improvement_value_column: str = 'improvement_value',
    footprint_area_column: str = 'sum_footprint_area_m2',
    parcel_area_column: str = 'area_ha',
    land_use_column: str = 'land_use_class',
    residential_classes: list[str] | None = None,
    peer_group_column: str = 'use_group_combined',
    footprint_ratio_threshold: float = 0.25,
    street_column: str = 'address_street',
    city_column: str = 'city',
    group_tiers: list[list[str]] | None = None,
    fallback_group_column: str = '_is_residential',
    statistic: str = 'mean',
    min_group_size: int = 5,
    output_land_value: str = 'land_value_imputed',
    output_improvement_value: str = 'improvement_value_imputed',
) -> CurateState:
    """Estimate a land value for parcels the source data left at 0/NaN.

    A parcel is a candidate when *land_value_column* is 0 or missing,
    *improvement_value_column* is present and positive (there is real assessed
    value on record -- a parcel missing both is a data gap, not a
    land-value-specific one, and is left untouched), *parcel_area_column* is a
    real positive value (needed to ever compute a dollar estimate), and there
    is evidence it should carry a land value: *land_use_column* is one of
    *residential_classes*, or this parcel's *footprint_area_column* exceeds
    *footprint_ratio_threshold* times the median *footprint_area_column*
    among parcels sharing its *peer_group_column* value.

    **Never confound *footprint_area_column* (total building footprint area)
    with *parcel_area_column* (the parcel's own lot area).** Land value is
    conventionally priced per unit of *land*, not per unit of building -- a
    small house on a large lot and a large house on a small lot can have very
    different footprint areas for similar lot sizes. *footprint_area_column*
    is used **only** for the footprint-share eligibility signal above ("is
    there a disproportionately large structure here"); the per-area rate and
    the final dollar estimate below always use *parcel_area_column* instead.

    No bare "total assessed value" column exists on the curated parcel entity
    (only *land_value_column*/*improvement_value_column* survive harmonize).
    Under the standard assessor identity ``total = land + improvement``, "land
    missing and improvement equals the total" reduces algebraically to "land
    missing" whenever improvement is itself present -- this step relies on that
    simplification rather than any separate total-value plumbing.

    The per-area estimate is a *local* ``land_value / parcel_area`` rate --
    never ``improvement_value``, and never divided by footprint area -- learned
    from donor parcels (real, positive *land_value_column*, real positive
    *parcel_area_column*) sharing an increasingly broad cohort, tried in order
    via *group_tiers*: each tier is a
    list of column names donors and candidates are grouped by (default a single
    tier, ``[street_column, city_column]``, where *street_column* defaults to
    ``'address_street'`` -- the genuinely parsed, canonical street name
    :func:`openplaces.io.harmonizer.addresses.reconcile_addresses_df` writes at
    harmonize time, not a name this step derives itself -- skipped
    automatically when *street_column*/*city_column* are absent, or when a
    candidate's own key has a missing component: a merge on a missing key
    never matches, so it simply falls through to the next tier). A tier's
    group rate is suppressed when fewer than *min_group_size* donors back it.
    After every configured tier, a final, unconditional fallback groups donors
    by *fallback_group_column* (default ``'_is_residential'``, a synthetic
    residential-vs-non-residential split reusing *residential_classes* --
    grouping this broadly, rather than by an exact category like
    *peer_group_column*, matters: a narrow category (e.g. a specific
    condominium use-subtype) can have too few *recorded* land values to be a
    reliable donor pool, and worse, per-unit condo land values are
    structurally smaller than a typical lot's, systematically biasing a
    narrowly-scoped fallback low). Since each curate call already processes
    exactly one admin-3 (county) unit (``process_by.admin_level: 3``), this
    fallback tier is implicitly county-scoped for free too, with no per-row
    admin id column required. Extending the tier chain (a future
    near-house-number match, an ``admin4_id`` or ``census_block_group_id``
    cohort) is a matter of adding another entry to *group_tiers* once the
    backing column exists; unresolved tiers are skipped gracefully rather than
    raising.

    *output_land_value*/*output_improvement_value* are the parcel's final,
    best-available land/improvement value, not sparse imputation-only
    columns: a row with a real *land_value_column* passes it straight through
    to *output_land_value* unchanged (real value wins, always); only a
    candidate row an estimate was actually found for gets
    ``rate * parcel_area`` instead. *output_improvement_value* mirrors
    *improvement_value_column* everywhere, **except** on that same
    estimated subset, where it becomes ``improvement_value -
    land_value_imputed`` (clipped at 0) instead -- subtracting only where
    *this step itself* derived the land estimate out of that same
    improvement value; a parcel whose land was already properly, separately
    assessed must never have it subtracted a second time. A row with neither
    a real value nor an estimate (missing and ineligible, or the group
    statistic was absent -- the same "left missing" behavior as
    :func:`imputers.impute_from_group_statistic`) stays missing in both
    outputs.

    Parameters
    ----------
    land_value_column, improvement_value_column : str, optional
        Bare parcel value columns (defaults ``'land_value'``,
        ``'improvement_value'``).
    footprint_area_column : str, optional
        Per-parcel total *building footprint* area in m2 (default
        ``'sum_footprint_area_m2'``, written by
        :func:`openplaces.io.harmonizer.attributes.summarize_footprint_morphology`).
        Used **only** for the footprint-share eligibility ratio -- never for
        the rate or the final dollar estimate; see *parcel_area_column*.
    parcel_area_column : str, optional
        The parcel's own polygon (lot) area in ha (default ``'area_ha'``,
        written by :func:`openplaces.io.curator.inferers.derive_metrics` from
        the parcel's real geometry). Used for the per-area rate and the final
        dollar estimate -- land value is priced per unit of land, not per
        unit of building.
    land_use_column : str, optional
        Parcel land-use class (default ``'land_use_class'``).
    residential_classes : list of str, optional
        Classes eligible on residential grounds alone (default: the standard
        residential subtypes, including ``'Condominium'``).
    peer_group_column : str, optional
        Cohort column for the footprint-area-share eligibility test (default
        ``'use_group_combined'``).
    footprint_ratio_threshold : float, optional
        A parcel is eligible on footprint-share grounds when its
        *footprint_area_column* exceeds this fraction of its peer group's
        median (default 0.25).
    street_column, city_column : str, optional
        Evidence columns for the default local grouping tier (defaults
        ``'address_street'``, ``'city'``).
    group_tiers : list of list of str, optional
        Ordered local-to-broad grouping tiers, tried before
        *fallback_group_column* (default a single ``[street_column,
        city_column]`` tier).
    fallback_group_column : str, optional
        Guaranteed final grouping column (default ``'_is_residential'``, a
        synthetic residential-vs-non-residential split -- see above for why
        not a finer category).
    statistic : {'mean', 'median', 'min', 'max'}, optional
        Cohort statistic for the per-area rate (default ``'mean'``).
    min_group_size : int, optional
        Minimum donor count for a group's rate to be used (default 5).
    output_land_value, output_improvement_value : str, optional
        Final, coalesced output column names (defaults ``'land_value_imputed'``,
        ``'improvement_value_imputed'``) -- real value passed through, imputed
        estimate only where the real value was missing and eligible.
    """
    curated = state.curated
    required = {
        land_value_column,
        improvement_value_column,
        footprint_area_column,
        parcel_area_column,
        land_use_column,
        peer_group_column,
    }
    missing = required - set(curated.columns)
    if missing:
        if state.verbose:
            print(f'  impute_land_value: {sorted(missing)} missing; skipping.')
        return state

    if statistic not in _RATE_STATISTICS:
        raise ValueError(
            f'Unknown statistic {statistic!r}; expected one of {_RATE_STATISTICS}.'
        )
    if residential_classes is None:
        residential_classes = _DEFAULT_RESIDENTIAL_CLASSES
    if group_tiers is None:
        group_tiers = [[street_column, city_column]]

    land_value = pd.to_numeric(curated[land_value_column], errors='coerce')
    improvement_value = pd.to_numeric(
        curated[improvement_value_column], errors='coerce'
    )
    footprint_area = pd.to_numeric(curated[footprint_area_column], errors='coerce')
    parcel_area = pd.to_numeric(curated[parcel_area_column], errors='coerce')

    # has_area gates both candidacy and donor eligibility on the LOT area --
    # that's what the rate/estimate below actually divides/multiplies by, not
    # footprint_area (used only for the eligibility signal further down).
    has_area = parcel_area.notna() & (parcel_area > 0)
    land_missing = land_value.isna() | (land_value <= 0)
    improvement_ok = improvement_value.notna() & (improvement_value > 0)
    is_residential = curated[land_use_column].astype(object).isin(residential_classes)

    peer_median = footprint_area.groupby(
        curated[peer_group_column], observed=True
    ).transform('median')
    footprint_heavy = (footprint_area > footprint_ratio_threshold * peer_median) & (
        peer_median > 0
    )

    candidate = (
        land_missing & improvement_ok & has_area & (is_residential | footprint_heavy)
    )

    is_donor = land_value.notna() & (land_value > 0) & has_area
    per_area = (land_value / parcel_area).where(is_donor)

    def _tier_frame(cols: list[str]) -> pd.DataFrame | None:
        data = {}
        for col in cols:
            if col == '_is_residential':
                data[col] = is_residential
            elif col in curated.columns:
                data[col] = curated[col]
            else:
                return None
        return pd.DataFrame(data, index=curated.index)

    estimate = pd.Series(np.nan, index=curated.index)
    tier_used = pd.Series(pd.NA, index=curated.index, dtype=object)

    for tier_cols in [*group_tiers, [fallback_group_column]]:
        still_needed = candidate & estimate.isna()
        if not still_needed.any():
            break
        tier_df = _tier_frame(tier_cols)
        if tier_df is None:
            continue
        key_present = tier_df.notna().all(axis=1)
        donor_mask = is_donor & key_present
        if not donor_mask.any():
            continue
        donor_tier = tier_df.loc[donor_mask].copy()
        donor_tier['_per_area'] = per_area.loc[donor_mask]
        grouped = donor_tier.groupby(tier_cols, observed=True)['_per_area']
        rate = grouped.agg(statistic)
        rate = rate.where(grouped.size() >= min_group_size).dropna()
        if rate.empty:
            continue
        mapped = tier_df.merge(
            rate.rename('_estimate').reset_index(), on=tier_cols, how='left'
        )['_estimate']
        mapped.index = curated.index

        token = (
            '_'.join(tier_cols)
            if tier_cols != [fallback_group_column]
            else tier_cols[0]
        )
        fill = still_needed & key_present & mapped.notna()
        estimate.loc[fill] = mapped.loc[fill]
        tier_used.loc[fill] = token

    # land_value_imputed/improvement_value_imputed are the parcel's final,
    # best-available land/improvement value: a passthrough of the real value
    # wherever one exists, overridden only on the rows this step itself
    # derived an estimate for. Subtraction (below) must never apply to a
    # passthrough row: a normally, separately-assessed parcel's
    # improvement_value was never conflated with land in the first place.
    has_estimate = candidate & estimate.notna()

    land_value_imputed = pd.Series(np.nan, index=curated.index)
    has_real_land = land_value.notna() & (land_value > 0)
    land_value_imputed.loc[has_real_land] = land_value.loc[has_real_land]
    land_value_imputed.loc[has_estimate] = (
        estimate.loc[has_estimate] * parcel_area.loc[has_estimate]
    )

    improvement_value_imputed = pd.Series(np.nan, index=curated.index)
    has_real_improvement = improvement_value.notna() & (improvement_value > 0)
    improvement_value_imputed.loc[has_real_improvement] = improvement_value.loc[
        has_real_improvement
    ]
    improvement_value_imputed.loc[has_estimate] = (
        improvement_value.loc[has_estimate] - land_value_imputed.loc[has_estimate]
    ).clip(lower=0)

    curated[output_land_value] = land_value_imputed
    curated[output_improvement_value] = improvement_value_imputed

    from openplaces.io.curator.provenance import record_source

    for token in tier_used.dropna().unique():
        record_source(curated, output_land_value, tier_used == token, token)

    state.curated = curated
    if state.verbose:
        n = int(has_estimate.sum())
        n_candidates = int(candidate.sum())
        print(
            f'  impute_land_value: {output_land_value} set for {n:,} of '
            f'{n_candidates:,} candidate rows.'
        )
    return state
