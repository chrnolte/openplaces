"""Registered curation steps that estimate a missing parcel land value.

Land-value imputation and inference is expected to grow into a substantial area
of this codebase, so it gets its own submodule rather than living alongside the
generic gap-fillers in :mod:`imputers`.
"""

from __future__ import annotations

import geopandas as gpd
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


def _incoming_source(curated, column: str) -> pd.Series:
    """Return *column*'s per-row provenance token, or a constant fallback.

    Falls back to the column's own name where no ``{column}_source``
    sidecar exists yet -- honest ("this came from the land_value column")
    and, more importantly, non-null, so a passthrough row is never left
    looking like a row nothing decided.
    """
    from openplaces.io.curator.provenance import source_column

    side = source_column(column)
    if side in curated.columns:
        return curated[side].astype(object)
    return pd.Series(column, index=curated.index, dtype=object)


def _share_of_lot(curated, lot_value, improvement_value) -> pd.Series:
    """Split each lot's estimated land value among the records on that lot.

    A condominium development is one lot carrying one land value, and the
    assessor writes one record per unit -- every record repeating the
    development's whole area. Multiplying the local rate by that area
    therefore charges each unit for the entire lot: measured in Beaufort
    NC, 169 of 360 condo records share a geometry with another, in groups
    of up to 44, one of them spanning 212.9 ha. This divides the lot's
    value instead of repeating it, which is the same conservation rule
    :data:`~openplaces.io.harmonizer.apportion.WHOLE_VALUE_COLUMNS`
    applies when a parcel's value reaches several buildings.

    **Lots are identified by identical geometry, never by identical
    area.** A platted subdivision sells many lots cut to the same size, so
    an area key would split their land values between unrelated
    neighbours: in the same county 635 Single-Family parcels share an area
    with another (groups up to 7) while only 18 share a *geometry* (groups
    of 2, which are true duplicates). Area identity would have introduced
    a worse bug than the one being fixed.

    Each record's share is proportional to its own improvement value,
    which mirrors how a condominium declaration assigns percentage
    interest in the common area -- a penthouse holds more of the land than
    a studio. Where a group's improvement values are all missing or zero
    there is nothing to weight by, and the lot is split evenly.

    Returns
    -------
    pandas.Series
        Per-record land value, aligned to *curated*. Identical to
        *lot_value* for any record that does not share its geometry, so a
        pipeline whose parcels are all distinct is unaffected.
    """
    if 'geometry' not in curated.columns:
        return lot_value
    try:
        key = gpd.GeoSeries(curated['geometry'], index=curated.index).to_wkb()
    except Exception:
        return lot_value

    shared = key.notna() & key.duplicated(keep=False)
    if not shared.any():
        return lot_value

    result = lot_value.copy()
    # Force plain float64 rather than a nullable dtype before any numpy
    # call: a source whose improvement value arrives as pandas `Float64`
    # propagates `pd.NA` through the groupby, and `np.where` raises
    # "boolean value of NA is ambiguous" on the mask instead of treating
    # it as False. New Hanover NC hit exactly that.
    weight = pd.Series(
        pd.to_numeric(improvement_value, errors='coerce').to_numpy(
            dtype='float64', na_value=np.nan
        ),
        index=curated.index,
    )
    weight = weight.where(weight > 0).fillna(0.0)
    group = key.where(shared)
    total = weight.groupby(group).transform('sum').to_numpy(dtype='float64')
    size = group.groupby(group).transform('size').to_numpy(dtype='float64')
    # Fall back to an even split only where no record on the lot carries a
    # usable improvement value; a partly-populated group still weights by
    # what it has. np.where evaluates both branches, so both divisions are
    # guarded rather than relying on the mask to skip them.
    safe_total = np.where(total > 0, total, 1.0)
    safe_size = np.where(size > 0, size, 1.0)
    fraction = np.where(total > 0, weight.to_numpy() / safe_total, 1.0 / safe_size)
    result.loc[shared] = (lot_value * pd.Series(fraction, index=curated.index)).loc[
        shared
    ]
    return result


@_register('impute_land_value')
def impute_land_value(
    state: CurateState,
    land_value_column: str = 'land_value',
    improvement_value_column: str = 'improvement_value',
    total_value_column: str = 'total_value',
    footprint_area_column: str = 'footprint_area_m2_in_parcel',
    parcel_area_column: str = 'area_ha',
    land_use_column: str = 'land_use_class',
    residential_classes: list[str] | None = None,
    peer_group_column: str = 'use_group_combined',
    footprint_ratio_threshold: float = 0.25,
    street_column: str = 'address_street',
    city_column: str = 'city',
    group_tiers: list[list[str]] | None = None,
    fallback_group_column: str = '_is_residential',
    statistic: str = 'median',
    min_donor_area_ha: float = 0.004,
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

    Under the standard assessor identity ``total = land + improvement``,
    a *total_value_column* worth having changes what this step should do,
    and since 2026-08 enough sources map one that both cases are live.
    Where a recorded total **exceeds** the improvement figure, the land
    component was recorded all along and is exactly ``total -
    improvement``: that is used directly, no rate involved, and the
    improvement figure is left alone because it already excludes land.
    Where the total **equals** the improvement figure, the source folded
    land into it -- the premise this step exists for -- so the learned
    rate is used and then capped at the total, because no parcel holds
    more land value than its whole assessed value. A parcel with no
    recorded total falls back to the uncapped rate, as before.

    The per-area estimate is a *local* ``land_value / parcel_area`` rate --
    never ``improvement_value``, and never divided by footprint area -- learned
    from donor parcels (real, positive *land_value_column*, real positive
    *parcel_area_column*) sharing an increasingly broad cohort, tried in order
    via *group_tiers*: each tier is a list of column names donors and
    candidates are grouped by (default two tiers, ``[street_column,
    city_column]`` then ``[city_column, fallback_group_column]``, where
    *street_column* defaults to ``'address_street'`` -- the genuinely parsed,
    canonical street name
    :func:`openplaces.io.harmonizer.addresses.reconcile_addresses_df` writes at
    harmonize time, not a name this step derives itself -- skipped
    automatically when a tier's own columns are absent, or when a
    candidate's own key has a missing component: a merge on a missing key
    never matches, so it simply falls through to the next tier). A tier's
    group rate is suppressed when fewer than *min_group_size* donors back it.
    After every configured tier, a final, unconditional fallback groups donors
    by *fallback_group_column* alone (default ``'_is_residential'``, a
    synthetic residential-vs-non-residential split reusing
    *residential_classes* -- grouping this broadly, rather than by an exact
    category like *peer_group_column*, matters: a narrow category (e.g. a
    specific condominium use-subtype) can have too few *recorded* land values
    to be a reliable donor pool, and worse, per-unit condo land values are
    structurally smaller than a typical lot's, systematically biasing a
    narrowly-scoped fallback low). Since each curate call already processes
    exactly one admin-3 (county) unit (``process_by.admin_level: 3``), this
    last-resort fallback tier is implicitly county-scoped for free too -- but
    a county can span towns with wildly different land economics (a real
    Middlesex County, MA measurement: one town's average recorded rate was
    ~3.4x the county-wide average), so the second, *city*-scoped tier exists
    specifically to catch a candidate whose own street tier failed (e.g. no
    same-street donor at all, or an upstream address-parsing gap left its
    *street_column* unmatched to any real neighbor) *before* diluting it
    across the whole county. Extending the tier chain further (a future
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
        Per-parcel total *building footprint* area in m2, clipped to each
        parcel's own boundary (default ``'footprint_area_m2_in_parcel'``,
        written by
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
        *fallback_group_column* alone (default two tiers: ``[street_column,
        city_column]``, then ``[city_column, fallback_group_column]`` --
        the latter catches a street-tier miss without diluting the rate
        across the whole county the way the final, unconditional fallback
        does).
    fallback_group_column : str, optional
        Guaranteed final grouping column (default ``'_is_residential'``, a
        synthetic residential-vs-non-residential split -- see above for why
        not a finer category).
    statistic : {'mean', 'median', 'min', 'max'}, optional
        Cohort statistic for the per-area rate (default ``'median'``).

        **Not the mean**, and the difference is not academic. A per-area
        rate is a ratio whose denominator can be near zero, so its
        distribution has a long right tail that no amount of donor
        cleaning removes: across the ten surveyed North Carolina counties
        the mean donor rate is **1,017x the median** ($273,028,985/ha
        against $268,485/ha). Using the mean inflated the region's imputed
        land value to **$64.96bn**, with a single 82 ha parcel imputed at
        $2.95bn; the median puts the same total at **$548m** and the worst
        row at $26m. Because an over-imputed land value is subtracted from
        the parcel's improvement value, that inflation also erased the
        structure value of 755 parcels ($94.0m); the median leaves 402
        ($33.7m). Prefer a rank statistic for any ratio a cohort is
        pooled over.
    total_value_column : str, optional
        Column holding the parcel's whole assessed value (default
        ``total_value``). Drives both the direct land recovery and the
        conservation cap described above; absent or non-positive, neither
        applies and the step behaves as it did before.
    min_donor_area_ha : float, optional
        Smallest lot area, in hectares, that may teach a per-area rate
        (default 0.004 ha, about 40 m2 -- smaller than a parking space,
        so not a lot anyone assessed). Guards the rate against parcels
        whose recorded area is a rounding artifact. With *statistic* left
        at its default this is belt-and-braces (the median is already
        robust to them, and the guard changes the region total by 0.01%);
        it earns its place for a small cohort, where a handful of
        degenerate donors can move the median itself.
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
        group_tiers = [
            [street_column, city_column],
            [city_column, fallback_group_column],
        ]

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

    # A donor teaches a per-area rate by division, so a parcel whose
    # recorded area is a rounding artifact rather than a lot teaches an
    # arbitrarily large one. Measured in Beaufort NC: the three donors
    # under 0.0001 ha carry a median rate of $39bn/ha, and the county's
    # single worst donor reaches $291bn/ha. Excluding them costs almost
    # nothing (they are a few hundred rows in ~675,000 donors) and is the
    # same shape of guard as the degenerate-join-key blanking in
    # io.harmonizer.links -- a value too extreme to be real is dropped
    # before it can teach anything, not after.
    is_donor = (
        land_value.notna()
        & (land_value > 0)
        & has_area
        & (parcel_area >= min_donor_area_ha)
    )
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
    # A recorded total is better evidence than any neighbour's rate, and
    # it splits the candidates into two genuinely different populations
    # (measured over 10,268 imputed rows in 76 rebuilt CHEER counties):
    #
    #  - total == improvement (65.2%): the source folded land into the
    #    improvement figure, which is the premise this whole step rests
    #    on. Nothing to recover, so the rate estimate stands -- but it is
    #    now capped, because land cannot exceed the parcel's whole value.
    #  - total > improvement (33.5%): a real land component was recorded
    #    all along and is exactly total - improvement. No estimate needed,
    #    and the improvement figure is already land-free, so it must NOT
    #    be reduced afterwards.
    #
    # On the second group the estimate was not merely unnecessary but
    # wrong in both directions: median $35,438 against a recorded
    # $119,698, and a worst case of $486,079,430 against $37,053,906.
    total_value = (
        pd.to_numeric(curated[total_value_column], errors='coerce')
        if total_value_column in curated.columns
        else pd.Series(np.nan, index=curated.index)
    )
    recorded_total = total_value.notna() & (total_value > 0)
    has_recorded_land = (
        candidate
        & recorded_total
        & improvement_value.notna()
        & (total_value > improvement_value)
    )
    has_estimate = candidate & estimate.notna() & ~has_recorded_land

    land_value_imputed = pd.Series(np.nan, index=curated.index)
    has_real_land = land_value.notna() & (land_value > 0)
    land_value_imputed.loc[has_real_land] = land_value.loc[has_real_land]
    # rate * area is the value of the LOT, which is not the same thing as
    # the value of the parcel record when several records share one lot.
    lot_value = estimate * parcel_area
    shared = _share_of_lot(curated, lot_value, improvement_value)
    # Conservation: a parcel cannot hold more land value than its own
    # recorded total. Only bites where a total was actually recorded.
    shared = shared.where(~recorded_total, np.minimum(shared, total_value))
    land_value_imputed.loc[has_estimate] = shared.loc[has_estimate]
    land_value_imputed.loc[has_recorded_land] = (
        total_value.loc[has_recorded_land] - improvement_value.loc[has_recorded_land]
    )

    improvement_value_imputed = pd.Series(np.nan, index=curated.index)
    has_real_improvement = improvement_value.notna() & (improvement_value > 0)
    improvement_value_imputed.loc[has_real_improvement] = improvement_value.loc[
        has_real_improvement
    ]
    # Subtract only where the land estimate was carved out of this same
    # improvement figure. A recorded-total row's improvement value already
    # excludes land, so subtracting again would charge it twice.
    improvement_value_imputed.loc[has_estimate] = (
        improvement_value.loc[has_estimate] - land_value_imputed.loc[has_estimate]
    ).clip(lower=0)

    curated[output_land_value] = land_value_imputed
    curated[output_improvement_value] = improvement_value_imputed

    from openplaces.io.curator.provenance import record_sources

    # Provenance for both outputs. Each is a passthrough of a real
    # assessed value on most rows and this step's own estimate on the
    # rest, and only this step knows which is which per row -- what
    # reaches a downstream step is two columns of dollars with nothing
    # to tell them apart. So record both sides: the incoming token
    # carried forward for a passthrough, and the peer-group tier that
    # produced the estimate, marked imputed, for the rest. The two
    # masks are disjoint by construction (has_estimate only ever holds
    # where the source left the value missing), so the write order
    # between them does not matter.
    record_sources(
        curated,
        output_land_value,
        _incoming_source(curated, land_value_column),
        mask=has_real_land & ~has_estimate,
    )
    record_sources(
        curated, output_land_value, tier_used, mask=has_estimate, imputed=True
    )
    # Its own token: this land value is arithmetic on the parcel's own
    # recorded figures, not a neighbour's rate, and a reader deciding how
    # far to trust a number needs to be able to tell those apart. Still
    # marked imputed -- openplaces filled the cell either way.
    record_sources(
        curated,
        output_land_value,
        pd.Series('total_minus_improvement', index=curated.index),
        mask=has_recorded_land,
        imputed=True,
    )

    # improvement_value_imputed is the parcel's own improvement figure
    # minus the land value imputed above, so on an estimated row it is
    # derived twice over and inherits that row's tier. It carried no
    # provenance at all before, and it is the column that reaches a
    # footprint as structure_value -- which is how the delivered
    # structure_value_source came to read `parcel` for a number no
    # assessor ever wrote.
    record_sources(
        curated,
        output_improvement_value,
        _incoming_source(curated, improvement_value_column),
        mask=has_real_improvement & ~has_estimate,
    )
    record_sources(
        curated,
        output_improvement_value,
        tier_used,
        mask=has_estimate,
        imputed=True,
    )

    state.curated = curated
    if state.verbose:
        n = int(has_estimate.sum())
        n_candidates = int(candidate.sum())
        print(
            f'  impute_land_value: {output_land_value} set for {n:,} of '
            f'{n_candidates:,} candidate rows.'
        )
    return state
