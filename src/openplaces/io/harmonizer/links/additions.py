"""Add spine rows a reference implies and the spine lacks
(infer_spine_additions), and the value-density helper it reads.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from openplaces.geo.ids import (
    add_openlocationcode_index,
)
from openplaces.io.harmonizer import (
    HarmonizeState,
    _register,
)
from openplaces.io.transform import make_index_unique
from openplaces.recipe import (
    source_id_from_recipe_id,
)


def _positive_value_density(values: pd.Series) -> pd.Series:
    """Coerce a value-per-hectare column to numeric, keeping only real ones.

    Anything that is not a strictly positive, finite number becomes NaN.
    Assessor sources spell "this parcel has no building" three different
    ways -- a null improvement value, a literal 0, and (via a zero-area
    geometry) an infinite density -- and treating any of them as a small
    positive number is what lets a vacant lot pass a value threshold.
    Collapsing all three to NaN here means callers can express "has a real
    improvement value" as a single ``notna()`` check.
    """
    numeric = pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan)
    return numeric.where(numeric > 0)


@_register('infer_spine_additions', phase='geometry')
def infer_spine_additions(
    state: HarmonizeState,
    entity_type: str | None = None,
    recipe_id: str | None = None,
    thresholds: dict | None = None,
) -> HarmonizeState:
    """Add spine entries inferred from a reference crosswalk.

    For each reference polygon that has no existing spine coverage and
    exceeds the improvement-value threshold, creates a new spine geometry
    equal to the reference polygon geometry.  The threshold is what keeps a
    vacant lot from becoming a building, so it is applied unconditionally:
    a candidate must carry a strictly positive, finite improvement value per
    hectare, above half its land-use group's *value_per_ha_quantile* and
    never below the reference-wide floor at that same quantile (measured
    only over parcels that do carry a real footprint and a positive value).
    A reference with no land-use group column at all is held to the
    reference-wide floor alone, and one where no footprint-carrying parcel
    reports a positive value is skipped with a warning rather than inferred
    against an uncalibrated threshold.

    The inferred GeoDataFrame is stored in
    ``state.metadata['inferred_from_<recipe_id>']`` for use by
    ``reconcile_attributes``.

    Parameters
    ----------
    entity_type : str, optional
        Selects all crosswalks matching this type via
        ``state.reference_types``.
    recipe_id : str, optional
        Explicit crosswalk key to use.  Takes precedence over entity_type.
    thresholds : dict, optional
        ``n_per_group_min`` (float, default 0.2) — minimum mean spine count
        per purpose group to be eligible for inference; not applied when the
        reference carries no group column.
        ``value_per_ha_quantile`` (float, default 0.05) — quantile of
        improvement_value_per_ha used as the lower inference bound.
    """
    if state.spine is None:
        return state

    thresholds = thresholds or {}
    n_min: float = thresholds.get('n_per_group_min', 0.2)
    q: float = thresholds.get('value_per_ha_quantile', 0.05)

    if recipe_id is None and entity_type is not None:
        candidates = list(state.get_crosswalks_by_type(entity_type).keys())
        if not candidates:
            if state.verbose:
                print(
                    f'  Infer: no crosswalk for entity_type={entity_type!r}; skipping.'
                )
            return state
        recipe_id = candidates[0]

    if recipe_id is None:
        warnings.warn('infer_spine_additions: no recipe_id resolved.')
        return state

    crosswalk = state.crosswalks.get(recipe_id)
    ref_polys = state.references.get(recipe_id)
    if crosswalk is None or ref_polys is None:
        warnings.warn(
            f'infer_spine_additions: crosswalk or reference for '
            f'{recipe_id!r} not found in state.'
        )
        return state

    if 'improvement_value_per_ha' not in ref_polys.columns:
        if state.verbose:
            print(
                '  Infer: improvement_value_per_ha not available; '
                'skipping parcel-based inference.'
            )
        return state

    # Parcels carry the use_* land-use vocabulary; fall back to purpose_* for a
    # building/footprint reference.  A reference with neither -- several county
    # parcel sources publish no land-use code at all -- gets a single ungrouped
    # threshold instead of a per-group one, never *no* threshold: a missing
    # group column used to switch both gates below to unconditionally True, so
    # every unmatched parcel, vacant land included, became an inferred
    # footprint (observed as ~20-37% parcel-sourced spine rows in the
    # group-less NC counties, against ~2-6% elsewhere).
    group_col = next(
        (c for c in ('use_group', 'purpose_group') if c in ref_polys.columns),
        None,
    )
    if group_col is None and state.verbose:
        print(
            f'  Infer: {recipe_id} carries no use_group/purpose_group; '
            'using an ungrouped value threshold.'
        )

    ref_stat_cols = [
        c
        for c in [
            group_col,
            'improvement_value_per_ha',
            'has_duplicate_geometry',
        ]
        if c is not None and c in ref_polys.columns
    ]

    has_dup = 'has_duplicate_geometry' in ref_polys.columns
    ref_no_dup = (
        ref_polys[~ref_polys['has_duplicate_geometry']] if has_dup else ref_polys
    )

    footprint_ref_data = crosswalk[['link']].join(
        ref_no_dup[ref_stat_cols],
        on='parcel_id',
        how='inner',
    )

    has_group = group_col is not None and group_col in footprint_ref_data.columns
    n_footprints_per_group = (
        (
            (
                footprint_ref_data[group_col].value_counts()
                / ref_polys[group_col].value_counts()
            )
            .fillna(0)
            .rename('n_footprints_mean')
        )
        if has_group
        else pd.Series(dtype=float)
    )

    imp_val_q_col = f'improvement_value_per_ha_q{q}'
    has_imp = 'improvement_value_per_ha' in footprint_ref_data.columns

    # A parcel's $0 (or missing) improvement value means "no building", not
    # "a very cheap building" -- sources spell the same fact both ways, and
    # some spell it only as 0.  Keeping those rows in the quantiles below
    # drags a mostly-vacant group's q-th percentile to 0, which silently
    # disarms the gate they are supposed to calibrate.  An infinite density
    # (a zero-area parcel) is likewise not evidence of a building.
    if has_imp and has_group:
        linked = footprint_ref_data[
            _positive_value_density(
                footprint_ref_data['improvement_value_per_ha']
            ).notna()
        ]
        imp_val_q_by_group = (
            linked.sample(frac=1)
            .reset_index()
            .drop_duplicates('parcel_id', keep=False)
            .groupby(group_col)['improvement_value_per_ha']
            .quantile((q, 0.5))
            .unstack()
            .rename(
                columns={
                    q: imp_val_q_col,
                    0.5: 'improvement_value_per_ha_median',
                }
            )
            if not linked.empty
            else None
        )
    else:
        imp_val_q_by_group = None

    parcel_ids_with_footprint = crosswalk.index.get_level_values('parcel_id').unique()
    mask_without_footprint = ~ref_polys.index.isin(parcel_ids_with_footprint)

    # Reference-wide lower bound, calibrated on the parcels that *do* carry a
    # real footprint: how little assessed improvement per hectare still comes
    # with a building here.  It is the sole threshold when there is no group
    # column, and the floor under every per-group threshold otherwise.
    imp_val_floor = np.nan
    if has_imp:
        floor_sample = _positive_value_density(
            ref_polys.loc[
                ref_polys.index.isin(parcel_ids_with_footprint),
                'improvement_value_per_ha',
            ]
        ).dropna()
        if not floor_sample.empty:
            imp_val_floor = floor_sample.quantile(q)

    if not has_imp or pd.isna(imp_val_floor):
        # No parcel has both a linked footprint and a positive improvement
        # value, so there is nothing to calibrate a threshold against (a
        # source that maps improvement_value but ships it empty, or reports
        # it only as part of a combined total).  Inferring on an
        # uncalibrated threshold is what produced the vacant-land
        # footprints; infer nothing instead, and say so.
        warnings.warn(
            f'infer_spine_additions: no parcel in {recipe_id!r} has both a '
            f'linked footprint and a positive improvement value for '
            f'{state.admin_id}, so no value threshold can be calibrated; '
            'skipping parcel-based inference.',
            stacklevel=2,
        )
        return state

    if state.verbose:
        print(f'  Infer: footprint if improvement_value > $ {imp_val_floor:,.0f}/ha')

    candidate_cols = [
        c
        for c in [
            group_col,
            'improvement_value',
            'improvement_value_per_ha',
            'has_duplicate_geometry',
            'geometry',
        ]
        if c is not None and c in ref_polys.columns
    ]
    ref_candidates = ref_polys[mask_without_footprint][candidate_cols].copy()

    if imp_val_q_by_group is not None:
        ref_candidates = ref_candidates.join(imp_val_q_by_group, on=group_col)
    if not n_footprints_per_group.empty:
        ref_candidates = ref_candidates.join(n_footprints_per_group, on=group_col)

    n_col_ok = (
        ref_candidates['n_footprints_mean'].gt(n_min)
        if 'n_footprints_mean' in ref_candidates.columns
        else pd.Series(True, index=ref_candidates.index)
    )

    # The value gate always applies.  Half the candidate's own group's q-th
    # percentile where that exists, never below the reference-wide floor;
    # the floor alone otherwise.  A group with no threshold of its own (an
    # unseen group, or one whose linked parcels all report 0) falls back to
    # the floor rather than to no gate -- note clip(lower=NaN) is a silent
    # no-op in pandas, so that NaN has to be filled explicitly.
    candidate_per_ha = _positive_value_density(
        ref_candidates['improvement_value_per_ha']
    )
    if imp_val_q_col in ref_candidates.columns:
        threshold = (
            pd.to_numeric(ref_candidates[imp_val_q_col], errors='coerce')
            .div(2)
            .clip(lower=imp_val_floor)
            .fillna(imp_val_floor)
        )
    else:
        threshold = pd.Series(imp_val_floor, index=ref_candidates.index)
    # notna() carries the "positive improvement value" requirement: an
    # unbuilt parcel is not a missing footprint, whichever way its source
    # spells the absence.
    val_col_ok = candidate_per_ha.notna() & candidate_per_ha.gt(threshold)

    mask_inferred = n_col_ok & val_col_ok
    ref_inferred = ref_candidates[mask_inferred]

    footprints_from_ref = add_openlocationcode_index(
        ref_inferred[['geometry']].reset_index(), name=state.spine.index.name
    )
    footprints_from_ref = footprints_from_ref[
        ~footprints_from_ref.index.isin(state.spine.index)
    ]
    _source_id = source_id_from_recipe_id(recipe_id)
    _et = entity_type or recipe_id.rsplit('_', 1)[-1].split('-', 1)[0]
    footprints_from_ref['geometry_source'] = f'{_et}.{_source_id}'
    # Record which reference entity seeded each inferred row. Provenance a
    # reader can use directly, and what lets an attribute-only successor
    # recipe rebuild metadata['inferred_from_{recipe_id}'] from the saved
    # spine alone (see load_geospine) -- the in-memory frame below does
    # not survive the geometry/attribute recipe split.
    footprints_from_ref['geometry_source_id'] = footprints_from_ref['parcel_id']

    state.spine = pd.concat(
        [
            state.spine,
            footprints_from_ref[['geometry', 'geometry_source', 'geometry_source_id']],
        ]
    ).sort_index()

    if state.spine.index.duplicated().any():
        n_dup = state.spine.index.duplicated().sum()
        warnings.warn(f'{n_dup} duplicate spine IDs for {state.admin_id}.')
        state.spine = make_index_unique(state.spine, sort_duplicates_by_area=True)

    if state.verbose:
        print(
            f'  Infer: +{len(footprints_from_ref):,d} reference-inferred spine entries'
        )
    if state.timer:
        state.timer.mark('Infer')

    state.metadata[f'inferred_from_{recipe_id}'] = footprints_from_ref
    return state
