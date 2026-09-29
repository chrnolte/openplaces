"""Each footprint's priority on its parcel
(classify_footprint_priority).
"""

from __future__ import annotations

import pandas as pd

from openplaces.io.harmonizer import HarmonizeState, _register


@_register('classify_footprint_priority')
def classify_footprint_priority(
    state: HarmonizeState,
    entity_type: str | None = None,
    thresholds: dict | None = None,
    **_params,
) -> HarmonizeState:
    """Classify each footprint's priority on its parcel.

    Assigns ``priority_on_parcel`` as ``'primary'``, ``'secondary'``, or
    ``'unknown'``.

    Uses dwelling-point and building-point evidence to assign roles within each
    parcel (Lochhead et al. 2026, Table 4):

    1. If any footprint on the parcel has dwelling-point evidence
       (``SourceGeometryType.single_dwelling_point``), those footprints are
       ``'primary'``; all others on the same parcel are ``'secondary'``,
       except the parcel's largest footprint when it is at least
       ``dwelling_override_ratio`` times the largest dwelling-evidence
       footprint and itself holds building-point evidence: it stays
       ``'primary'`` alongside them (see *thresholds*).
    2. Else if any footprint has single-building-point evidence
       (``SourceGeometryType.single_building_point``, e.g. NSI), those are
       ``'primary'``; all others are ``'secondary'``.
    3. If no footprint on a multi-footprint parcel has evidence, the
       largest footprint is ``'primary'``, as is any other footprint at
       least ``fallback_always_primary_m2`` square meters (default 355,
       about twice the median single-family primary footprint measured
       on CHEER evidence); the rest are ``'secondary'``.
    4. Footprints that are the sole geometry on their parcel are always
       ``'primary'``.
    5. Footprints not linked to any parcel are ``'unknown'``, unless they
       carry dwelling-point evidence — those are promoted to ``'primary'``.
    6. A synthetic, parcel-derived fallback geometry (``geometry_source``
       starting with ``'{entity_type}.'``, set by
       :func:`~openplaces.io.harmonizer.links.infer_spine_additions`) is
       always ``'primary'``, overriding the above: it stands in for the
       parcel's one inferred building and was never eligible for the
       crosswalk-seeded evidence rules (it postdates the footprint-parcel
       crosswalk that seeds them).

    Parameters
    ----------
    entity_type : str, optional
        Entity type used to locate the parcel crosswalk in ``state.crosswalks``.
        Defaults to ``'parcel'``.
    thresholds : dict, optional
        ``fallback_always_primary_m2`` (float or None, default 355):
        size above which a no-evidence footprint is always promoted to
        primary alongside its parcel's largest; None keeps only the
        largest-footprint promotion.

        ``dwelling_override_ratio`` (float or None, default None): on a
        parcel with dwelling-point evidence, keep the largest footprint
        primary when its area is at least this multiple of the largest
        dwelling-evidence footprint's. None disables the rule.

        ``dwelling_override_requires_building_point`` (bool, default
        True): apply that rule only when the largest footprint also
        holds building-point evidence, i.e. only where the two point
        sources disagree about which structure is the building.
    """
    from openplaces.core.schema import SourceGeometryType as _SGT

    if state.spine is None:
        return state

    spine_id_col = state.spine.index.name
    entity_type = entity_type or 'parcel'

    # A synthetic, reference-derived fallback row (added by
    # infer_spine_additions after the crosswalk below was built, so it can
    # never appear in it) stands in for the reference entity's one inferred
    # building and is always 'primary', regardless of what the
    # crosswalk/evidence rules below would otherwise assign it. geometry_source
    # is prefixed with the entity_type infer_spine_additions was called with
    # (e.g. 'parcel.spine'); matching on entity_type specifically (not just
    # any '.') avoids misclassifying a fallback synthesized from a different
    # reference entity_type.
    is_synthetic = (
        state.spine['geometry_source']
        .astype('string')
        .str.startswith(f'{entity_type}.', na=False)
        if 'geometry_source' in state.spine.columns
        else pd.Series(False, index=state.spine.index)
    )

    parcel_crosswalks = state.get_crosswalks_by_type(entity_type)
    if not parcel_crosswalks:
        if state.verbose:
            print(
                f'  classify_footprint_priority: no {entity_type} crosswalk; skipping.'
            )
        return state

    parcel_recipe_id = next(iter(parcel_crosswalks))
    crosswalk = parcel_crosswalks[parcel_recipe_id]

    fp_parcel = crosswalk.reset_index()[[spine_id_col, 'parcel_id']].dropna(
        subset=['parcel_id']
    )
    fp_parcel = fp_parcel[fp_parcel[spine_id_col].isin(state.spine.index)]

    parcel_fp_count = fp_parcel.groupby('parcel_id')[spine_id_col].transform('count')
    multi_fp = fp_parcel[parcel_fp_count > 1].copy()

    # Seed: parcel-linked → 'primary', unlinked → 'unknown'.
    # Single-footprint parcels keep 'primary' and never enter the loop below.
    role = pd.Series('unknown', index=state.spine.index, dtype=object)
    role.loc[role.index.isin(set(fp_parcel[spine_id_col]))] = 'primary'
    role.loc[is_synthetic] = 'primary'

    if multi_fp.empty:
        state.spine['priority_on_parcel'] = pd.Categorical(
            role, categories=['primary', 'secondary', 'unknown']
        )
        return state

    # Collect point-evidence sets.
    address_evidence: set = set()
    building_point_evidence: set = set()
    for rid, sgt in state.source_geometry_types.items():
        if sgt not in {_SGT.single_dwelling_point, _SGT.single_building_point}:
            continue
        cw = state.crosswalks.get(rid)
        if cw is None:
            continue
        linked_ids = (
            cw[spine_id_col].dropna().unique()
            if spine_id_col in cw.columns
            else cw.index[cw.index.isin(state.spine.index)]
        )
        if sgt == _SGT.single_dwelling_point:
            address_evidence.update(linked_ids)
        elif sgt == _SGT.single_building_point:
            building_point_evidence.update(linked_ids)

    # Geometry fallback for parcels with no point evidence at all. All-
    # secondary was the old behavior, and it starves such parcels in value
    # apportionment (only primary footprints receive the parcel's
    # improvement value): every multi-building parcel in a region with no
    # NSI/Overture points lost its structure value entirely. Calibrated
    # 2026-08-25 on 46,422 evidence-labeled multi-footprint parcels
    # across five CHEER counties (NC-CAM/CAR/ONS, TX-ARA/NUE): the
    # largest footprint is an evidence primary on 90.8% of parcels, and
    # additional footprints at or above ~2x the median single-family
    # primary footprint (177 m2 -> 355 m2 default) are primary 84% of
    # the time; adding a dominance guard on those extra promotions only
    # lowered agreement. Threshold override:
    # thresholds: {fallback_always_primary_m2: <m2>} (null disables the
    # extra promotions; the largest footprint stays primary regardless).
    from openplaces.geo.polygon import get_areas

    thresholds = thresholds or {}
    always_primary_m2 = thresholds.get('fallback_always_primary_m2', 355.0)
    # An address point that lands on a garage or shed would otherwise
    # make it the parcel's only primary and demote the house, which
    # then receives no share of the parcel's improvement value. Tested
    # 2026-09-17 against roll heated/living floor area on inverted
    # single-dwelling parcels (the largest footprint secondary, a
    # smaller one primary): where the largest is at least 3x the
    # dwelling-point footprint and holds an NSI point, its floor area
    # matches the roll's better on 68% of NC parcels (n 1,322) and 89%
    # of TX parcels (n 7,008); at 2x only 58% in NC. Both footprints
    # stay primary rather than swapping roles, because the smaller one
    # is the dwelling on the remaining parcels (and can be an
    # accessory dwelling on any of them); value then splits by area.
    override_ratio = thresholds.get('dwelling_override_ratio')
    override_needs_building_point = thresholds.get(
        'dwelling_override_requires_building_point', True
    )
    areas_m2 = (
        state.spine['area_ha'] * 10_000
        if 'area_ha' in state.spine.columns
        else get_areas(state.spine, unit='m2')
    )

    for parcel_id, group in multi_fp.groupby('parcel_id'):
        fp_ids = set(group[spine_id_col])
        has_addr = fp_ids & address_evidence
        has_bldg = fp_ids & building_point_evidence

        if has_addr:
            # Dwelling evidence wins: dwelling-linked footprints are primary;
            # everything else on this parcel is secondary, bar the
            # size override above.
            keep = set(has_addr)
            if override_ratio is not None:
                fp_areas = areas_m2.reindex(list(fp_ids)).fillna(0.0)
                largest = fp_areas.idxmax()
                dwelling_max = fp_areas.reindex(list(has_addr)).max()
                if (
                    largest not in has_addr
                    and dwelling_max > 0
                    and fp_areas[largest] >= override_ratio * dwelling_max
                    and (
                        not override_needs_building_point
                        or largest in building_point_evidence
                    )
                ):
                    keep.add(largest)
            for fp_id in fp_ids - keep:
                role[fp_id] = 'secondary'
        elif has_bldg:
            # NSI evidence: NSI-linked footprints are primary, rest secondary.
            for fp_id in fp_ids - has_bldg:
                role[fp_id] = 'secondary'
        else:
            # No point evidence on this parcel: promote by geometry (see
            # the calibration note above), demote the rest.
            fp_areas = areas_m2.reindex(list(fp_ids)).fillna(0.0)
            keep = {fp_areas.idxmax()}
            if always_primary_m2 is not None:
                keep |= set(fp_areas.index[fp_areas >= always_primary_m2])
            for fp_id in fp_ids - keep:
                role[fp_id] = 'secondary'

    # Promote unlinked footprints with dwelling evidence from 'unknown' to 'primary'.
    for fp_id in address_evidence:
        if fp_id in state.spine.index and role[fp_id] == 'unknown':
            role[fp_id] = 'primary'

    role.loc[is_synthetic] = 'primary'

    state.spine['priority_on_parcel'] = pd.Categorical(
        role, categories=['primary', 'secondary', 'unknown']
    )
    if state.verbose:
        counts = state.spine['priority_on_parcel'].value_counts()
        print(
            '  classify_footprint_priority: '
            + ', '.join(f'{k}={v:,d}' for k, v in counts.items())
        )
    if state.timer:
        state.timer.mark('Classify')
    return state
