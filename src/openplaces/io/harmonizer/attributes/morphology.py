"""Per-parcel footprint morphology summary
(summarize_footprint_morphology).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from openplaces.io.harmonizer import HarmonizeState, _register


@_register('summarize_footprint_morphology', phase='geometry')
def summarize_footprint_morphology(
    state: HarmonizeState,
    footprint_recipe_id: str,
    small_area_max_m2: float = 185.0,
    elongated_aspect_min: float = 2.0,
    on: str | None = 'parcel_id',
    min_overlap_m2: float = 10.0,
    overlap_column: str = 'area_intersection_m2_parcel',
    priority_column: str = 'priority_on_parcel',
    dwelling_column: str = 'n_dwellings_overture',
    span_column: str = 'n_parcels_per_footprint',
) -> HarmonizeState:
    """Attach per-parcel footprint morphology aggregates to the parcel spine.

    Footprint-to-parcel linkage is the harmonizer's job; this reads the footprint
    entity, assigns each footprint to a parcel (by a shared id column when *on* is
    present on both sides, else by spatial containment of the footprint's
    representative point), and writes the per-parcel counts the parcel land-use
    classifier consumes downstream: ``n_footprints_per_parcel``,
    ``n_small_elongated_footprints_per_parcel`` (manufactured-home-shaped),
    ``max_footprint_area_m2``, ``footprint_area_m2_dominant``,
    ``footprint_area_m2_in_parcel``, ``n_primary_footprints_per_parcel``,
    ``footprint_area_m2_primary``, ``max_parcels_per_footprint``, and
    ``max_dwellings_per_footprint``. The classification itself is parcel-curate
    work.

    ``max_parcels_per_footprint`` and ``max_dwellings_per_footprint`` are scoped
    to *dwelling-confirmed* footprints only (*is_primary_candidate* — see below —
    AND a positive *dwelling_column*): ``priority_on_parcel == 'primary'`` alone
    is not evidence of a real dwelling — it's also assigned to a sole footprint
    on its own parcel, an NSI-only-evidence footprint, or a synthetic fallback
    row, none of which confirm occupancy. ``max_parcels_per_footprint`` is the
    largest *span_column* value among those confirmed footprints (does any of
    this parcel's confirmed footprints span multiple parcels — a real,
    non-FEMA-geometry building-shape signal); ``max_dwellings_per_footprint`` is
    the largest confirmed dwelling count on any single one of them (a footprint
    itself holding multiple dwellings, independent of parcel boundaries).

    ``on`` defaults to the globally-unique ``parcel_id`` rather than
    ``parcel_id_local``: the latter is only locally cross-comparable and can
    collide across genuinely distinct parcels within one admin unit (e.g. every
    unit of a condo complex sharing one assessor PIN), which would silently
    merge their footprint counts together.

    ``n_footprints_per_parcel`` counts every footprint row that clears
    *min_overlap_m2* (its overlap with this parcel, via *overlap_column*, when
    present), including a parcel-derived synthetic fallback geometry
    (``geometry_source`` containing ``.``, set by :func:`infer_spine_additions`)
    regardless of overlap size when that is the parcel's only footprint — a
    fallback's geometry is the parcel boundary, so its "overlap" is trivially the
    whole parcel and it needs no floor. ``n_small_elongated_footprints_per_parcel``,
    ``max_footprint_area_m2``, and ``footprint_area_m2_dominant`` additionally
    exclude synthetic rows entirely: a fallback's area/aspect ratio are not
    meaningful size/shape evidence. Like ``max_footprint_area_m2``,
    ``footprint_area_m2_dominant`` stays ``NaN`` (not ``0``) for a parcel with no
    real, non-synthetic footprint at all.
    ``n_primary_footprints_per_parcel`` additionally excludes footprints whose
    *priority_column* value (when present) is ``'secondary'`` — a real, but
    accessory, structure (garage, shed) that clears the overlap floor but isn't a
    distinct home; synthetic fallback rows still count here too.
    ``footprint_area_m2_primary`` sums real footprint area over that same
    non-secondary subset (excluding synthetic rows, like
    ``footprint_area_m2_dominant``).

    ``footprint_area_m2_dominant`` and ``footprint_area_m2_primary`` both sum
    each contributing footprint's full, unclipped polygon area — even when
    part of that footprint's geometry actually lies outside the parcel,
    because the footprint straddles a boundary. **Never divide either by
    parcel area to estimate footprint coverage share — the ratio can exceed
    1.** ``footprint_area_m2_in_parcel`` is the coverage-safe alternative:
    for every footprint that clears *min_overlap_m2* against *this* parcel
    specifically (including footprints whose dominant parcel is a different,
    neighboring one but that still spill a real sliver onto this parcel), it
    sums that footprint's geometry clipped to this parcel's own boundary (a
    real geometric intersection, via :func:`openplaces.geo.polygon.overlay_polygons`).
    It is bounded by construction to at most this parcel's own area, and
    (like the two sums above) excludes synthetic rows and stays ``NaN`` for a
    parcel with no real footprint coverage at all.

    Parameters
    ----------
    footprint_recipe_id : str
        Footprint entity recipe to read (geometry required).
    small_area_max_m2, elongated_aspect_min : float, optional
        A footprint counts as small-and-elongated (manufactured-home morphology)
        when its area is at most *small_area_max_m2* and its oriented aspect ratio
        is at least *elongated_aspect_min*.
    on : str, optional
        Shared id for an id-based assignment (default ``parcel_id``): a footprint
        entity column matched against either a same-named spine column or (for a
        parcel spine, whose own true id lives on the index during this step --
        see ``resolve_spine``) the spine's index. When absent on either side, a
        spatial within-join is used instead (which has no overlap-area or
        priority data to apply *min_overlap_m2*/*priority_column* with, so every
        contained footprint counts toward all four outputs). When a
        ``{on}_all`` column is present (e.g. ``parcel_id_all``, written by
        :func:`_attribute_polygon_reference`), every pipe-joined id counts
        toward this parcel's outputs, not just *on*'s single dominant one --
        the fix for a footprint spanning many parcels (e.g. one real building
        over many tiny condo-unit parcels) that would otherwise only ever
        credit whichever parcel it overlaps most.
    min_overlap_m2 : float, optional
        Minimum overlap (m²) with this parcel, from *overlap_column*, for a
        non-synthetic footprint to count toward any of the four outputs (default
        10, matching this codebase's ``area_intersection_m2_min`` convention).
        Excludes a sliver footprint whose only detected parcel candidate happens
        to be this one.
    overlap_column : str, optional
        Footprint-entity column holding each footprint's overlap area (m²) with
        its dominant parcel (default ``area_intersection_m2_parcel``, written by
        :func:`_attribute_polygon_reference`). Ignored (no floor applied) if
        absent from the footprint entity.
    priority_column : str, optional
        Footprint-entity column holding each footprint's structural role on its
        parcel (default ``priority_on_parcel``, written by
        :func:`classify_footprint_priority`). Ignored (no exclusion applied) if
        absent from the footprint entity.
    dwelling_column : str, optional
        Footprint-entity column holding each footprint's confirmed dwelling
        count (default ``n_dwellings_overture``). A footprint counts toward
        ``max_parcels_per_footprint``/``max_dwellings_per_footprint`` only when
        this is positive. Ignored (neither output is set) if absent from the
        footprint entity.
    span_column : str, optional
        Footprint-entity column holding how many parcels that footprint spans
        (default ``n_parcels_per_footprint``, written by
        :func:`_attribute_polygon_reference` when the footprint spine attributes
        the parcel reference). Ignored (``max_parcels_per_footprint`` not set) if
        absent from the footprint entity.
    """
    import geopandas as gpd

    from openplaces.geo.polygon import local_metric_crs, overlay_polygons
    from openplaces.io.harmonizer.spine import get_oriented_dims
    from openplaces.io.readers import get_entities

    if state.spine is None:
        return state
    spine = state.spine

    # missing='ignore': an admin unit can genuinely have no footprint
    # coverage at all (no OBM/Microsoft source ingested there), the same
    # tolerated case this function's own empty check below already expects.
    footprints = get_entities(
        footprint_recipe_id, state.admin_id, geom=True, missing='ignore'
    )
    if footprints is None or len(footprints) == 0:
        if state.verbose:
            print('  summarize_footprint_morphology: no footprints; skipping.')
        return state

    geom_m = footprints.geometry.to_crs(local_metric_crs(footprints))
    area = geom_m.area.to_numpy()
    dims = geom_m.map(get_oriented_dims)
    length = np.array([d[1] for d in dims])
    width = np.clip(np.array([d[2] for d in dims]), 1e-6, None)
    aspect = length / width
    small_elong = (area <= small_area_max_m2) & (aspect >= elongated_aspect_min)

    # Parcel-derived synthetic fallback geometries (geometry_source containing
    # '.', set by infer_spine_additions) are the parcel boundary, not a building
    # outline, so their area/aspect are excluded from the size/shape aggregates
    # (max_footprint_area_m2, n_small_elongated_footprints_per_parcel). They
    # still count toward n_footprints_per_parcel below: that count exists
    # because independent value evidence already implied a building is there.
    if 'geometry_source' in footprints.columns:
        is_synthetic = (
            footprints['geometry_source']
            .astype('string')
            .str.contains(r'\.', regex=True, na=False)
            .to_numpy()
        )
    else:
        is_synthetic = np.zeros(len(footprints), dtype=bool)
    real_area = np.where(is_synthetic, np.nan, area)
    real_small_elong = small_elong & ~is_synthetic

    # A non-synthetic footprint must clear the overlap floor to count toward any
    # output (a sliver footprint whose only detected parcel candidate is this one
    # is not real evidence of a structure on it); synthetic fallback rows bypass
    # this, same reasoning as real_area/real_small_elong above. Missing
    # overlap_column (e.g. a not-yet-regenerated footprint entity) disables the
    # floor entirely, matching the function's prior behavior.
    if overlap_column in footprints.columns:
        overlap = pd.to_numeric(footprints[overlap_column], errors='coerce').to_numpy()
        meets_floor = is_synthetic | (overlap >= min_overlap_m2)
    else:
        meets_floor = np.ones(len(footprints), dtype=bool)

    # A footprint marked 'secondary' (an accessory structure, per
    # classify_footprint_priority's dwelling/NSI-point evidence) clears the floor
    # but isn't a distinct home; excluded only from the stricter
    # n_primary_footprints_per_parcel below. Synthetic rows bypass this too.
    if priority_column in footprints.columns:
        not_secondary = (
            is_synthetic
            | (footprints[priority_column].astype('string') != 'secondary').to_numpy()
        )
    else:
        not_secondary = np.ones(len(footprints), dtype=bool)
    is_primary_candidate = meets_floor & not_secondary

    # Confirmed-dwelling subset for max_parcels_per_footprint/
    # max_dwellings_per_footprint: is_primary_candidate alone lets through
    # sole-footprint-on-parcel, NSI-only, and synthetic rows with no dwelling
    # evidence at all, so a positive dwelling_column is required on top of it.
    if dwelling_column in footprints.columns:
        dwellings = (
            pd.to_numeric(footprints[dwelling_column], errors='coerce')
            .fillna(0)
            .to_numpy()
        )
        dwelling_confirmed = is_primary_candidate & (dwellings > 0)
    else:
        dwellings = np.zeros(len(footprints))
        dwelling_confirmed = np.zeros(len(footprints), dtype=bool)

    if span_column in footprints.columns:
        n_parcels_span = pd.to_numeric(
            footprints[span_column], errors='coerce'
        ).to_numpy()
    else:
        n_parcels_span = np.full(len(footprints), np.nan)

    # A parcel spine's own true id (parcel_id) lives on the index during this
    # step under a renamed, working index name (e.g. 'spine_id') --
    # resolve_spine records the *original* name in
    # state.metadata['spine_index_name'] and only restores it at save time --
    # so 'on' may match a plain column, the spine's current index name, or
    # that recorded original name, not just a column.
    on_in_spine = on and (
        on in spine.columns
        or on == spine.index.name
        or on == state.metadata.get('spine_index_name')
    )
    if on and on in footprints.columns and on_in_spine:
        # {on}_all (written by _attribute_polygon_reference, e.g.
        # 'parcel_id_all') pipe-joins every reference this footprint links to
        # in the trimmed crosswalk, not just its single dominant one -- a
        # footprint spanning several parcels (e.g. one real building over
        # many tiny condo-unit parcels) would otherwise only ever credit
        # whichever parcel it overlaps most, leaving every other linked
        # parcel silently uncounted by every output below. Falls back to the
        # plain dominant-only column when the reference recipe hasn't been
        # regenerated with it yet.
        on_all_col = f'{on}_all'
        pid_source = (
            footprints[on_all_col]
            if on_all_col in footprints.columns
            else footprints[on]
        )
        per_fp = pd.DataFrame(
            {
                '_pid': pid_source.astype('string').to_numpy(),
                '_a': real_area,
                '_se': real_small_elong,
                '_meets_floor': meets_floor,
                '_primary': is_primary_candidate,
                '_dwellings': dwellings,
                '_confirmed': dwelling_confirmed,
                '_span': n_parcels_span,
            }
        ).dropna(subset=['_pid'])
        if on_all_col in footprints.columns:
            per_fp = per_fp.assign(_pid=per_fp['_pid'].str.split('|')).explode('_pid')
        key = (
            spine[on].astype('string')
            if on in spine.columns
            else spine.index.to_series().astype('string')
        )

        grp = per_fp[per_fp['_meets_floor']].groupby('_pid')
        n_fp = key.map(grp.size())
        n_se = key.map(grp['_se'].sum())
        max_a = key.map(grp['_a'].max())
        # min_count=1: an all-synthetic (all-NaN) or footprint-less group must
        # stay NaN here too, matching max_a's "no real footprint" semantics,
        # not silently become 0 (pandas' default sum-of-nothing).
        sum_a = key.map(grp['_a'].sum(min_count=1))

        grp_primary = per_fp[per_fp['_primary']].groupby('_pid')
        n_primary = key.map(grp_primary.size())
        sum_a_primary = key.map(grp_primary['_a'].sum(min_count=1))

        grp_confirmed = per_fp[per_fp['_confirmed']].groupby('_pid')
        max_dwellings = key.map(grp_confirmed['_dwellings'].max())
        max_span = key.map(grp_confirmed['_span'].max())
    else:
        reps = gpd.GeoDataFrame(
            {
                '_a': real_area,
                '_se': real_small_elong,
                '_meets_floor': meets_floor,
                '_primary': is_primary_candidate,
                '_dwellings': dwellings,
                '_confirmed': dwelling_confirmed,
                '_span': n_parcels_span,
            },
            geometry=footprints.geometry.representative_point(),
            crs=footprints.crs,
        ).to_crs(spine.crs)
        # Carry the spine id as an explicit column so the join does not depend on
        # the sjoin right-index column name (which varies with the index name).
        spine_geom = spine[[spine.geometry.name]].copy()
        spine_geom['_spine_id'] = spine.index
        joined = gpd.sjoin(reps, spine_geom, predicate='within', how='left').dropna(
            subset=['_spine_id']
        )

        grp = joined[joined['_meets_floor']].groupby('_spine_id')
        n_fp = grp.size().reindex(spine.index)
        n_se = grp['_se'].sum().reindex(spine.index)
        max_a = grp['_a'].max().reindex(spine.index)
        sum_a = grp['_a'].sum(min_count=1).reindex(spine.index)

        grp_primary = joined[joined['_primary']].groupby('_spine_id')
        n_primary = grp_primary.size().reindex(spine.index)
        sum_a_primary = grp_primary['_a'].sum(min_count=1).reindex(spine.index)

        grp_confirmed = joined[joined['_confirmed']].groupby('_spine_id')
        max_dwellings = grp_confirmed['_dwellings'].max().reindex(spine.index)
        max_span = grp_confirmed['_span'].max().reindex(spine.index)

    # footprint_area_m2_in_parcel: real, clipped footprint area actually
    # inside this parcel's own boundary, across every non-synthetic footprint
    # that touches it above min_overlap_m2 -- including footprints whose
    # *dominant* parcel is a different, neighboring one but that still spill
    # a real sliver onto this parcel. Unlike footprint_area_m2_dominant/
    # footprint_area_m2_primary (full, unclipped footprint area, credited
    # only to the one dominant parcel), this is bounded by construction to at
    # most this parcel's own area -- computed independently of the id-join
    # vs. spatial-fallback branch above, since it needs every touching
    # parcel, not just each footprint's single assigned one.
    non_synth = footprints.loc[~is_synthetic]
    if len(non_synth) == 0:
        in_parcel_sum = pd.Series(np.nan, index=spine.index)
    else:
        pairs = overlay_polygons(
            spine,
            non_synth.to_crs(spine.crs),
            how='intersection',
            area_intersection=True,
            geom=False,
            suffixes=('_spine', '_footprint'),
        )
        if len(pairs) == 0:
            in_parcel_sum = pd.Series(np.nan, index=spine.index)
        else:
            clipped = pairs['area_intersection_m2']
            clipped = clipped[clipped >= min_overlap_m2]
            in_parcel_sum = clipped.groupby(level=0).sum().reindex(spine.index)

    # A parcel with no footprints has zero footprint area, not an
    # unknown one: the count below says 0 for it, and an area sum that
    # said null for the same parcel disagreed with the count (measured
    # on Lake County FL, 2026-09-08: null on 98.1% of footprint-less
    # parcels). A vacant-land analysis selects exactly those rows, so
    # the null emptied its design matrix. A parcel that does have
    # footprints but no clipped overlap above the floor keeps null,
    # which is the unknown the null was meant for.
    n_fp = n_fp.fillna(0).astype('int64')
    n_primary = n_primary.fillna(0).astype('int64')
    sum_a = sum_a.mask((n_fp == 0) & sum_a.isna(), 0.0)
    in_parcel_sum = in_parcel_sum.mask((n_fp == 0) & in_parcel_sum.isna(), 0.0)
    sum_a_primary = sum_a_primary.mask((n_primary == 0) & sum_a_primary.isna(), 0.0)

    spine['n_footprints_per_parcel'] = n_fp
    spine['n_small_elongated_footprints_per_parcel'] = n_se.fillna(0).astype('int64')
    spine['max_footprint_area_m2'] = max_a
    spine['footprint_area_m2_dominant'] = sum_a
    spine['footprint_area_m2_in_parcel'] = in_parcel_sum
    spine['n_primary_footprints_per_parcel'] = n_primary
    spine['footprint_area_m2_primary'] = sum_a_primary
    spine['max_dwellings_per_footprint'] = max_dwellings.fillna(0).astype('int64')
    spine['max_parcels_per_footprint'] = max_span.fillna(0).astype('int64')
    state.spine = spine

    if state.verbose:
        print(
            '  summarize_footprint_morphology: '
            f'{int((spine["n_small_elongated_footprints_per_parcel"] > 0).sum()):,} '
            f'parcels with manufactured-home-shaped footprints.'
        )
    if state.timer:
        state.timer.mark('Summarize')
    return state
