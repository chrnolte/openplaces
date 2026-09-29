"""Consolidate the footprints of a condominium cluster onto one outline
(consolidate_condo_cluster_footprints).
"""

from __future__ import annotations

import warnings

import geopandas as gpd
import pandas as pd
import shapely

from openplaces.geo.ids import (
    add_openlocationcode_index,
)
from openplaces.geo.link import get_entity_link_path
from openplaces.geo.polygon import (
    get_areas,
    local_metric_crs,
)
from openplaces.io import to_parquet
from openplaces.io.aggregate import read_file_metadata
from openplaces.io.harmonizer import (
    HarmonizeState,
    _register,
)
from openplaces.io.harmonizer.links._shared import (
    _COVERAGE_SCORE_EPS,
    _LINK_METADATA_KEY,
    _REAL_FOOTPRINT_TOUCH_TOLERANCE_M,
)
from openplaces.io.harmonizer.links.spatial import (
    _superseded_by_consolidation,
)
from openplaces.io.transform import make_index_unique
from openplaces.recipe import (
    get_recipe_id,
    source_id_from_recipe_id,
)


@_register('consolidate_condo_cluster_footprints', phase='geometry')
def consolidate_condo_cluster_footprints(
    state: HarmonizeState,
    entity_type: str | None = 'parcel',
    recipe_id: str | None = None,
    cluster_thresholds: dict | None = None,
    coverage_power: float = -2.0,
    min_coverage_score: float = 0.5,
) -> HarmonizeState:
    """Collapse a stacked-condo building's parcels to one footprint row.

    Runs the same parcel-side clustering as
    :func:`~openplaces.io.harmonizer.attributes.detect_condo_building_clusters`
    (the ``'touches'``-adjacency, tiny-unit-parcel pattern), but against the
    raw parcel reference this footprint spine's own :func:`link_to_reference`
    step already loaded (``state.references``/``state.crosswalks``) --
    necessarily a separate pass, not a read of the parcel spine's own output,
    since the parcel spine's ``summarize_footprint_morphology`` step reads
    *this* recipe's saved output, and a footprint-spine step reading the
    parcel spine's output in the same admin run would be circular.

    For each qualifying cluster: picks **one** geometry source rather than
    combining them. Any real footprint fragments already linked to the
    cluster's parcels are unioned together and, if that union adequately
    covers the cluster's own parcels, it is used as-is; otherwise the union
    of the cluster's own parcel geometries is used instead. "Adequately
    covers" is a single smooth score (see ``_coverage_score`` below) rather
    than a hard cutoff, since real data (a 12-unit cluster, 11 parcels
    97.9-100% covered, one at 30.9%) showed that a lone straggler parcel
    can otherwise sink an excellent match. A real fragment's geometry is
    first trimmed to whichever of its connected parts actually touch the
    cluster's own parcels, since a spine id's crosswalk link can be shared
    with an unrelated, non-adjacent cluster (confirmed on real data:
    `link_to_reference`'s permissive ``min_fraction_of_largest`` trim
    allows one footprint id to link to parcels in two different clusters)
    and its full raw geometry must never be carried wholesale into an
    unrelated cluster's output.

    Separately, if two or more of :func:`_cluster_condo_parcels`'s
    parcel-topology clusters each *independently* qualify for the *same*
    real footprint (confirmed on real data: one ~18-unit building's
    parcels form 3 disconnected touching-groups, each dominated by the
    same single OBM polygon), they are merged into one output row before
    the pick-one-source decision above runs. Without this, each cluster
    would independently claim the *entire* shared polygon, producing
    duplicate/near-identical rows that a later overlap-resolution step
    silently drops -- real, well-evidenced buildings losing footprint
    coverage entirely, not just an imprecise shape. See the ``merge_groups``
    step below.

    Writes **one** new consolidated spine row per (possibly merged)
    cluster, crosswalked to every parcel in it. The original per-fragment
    real rows are dropped (superseded), and every cluster parcel's
    crosswalk entry is rewritten to point at the new row -- so
    :func:`infer_spine_additions`, which must run *after* this step,
    correctly treats every cluster parcel as already covered and does not
    also generate a per-unit synthetic fallback for whichever units the
    original fragments didn't dominate.

    Explicitly not a goal: preserving a 1:1 unit-to-footprint-row mapping.
    A condo unit's individual identity is expected to live at the parcel/
    property layer (``n_properties_per_parcel``, the harmonized property
    spine), not the footprint layer -- this step deliberately trades away
    per-unit footprint rows for one geometrically coherent building shape.

    Parameters
    ----------
    entity_type : str, optional
        Selects the crosswalk/reference the same way
        :func:`infer_spine_additions` does (default ``'parcel'``). Ignored
        when *recipe_id* is given.
    recipe_id : str, optional
        Explicit crosswalk key. Takes precedence over *entity_type*.
    cluster_thresholds : dict, optional
        Forwarded to
        :func:`~openplaces.io.harmonizer.attributes._cluster_condo_parcels`
        (``max_unit_area_ha``, ``max_hub_area_ha``, ``max_hub_aspect_ratio``,
        ``min_group_size``, ``max_group_size``) -- same defaults as
        ``detect_condo_building_clusters``.
    coverage_power : float, optional
        Exponent *p* of the area-weighted generalized-mean coverage score
        (default -2.0; must be negative). At ``p=1`` the score would equal
        the plain area-weighted average per-parcel coverage; as
        ``p -> -inf`` it converges to the harshest possible test, the
        per-parcel minimum. Negative ``p`` smoothly interpolates: a
        low-weight straggler parcel gets outvoted by well-covered peers
        while still pulling the score down, and a parcel with ~0 coverage
        still drives the whole score toward 0 (a tiny base raised to a
        negative power dominates the weighted sum).
    min_coverage_score : float, optional
        Minimum coverage score (default 0.5) for the real footprint union
        to be preferred over the parcel union.
    """
    if state.spine is None:
        return state

    if recipe_id is None and entity_type is not None:
        candidates = list(state.get_crosswalks_by_type(entity_type).keys())
        if not candidates:
            if state.verbose:
                print(
                    '  consolidate_condo_cluster_footprints: no crosswalk for '
                    f'entity_type={entity_type!r}; skipping.'
                )
            return state
        recipe_id = candidates[0]
    if recipe_id is None:
        return state

    crosswalk = state.crosswalks.get(recipe_id)
    ref_polys = state.references.get(recipe_id)
    if crosswalk is None or ref_polys is None:
        return state

    spine_id_col = state.spine.index.name
    if spine_id_col is None or spine_id_col not in crosswalk.index.names:
        return state

    from openplaces.io.harmonizer.attributes import _cluster_condo_parcels

    ref_for_clustering = ref_polys.copy()
    if 'area_ha' not in ref_for_clustering.columns:
        ref_for_clustering['area_ha'] = get_areas(ref_for_clustering, 'ha')

    result = _cluster_condo_parcels(
        ref_for_clustering, verbose=state.verbose, **(cluster_thresholds or {})
    )
    if result is None:
        return state
    component, hub_ids = result

    spine = state.spine
    _source_id = source_id_from_recipe_id(recipe_id)
    _et = entity_type or recipe_id.rsplit('_', 1)[-1].split('-', 1)[0]

    def _evaluate_cluster(cluster_pids: list) -> dict:
        """Real-vs-parcel geometry choice for one cluster's parcel set.

        Shared by the per-original-cluster pass (to decide merges below)
        and the final pass over merged clusters -- simplest correct way to
        get the right combined real-footprint geometry for a merged group
        without hand-merging partial results: re-deriving from the
        combined parcel set naturally re-collects every relevant spine id
        from the crosswalk.
        """
        linked_mask = crosswalk.index.get_level_values('parcel_id').isin(cluster_pids)
        linked_spine_ids = (
            crosswalk[linked_mask].index.get_level_values(spine_id_col).unique()
        )
        real_geoms = (
            spine.loc[spine.index.intersection(linked_spine_ids), 'geometry']
            .dropna()
            .tolist()
        )
        # The parcel-geometry fallback covers units no real footprint
        # reaches -- it must never include a hub/common-area parcel's own
        # polygon (the surrounding lot, typically an order of magnitude
        # larger than a real unit), or the consolidated shape balloons to
        # roughly the whole lot instead of the building. A hub's real
        # footprint fragments (if any) still enter via real_geoms above,
        # and it still keeps its crosswalk link below -- only this shape
        # fallback excludes it.
        unit_pids = [pid for pid in cluster_pids if pid not in hub_ids]
        unit_polys = ref_polys.loc[
            ref_polys.index.intersection(unit_pids or cluster_pids), 'geometry'
        ]
        parcel_geom = unit_polys.union_all()

        # Never output a geometry that mixes a real footprint with parcel
        # boundaries -- pick one source per cluster. A spine id's crosswalk
        # link can be shared with an unrelated, non-adjacent cluster (the
        # crosswalk's own min_fraction_of_largest trim is permissive
        # enough to allow this), so first drop any connected piece of the
        # real geometry that doesn't actually touch this cluster's own
        # parcels -- carrying in a stray, disjoint chunk of someone else's
        # building would corrupt this cluster's shape even if the
        # remaining, genuinely-touching part is trustworthy. Spine/parcel
        # geometry is stored geographic, so the touch-tolerance buffer and
        # the coverage score below both need a local metric reprojection
        # (mirrors _cluster_condo_parcels's own use of local_metric_crs).
        real_union_raw = (
            gpd.GeoSeries(real_geoms, crs=spine.crs).union_all() if real_geoms else None
        )
        real_union = None
        coverage_score = 0.0
        if real_union_raw is not None:
            crs_m = local_metric_crs(gpd.GeoSeries([parcel_geom], crs=spine.crs))
            parcel_geom_m = (
                gpd.GeoSeries([parcel_geom], crs=spine.crs).to_crs(crs_m).iloc[0]
            )
            parts = shapely.get_parts(real_union_raw)
            parts_m = gpd.GeoSeries(parts, crs=spine.crs).to_crs(crs_m)
            touch_zone_m = parcel_geom_m.buffer(_REAL_FOOTPRINT_TOUCH_TOLERANCE_M)
            keep = parts_m.intersects(touch_zone_m).to_numpy()
            if keep.any():
                real_union = shapely.union_all(parts[keep])
                real_union_m = shapely.union_all(parts_m.to_numpy()[keep])

                # Prefer the real footprint only if it credibly represents
                # the whole cluster -- an area-weighted generalized mean
                # of every unit parcel's own coverage fraction (see the
                # coverage_power docstring for why this replaces a hard
                # group-coverage-and-per-parcel-minimum pair of cutoffs).
                unit_polys_m = unit_polys.to_crs(crs_m)
                if parcel_geom_m.area > 0:
                    areas_m = unit_polys_m.area
                    weights = areas_m / areas_m.sum()
                    # GEOS can fail to node an intersection whose operands
                    # were built from individually valid inputs (seen in
                    # Webb County, TX: one cluster's ring collapse crashed
                    # the whole county's harmonize). Repair and retry; if
                    # the repair fails too, this cluster simply does not
                    # consolidate (coverage_score stays 0.0), which is the
                    # conservative fallback, not an error.
                    from shapely.errors import GEOSException

                    fracs = None
                    try:
                        fracs = (
                            unit_polys_m.intersection(real_union_m).area / areas_m
                        ).clip(lower=_COVERAGE_SCORE_EPS)
                    except GEOSException:
                        try:
                            real_union_m = shapely.make_valid(real_union_m)
                            fracs = (
                                unit_polys_m.make_valid()
                                .intersection(real_union_m)
                                .area
                                / areas_m
                            ).clip(lower=_COVERAGE_SCORE_EPS)
                        except GEOSException:
                            warnings.warn(
                                'consolidate_condo_cluster_footprints: GEOS '
                                'failed on one cluster even after repair; '
                                'leaving it unconsolidated.'
                            )
                    if fracs is not None:
                        coverage_score = float(
                            (weights * fracs**coverage_power).sum()
                            ** (1.0 / coverage_power)
                        )
        return {
            'cluster_pids': cluster_pids,
            'linked_spine_ids': linked_spine_ids,
            'real_union': real_union,
            'parcel_geom': parcel_geom,
            'passes': real_union is not None and coverage_score >= min_coverage_score,
        }

    cluster_pid_lists = [
        list(pids) for pids in component.groupby(component).groups.values()
    ]
    evaluations = [_evaluate_cluster(pids) for pids in cluster_pid_lists]

    # Merge clusters that each independently qualify for the same real
    # footprint -- confirmed on real data: an ~18-unit building's parcels
    # form 3 disconnected touching-groups (no shared hub ties them), each
    # separately proving strong coverage against the same single OBM
    # polygon. Left unmerged, each would claim the entire shared polygon,
    # producing duplicate/near-identical rows that a later overlap-
    # resolution step silently drops (real footprint coverage lost
    # entirely, not just an imprecise shape). A spine id that only one
    # side actually passes with can't trigger a merge on its own -- both
    # sides must independently prove strong coverage first.
    merge_parent = list(range(len(cluster_pid_lists)))

    def _find_root(i: int) -> int:
        while merge_parent[i] != i:
            merge_parent[i] = merge_parent[merge_parent[i]]
            i = merge_parent[i]
        return i

    def _union_roots(i: int, j: int) -> None:
        ri, rj = _find_root(i), _find_root(j)
        if ri != rj:
            merge_parent[max(ri, rj)] = min(ri, rj)

    passing_spine_id_clusters: dict = {}
    for i, ev in enumerate(evaluations):
        if not ev['passes']:
            continue
        for sid in ev['linked_spine_ids']:
            passing_spine_id_clusters.setdefault(sid, []).append(i)
    n_merges = 0
    for idxs in passing_spine_id_clusters.values():
        for idx in idxs[1:]:
            if _find_root(idx) != _find_root(idxs[0]):
                n_merges += 1
            _union_roots(idxs[0], idx)

    merge_groups: dict = {}
    for i in range(len(cluster_pid_lists)):
        merge_groups.setdefault(_find_root(i), []).append(i)

    new_rows = []
    crosswalk_additions = []
    superseded_spine_ids: set = set()
    consolidated_pids: set = set()

    for idxs in merge_groups.values():
        cluster_pids = sum((cluster_pid_lists[i] for i in idxs), [])
        ev = _evaluate_cluster(cluster_pids) if len(idxs) > 1 else evaluations[idxs[0]]
        consolidated = ev['real_union'] if ev['passes'] else ev['parcel_geom']

        new_row = add_openlocationcode_index(
            gpd.GeoDataFrame({'geometry': [consolidated]}, crs=spine.crs),
            name=spine_id_col,
        )
        new_row['geometry_source'] = f'condo_cluster.{_et}.{_source_id}'
        new_rows.append(new_row)
        new_id = new_row.index[0]

        for pid in cluster_pids:
            area_m2 = get_areas(ref_polys.loc[[pid]], 'm2').iloc[0]
            crosswalk_additions.append(
                {
                    spine_id_col: new_id,
                    'parcel_id': pid,
                    'link': 'condo cluster',
                    'area_intersection_m2': area_m2,
                }
            )
        superseded_spine_ids.update(ev['linked_spine_ids'])
        consolidated_pids.update(cluster_pids)

    if not new_rows:
        return state

    state.spine = pd.concat(
        [spine.drop(index=spine.index.intersection(superseded_spine_ids)), *new_rows]
    ).sort_index()
    if state.spine.index.duplicated().any():
        state.spine = make_index_unique(state.spine, sort_duplicates_by_area=True)

    additions_df = pd.DataFrame(crosswalk_additions).set_index(
        [spine_id_col, 'parcel_id']
    )
    superseded = _superseded_by_consolidation(
        crosswalk.index, spine_id_col, superseded_spine_ids, consolidated_pids
    )
    state.crosswalks[recipe_id] = pd.concat(
        [crosswalk[~superseded], additions_df]
    ).sort_index()

    overlay = state.overlays.get(recipe_id)
    if overlay is not None:
        # Supersede the overlay by the same rule as the crosswalk above,
        # which drops every row for a consolidated parcel. Filtering on
        # the spine id alone kept the pre-consolidation geometric row
        # wherever its footprint was not itself superseded, so the
        # overlay -- and the sidecar written from it -- carried that pair
        # twice: once as the old overlap and once as the new 'condo
        # cluster' link. That is a genuine pair of different records, so
        # nothing downstream could tell it from a conflict, and the
        # uniqueness guard failed 7 of the 86 CHEER counties on it
        # (Carteret NC 488 pairs, Pender NC 96, Webb TX 7).
        overlay_superseded = _superseded_by_consolidation(
            overlay.index, spine_id_col, superseded_spine_ids, consolidated_pids
        )
        state.overlays[recipe_id] = pd.concat(
            [overlay.loc[~overlay_superseded], additions_df[['area_intersection_m2']]]
        ).sort_index()

    # Re-persist link_to_reference's identity-overlay sidecar (save_link:
    # true), if one exists, so curate-stage readers -- apportion_curated_
    # values, collect_link_ids -- see this cluster's consolidated links
    # too. Without this, the sidecar (written by link_to_reference *before*
    # this step ran) has no rows at all for the new consolidated footprint,
    # and its 'condo_cluster.*' geometry_source prefix never matches
    # apportion_curated_values' own synthetic-row carve-out (which only
    # recognizes infer_spine_additions's '{entity_type}.*' rows) -- so the
    # footprint silently gets no parcel link and resolves to a $0 value
    # despite every real parcel underneath it having a positive one.
    # Rewritten as a direct read-modify-write of the existing parquet
    # (reusing its already-stored fingerprint verbatim) rather than via
    # _write_link_sidecar, which would need a *snapped* argument to
    # reproduce this recipe's own snap_chains link_chain labels -- info
    # this step has no reason to recompute when every pre-existing,
    # non-superseded row's own columns (including link_chain) can simply
    # be carried through untouched.
    sidecar_path = get_entity_link_path(
        get_recipe_id(state.recipe), recipe_id, state.admin_id
    )
    if sidecar_path.exists():
        stored_raw = read_file_metadata(sidecar_path).get(_LINK_METADATA_KEY)
        if stored_raw is not None:
            existing = pd.read_parquet(sidecar_path).set_index(
                [spine_id_col, 'parcel_id']
            )
            # Same supersession rule as the crosswalk and the overlay:
            # by consolidated parcel, not by spine id alone. This is the
            # copy that reaches disk, so filtering it more narrowly than
            # the crosswalk is what put a pair in the sidecar twice --
            # once as its pre-consolidation overlap and once as the new
            # 'condo cluster' link -- and failed every recipe that
            # reloads the sidecar rather than recomputing the overlay.
            kept = existing.loc[
                ~_superseded_by_consolidation(
                    existing.index,
                    spine_id_col,
                    superseded_spine_ids,
                    consolidated_pids,
                )
            ]
            merged = pd.concat(
                [kept, additions_df[['area_intersection_m2', 'link']]]
            ).sort_index()
            to_parquet(
                merged.reset_index(),
                sidecar_path,
                file_metadata={_LINK_METADATA_KEY: stored_raw},
            )
            if state.verbose:
                print(
                    '  consolidate_condo_cluster_footprints: re-persisted link '
                    f'sidecar {sidecar_path.name}'
                )

    if state.verbose:
        merge_note = (
            f', {n_merges:,} merged via a shared real footprint' if n_merges else ''
        )
        print(
            f'  consolidate_condo_cluster_footprints: {len(new_rows):,} building '
            f'clusters consolidated ({len(superseded_spine_ids):,} fragment rows '
            f'replaced{merge_note}).'
        )
    if state.timer:
        state.timer.mark('Consolidate condo clusters')
    return state
