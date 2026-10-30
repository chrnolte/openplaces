"""Consolidate the footprints of a condominium cluster onto one outline
(consolidate_condo_cluster_footprints).
"""

from __future__ import annotations

import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely

from openplaces.geo.ids import (
    get_openlocationcodes,
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
from openplaces.io.harmonizer.written import note_written
from openplaces.io.transform import make_index_unique
from openplaces.recipe import (
    get_recipe_id,
    source_id_from_recipe_id,
)

# The helpers below replace per-cluster pandas calls whose cost grew
# with the size of the whole unit (a crosswalk mask, an index
# intersection, one reprojection per input). Each returns exactly what
# the call it replaces returned, in the same order: a geometry union or
# area can change in its last bits when its inputs come in a different
# order, and the step's output must not move.


def _crosswalk_rows_by_parcel(crosswalk: pd.DataFrame) -> dict:
    """Row positions of each parcel's links, in crosswalk order.

    Parameters
    ----------
    crosswalk : DataFrame
        Indexed by (spine id, 'parcel_id').

    Returns
    -------
    dict
        Parcel id to an ascending array of row positions.
    """
    pids = crosswalk.index.get_level_values('parcel_id')
    return pd.Series(np.arange(len(pids))).groupby(pids.to_numpy()).indices


def _linked_spine_ids(
    cluster_pids: list, crosswalk_rows: dict, crosswalk_spine_ids: pd.Index
) -> pd.Index:
    """Spine ids linked to any of *cluster_pids*, in crosswalk order.

    Equal to masking the crosswalk with an `isin` on its parcel level
    and taking the unique spine ids of the rows left.

    Parameters
    ----------
    cluster_pids : list
        Parcel ids of one cluster; a repeated id counts once.
    crosswalk_rows : dict
        From :func:`_crosswalk_rows_by_parcel`.
    crosswalk_spine_ids : Index
        The crosswalk's spine-id level.

    Returns
    -------
    Index
        Unique spine ids, in the order they first occur in the crosswalk.
    """
    rows = [crosswalk_rows[pid] for pid in cluster_pids if pid in crosswalk_rows]
    positions = np.unique(np.concatenate(rows)) if rows else np.empty(0, np.intp)
    return crosswalk_spine_ids[positions].unique()


def _take_by_labels(series: pd.Series, labels) -> pd.Series:
    """Rows of *series* whose label is in *labels*, in *series* order.

    Equal to `series.loc[series.index.intersection(labels)]`, which
    deduplicates the index on every call, a cost that grows with the
    unit. This looks the labels up in the index's own hash table, built
    once.

    Parameters
    ----------
    series : Series
        Any series; with a non-unique index the intersection is used.
    labels : array-like
        Labels to select; missing and repeated labels are ignored.

    Returns
    -------
    Series
        The selected rows.
    """
    if not series.index.is_unique:
        return series.loc[series.index.intersection(labels)]
    positions = series.index.get_indexer(pd.unique(np.asarray(labels, dtype=object)))
    return series.iloc[np.sort(positions[positions >= 0])]


def _fast_source_crs(crs):
    """*crs*, or EPSG:4326 itself where *crs* is equivalent to it.

    A CRS read from GeoParquet is PROJJSON, which pyproj parses again
    for every transformation it builds from it, once per cluster. The
    same CRS given by its EPSG code builds in a fraction of the time
    and resolves to the same operation, so the coordinates come out
    identical.

    Measured 2026-10-22 on US-FL-BR: 109 s of 203 s evaluating clusters.

    Parameters
    ----------
    crs : CRS or None
        A geometry set's CRS.

    Returns
    -------
    CRS or None
        The CRS to reproject from.
    """
    wgs84 = pyproj.CRS.from_epsg(4326)
    return wgs84 if crs is not None and pyproj.CRS(crs) == wgs84 else crs


def _to_crs_together(crs_from, crs_to, *parts) -> list[gpd.GeoSeries]:
    """Reproject several geometry sets in one call.

    Reprojection is point by point, so concatenating the inputs gives
    each geometry the coordinates its own call would, and resolves the
    transformation once instead of once per input.

    Parameters
    ----------
    crs_from, crs_to : CRS
        Source and target CRS; every part must be in *crs_from*.
    *parts : GeoSeries or array-like of geometries
        The geometry sets.

    Returns
    -------
    list of GeoSeries
        One per input, each keeping a GeoSeries input's index.
    """
    values = [np.asarray(getattr(p, 'values', p), dtype=object) for p in parts]
    joined = gpd.GeoSeries(np.concatenate(values), crs=crs_from).to_crs(crs_to)
    out, start = [], 0
    for part, value in zip(parts, values):
        end = start + len(value)
        index = part.index if isinstance(part, pd.Series) else None
        out.append(gpd.GeoSeries(joined.values[start:end], index=index, crs=crs_to))
        start = end
    return out


def _first_area_m2_by_label(gdf: gpd.GeoDataFrame, labels) -> dict:
    """Area in square meters of the first row carrying each label.

    Equal to `get_areas(gdf.loc[[label]], 'm2').iloc[0]` per label, in
    one projection of all rows instead of one per label.

    Parameters
    ----------
    gdf : GeoDataFrame
        Polygons, indexed by label.
    labels : iterable
        Labels to measure; each must be in *gdf*'s index.

    Returns
    -------
    dict
        Label to area.
    """
    first = ~gdf.index.duplicated(keep='first')
    selected = gdf.loc[first & gdf.index.isin(list(labels))]
    return get_areas(selected, 'm2').to_dict()


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
    from openplaces.io.harmonizer.spine import in_unit_extent

    # Only parcels within the unit's extent: one outside it is a stray
    # the parcel geospine drops by the same rule, and its cluster's
    # outline would lie in another unit.
    ref_for_clustering = ref_polys[in_unit_extent(ref_polys, state.admin_id)].copy()
    if 'area_ha' not in ref_for_clustering.columns:
        ref_for_clustering['area_ha'] = get_areas(ref_for_clustering, 'ha')

    if state.timer:
        state.timer.mark('Condo clusters: prepare parcels')
    result = _cluster_condo_parcels(
        ref_for_clustering, verbose=state.verbose, **(cluster_thresholds or {})
    )
    if state.timer:
        state.timer.mark('Condo clusters: detect clusters')
    if result is None:
        return state
    component, hub_ids = result

    spine = state.spine
    _source_id = source_id_from_recipe_id(recipe_id)
    _et = entity_type or recipe_id.rsplit('_', 1)[-1].split('-', 1)[0]
    # Built once: a per-cluster scan of the whole crosswalk cost time in
    # proportion to clusters times links.
    crosswalk_rows = _crosswalk_rows_by_parcel(crosswalk)
    crosswalk_spine_ids = crosswalk.index.get_level_values(spine_id_col)
    # Parcels reproject together with the footprints only when both are
    # stored in one CRS.
    same_crs = ref_polys.crs == spine.crs
    from_crs = _fast_source_crs(spine.crs)

    def _evaluate_cluster(cluster_pids: list) -> dict:
        """Real-vs-parcel geometry choice for one cluster's parcel set.

        Shared by the per-original-cluster pass (to decide merges below)
        and the final pass over merged clusters -- simplest correct way to
        get the right combined real-footprint geometry for a merged group
        without hand-merging partial results: re-deriving from the
        combined parcel set naturally re-collects every relevant spine id
        from the crosswalk.
        """
        linked_spine_ids = _linked_spine_ids(
            cluster_pids, crosswalk_rows, crosswalk_spine_ids
        )
        real_geoms = _take_by_labels(spine.geometry, linked_spine_ids).dropna().tolist()
        # The parcel-geometry fallback covers units no real footprint
        # reaches -- it must never include a hub/common-area parcel's own
        # polygon (the surrounding lot, typically an order of magnitude
        # larger than a real unit), or the consolidated shape balloons to
        # roughly the whole lot instead of the building. A hub's real
        # footprint fragments (if any) still enter via real_geoms above,
        # and it still keeps its crosswalk link below -- only this shape
        # fallback excludes it.
        unit_pids = [pid for pid in cluster_pids if pid not in hub_ids]
        unit_polys = _take_by_labels(ref_polys.geometry, unit_pids or cluster_pids)
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
            parts = shapely.get_parts(real_union_raw)
            if same_crs:
                parcel_geom_m, parts_m, unit_polys_m = _to_crs_together(
                    from_crs, crs_m, [parcel_geom], parts, unit_polys
                )
            else:
                parcel_geom_m, parts_m = _to_crs_together(
                    from_crs, crs_m, [parcel_geom], parts
                )
                unit_polys_m = unit_polys.to_crs(crs_m)
            parcel_geom_m = parcel_geom_m.iloc[0]
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
    if state.timer:
        state.timer.mark(
            'Condo clusters: evaluate clusters', n_clusters=len(cluster_pid_lists)
        )

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

    group_pids = []
    for idxs in merge_groups.values():
        cluster_pids = sum((cluster_pid_lists[i] for i in idxs), [])
        ev = _evaluate_cluster(cluster_pids) if len(idxs) > 1 else evaluations[idxs[0]]
        new_rows.append(ev['real_union'] if ev['passes'] else ev['parcel_geom'])
        group_pids.append(cluster_pids)
        superseded_spine_ids.update(ev['linked_spine_ids'])
        consolidated_pids.update(cluster_pids)

    if state.timer:
        state.timer.mark('Condo clusters: build consolidated rows')
    if not new_rows:
        return state

    # One id and one area pass over every new row, not one per cluster
    # and per parcel: both are computed row by row, so the batch gives
    # each row the value it would get alone. Ids come without the
    # duplicate suffix, which a lone row never gets either; a collision
    # with another row is resolved below, exactly as before.
    new_spine = gpd.GeoDataFrame({'geometry': new_rows}, crs=spine.crs)
    new_spine.index = pd.Index(
        get_openlocationcodes(new_spine, handle_duplicates=False), name=spine_id_col
    )
    new_spine['geometry_source'] = f'condo_cluster.{_et}.{_source_id}'
    pid_area_m2 = _first_area_m2_by_label(ref_polys, consolidated_pids)
    for new_id, cluster_pids in zip(new_spine.index, group_pids):
        for pid in cluster_pids:
            crosswalk_additions.append(
                {
                    spine_id_col: new_id,
                    'parcel_id': pid,
                    'link': 'condo cluster',
                    'area_intersection_m2': pid_area_m2[pid],
                }
            )

    if state.timer:
        state.timer.mark('Condo clusters: build consolidated rows')

    state.spine = pd.concat(
        [spine.drop(index=spine.index.intersection(superseded_spine_ids)), new_spine]
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
            note_written(sidecar_path)
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
        state.timer.mark('Condo clusters: rewrite spine, crosswalk and sidecar')
    return state
