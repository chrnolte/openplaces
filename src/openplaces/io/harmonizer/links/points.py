"""Point references (dwellings, NSI buildings, address points): quality
ordering, multipoint aggregation, duplicate flags and the spatial
point join.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd

from openplaces.geo.link import get_entity_link_path
from openplaces.geo.polygon import (
    get_areas,
)
from openplaces.io.harmonizer import (
    HarmonizeState,
    _rename_right_index,
)
from openplaces.io.harmonizer.links.sidecars import (
    _link_fingerprint,
    _load_point_link_sidecar,
    _write_point_link_sidecar,
)
from openplaces.io.readers import get_entities
from openplaces.io.transform import remap
from openplaces.recipe import (
    get_recipe_id,
    raise_if_coverage_complete,
)


def _dedup_address_points(
    ref: gpd.GeoDataFrame,
    unit_col: str,
    address_col: str,
    housenumber_col: str,
) -> gpd.GeoDataFrame:
    """Deduplicate Overture-style address points for multi-dwelling buildings.

    Groups by (address_col, housenumber_col) — the base address without unit.

    *  When a group contains **unit-specific** records (e.g. "Apt 1", "Apt 2")
       **and** a no-unit record for the same base address, the no-unit record is
       dropped as a redundant aggregate.  Each unit-specific record is kept and
       tagged ``n_dwellings = 1``; downstream :func:`_aggregate_multipoint`
       then sums these into the total per footprint.
    *  When a group contains **only** a no-unit record (single address, no
       unit breakdown), it is kept unchanged.

    Disabled automatically when *unit_col* is absent from *ref*.
    """
    if unit_col not in ref.columns:
        return ref

    key_cols = [c for c in [address_col, housenumber_col] if c in ref.columns]
    if not key_cols:
        return ref

    ref = ref.copy()
    has_unit = ref[unit_col].notna() & (ref[unit_col].astype(str).str.strip() != '')

    ref['_has_unit'] = has_unit
    ref['_group_has_unit'] = ref.groupby(key_cols)['_has_unit'].transform('any')

    mask_drop = ~has_unit & ref['_group_has_unit']
    ref = ref[~mask_drop]

    if 'n_dwellings' not in ref.columns:
        ref['n_dwellings'] = 1.0
    else:
        ref.loc[ref['n_dwellings'].isna(), 'n_dwellings'] = 1.0

    return ref.drop(columns=['_has_unit', '_group_has_unit'])


def _build_size_limit_dict(
    within: pd.DataFrame,
    spine: gpd.GeoDataFrame,
    spine_id_col: str,
    min_samples: int = 10,
) -> dict[str, tuple[float, float]]:
    """Build per-occupancy-class footprint area bounds from Pass 1 (within) matches.

    Computes mean ± 2σ of footprint area (m²) for each ``occupancy_type``
    class in *within*.  Classes with fewer than *min_samples* observations are
    excluded (no size constraint for those classes).

    Parameters
    ----------
    within : DataFrame
        Pass 1 crosswalk (point index, spine_id_col column, occupancy_type column).
    spine : GeoDataFrame
        Spine GeoDataFrame (used to look up footprint areas).
    spine_id_col : str
        Column in *within* that holds the matched footprint ID.
    min_samples : int
        Minimum observations required to compute bounds for a class.
    """
    if 'occupancy_type' not in within.columns or spine_id_col not in within.columns:
        return {}

    areas_m2 = get_areas(spine, unit='m2')
    fp_area = within[spine_id_col].map(areas_m2)

    limits: dict[str, tuple[float, float]] = {}
    for occ, grp_areas in fp_area.groupby(within['occupancy_type']):
        vals = grp_areas.dropna()
        if len(vals) < min_samples:
            continue
        mu, sigma = float(vals.mean()), float(vals.std())
        limits[str(occ)] = (max(0.0, mu - 2 * sigma), mu + 2 * sigma)
    return limits


def _filter_by_size_limit(
    near: pd.DataFrame,
    size_limit_dict: dict[str, tuple[float, float]],
    spine: gpd.GeoDataFrame,
    spine_id_col: str,
) -> pd.DataFrame:
    """Drop proximity-matched pairs whose footprint area falls outside class bounds."""
    if not size_limit_dict or 'occupancy_type' not in near.columns:
        return near
    if spine_id_col not in near.columns:
        return near

    areas_m2 = get_areas(spine, unit='m2')
    fp_area = near[spine_id_col].map(areas_m2)
    lo = near['occupancy_type'].map({k: v[0] for k, v in size_limit_dict.items()})
    hi = near['occupancy_type'].map({k: v[1] for k, v in size_limit_dict.items()})
    no_limit = lo.isna()
    in_range = (fp_area >= lo) & (fp_area <= hi)
    return near[no_limit | in_range]


#: Quality order for picking one representative among several linked
#: reference points: source label ascending (the ranking baked into the
#: label), structure value descending (the larger structure wins).
_POINT_QUALITY_ORDER = {'source': True, 'structure_value': False}


def _point_quality_sort(frame: pd.DataFrame) -> tuple[list[str], list[bool]]:
    """Return the sort columns and directions for a quality ranking.

    Each column keeps its own direction, so a frame carrying only
    'structure_value' still sorts it descending. Slicing a fixed
    ascending list positionally instead paired the surviving column with
    the first direction, which picked the *lowest*-value point as the
    representative wherever 'source' was absent.

    Parameters
    ----------
    frame : pandas.DataFrame
        Linked reference points to be ranked.

    Returns
    -------
    tuple of (list of str, list of bool)
        Present sort columns and their ascending flags, both empty when
        the frame carries neither.
    """
    cols = [c for c in _POINT_QUALITY_ORDER if c in frame.columns]
    return cols, [_POINT_QUALITY_ORDER[c] for c in cols]


def _aggregate_multipoint(
    linked: pd.DataFrame,
    spine_id_col: str,
    source_geometry_type,
    verbose: bool = False,
) -> pd.DataFrame:
    """Aggregate multiple points linked to the same footprint into one row.

    Mirrors Lochhead et al. (2026) ``merge_into_group`` / ``merge_occ_type``
    logic.

    For ``single_building_point`` sources (e.g. NSI): sums unit counts across
    the matched occupancy classes via ``_OCC_UNITS`` and re-classifies the
    total using :func:`~openplaces.io.harmonizer.attributes.reverse_occ_units`.
    For ``single_dwelling_point`` sources (address points): sums
    ``n_dwellings`` across all matched points per footprint.

    The highest-quality representative row (by existing sort order) is kept as
    the output row; its ``purpose_subgroup`` and ``n_dwellings`` are
    updated in-place.
    """
    from openplaces.core.schema import SourceGeometryType as _SGT
    from openplaces.io.harmonizer.attributes import _OCC_UNITS, reverse_occ_units

    if spine_id_col not in linked.columns:
        return linked

    dup_fp = linked[spine_id_col].duplicated(keep=False)
    if not dup_fp.any():
        return linked

    singles = linked[~dup_fp]
    multis = linked[dup_fp]

    _is_nsi = source_geometry_type == _SGT.single_building_point
    _is_addr = source_geometry_type == _SGT.single_dwelling_point

    agg_idx: list = []
    agg_rows: list = []
    n_aggregated = 0

    for _fp_id, group in multis.groupby(spine_id_col, sort=False):
        sort_cols, sort_ascending = _point_quality_sort(group)
        if sort_cols:
            group = group.sort_values(sort_cols, ascending=sort_ascending)
        rep = group.iloc[0].copy()

        if len(group) > 1:
            n_aggregated += 1
            if _is_nsi and 'occupancy_type' in group.columns:
                total = group['occupancy_type'].map(_OCC_UNITS).fillna(0.0).sum()
                if total > 0:
                    rep['occupancy_type'] = reverse_occ_units(total)
                    rep['n_dwellings'] = float(round(total))
            elif _is_addr:
                if 'n_dwellings' in group.columns:
                    total = (
                        pd.to_numeric(group['n_dwellings'], errors='coerce')
                        .fillna(1.0)
                        .sum()
                    )
                else:
                    # each single_dwelling_point represents one unit
                    total = float(len(group))
                if total > 0:
                    rep['n_dwellings'] = float(total)
                for _col in group.columns:
                    if (
                        not pd.api.types.is_numeric_dtype(group[_col])
                        and not pd.api.types.is_bool_dtype(group[_col])
                        and not pd.api.types.is_datetime64_any_dtype(group[_col])
                    ):
                        _seen = dict.fromkeys(
                            str(v) for v in group[_col] if pd.notna(v)
                        )
                        if _seen:
                            rep[_col] = '; '.join(_seen)

        agg_idx.append(group.index[0])
        agg_rows.append(rep)

    agg_df = pd.DataFrame(agg_rows, index=agg_idx)
    result = pd.concat([singles, agg_df])

    if verbose and n_aggregated > 0:
        print(
            f'  Aggregate: {n_aggregated:,d} footprints with >1 point; units aggregated'
        )
    return result


def flag_duplicate_points(
    ref: pd.DataFrame,
    key_col: str,
    ignore_sources: list[str],
) -> pd.Series:
    """Flag colocated duplicate points from low-rank sources.

    Within groups of two or more points sharing *key_col* (e.g. NSI's
    ``building_id_ubid``, or the ``_olc`` location cell), rows whose
    ``source`` is in *ignore_sources* are labeled
    ``'colocated low-rank source'`` when the group also contains at least one
    source outside that set — a higher-level record to defer to. A group made
    up entirely of ignorable sources stays unflagged (nothing better exists),
    as does any point at a unique location. Returns an object Series aligned
    to *ref* (null = kept); rows are never dropped here — the exclusion is
    applied where the evidence is merged onto a spine
    (:func:`~openplaces.io.harmonizer.attributes.reconcile_attributes`).
    """
    resolution = pd.Series(pd.NA, index=ref.index, dtype=object)
    if not ignore_sources or 'source' not in ref.columns or key_col not in ref.columns:
        return resolution

    key = ref[key_col].astype('string')
    in_group = key.notna() & key.duplicated(keep=False)
    if not in_group.any():
        return resolution

    source = ref['source'].astype(object)
    ignorable = source.isin(list(ignore_sources))
    # Rows with a null key fall out of the groupby; treat them as having no
    # better sibling (they are not in a group anyway).
    has_better = (~ignorable).groupby(key).transform('any').fillna(False).astype(bool)
    flagged = in_group & ignorable & has_better
    resolution[flagged] = 'colocated low-rank source'
    return resolution


def _point_step_config(
    thresholds: dict,
    remap_id: str | None,
    source_geometry_type=None,
) -> dict:
    """Build the step-config half of a spatial_point link fingerprint.

    Shared by the writer (:func:`_link_spatial_point`) and the geospine
    loader, which have to spell the same dict or every attribute run
    fails closed against a sidecar that is in fact current.

    The source geometry type is included only where it changes what the
    sidecar holds, which is when aggregate_multipoint is on: it selects
    between the building-point and dwelling-point aggregation branches,
    and changing it used to reload a sidecar built by the other. Off that
    path it touches nothing the sidecar carries, and including it there
    would invalidate every sidecar already on disk for no gain.

    Parameters
    ----------
    thresholds : dict
        The step's thresholds block.
    remap_id : str or None
        Value-crosswalk recipe applied to the reference before linking.
    source_geometry_type : SourceGeometryType or str or None, optional
        The step's declared source geometry type, if any.

    Returns
    -------
    dict
        Step config to hand to :func:`_link_fingerprint`.
    """
    config = {
        'join': 'spatial_point',
        'thresholds': thresholds,
        'remap_id': remap_id,
    }
    if thresholds.get('aggregate_multipoint'):
        config['source_geometry_type'] = (
            None if source_geometry_type is None else str(source_geometry_type)
        )
    return config


def _link_spatial_point(
    state: HarmonizeState,
    recipe_id: str,
    entity_type: str | None,
    remap_id: str | None,
    thresholds: dict,
    save_link: bool = False,
) -> HarmonizeState:
    """Point-in-polygon join: reference points → spine entities.

    Four-pass attribution (Lochhead et al. 2026, Table 3):

    1. ``predicate='within'`` — direct containment (Step 2).
    2. ``sjoin_nearest`` up to *proximity_m* (default 10 m) — inner proximity
       fallback for near-miss points (Step 5).
    3. ``sjoin_nearest`` up to *far_proximity_m* (default 100 m), constrained
       to the same parcel as the point (Step 6).  Set to 0 to disable.
    4. ``sjoin_nearest`` up to *unbounded_proximity_m* (default 0 = disabled)
       — nearest-footprint fallback with no parcel constraint (Step 7).

    Parcel-derived footprints (``geometry_source`` starting with ``'parcel'``, the
    parcel-shaped fallbacks added by :func:`infer_spine_additions`) participate in
    Pass 1 only: they acquire points by strict containment and are excluded from
    the proximity passes, since a parcel-shaped polygon is not a real building
    outline and must not grab nearby points.

    Optional pre-processing and post-processing steps controlled via
    *thresholds*:

    ``dedup_addresses`` (bool)
        Run :func:`_dedup_address_points` before spatial linking.  Groups by
        base address (street + housenumber), keeps the unit-less record as the
        spatial representative, and counts unit siblings as ``n_dwellings``.
        Column name overrides: ``dedup_unit_number_col`` (default ``'unit_number'``),
        ``dedup_address_street_col`` (default ``'address_street'``),
        ``dedup_address_number_col`` (default ``'address_number'``).
    ``use_size_limit`` (bool)
        After Passes 2–3, drop pairs whose footprint area is outside the
        mean ± 2σ bounds derived from Pass 1 matches per occupancy class
        (:func:`_build_size_limit_dict` / :func:`_filter_by_size_limit`).
    ``aggregate_multipoint`` (bool)
        After linking, aggregate multiple points per footprint into one row
        (:func:`_aggregate_multipoint`) rather than silently dropping
        lower-priority duplicates.

    After linking, joins all points to the first polygon reference in
    ``state.overlays`` to attach a polygon reference ID (e.g. parcel_id).

    With *save_link* (default True) the final flat crosswalk is persisted
    geometry-free at the canonical entity-link path and reloaded on later
    runs while its footer fingerprint still matches, skipping every
    linking pass (see :func:`_write_point_link_sidecar`).
    """
    _EA_CRS = 'EPSG:6933'
    proximity_m: float = thresholds.get('proximity_m', 10.0)
    far_proximity_m: float = thresholds.get('far_proximity_m', 100.0)
    unbounded_m: float = thresholds.get('unbounded_proximity_m', 0.0)
    dedup_addresses: bool = bool(thresholds.get('dedup_addresses', False))
    use_size_limit: bool = bool(thresholds.get('use_size_limit', False))
    aggregate_mp: bool = bool(thresholds.get('aggregate_multipoint', False))

    # missing='warn': a reference recipe (e.g. dwelling-overture-2025) can
    # genuinely have zero coverage for this admin unit -- many rural U.S.
    # counties have no Overture address points at all, not merely an
    # unfinished ingest -- so this is a normal, expected admin-scoped gap,
    # not an error.
    ref = get_entities(recipe_id, state.admin_id, geom=True, missing='warn')
    if ref is None or len(ref) == 0:
        raise_if_coverage_complete(recipe_id, state.admin_id)
        # Record the reference's type even with nothing to link:
        # reconcile_attributes needs it to name this source's evidence
        # columns, which it writes as nulls so a zero-coverage unit's
        # spine carries the same columns as its neighbors'.
        if entity_type:
            state.reference_types[recipe_id] = entity_type
        if state.verbose:
            print(
                f'  Link ({entity_type or "point"}): no {recipe_id} for '
                f'{state.admin_id}; skipping.'
            )
        return state
    if remap_id:
        ref = remap(ref, remap_id)
    if state.verbose:
        print(f'  Load: {len(ref):,d} {entity_type or "point"} ({recipe_id})')

    # Location key for detecting two points at (near-)identical coordinates
    # (e.g. an ESRI-sourced point duplicating a Parcel-sourced one at the same
    # address) regardless of which linking pass matches each one. Computed here,
    # before geometry is dropped by Pass 1's sjoin below, as a plain non-geometry
    # column so it rides through every pass the same way `source` does.
    # codelength=11 gives a ~2.5-3 m snapping cell (avoids float/rounding
    # false-negatives); handle_duplicates=False keeps colocated points' codes
    # equal (the whole point of using this as a grouping key).
    from openplaces.geo.ids import get_openlocationcodes

    ref = ref.copy()
    ref['_olc'] = get_openlocationcodes(ref, codelength=11, handle_duplicates=False)

    # Duplicate resolution BEFORE any merging to footprints/parcels: flag (not
    # drop) colocated duplicate points per the recipe-chosen rule. The label
    # rides through every linking pass onto state.crosswalks; the actual
    # exclusion is applied where NSI evidence is merged onto the spine
    # (_attribute_point_reference), so the resolution stays inspectable and
    # the method swappable per recipe.
    resolve_dup = thresholds.get('resolve_duplicates')
    if resolve_dup:
        key_col = resolve_dup.get('key', 'building_id_ubid')
        key_col = '_olc' if key_col == 'olc' else key_col
        ignore_sources = resolve_dup.get('ignore_sources', [])
        ref['duplicate_resolution'] = flag_duplicate_points(
            ref, key_col, ignore_sources
        )
        if state.verbose:
            n_flagged = int(ref['duplicate_resolution'].notna().sum())
            if n_flagged:
                print(
                    f'  Dedup (colocated): {n_flagged:,d} low-rank duplicate '
                    'point(s) flagged'
                )

    # Address deduplication: keep building-level representative, count unit siblings
    if dedup_addresses:
        n_before = len(ref)
        ref = _dedup_address_points(
            ref,
            unit_col=thresholds.get('dedup_unit_number_col', 'unit_number'),
            address_col=thresholds.get('dedup_address_street_col', 'address_street'),
            housenumber_col=thresholds.get(
                'dedup_address_number_col', 'address_number'
            ),
        )
        if state.verbose and n_before - len(ref) > 0:
            print(
                f'  Dedup (address): {len(ref):,d} after merging '
                f'{n_before - len(ref):,d} duplicates'
            )

    spine_id_col = state.spine.index.name

    sidecar_path = None
    fingerprint = None
    linked = None
    if save_link:
        sidecar_path = get_entity_link_path(
            get_recipe_id(state.recipe), recipe_id, state.admin_id
        )
        fingerprint = _link_fingerprint(
            state,
            recipe_id,
            _point_step_config(
                thresholds,
                remap_id,
                state.source_geometry_types.get(recipe_id),
            ),
        )
        if not state.reprocess:
            linked = _load_point_link_sidecar(
                sidecar_path, fingerprint, verbose=state.verbose
            )
    computed_fresh = linked is None

    if computed_fresh:
        # Pass 1 — within (Lochhead Step 2)
        within = gpd.sjoin(ref, state.spine[['geometry']]).drop(columns='geometry')
        within = _rename_right_index(within, spine_id_col, spine_id_col)
        attributed_idx: set = set(within.index)
        n_pass1 = len(attributed_idx)

        # Parcel-derived footprints (geometry_source like 'parcel.<source>', set by
        # infer_spine_additions) are parcel-shaped fallbacks, not true building
        # outlines, so they link points by strict containment only (Pass 1). Exclude
        # them from the proximity passes below so a parcel-shaped polygon never grabs
        # a nearby point — the only valid criterion for them is 'within'.
        if 'geometry_source' in state.spine.columns:
            parcel_derived = (
                state.spine['geometry_source']
                .astype('string')
                .str.startswith('parcel')
                .fillna(False)
            )
            proximity_spine = state.spine.loc[~parcel_derived, ['geometry']]
        else:
            proximity_spine = state.spine[['geometry']]

        # Build per-class size limits from Pass 1 for use in Passes 2–3
        size_limit_dict: dict[str, tuple[float, float]] = {}
        if use_size_limit:
            size_limit_dict = _build_size_limit_dict(within, state.spine, spine_id_col)

        # Pass 2 — inner proximity, default 10 m (Step 5)
        if proximity_m > 0:
            unlinked = ref[~ref.index.isin(attributed_idx)]
            if not unlinked.empty:
                spine_proj = proximity_spine.to_crs(_EA_CRS)
                unlinked_proj = unlinked.to_crs(_EA_CRS)
                near = gpd.sjoin_nearest(
                    unlinked_proj,
                    spine_proj,
                    how='left',
                    max_distance=proximity_m,
                    distance_col='_dist',
                )
                near = _rename_right_index(near, spine_id_col, spine_id_col)
                near = near[near[spine_id_col].notna()].drop(columns='_dist')
                near = near.set_crs(ref.crs, allow_override=True)
                if size_limit_dict:
                    near = _filter_by_size_limit(
                        near, size_limit_dict, state.spine, spine_id_col
                    )
                within = pd.concat([within, near])
                attributed_idx |= set(near.index)

        # Pass 3 — outer proximity, default 100 m, same-parcel constraint (Step 6)
        overlay_ids = list(state.overlays.keys())
        poly_ref: gpd.GeoDataFrame | None = None
        if far_proximity_m > 0 and overlay_ids:
            poly_ref = state.references.get(overlay_ids[0])
            unlinked = ref[~ref.index.isin(attributed_idx)]
            if not unlinked.empty and poly_ref is not None:
                poly_ref_id_col = poly_ref.index.name
                pts_parcel = gpd.sjoin(
                    unlinked[['geometry']], poly_ref[['geometry']], how='left'
                ).drop(columns='geometry')
                pts_parcel = _rename_right_index(
                    pts_parcel, poly_ref_id_col, '_pt_parcel'
                )

                fp_parcel = (
                    state.crosswalks[overlay_ids[0]]
                    .reset_index()[[spine_id_col, poly_ref_id_col]]
                    .drop_duplicates(spine_id_col)
                    .set_index(spine_id_col)[poly_ref_id_col]
                    .rename('_fp_parcel')
                )

                spine_proj = proximity_spine.to_crs(_EA_CRS)
                unlinked_proj = unlinked.to_crs(_EA_CRS)
                far = gpd.sjoin_nearest(
                    unlinked_proj,
                    spine_proj,
                    how='left',
                    max_distance=far_proximity_m,
                    distance_col='_dist',
                )
                far = _rename_right_index(far, spine_id_col, spine_id_col)
                far = far[far[spine_id_col].notna()].drop(columns='_dist')
                far = far.set_crs(ref.crs, allow_override=True)
                far = far.join(pts_parcel[['_pt_parcel']])
                far = far.join(fp_parcel, on=spine_id_col)
                far = far[
                    far['_pt_parcel'].notna() & (far['_pt_parcel'] == far['_fp_parcel'])
                ].drop(columns=['_pt_parcel', '_fp_parcel'])
                if size_limit_dict:
                    far = _filter_by_size_limit(
                        far, size_limit_dict, state.spine, spine_id_col
                    )
                within = pd.concat([within, far])
                attributed_idx |= set(far.index)

        # Pass 4 — unbounded nearest-footprint fallback, no parcel constraint (Step 7)
        if unbounded_m > 0:
            unlinked = ref[~ref.index.isin(attributed_idx)]
            if not unlinked.empty:
                spine_proj_p4 = proximity_spine.to_crs(_EA_CRS)
                unlinked_proj_p4 = unlinked.to_crs(_EA_CRS)
                far2 = gpd.sjoin_nearest(
                    unlinked_proj_p4,
                    spine_proj_p4,
                    how='left',
                    max_distance=unbounded_m,
                    distance_col='_dist',
                )
                far2 = _rename_right_index(far2, spine_id_col, spine_id_col)
                far2 = far2[far2[spine_id_col].notna()].drop(columns='_dist')
                far2 = far2.set_crs(ref.crs, allow_override=True)
                within = pd.concat([within, far2])
                attributed_idx |= set(far2.index)

        linked = within

        # Deduplicate: one spine entity per point (keep highest-quality source first)
        sort_cols, sort_ascending = _point_quality_sort(linked)
        if sort_cols:
            linked = linked.sort_values(sort_cols, ascending=sort_ascending)
        n_repeated = int(linked.index.duplicated().sum())
        if n_repeated:
            linked = linked[~linked.index.duplicated()].copy()
            if state.verbose:
                print(
                    f'  Deduplicate: {n_repeated:,d} reference point(s) matched '
                    f'more than one {spine_id_col}; kept the first by source '
                    f'priority ({len(linked):,d} remain)'
                )

        # Filter: for footprints that already have a same-parcel dwelling point,
        # drop dwelling points that are on a different parcel.
        _poly_ref_filter = poly_ref
        if _poly_ref_filter is None and overlay_ids:
            _poly_ref_filter = state.references.get(overlay_ids[0])
        if _poly_ref_filter is not None and not linked.empty:
            _prf_id = _poly_ref_filter.index.name
            _ref_sub = ref.loc[ref.index.isin(linked.index), ['geometry']]
            _pts_poly = gpd.sjoin(
                _ref_sub, _poly_ref_filter[['geometry']], how='left'
            ).drop(columns='geometry')
            _pts_poly = _rename_right_index(_pts_poly, _prf_id, '_pt_parcel')
            n_repeated = int(_pts_poly.index.duplicated().sum())
            if n_repeated:
                _pts_poly = _pts_poly[~_pts_poly.index.duplicated()].copy()
                if state.verbose:
                    print(
                        f'  Cross-parcel filter: {n_repeated:,d} reference '
                        f'point(s) fell in more than one {_prf_id}; kept the '
                        'first'
                    )
            _fp_parcel_sets = (
                state.crosswalks[overlay_ids[0]]
                .reset_index()[[spine_id_col, _prf_id]]
                .groupby(spine_id_col)[_prf_id]
                .agg(set)
            )
            _pt_parcel = linked.join(_pts_poly[['_pt_parcel']])['_pt_parcel']
            _fp_parcel_set = linked[spine_id_col].map(_fp_parcel_sets)
            _is_same_parcel = pd.Series(
                [
                    (pd.notna(pt) and isinstance(fps, set) and pt in fps)
                    for pt, fps in zip(_pt_parcel, _fp_parcel_set)
                ],
                index=linked.index,
                dtype=bool,
            )
            _is_cross_parcel = pd.Series(
                [
                    (pd.notna(pt) and isinstance(fps, set) and pt not in fps)
                    for pt, fps in zip(_pt_parcel, _fp_parcel_set)
                ],
                index=linked.index,
                dtype=bool,
            )
            _fp_has_same = _is_same_parcel.groupby(linked[spine_id_col]).transform(
                'any'
            )
            _mask_drop = _fp_has_same & _is_cross_parcel
            if _mask_drop.any():
                n_cross = int(_mask_drop.sum())
                linked = linked[~_mask_drop].copy()
                if state.verbose:
                    print(
                        f'  Filter (cross-parcel): {n_cross:,d} cross-parcel link(s) '
                        f'dropped ({len(linked):,d} remain)'
                    )

        # Aggregate multiple points per footprint (Lochhead merge_into_group)
        if aggregate_mp:
            sgt = state.source_geometry_types.get(recipe_id)
            linked = _aggregate_multipoint(
                linked, spine_id_col, sgt, verbose=state.verbose
            )

        # Attach polygon reference ID (e.g. parcel_id) to each linked point
        if poly_ref is None and overlay_ids:
            poly_ref = state.references.get(overlay_ids[0])
        if poly_ref is not None:
            poly_ref_id_col = poly_ref.index.name
            ref_on_poly = gpd.sjoin(ref, poly_ref[['geometry']], how='left').drop(
                columns='geometry'
            )
            ref_on_poly = _rename_right_index(
                ref_on_poly, poly_ref_id_col, poly_ref_id_col
            )
            n_repeated = int(ref_on_poly.index.duplicated().sum())
            if n_repeated:
                ref_on_poly = ref_on_poly[~ref_on_poly.index.duplicated()].copy()
                if state.verbose:
                    print(
                        f'  Reference join ({recipe_id}): {n_repeated:,d} point(s) '
                        f'fell in more than one {poly_ref_id_col}; kept the first'
                    )
            if poly_ref_id_col in ref_on_poly.columns:
                linked = linked.join(ref_on_poly[[poly_ref_id_col]])

        # Drop low-quality NSI duplicates when a higher-quality source
        # covers the same entity
        if 'source' in linked.columns and overlay_ids:
            poly_ref_id_col = (
                state.references[overlay_ids[0]].index.name if overlay_ids else None
            )
            group_cols = [
                c for c in [spine_id_col, poly_ref_id_col] if c and c in linked.columns
            ]
            if group_cols:
                low_quality = {'ESRI', 'HAZUS/NSI-2015'}
                mask_dup = linked[group_cols].duplicated(keep=False)
                first_source = linked.groupby(group_cols, sort=False)[
                    'source'
                ].transform('first')
                mask_to_drop = (
                    mask_dup
                    & linked['source'].isin(low_quality)
                    & first_source.eq('Parcel')
                )
                linked = linked[~mask_to_drop]

        # Flag ESRI points that share a location with a differently-sourced point
        # (e.g. a home-office duplicate of a Parcel-sourced record at the same
        # address) -- broader and more targeted than the entity-membership drop
        # above: it fires purely on coordinates, regardless of whether the two
        # points happen to link to the same footprint/parcel, and regardless of
        # what the other source is (not just 'Parcel'). Consumers that sum/count
        # NSI evidence into an upward Single->Multi-Family correction (e.g.
        # n_dwellings in _attribute_point_reference) should exclude flagged rows;
        # other uses of `linked` are unaffected -- rows are flagged, not dropped.
        if 'source' in linked.columns and '_olc' in linked.columns:
            colocated_sources = (
                linked.groupby('_olc')['source'].transform('nunique') > 1
            )
            linked['exclude_from_upward_correction'] = colocated_sources & linked[
                'source'
            ].eq('ESRI')

        if state.verbose:
            n_linked = (
                linked[spine_id_col].notna().sum()
                if spine_id_col in linked.columns
                else len(linked)
            )
            n_proximity = len(attributed_idx) - n_pass1
            print(
                f'  Link (point): {n_linked:,d} points linked'
                + (f' ({n_proximity:,d} via proximity)' if n_proximity > 0 else '')
            )
    if save_link and computed_fresh:
        _write_point_link_sidecar(
            sidecar_path, linked, fingerprint, verbose=state.verbose
        )

    if state.timer:
        state.timer.mark('Link (point)')

    state.references[recipe_id] = ref
    state.crosswalks[recipe_id] = linked
    if entity_type:
        state.reference_types[recipe_id] = entity_type
    return state
