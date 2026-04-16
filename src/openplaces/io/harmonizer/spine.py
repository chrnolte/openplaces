"""
Pipeline steps for building and refining the primary entity spine:
  - resolve_spine: merge multiple source GeoDataFrames via IoU dedup
  - split_by_reference: split spine geometries at reference boundaries [stub]
"""

from __future__ import annotations

import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.strtree import STRtree

from openplaces.diagnostics import find_recipes
from openplaces.geo.polygon import overlay_polygons
from openplaces.io.harmonizer import HarmonizeState, _register
from openplaces.io.readers import get_entities


def get_oriented_dims(geom) -> tuple[float, float, float]:
    """Return ``(angle_deg % 180, length, width)`` of the minimum bounding rectangle."""
    obb = geom.minimum_rotated_rectangle
    coords = np.array(obb.exterior.coords)[:-1]  # 4 corners
    d0 = coords[1] - coords[0]
    d1 = coords[2] - coords[1]
    l0, l1 = float(np.linalg.norm(d0)), float(np.linalg.norm(d1))
    if l0 >= l1:
        return float(np.degrees(np.arctan2(d0[1], d0[0])) % 180), l0, l1
    return float(np.degrees(np.arctan2(d1[1], d1[0])) % 180), l1, l0


def project_onto_axis(geom, cos_a: float, sin_a: float) -> tuple[float, float]:
    """Return ``(min, max)`` projection of polygon coordinates onto ``(cos_a, sin_a)``.

    Works for both the long-axis direction (to compute projection interval)
    and the perpendicular direction (to compute lateral centroid offset).
    No rotation is applied — the projection is a direct dot product.
    """
    if hasattr(geom, 'exterior'):
        coords = np.array(geom.exterior.coords)
    else:
        # MultiPolygon — collect coordinates from all parts.
        coords = np.concatenate([np.array(g.exterior.coords) for g in geom.geoms])
    dots = coords[:, 0] * cos_a + coords[:, 1] * sin_a
    return float(dots.min()), float(dots.max())


def drop_elongated_duplicates(
    spine: gpd.GeoDataFrame,
    to_add: gpd.GeoDataFrame,
    aspect_ratio_min: float = 2.5,
    angle_tol_deg: float = 15.0,
    long_overlap_min: float = 0.5,
    lateral_sep_ratio_max: float = 2.0,
) -> gpd.GeoDataFrame:
    """Remove from *to_add* candidates that are elongated duplicates of *spine*.

    Catches displaced thin-rectangle buildings (mobile homes, trailers) that
    escape IoU-based deduplication because the positional offset is perpendicular
    to the long axis rather than along it.  The two footprints do **not** need to
    overlap — a close lateral neighbour with a matching long axis is enough.

    Two polygons are declared duplicates when all of the following hold:

    * Both have minimum-bounding-rectangle aspect ratio ≥ *aspect_ratio_min*
    * Long-axis orientations agree to within *angle_tol_deg* degrees (mod 180°)
    * 1-D projections onto the shared long axis overlap by ≥ *long_overlap_min*
      (fraction of the shorter polygon's projected length)
    * Lateral (perpendicular) distance between centroid projections is
      < *lateral_sep_ratio_max* × average width

    Parameters
    ----------
    aspect_ratio_min : float
        Minimum OBB aspect ratio (length / width) to classify as elongated.
    angle_tol_deg : float
        Maximum long-axis angle difference (degrees, mod 180°) to be aligned.
    long_overlap_min : float
        Minimum fraction of the shorter polygon's length covered by the
        long-axis projection overlap.
    lateral_sep_ratio_max : float
        Maximum lateral separation as a multiple of the average polygon width.
    """
    if to_add.empty or spine.empty:
        return to_add

    # Compute OBB dimensions for all candidates.
    cand_dims = to_add.geometry.map(get_oriented_dims)
    cand_angle = cand_dims.map(lambda x: x[0])
    cand_width = cand_dims.map(lambda x: x[2])
    cand_aspect = cand_dims.map(lambda x: x[1]) / cand_width.clip(lower=1e-6)

    mask_elong_cand = cand_aspect >= aspect_ratio_min
    if not mask_elong_cand.any():
        return to_add

    # Compute OBB dimensions for spine and keep only elongated entries.
    spine_dims = spine.geometry.map(get_oriented_dims)
    spine_angle = spine_dims.map(lambda x: x[0])
    spine_width = spine_dims.map(lambda x: x[2])
    spine_aspect = spine_dims.map(lambda x: x[1]) / spine_width.clip(lower=1e-6)

    mask_elong_spine = spine_aspect >= aspect_ratio_min
    if not mask_elong_spine.any():
        return to_add

    spine_elong = spine[mask_elong_spine]
    spine_geoms_arr = np.array(spine_elong.geometry.values)
    spine_angles_arr = spine_angle[mask_elong_spine].values
    spine_widths_arr = spine_width[mask_elong_spine].values
    tree = STRtree(spine_geoms_arr)

    dup_ids: set = set()
    for cand_id in to_add.index[mask_elong_cand]:
        cand_geom = to_add.geometry[cand_id]
        angle = cand_angle[cand_id]
        width = cand_width[cand_id]

        # Spatial proximity search via buffered candidate.
        hits = tree.query(
            cand_geom.buffer(width * lateral_sep_ratio_max),
            predicate='intersects',
        )
        if len(hits) == 0:
            continue

        # Long-axis unit vector and its perpendicular.
        rad = np.radians(angle)
        cos_l, sin_l = float(np.cos(rad)), float(np.sin(rad))
        cos_p, sin_p = -sin_l, cos_l  # perpendicular

        # Long-axis projection interval for the candidate.
        cmin_l, cmax_l = project_onto_axis(cand_geom, cos_l, sin_l)
        # Centroid lateral projection for the candidate.
        c_lat = (
            project_onto_axis(cand_geom, cos_p, sin_p)[0]
            + project_onto_axis(cand_geom, cos_p, sin_p)[1]
        ) / 2

        for i in hits:
            s_angle = spine_angles_arr[i]
            s_width = spine_widths_arr[i]

            # Angle alignment (mod 180°).
            diff = abs(angle - s_angle)
            if min(diff, 180.0 - diff) > angle_tol_deg:
                continue

            spine_geom = spine_geoms_arr[i]

            # Long-axis projection overlap.
            smin_l, smax_l = project_onto_axis(spine_geom, cos_l, sin_l)
            overlap_len = max(0.0, min(cmax_l, smax_l) - max(cmin_l, smin_l))
            min_long = min(cmax_l - cmin_l, smax_l - smin_l)
            if min_long <= 0 or overlap_len / min_long < long_overlap_min:
                continue

            # Lateral separation between centroid projections.
            smin_p, smax_p = project_onto_axis(spine_geom, cos_p, sin_p)
            s_lat = (smin_p + smax_p) / 2
            lateral = abs(c_lat - s_lat)
            if lateral < lateral_sep_ratio_max * (width + s_width) / 2:
                dup_ids.add(cand_id)
                break

    if dup_ids:
        to_add = to_add[~to_add.index.isin(dup_ids)]
    return to_add


@_register('resolve_spine')
def resolve_spine(
    state: HarmonizeState,
    sources: list[dict] | None = None,
    thresholds: dict | None = None,
) -> HarmonizeState:
    """Build the primary entity spine from multiple prioritized sources.

    Merges GeoDataFrames from each entry in *sources* in order.  A geometry
    from a lower-priority source is added to the spine only if its IoU with
    every existing spine geometry is below ``overlap_iou_max`` (default 0.02).

    Source entries may include an ``auto_discover: true`` sentinel entry,
    which is replaced at runtime by all ingest recipes of the same entity
    type that are scoped to child admin_ids of the recipe's ``admin_id``.

    Parameters
    ----------
    sources : list of dict
        Ordered source entries.  Each entry is either
        ``{'recipe_id': str, 'label': str}`` or ``{'auto_discover': True}``.
    thresholds : dict, optional
        ``overlap_iou_max`` (float, default 0.02) — maximum IoU to treat two
        footprints as overlapping duplicates.
        ``elongated_aspect_min`` (float) — when set, enables the elongated-duplicate
        filter via :func:`drop_elongated_duplicates`.  Also accepts
        ``elongated_angle_tol``, ``elongated_long_overlap_min``,
        ``elongated_lateral_sep_ratio``.
    """
    if not sources:
        warnings.warn(f'resolve_spine: no sources configured for {state.admin_id}.')
        return state

    thresholds = thresholds or {}
    overlap_iou_max: float = thresholds.get('overlap_iou_max', 0.02)

    resolved = _expand_auto_discover(sources, state)
    if not resolved:
        warnings.warn(f'resolve_spine: no sources resolved for {state.admin_id}.')
        return state

    # Load all sources
    source_gdfs: dict[str, gpd.GeoDataFrame] = {}
    for src in resolved:
        recipe_id = src['recipe_id']
        label = src.get('label', recipe_id)
        try:
            gdf = get_entities(recipe_id, state.admin_id, geom=True)
            source_gdfs[label] = gdf
            if state.verbose:
                print(f'  Load {label}: {len(gdf):,d} footprints')
        except Exception as exc:
            warnings.warn(f'Could not load {recipe_id} for {state.admin_id}: {exc}')

    if not source_gdfs:
        warnings.warn(f'resolve_spine: no footprints loaded for {state.admin_id}.')
        return state

    if state.timer:
        state.timer.mark('Load')

    # Build spine starting from the first source
    first_label = resolved[0].get('label', resolved[0]['recipe_id'])
    spine: gpd.GeoDataFrame = source_gdfs[first_label][['geometry']].copy()
    spine['source'] = first_label

    for src in resolved[1:]:
        label = src.get('label', src['recipe_id'])
        if label not in source_gdfs:
            continue
        candidate = source_gdfs[label]

        overlap = overlay_polygons(
            spine,
            candidate,
            suffixes=('_spine', f'_{label}'),
            iou=True,
        ).sort_values('iou', ascending=False)

        overlap_ids = overlap[
            overlap['iou'].gt(overlap_iou_max)
        ].index.get_level_values(f'{spine.index.name}_{label}')

        to_add = candidate[~candidate.index.isin(overlap_ids)][['geometry']].copy()

        if thresholds.get('elongated_aspect_min'):
            n_before = len(to_add)
            to_add = drop_elongated_duplicates(
                spine,
                to_add,
                aspect_ratio_min=thresholds.get('elongated_aspect_min', 2.5),
                angle_tol_deg=thresholds.get('elongated_angle_tol', 15.0),
                long_overlap_min=thresholds.get('elongated_long_overlap_min', 0.5),
                lateral_sep_ratio_max=thresholds.get(
                    'elongated_lateral_sep_ratio', 2.0
                ),
            )
            n_elong_dropped = n_before - len(to_add)
        else:
            n_elong_dropped = 0

        to_add['source'] = label
        spine = pd.concat([spine, to_add]).sort_index()

        if state.verbose:
            msg = (
                f'  Merge {label}: +{len(to_add):,d} ({len(overlap_ids):,d} overlapping'
            )
            if n_elong_dropped:
                msg += f', {n_elong_dropped:,d} elongated duplicates'
            print(msg + ')')

    if state.timer:
        state.timer.mark('Merge')

    state.spine = spine
    return state


def _expand_auto_discover(
    sources: list[dict],
    state: HarmonizeState,
) -> list[dict]:
    """Replace any ``auto_discover: true`` sentinel with discovered recipes."""
    sentinel_idx = next(
        (i for i, s in enumerate(sources) if s.get('auto_discover')),
        None,
    )
    existing_ids = {s.get('recipe_id') for s in sources if not s.get('auto_discover')}

    recipe = state.recipe
    recipe_admin_str = str(recipe['admin_id'])
    admin_str = str(state.admin_id) if state.admin_id is not None else ''
    entity_obj = recipe.get('entity')
    entity_type = str(entity_obj.entity_type) if entity_obj is not None else ''

    df = find_recipes(entity_type, stage='ingest')
    discovered: list[dict] = []
    for _, row in df.iterrows():
        rid_str = row['admin_id']
        if rid_str and rid_str != recipe_admin_str and admin_str.startswith(rid_str):
            prefix = f'{rid_str}_'
            child_id = (
                f'{prefix}{row["entity_type"]}-{row["source_id"]}-{row["version"]}'
            )
            if child_id not in existing_ids:
                discovered.append({'recipe_id': child_id, 'label': row['source_id']})
                existing_ids.add(child_id)

    if sentinel_idx is not None:
        return sources[:sentinel_idx] + discovered + sources[sentinel_idx + 1 :]
    return sources + discovered


@_register('split_by_reference')
def split_by_reference(
    state: HarmonizeState,
    entity_type: str | None = None,
    recipe_id: str | None = None,
    thresholds: dict | None = None,
) -> HarmonizeState:
    """Split spine geometries at reference polygon boundaries [stub].

    Splits contiguous spine geometries (e.g., large building envelopes that
    span multiple parcels) at reference polygon boundaries to reflect
    differences in age, ownership, or use across the merged footprint.

    Intended use: urban rowhouses / townhouses where a single contiguous
    footprint covers multiple independently owned units that differ in
    renovation history, assessed value, etc.

    Parameters
    ----------
    entity_type : str, optional
        Entity type of the reference dataset (e.g. ``'parcel'``).
    recipe_id : str, optional
        Explicit reference recipe ID (takes precedence over entity_type).
    thresholds : dict, optional
        Step-specific thresholds (e.g. ``min_area_m2``).

    Raises
    ------
    NotImplementedError
        Always — this step is not yet implemented.
    """
    raise NotImplementedError(
        "The 'split_by_reference' harmonization step is not yet implemented. "
        'Remove it from the recipe pipeline or implement it in '
        'openplaces/io/harmonizer/spine.py.'
    )
