"""
Pipeline steps for building and refining the primary entity spine:
  - resolve_spine: merge multiple source GeoDataFrames via IoU dedup
  - union_spine_sources: concatenate multiple source tables for a non-spatial entity
  - link_geographic_ids: assign each row's containing polygon id from one or
    more space-partitioning reference layers (admin units, Census geographies)
  - split_by_reference: split spine geometries at reference boundaries [stub]
"""

from __future__ import annotations

import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.strtree import STRtree

from openplaces.diagnostics import find_recipes
from openplaces.geo.polygon import get_areas, overlay_polygons, points_from_coords
from openplaces.io.harmonizer import (
    HarmonizeState,
    _record_source,
    _register,
    restrict_to_admin_by_name,
)
from openplaces.io.readers import get_admin, get_entities
from openplaces.io.transform import make_index_unique


def get_oriented_dims(geom) -> tuple[float, float, float]:
    """Return ``(angle_deg % 180, length, width)`` of the minimum bounding rectangle.

    A missing or empty geometry returns all zeros rather than raising.
    Blank geometry is a legitimate state here -- `fix_polygons` blanks a
    polygon whose coordinates are non-finite instead of handing it to
    `shapely.make_valid`, which would raise -- and every caller feeds the
    result into an aspect-ratio test. Zero width makes that ratio zero,
    so such a row simply fails the elongation test instead of aborting
    the whole geospine.
    """
    if geom is None or geom.is_empty:
        return 0.0, 0.0, 0.0
    obb = geom.minimum_rotated_rectangle
    if obb is None or obb.is_empty or not hasattr(obb, 'exterior'):
        # A degenerate polygon (a point, or a zero-area sliver) has a
        # rectangle that is not itself a polygon, so it has no corners
        # to measure.
        return 0.0, 0.0, 0.0
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

    Catches displaced thin-rectangle buildings (manufactured homes, trailers) that
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


def _buffer_reject_ids(
    spine: gpd.GeoDataFrame,
    candidate: gpd.GeoDataFrame,
    min_area_m2: float,
    base_m: float,
    area_scale: float,
) -> pd.Index:
    """Candidate ids that fall within a size-scaled keep-out zone of *spine*.

    Only spine footprints at least *min_area_m2* get a zone at all (also the
    main performance lever: it excludes the majority of ordinary footprints
    from ever being buffered). Reprojects once to an equal-area metric CRS for
    correct distance math, uses a coarse (few-vertex) buffer polygon — plenty
    accurate for a keep-out test — and a cheap ``intersects`` spatial join
    rather than exact-overlap/IoU computation.
    """
    if spine.empty or candidate.empty:
        return candidate.index[:0]

    spine_areas_m2 = get_areas(spine, unit='m2')
    protect = spine_areas_m2 >= min_area_m2
    if not protect.any():
        return candidate.index[:0]

    protected = spine.loc[protect, ['geometry']].to_crs('epsg:6933')
    buffer_m = base_m + area_scale * np.sqrt(spine_areas_m2.loc[protect].to_numpy())
    zones = gpd.GeoDataFrame(
        geometry=protected.geometry.buffer(buffer_m, resolution=2),
        crs='epsg:6933',
    )
    cand_m = candidate[['geometry']].to_crs('epsg:6933')
    near = gpd.sjoin(cand_m, zones, predicate='intersects', how='inner')
    return pd.Index(near.index.unique())


@_register('resolve_spine', phase='geometry')
def resolve_spine(
    state: HarmonizeState,
    sources: list[dict] | None = None,
    thresholds: dict | None = None,
    keep_columns: list[str] | None = None,
    track_provenance: list[str] | None = None,
) -> HarmonizeState:
    """Build the primary entity spine from multiple prioritized sources.

    Merges GeoDataFrames from each entry in *sources* in order.  A geometry
    from a lower-priority source is added to the spine only if its IoU with
    every existing spine geometry is below ``overlap_iou_max`` (default 0.02).

    Source entries may include an ``auto_discover: true`` sentinel entry,
    which is replaced at runtime by all ingest recipes of the same entity
    type that are scoped to child admin_ids of the recipe's ``admin_id``,
    excluding any recipe with ``exclude_from_auto_discover: true`` (a
    reference dataset meant to be consumed only via an explicit crosswalk,
    not folded into the canonical spine's geometry).

    Parameters
    ----------
    sources : list of dict
        Ordered source entries.  Each entry is either
        ``{'recipe_id': str, 'label': str}`` or ``{'auto_discover': True}``.
    thresholds : dict, optional
        ``overlap_iou_max`` (float, default 0.02) — maximum IoU to treat two
        footprints as overlapping duplicates.
        ``min_area_m2`` (float, default 0.0) — drop footprints smaller than this
        area (in m²) before IoU deduplication.  0.0 disables the filter.
        ``elongated_aspect_min`` (float) — when set, enables the elongated-duplicate
        filter via :func:`drop_elongated_duplicates`.  Also accepts
        ``elongated_angle_tol``, ``elongated_long_overlap_min``,
        ``elongated_lateral_sep_ratio``.
        ``buffer_base_m`` / ``buffer_area_scale`` (float, default 0.0 — off) — a
        second, independent rejection alongside the IoU test: a lower-priority
        candidate is also dropped if it falls within
        ``buffer_base_m + buffer_area_scale * sqrt(area_m2)`` of an
        already-accepted spine footprint (its own area), so a large building
        gets a bigger keep-out zone than a small one and IoU's insensitivity to
        size mismatch (a sliver clipping a huge building can have a tiny IoU)
        no longer lets duplicates through.  ``buffer_min_area_m2`` (float,
        default 0.0) restricts which spine footprints get a keep-out zone at
        all — set it to the rough size of an ordinary house to protect only
        large/likely-multi-unit buildings (also the main performance lever,
        since it excludes most footprints from ever being buffered).
    track_provenance : list of str, optional
        Subset of *keep_columns* to seed per-cell source provenance for: a
        ``{column}_source`` sidecar recording which resolved source's own
        row supplied that cell's value (its ``label``/``source_id``), via
        :func:`openplaces.io.harmonizer._record_source`. Unlike
        :func:`~openplaces.io.harmonizer.links.link_by_id`'s same-named
        parameter (which records provenance for values it joins later),
        this seeds provenance for the values *this* function writes
        directly here, so a later ``link_by_id(auto_discover=True,
        track_provenance=[...])`` step covering the same columns has a
        correct baseline to build on rather than a blank sidecar.
    """
    if not sources:
        warnings.warn(f'resolve_spine: no sources configured for {state.admin_id}.')
        return state

    thresholds = thresholds or {}
    overlap_iou_max: float = thresholds.get('overlap_iou_max', 0.02)
    buffer_min_area_m2: float = thresholds.get('buffer_min_area_m2', 0.0)
    buffer_base_m: float = thresholds.get('buffer_base_m', 0.0)
    buffer_area_scale: float = thresholds.get('buffer_area_scale', 0.0)
    apply_buffer = buffer_base_m > 0 or buffer_area_scale > 0

    resolved = _expand_auto_discover(sources, state)
    if not resolved:
        warnings.warn(f'resolve_spine: no sources resolved for {state.admin_id}.')
        return state

    # Load all sources, keyed by recipe_id: `label` (default the source_id)
    # is not guaranteed unique -- two recipes of the same entity_type can
    # share a source_id, e.g. a geometry recipe and a same-source
    # attribute-only roll distinguished only by version. Keying by label
    # would let the second silently alias the first in `source_gdfs`.
    source_gdfs: dict[str, gpd.GeoDataFrame] = {}
    loaded: list[dict] = []
    for src in resolved:
        recipe_id = src['recipe_id']
        label = src.get('label', recipe_id)
        try:
            gdf = get_entities(recipe_id, state.admin_id, geom=True)
            source_gdfs[recipe_id] = gdf
            loaded.append(src)
            if state.verbose:
                print(f'  Load {label}: {len(gdf):,d} footprints')
        except Exception as exc:
            warnings.warn(f'Could not load {recipe_id} for {state.admin_id}: {exc}')

    if not source_gdfs:
        warnings.warn(f'resolve_spine: no footprints loaded for {state.admin_id}.')
        return state

    if state.timer:
        state.timer.mark('Load')

    min_area_m2: float = thresholds.get('min_area_m2', 0.0)
    if min_area_m2 > 0:
        for src in loaded:
            recipe_id = src['recipe_id']
            label = src.get('label', recipe_id)
            areas = get_areas(source_gdfs[recipe_id], unit='m2')
            before = len(source_gdfs[recipe_id])
            source_gdfs[recipe_id] = source_gdfs[recipe_id][areas >= min_area_m2].copy()
            dropped = before - len(source_gdfs[recipe_id])
            if dropped and state.verbose:
                print(
                    f'  Filter {label}: dropped {dropped:,d} footprints < '
                    f'{min_area_m2} m²'
                )

    # Build spine starting from the first source. ``keep_columns`` carries
    # source attributes (e.g. parcel_id_local) onto the spine for non-footprint
    # entities; absent columns are ignored.
    keep_columns = keep_columns or []
    track_cols = set(track_provenance or []) & set(keep_columns)

    def _spine_cols(gdf):
        return ['geometry'] + [c for c in keep_columns if c in gdf.columns]

    # loaded[0] is the highest-priority source by admin specificity that
    # actually loaded (not necessarily resolved[0] -- e.g. Craven County's
    # parcel roll is a geometry-less attribute table by design, its own
    # property recipe carries the geometry instead), keyed by recipe_id
    # since `label` (default the source_id) is not guaranteed unique.
    first_recipe_id = loaded[0]['recipe_id']
    first_label = loaded[0].get('label', first_recipe_id)
    spine: gpd.GeoDataFrame = source_gdfs[first_recipe_id][
        _spine_cols(source_gdfs[first_recipe_id])
    ].copy()
    spine['geometry_source'] = first_label
    for col in track_cols & set(spine.columns):
        _record_source(spine, col, spine[col].notna(), first_label)

    for src in loaded[1:]:
        recipe_id = src['recipe_id']
        label = src.get('label', recipe_id)
        candidate = source_gdfs[recipe_id]

        overlap = overlay_polygons(
            spine,
            candidate,
            suffixes=('_spine', f'_{label}'),
            iou=True,
        ).sort_values('iou', ascending=False)

        overlap_ids = overlap[
            overlap['iou'].gt(overlap_iou_max)
        ].index.get_level_values(1)

        if apply_buffer:
            near_ids = _buffer_reject_ids(
                spine, candidate, buffer_min_area_m2, buffer_base_m, buffer_area_scale
            )
        else:
            near_ids = candidate.index[:0]

        to_add = candidate[
            ~candidate.index.isin(overlap_ids) & ~candidate.index.isin(near_ids)
        ][_spine_cols(candidate)].copy()

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

        to_add['geometry_source'] = label
        for col in track_cols & set(to_add.columns):
            _record_source(to_add, col, to_add[col].notna(), label)
        if candidate.index.name != spine.index.name:
            warnings.warn(
                f'resolve_spine: index name mismatch — spine has '
                f'{spine.index.name!r} but {label!r} has {candidate.index.name!r}. '
                f'Data may need to be re-ingested.'
            )
            to_add.index.name = spine.index.name
        spine = pd.concat([spine, to_add]).sort_index()

        if state.verbose:
            msg = (
                f'  Merge {label}: +{len(to_add):,d} ({len(overlap_ids):,d} overlapping'
            )
            if apply_buffer:
                msg += f', {len(near_ids):,d} near-buffer'
            if n_elong_dropped:
                msg += f', {n_elong_dropped:,d} elongated duplicates'
            print(msg + ')')

    if state.timer:
        state.timer.mark('Merge')

    # Record which ingest recipes already contributed geometry, and which
    # columns came straight from each one's own row (keep_columns), so a
    # later `link_by_id(auto_discover=True)` can avoid re-deriving those
    # same columns by re-joining a source onto itself via its (non-unique)
    # local join key -- see that function's auto_discover branch for why.
    # Other columns (e.g. improvement_value) that keep_columns did not carry
    # still need that join, self-join or not.
    state.metadata['spine_source_recipe_ids'] = {s['recipe_id'] for s in resolved}
    state.metadata['spine_keep_columns'] = set(keep_columns)

    # A source can ship rows whose geometries hash to the same geo_id
    # (seen: 6 duplicated parcel ids in Fort Bend, TX), and a duplicated
    # spine index breaks every downstream index-aligned operation in turn.
    # Uniquify here, at the spine's birth, with the same helper and
    # semantics infer_spine_additions already applies to its own additions.
    if spine.index.duplicated().any():
        n_dup = int(spine.index.duplicated().sum())
        if all(isinstance(value, str) for value in spine.index):
            warnings.warn(
                f'{n_dup} duplicate spine IDs for {state.admin_id}; making unique.'
            )
            spine = make_index_unique(spine, sort_duplicates_by_area=True)
        else:
            # make_index_unique suffixes string ids only. A non-string
            # index never occurs on a real spine (ids are geo_id/OLC
            # strings); leave it, with the warning, rather than failing.
            warnings.warn(
                f'{n_dup} duplicate spine IDs for {state.admin_id}; index is '
                'not string-typed, leaving duplicates in place.'
            )

    state.spine = spine

    # Spatial-overlay links name the reference-id index level 'parcel_id'. When
    # the spine is itself a parcel entity its index is also 'parcel_id', so the
    # two collide in the overlay MultiIndex (and every consumer that groups by
    # the reference level). Give the spine a distinct working index name for the
    # rest of the pipeline; the harmonizer's save step restores the original.
    if state.spine.index.name == 'parcel_id':
        state.metadata['spine_index_name'] = state.spine.index.name
        state.spine.index = state.spine.index.rename('spine_id')

    return state


@_register('union_spine_sources', phase='geometry')
def union_spine_sources(
    state: HarmonizeState,
    sources: list[dict] | None = None,
) -> HarmonizeState:
    """Build a non-spatial primary entity spine by concatenating sources.

    Non-spatial peer of :func:`resolve_spine`, for entities with no geometry
    (e.g. a transaction spine). ``resolve_spine``'s IoU-overlap dedup exists
    because multiple sources genuinely *can* describe the same physical
    parcel; sources for an entity like ``transaction`` are always disjoint by
    admin scope instead (a sale recorded in one state's feed can't also
    appear in another's), so this is a straightforward concatenation, not a
    priority-merge -- no dedup logic beyond what ``pd.concat`` needs.

    Parameters
    ----------
    sources : list of dict
        Ordered source entries, each either ``{'recipe_id': str, 'label':
        str}`` or the ``{'auto_discover': True, 'entity_type': str}``
        sentinel (see :func:`resolve_spine`).
    """
    if not sources:
        warnings.warn(
            f'union_spine_sources: no sources configured for {state.admin_id}.'
        )
        return state

    resolved = _expand_auto_discover(sources, state)
    if not resolved:
        warnings.warn(f'union_spine_sources: no sources resolved for {state.admin_id}.')
        return state

    parts = []
    loaded_recipe_ids = set()
    for src in resolved:
        recipe_id = src['recipe_id']
        label = src.get('label', recipe_id)
        layer = src.get('layer')
        try:
            # missing='ignore': a source scoped finer than state.admin_id
            # (e.g. a per-town transaction crawl) may only have partial
            # coverage -- some child units genuinely not yet ingested rather
            # than an error -- so load whatever's available instead of
            # failing the whole admin unit. layer selects a bundled
            # additional_layers table (e.g. 'property') on a recipe whose
            # own primary entity is a different type.
            df = get_entities(recipe_id, state.admin_id, layer=layer, missing='ignore')
        except Exception as exc:
            warnings.warn(f'Could not load {recipe_id} for {state.admin_id}: {exc}')
            continue
        df = restrict_to_admin_by_name(df, recipe_id, state.admin_id)
        if df.empty:
            continue
        df = df.copy()
        df['source'] = label
        parts.append(df)
        loaded_recipe_ids.add(recipe_id)
        if state.verbose:
            print(f'  Load {label}: {len(df):,d} rows')

    if not parts:
        warnings.warn(f'union_spine_sources: no rows loaded for {state.admin_id}.')
        return state

    if state.timer:
        state.timer.mark('Load')

    spine = pd.concat(parts, ignore_index=True, sort=False)
    state.metadata['spine_source_recipe_ids'] = loaded_recipe_ids
    state.spine = spine
    if state.timer:
        state.timer.mark('Union')
    if state.verbose:
        print(f'  union_spine_sources: {len(spine):,d} total rows')
    return state


def _expand_auto_discover(
    sources: list[dict],
    state: HarmonizeState,
) -> list[dict]:
    """Replace any ``auto_discover: true`` sentinel with discovered recipes.

    Standalone ingest recipes of *entity_type* are found via ``find_recipes``
    as before, ordered most-specific-admin-id first (newest version breaking
    a tie within the same specificity) -- ``resolve_spine`` treats
    ``sources[0]`` as primary for an IoU-tie, so a county-scoped recipe
    should out-prioritize a statewide one covering the same admin unit,
    mirroring :func:`openplaces.io.harmonizer.discover._best_recipe_for`'s
    single-winner precedent. Additionally, entries bundled as an
    ``additional_layers`` entry inside a *different* host entity's recipe
    (e.g. MassGIS's ``property`` table, bundled inside its ``parcel`` recipe
    rather than registered on its own) are found via
    :func:`openplaces.recipe.find_additional_layer_recipes` -- necessary for
    entity types like ``property`` that, today, have no standalone ingest
    recipe anywhere and would otherwise never resolve. These are appended
    after the standalone recipes, in their own (unordered) discovery order.
    """
    from openplaces.recipe import find_additional_layer_recipes

    sentinel_idx = next(
        (i for i, s in enumerate(sources) if s.get('auto_discover')),
        None,
    )
    existing_ids = {s.get('recipe_id') for s in sources if not s.get('auto_discover')}

    recipe = state.recipe
    recipe_admin_str = str(recipe['admin_id'])
    admin_str = str(state.admin_id) if state.admin_id is not None else ''
    entity_obj = recipe.get('entity')

    sentinel = next((s for s in sources if s.get('auto_discover')), {})
    sentinel_entity_type = sentinel.get('entity_type')
    entity_type = sentinel_entity_type or (
        str(entity_obj.entity_type) if entity_obj is not None else ''
    )

    df = find_recipes(entity_type, stage='ingest')
    ranked: list[tuple[int, str, dict]] = []
    for _, row in df.iterrows():
        if row['exclude_from_auto_discover']:
            continue
        rid_str = row['admin_id']
        if rid_str and rid_str != recipe_admin_str and admin_str.startswith(rid_str):
            prefix = f'{rid_str}_'
            child_id = (
                f'{prefix}{row["entity_type"]}-{row["source_id"]}-{row["version"]}'
            )
            if child_id not in existing_ids:
                specificity = rid_str.count('-') + 1
                ranked.append(
                    (
                        specificity,
                        str(row['version']),
                        {'recipe_id': child_id, 'label': row['source_id']},
                    )
                )
                existing_ids.add(child_id)
    ranked.sort(key=lambda t: (t[0], t[1]), reverse=True)
    discovered: list[dict] = [entry for _specificity, _version, entry in ranked]

    if state.admin_id is not None:
        for match in find_additional_layer_recipes(entity_type, state.admin_id):
            key = (match['recipe_id'], match['layer'])
            if key in existing_ids:
                continue
            discovered.append(
                {
                    'recipe_id': match['recipe_id'],
                    'label': match['label'],
                    'layer': match['layer'],
                }
            )
            existing_ids.add(key)

    if sentinel_idx is not None:
        return sources[:sentinel_idx] + discovered + sources[sentinel_idx + 1 :]
    return sources + discovered


@_register('derive_geometry_attributes', phase='geometry')
def derive_geometry_attributes(
    state: HarmonizeState, area_unit: str = 'ha'
) -> HarmonizeState:
    """Compute this entity's own centroid lat/long and area, once.

    Runs immediately after the spine's geometry is finalized (right after
    resolve_spine, including any synthetic fallback rows added by
    infer_spine_additions), so downstream harmonize and curate steps reuse
    the lat/long/area_{area_unit} columns instead of recomputing them. Thin
    wrapper around geo.polygon.add_geometry_derivatives -- the same
    function ingest recipes opt into via add_geometry_derivatives: true --
    so both entry points share one implementation.

    Parameters
    ----------
    area_unit : str, optional
        Area unit for the output area_{area_unit} column (default 'ha').
    """
    from openplaces.core.schema import is_synthetic_geometry
    from openplaces.geo.polygon import add_geometry_derivatives

    if state.spine is None:
        return state
    spine = state.spine
    area_mask = ~is_synthetic_geometry(spine, state.recipe.get('entity'))
    state.spine = add_geometry_derivatives(
        spine, state.timer, area_unit=area_unit, area_mask=area_mask
    )
    return state


def _resolve_geographic_reference(state, link):
    """Load the reference layer for one `link_geographic_ids` entry.

    Returns a two-column GeoDataFrame (``geometry``, ``_ref_value``) or
    ``None`` when the reference has no coverage for this admin unit.
    """
    admin_level = link.get('admin_level')
    recipe_id = link.get('recipe_id')
    if (admin_level is None) == (recipe_id is None):
        raise ValueError(
            "link_geographic_ids: each link needs exactly one of 'admin_level' "
            "or 'recipe_id' "
            f'(output_column={link.get("output_column")!r}).'
        )
    id_column = link.get('id_column')

    if admin_level is not None:
        ref = get_admin(state.admin_id, admin_level, geom=True)
    else:
        bbox = tuple(state.spine.total_bounds) if state.spine is not None else None
        ref = get_entities(
            recipe_id, state.admin_id, geom=True, bbox=bbox, missing='warn'
        )
    if ref is None or len(ref) == 0:
        return None

    values = ref[id_column] if id_column else pd.Series(ref.index, index=ref.index)
    out = ref[['geometry']].copy()
    out['_ref_value'] = values.to_numpy()
    return out


def _join_largest_overlap(
    spine_subset: gpd.GeoDataFrame, ref: gpd.GeoDataFrame
) -> pd.Series:
    """Pass-2 fallback: the reference polygon with the largest intersection area.

    Used only for rows whose centroid missed every reference polygon in Pass
    1 (typically a boundary/rounding sliver) -- an identity overlay of the
    row's *actual* geometry, keeping the dominant (largest-area) match.
    """
    if spine_subset.empty:
        return pd.Series(pd.NA, index=spine_subset.index, dtype=object)

    spine_id_name = spine_subset.index.name or 'index'
    overlay = overlay_polygons(
        spine_subset[['geometry']],
        ref[['geometry', '_ref_value']],
        columns=['_ref_value'],
        how='identity',
        area_intersection=True,
        suffixes=('_spine', '_ref'),
    )
    overlay = overlay[overlay['_ref_value'].notna()]
    if overlay.empty:
        return pd.Series(pd.NA, index=spine_subset.index, dtype=object)

    overlay = overlay.sort_values('area_intersection_m2', ascending=False)
    overlay = overlay[~overlay.index.get_level_values(spine_id_name).duplicated()]
    return pd.Series(
        overlay['_ref_value'].to_numpy(),
        index=overlay.index.get_level_values(spine_id_name),
    ).reindex(spine_subset.index)


def _join_containing_polygon(
    spine_subset: gpd.GeoDataFrame, ref: gpd.GeoDataFrame
) -> pd.Series:
    """Two-pass containing-polygon join for *spine_subset* against *ref*.

    Pass 1: point-in-polygon on each row's own centroid (``lat``/``long``).
    Pass 2 (only for Pass-1 misses): dominant-overlap fallback, see
    :func:`_join_largest_overlap`.
    """
    if spine_subset.empty:
        return pd.Series(pd.NA, index=spine_subset.index, dtype=object)

    points = points_from_coords(spine_subset[['lat', 'long']].copy(), x='long', y='lat')
    joined = gpd.sjoin(
        points, ref[['geometry', '_ref_value']], how='left', predicate='within'
    )
    joined = joined[~joined.index.duplicated()]
    result = joined['_ref_value'].reindex(spine_subset.index)

    missing = result.index[result.isna()]
    if len(missing):
        result.update(_join_largest_overlap(spine_subset.loc[missing], ref))
    return result


def _pick_unanimous(series: pd.Series):
    """Return the single distinct non-null value in *series*, else ``pd.NA``."""
    values = pd.unique(series.dropna())
    return values[0] if len(values) == 1 else pd.NA


def _inherit_geographic_ids(
    state: HarmonizeState,
    output_columns: list[str],
    inherit_from: dict,
) -> tuple[dict[str, pd.Series], list[dict]]:
    """Roll up already-computed geographic ids from a linked recipe's output.

    Groups the linked recipe's rows by ``inherit_from['group_by']`` (default
    ``'parcel_id'``) when that column is present on the linked output, and
    joins the result onto this spine by value (the shared id's underlying
    values, regardless of this spine's own current index *name* -- see
    ``resolve_spine``'s ``parcel_id`` -> ``spine_id`` rename). Falls back to
    spatial containment of the linked entity's centroid within this spine's
    geometry when no shared column is present, mirroring
    :func:`~openplaces.io.harmonizer.attributes.summarize_footprint_morphology`'s
    identical id-or-spatial fallback for this same footprint/parcel
    relationship.

    A group's value is only inherited where every row in the group agrees
    (or there is exactly one); disagreeing groups are left null so the
    caller's own direct join resolves them instead.

    Returns
    -------
    tuple of (dict[str, pandas.Series], list of dict)
        Per-column inherited values (aligned to ``state.spine.index``, null
        where not inherited), and a list of disagreement records
        (``{'output_column', 'group_key', 'values'}``) for diagnostics.
    """
    recipe_id = inherit_from['recipe_id']
    group_by = inherit_from.get('group_by', 'parcel_id')
    spine = state.spine

    linked = get_entities(recipe_id, state.admin_id, geom=True, missing='ignore')
    empty = {
        col: pd.Series(pd.NA, index=spine.index, dtype=object) for col in output_columns
    }
    if linked is None or len(linked) == 0:
        return empty, []

    available = [c for c in output_columns if c in linked.columns]
    if not available:
        return empty, []

    if group_by in linked.columns:
        grouped = linked.groupby(group_by)
    elif {'lat', 'long'}.issubset(linked.columns):
        points = points_from_coords(
            linked[['lat', 'long'] + available].copy(), x='long', y='lat'
        )
        joined = gpd.sjoin(points, spine[['geometry']], how='inner', predicate='within')
        spine_id_name = spine.index.name or 'index'
        grouped = joined.groupby(spine_id_name)
    else:
        return empty, []

    # `grouped` stays column-unselected so each `grouped[col]` below is a
    # fresh sub-selection -- selecting a column from an *already*
    # column-selected DataFrameGroupBy raises in pandas.
    rollup = grouped[available].agg(_pick_unanimous)

    conflicts: list[dict] = []
    for col in available:
        counts = grouped[col].nunique(dropna=True)
        for key in counts.index[counts > 1]:
            values = pd.unique(grouped.get_group(key)[col].dropna())
            conflicts.append(
                {'output_column': col, 'group_key': key, 'values': list(values)}
            )

    inherited = {
        col: (
            rollup[col].reindex(spine.index)
            if col in rollup.columns
            else pd.Series(pd.NA, index=spine.index, dtype=object)
        )
        for col in output_columns
    }
    return inherited, conflicts


@_register('link_geographic_ids', phase='geometry')
def link_geographic_ids(
    state: HarmonizeState,
    links: list[dict] | None = None,
    inherit_from: dict | None = None,
) -> HarmonizeState:
    """Assign each spine row the id of the reference polygon containing it.

    For each entry in *links*, joins against a reference layer that
    partitions space without overlaps (an admin level, or a tile-type
    reference entity such as a Census statistical geography) and writes the
    matching reference id as a new/overwritten spine column.

    Pass 1 is a point-in-polygon join on each row's own centroid (``lat``/
    ``long``, from :func:`derive_geometry_attributes`, which must run
    first). Any row whose centroid misses every reference polygon falls
    back to Pass 2: an identity overlay of the row's *actual* geometry
    against the reference layer, keeping the reference polygon with the
    largest intersection area. Rows unmatched by both passes are left null
    -- a visible coverage gap, not a masked one.

    Parameters
    ----------
    links : list of dict
        Each entry configures one output column:

        ``output_column`` (str, required)
            Name for the new/overwritten spine column.
        ``admin_level`` (int, optional)
            Join against ``get_admin(state.admin_id, admin_level, geom=True)``;
            the joined value is that reference's own index. Mutually
            exclusive with ``recipe_id``.
        ``recipe_id`` (str, optional)
            Join against ``get_entities(recipe_id, state.admin_id, geom=True,
            bbox=spine.total_bounds)``. Mutually exclusive with
            ``admin_level``.
        ``id_column`` (str, optional)
            Reference column to copy as the output value; defaults to the
            reference's own index.
    inherit_from : dict, optional
        ``{'recipe_id': str, 'group_by': str}`` (``group_by`` defaults to
        ``'parcel_id'``). When set, every output column is first filled by
        rolling up *recipe_id*'s already-harmonized output via
        :func:`_inherit_geographic_ids`; only rows left null run the direct
        two-pass join above. Avoids recomputing the same join twice for a
        parcel spine whose footprint spine already resolved it (see
        ``US_parcel-spine-2026``'s wiring).

    If an output column already exists on the spine with non-null values
    (e.g. a source-native ``admin4_id`` carried through ``resolve_spine``'s
    ``keep_columns``), rows where the pre-existing value disagrees with the
    freshly computed one are reported via ``warnings.warn`` (a bounded
    sample) before the column is overwritten.
    """
    if state.spine is None or not links:
        return state
    spine = state.spine
    if not {'lat', 'long'}.issubset(spine.columns):
        warnings.warn(
            'link_geographic_ids: spine has no lat/long columns -- run '
            'derive_geometry_attributes first. Skipping.'
        )
        return state

    output_columns = [link['output_column'] for link in links]
    inherited: dict[str, pd.Series] = {}
    inherit_conflicts: list[dict] = []
    if inherit_from:
        inherited, inherit_conflicts = _inherit_geographic_ids(
            state, output_columns, inherit_from
        )
        if inherit_conflicts:
            state.metadata.setdefault('geographic_id_inheritance_conflicts', []).extend(
                inherit_conflicts
            )

    for link in links:
        output_column = link['output_column']
        inherited_col = inherited.get(output_column)
        computed = (
            inherited_col.copy()
            if inherited_col is not None
            else pd.Series(pd.NA, index=spine.index, dtype=object)
        )

        # Only resolves (loads) the reference layer if inheritance left a
        # residual to fill -- the whole point of `inherit_from` is to avoid
        # this load/join entirely when it isn't needed.
        missing = computed.index[computed.isna()]
        if len(missing):
            ref = _resolve_geographic_reference(state, link)
            if ref is None:
                if state.verbose:
                    print(
                        f'  link_geographic_ids: no reference for '
                        f"'{output_column}'; leaving {len(missing):,} row(s) null."
                    )
            else:
                computed.update(_join_containing_polygon(spine.loc[missing], ref))

        if output_column in spine.columns:
            existing = spine[output_column]
            disagree = existing.notna() & computed.notna() & existing.ne(computed)
            if disagree.any():
                sample = pd.DataFrame(
                    {'existing': existing[disagree], 'computed': computed[disagree]}
                ).head(10)
                warnings.warn(
                    f'link_geographic_ids: {int(disagree.sum()):,} row(s) disagree '
                    f"on '{output_column}' between the existing value and the "
                    f'freshly computed spatial join; overwriting with the '
                    f'spatial join. Examples:\n{sample}'
                )

        spine[output_column] = computed
        if state.verbose:
            n = int(computed.notna().sum())
            print(
                f'  link_geographic_ids: {output_column} resolved '
                f'{n:,} of {len(spine):,}.'
            )

    from openplaces.io.harmonizer.diagnostics import (
        save_geographic_id_inheritance_conflicts,
    )

    save_geographic_id_inheritance_conflicts(state)

    state.spine = spine
    return state


@_register('split_by_reference', phase='geometry')
def split_by_reference(
    state: HarmonizeState,
    entity_type: str | None = None,
    recipe_id: str | None = None,
    thresholds: dict | None = None,
) -> HarmonizeState:
    """Split spine geometries at reference polygon boundaries [stub].

    Splits contiguous spine geometries (e.g., large building footprints that
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
