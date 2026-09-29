"""Load a reference and build the spine-to-reference crosswalk by
overlay, then persist it as the link sidecar (link_to_reference).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from openplaces.core.attribute_registry import (
    load_registry,
)
from openplaces.core.schema import (
    AdminId,
    SourceGeometryType,
    admin_scope_covers,
)
from openplaces.diagnostics import find_recipes
from openplaces.geo.ids import (
    get_geo_ids,
)
from openplaces.geo.link import get_entity_link_path
from openplaces.geo.polygon import (
    get_areas,
    overlay_polygons,
)
from openplaces.io.aggregate import aggregate_rows
from openplaces.io.harmonizer import (
    HarmonizeState,
    _register,
)
from openplaces.io.harmonizer.links._shared import (
    _CROSSWALK_COLS,
)
from openplaces.io.harmonizer.links.points import (
    _link_spatial_point,
)
from openplaces.io.harmonizer.links.sidecars import (
    _link_fingerprint,
    _load_link_sidecar,
    _write_link_sidecar,
)
from openplaces.io.readers import get_entities
from openplaces.recipe import (
    get_output_path,
    get_recipe_id,
    raise_if_coverage_complete,
)
from openplaces.table import require_unique_index


@_register('link_to_reference', phase='geometry')
def link_to_reference(
    state: HarmonizeState,
    join: str = 'spatial_overlay',
    entity_type: str | None = None,
    recipe_id: str | None = None,
    thresholds: dict | None = None,
    remap_id: str | None = None,
    source_geometry_type: str | None = None,
    aggregation_function=None,
    sort_by: str | None = None,
    list_columns: list[str] | None = None,
    save_link: bool = True,
) -> HarmonizeState:
    """Load a reference dataset and build a spine ↔ reference crosswalk.

    Populates ``state.references[recipe_id]``,
    ``state.crosswalks[recipe_id]``, ``state.overlays[recipe_id]`` (for
    ``spatial_overlay`` joins), ``state.reference_types[recipe_id]``, and
    ``state.source_geometry_types[recipe_id]`` (when *source_geometry_type*
    is provided).

    Parameters
    ----------
    join : str
        How to join the reference to the spine:

        ``'spatial_overlay'``
            Polygon-on-polygon identity overlay.  Produces a crosswalk table
            with IoU and area-intersection columns.  Populates
            ``state.overlays[recipe_id]`` with the full geometry-bearing
            overlay result for use by later steps.
        ``'spatial_point'``
            Point-in-polygon sjoin.  Joins reference points to the spine
            entities, and unlinked points to any polygon reference already
            in ``state.references`` matching the reference's entity type.

    entity_type : str, optional
        Auto-discover the best ingest recipe of this entity type for the
        current ``admin_id``.  Ignored when ``recipe_id`` is given.
    recipe_id : str, optional
        Explicit reference recipe ID.  Takes precedence over ``entity_type``.
    thresholds : dict, optional
        For ``spatial_overlay``:
        ``min_fraction_of_largest`` (float, default 1/6) — minimum fraction
        of the largest spine-reference intersection to keep a secondary link.
        ``area_intersection_m2_min`` (float, default 10) — minimum intersection
        area in m² to keep a link.
        For ``spatial_point``:
        ``proximity_m`` (float, default 10) — radius for inner proximity pass.
        ``far_proximity_m`` (float, default 100) — radius for outer proximity
        pass (same-parcel constraint applied).  Set to 0 to disable.
    remap_id : str, optional
        Recipe ID for a column-remap table applied to the reference after
        loading (see :func:`openplaces.io.transform.remap`).
    source_geometry_type : str, optional
        :class:`~openplaces.core.schema.SourceGeometryType` value describing
        what this source represents spatially (e.g. ``'single_building_point'``).
        Stored in ``state.source_geometry_types`` for use by downstream steps
        such as ``classify_footprint_priority``.
    aggregation_function : None, callable, or dict, optional
        Controls how duplicate ``geo_id`` rows in the reference are reduced to
        one row before joining.  ``None`` (default) applies the aggregation
        function from the attribute registry.  A dict maps column names to
        callables; columns absent from the dict fall back to the registry
        default.  Only used for ``spatial_overlay`` joins.
    sort_by : str, optional
        Column to sort reference rows by descending before aggregation.
        Falls back to geometry area when the column is absent and the
        reference is a GeoDataFrame.  Only used for ``spatial_overlay`` joins.
    list_columns : list of str, optional
        Column names for which an extra ``{col}_list`` column is added to the
        aggregated reference, collecting all values per ``geo_id`` into a list.
        Normal scalar aggregation for each column still applies alongside.
        Only used for ``spatial_overlay`` joins.
    save_link : bool, optional
        Persist the join product as a sidecar parquet at the canonical
        entity-link path (default True; every link product is a
        first-class table of the normalized store, so a recipe opts
        *out*, not in). For ``spatial_overlay``: the full many-to-many
        identity overlay (geometry-free, every spine-reference pair
        including sub-threshold slivers, with the crosswalk's link label
        joined on). For ``spatial_point``: the final flat crosswalk (all
        reference columns plus the matched spine id, pass provenance,
        and duplicate flags), geometry-free. On later runs the sidecar
        is reloaded instead of recomputing the join — the overlay is the
        single most expensive harmonize step — iff its footer
        fingerprint (step config, the configs of every prior
        geometry-phase pipeline step, plus size/mtime of the ingest
        inputs) still matches; a deleted input with a tombstone receipt
        stays verifiable. After an overlay reload,
        ``state.overlays[recipe_id]`` carries no geometry column (only
        the area/IoU columns are consumed downstream).
    """
    if state.spine is None:
        warnings.warn('link_to_reference: spine is None; skipping.')
        return state

    resolved_recipe_id, resolved_entity_type = _resolve_reference_recipe(
        recipe_id, entity_type, state.admin_id
    )
    if resolved_recipe_id is None:
        if state.verbose:
            print(
                f'  Link ({join}): no reference recipe found for '
                f'entity_type={entity_type!r} and {state.admin_id}. '
                'Skipping.'
            )
        return state

    if source_geometry_type is not None:
        state.source_geometry_types[resolved_recipe_id] = SourceGeometryType(
            source_geometry_type
        )

    if join == 'spatial_overlay':
        return _link_spatial_overlay(
            state,
            resolved_recipe_id,
            resolved_entity_type,
            thresholds or {},
            aggregation_function,
            sort_by,
            list_columns,
            save_link,
        )
    elif join == 'spatial_point':
        return _link_spatial_point(
            state,
            resolved_recipe_id,
            resolved_entity_type,
            remap_id,
            thresholds or {},
            save_link,
        )
    else:
        raise ValueError(
            f"Unknown join mode: '{join}'. "
            "Expected 'spatial_overlay' or 'spatial_point'."
        )


def _resolve_reference_recipe(
    recipe_id: str | None,
    entity_type: str | None,
    admin_id: AdminId | None,
) -> tuple[str | None, str | None]:
    """Return (resolved_recipe_id, entity_type) for a reference step."""
    if recipe_id is not None:
        derived_type = entity_type
        if derived_type is None:
            base = recipe_id.split('_')[-1]
            derived_type = base.split('-')[0]
        return recipe_id, derived_type
    if entity_type is not None and admin_id is not None:
        found_id = _find_reference_recipe(entity_type, admin_id)
        return found_id, entity_type
    return None, None


def _find_reference_recipe(entity_type: str, admin_id: AdminId) -> str | None:
    """Auto-discover the best ingest recipe for *entity_type* and *admin_id*.

    Scans all stage=``'ingest'`` recipes of the given entity type and returns
    the recipe_id of the one whose most specific ``admin_id`` is a parent
    of (or equal to) *admin_id*.
    """
    df = find_recipes(entity_type, stage='ingest')
    if df.empty:
        return None
    admin_str = str(admin_id)
    candidates = []
    for _, row in df.iterrows():
        rid_str = row['admin_id']
        # Containment by level, not by string prefix: a recipe scoped to
        # the pre-2026 'US-NC-WA' (Wake) is not a parent of 'US-NC-WAR'
        # (Warren). The id is read, not rebuilt, so a recipe carrying a
        # filename suffix keeps it (CO_parcel-igac-2026_rural).
        if admin_scope_covers(rid_str, admin_str):
            level = rid_str.count('-') + 1 if rid_str else 0
            recipe_id = row['recipe_id']

            # Since this reference recipe is auto-discovered for a spatial join,
            # we check if it has geometry. If it was ingested, a companion
            # `_geo.parquet` file will exist.
            has_geo = False
            try:
                path = get_output_path(recipe_id, admin_id)
                geo_path = path.with_stem(path.stem + '_geo')
                if geo_path.exists():
                    has_geo = True
            except Exception:
                pass
            candidates.append((level, has_geo, recipe_id))

    if not candidates:
        return None

    # Prefer candidates that have geometry. If none do (e.g. before ingestion),
    # fall back to all candidates.
    geo_candidates = [c for c in candidates if c[1]]
    if geo_candidates:
        best_candidate = max(geo_candidates, key=lambda x: x[0])
    else:
        best_candidate = max(candidates, key=lambda x: x[0])

    return best_candidate[2]


def _prepare_reference(
    ref_raw,
    recipe_id: str,
    entity_type: str | None,
    state: HarmonizeState,
    aggregation_function=None,
    sort_by: str | None = None,
    list_columns: list[str] | None = None,
):
    """Prepare a raw polygon reference for overlaying or attribution.

    Shared by :func:`_link_spatial_overlay` and the geospine loader step,
    so an attribute-only recipe reloading a persisted overlay reproduces
    exactly the reference table the overlay was computed against: geo_id
    deduplication, numeric coercion, combined land-use labels, areas,
    per-geo_id aggregation with collision renames, and the derived
    improvement_value_per_ha.

    Returns the aggregated reference indexed by ``parcel_id`` (the geo_id
    under the crosswalk's reference-level name).
    """
    ref = ref_raw.copy()

    # A source can carry a handful of degenerate non-polygon geometries
    # (digitizing artifacts, e.g. a 2-point LineString parcel boundary, or a
    # null geometry) that would otherwise crash geopandas.overlay's
    # mixed-geometry-type check for the entire admin unit. The consumer
    # is a polygon-on-polygon identity overlay, so drop them here instead.
    valid_polygon = ref.geometry.notna() & ref.geometry.geom_type.isin(
        ('Polygon', 'MultiPolygon')
    )
    if (~valid_polygon).any():
        if state.verbose:
            print(
                f'  Link (overlay): dropped {(~valid_polygon).sum()} '
                f'non-polygon/null {entity_type or "ref"} geometries'
            )
        ref = ref[valid_polygon]

    # geo_id is generated at ingest only for parcels; footprint/building
    # references (e.g. FEMA) arrive without it, so derive the same stable
    # geometry-hash id here to dedup identical geometries below.
    if 'geo_id' not in ref.columns:
        ref['geo_id'] = get_geo_ids(ref, handle_duplicates=False)

    registry = load_registry()
    numeric_attrs = set(registry.index[registry['data_type'].isin(['float', 'int'])])
    for numeric_col in numeric_attrs:
        if numeric_col in ref.columns:
            ref[numeric_col] = pd.to_numeric(ref[numeric_col], errors='coerce')

    # Build the combined group label for whichever land-use vocabulary the
    # reference carries: parcels use use_group ("what it is used for"),
    # buildings/footprints use purpose_group ("what it was built for").
    for _base in ('use_group', 'purpose_group'):
        if _base in ref.columns:
            _sub = _base.replace('_group', '_subgroup')
            _label = ref[_base].astype(str).fillna('n/a')
            if _sub in ref.columns:
                _label = _label + ' | ' + ref[_sub].astype(str).fillna('n/a')
            ref[f'{_base}_combined'] = pd.Categorical(_label)
    ref['has_duplicate_geometry'] = ref['geo_id'].duplicated(keep=False)
    if entity_type in ('footprint', 'building'):
        metric_unit, imperial_unit = 'm2', 'sqft'
    elif entity_type == 'admin':
        metric_unit, imperial_unit = 'km2', 'sqmi'
    else:
        metric_unit, imperial_unit = 'ha', 'ac'

    metric_col = f'area_{metric_unit}'
    imperial_col = f'area_{imperial_unit}'
    ref[metric_col] = get_areas(ref, metric_unit)
    ref[imperial_col] = get_areas(ref, imperial_unit)

    ref_polys = ref[~ref['geo_id'].duplicated()][
        [
            c
            for c in [
                'geometry',
                'geo_id',
                metric_col,
                imperial_col,
                'has_duplicate_geometry',
            ]
            if c in ref.columns
        ]
    ].copy()
    if 'geo_id' in ref_polys.columns:
        ref_polys.index = ref_polys['geo_id'].rename('parcel_id')

    ref_agg = aggregate_rows(
        ref,
        by='geo_id',
        aggregation_function=aggregation_function,
        sort_by=sort_by,
        list_columns=list_columns,
    )
    if ref_agg is not None:
        ref_agg['n_parcels'] = ref.groupby('geo_id').size()
        collision_cols = [
            c for c in ref_polys.columns if c in ref_agg.columns and c != 'geo_id'
        ]
        if collision_cols:
            from openplaces.io.harmonizer.attributes import _resolve_suffix

            ref_suffix = _resolve_suffix(recipe_id, entity_type, state, default='_ref')
            ref_polys = ref_polys.rename(
                columns={c: f'{c}_geometry' for c in collision_cols}
            )
            ref_agg = ref_agg.rename(
                columns={c: f'{c}{ref_suffix}' for c in collision_cols}
            )
        ref_polys = ref_polys.join(ref_agg)
    area_ha_col = next(
        (c for c in ('area_ha', 'area_ha_geometry') if c in ref_polys.columns), None
    )
    if 'improvement_value' in ref_polys.columns and area_ha_col is not None:
        ref_polys['improvement_value_per_ha'] = (
            ref_polys['improvement_value'] / ref_polys[area_ha_col]
        )
    return ref_polys


def _link_spatial_overlay(
    state: HarmonizeState,
    recipe_id: str,
    entity_type: str | None,
    thresholds: dict,
    aggregation_function=None,
    sort_by: str | None = None,
    list_columns: list[str] | None = None,
    save_link: bool = False,
) -> HarmonizeState:
    """Polygon-on-polygon identity overlay; builds spine-reference crosswalk."""
    min_fraction = thresholds.get('min_fraction_of_largest', 1 / 6)
    area_min_m2 = thresholds.get('area_intersection_m2_min', 10)
    spine_id_col = state.spine.index.name

    # missing='warn': a reference recipe can genuinely have zero coverage for
    # this admin unit (see _link_spatial_point's identical handling) -- an
    # expected admin-scoped gap, not an error.
    ref_raw = get_entities(recipe_id, state.admin_id, geom=True, missing='warn')
    if ref_raw is None or len(ref_raw) == 0:
        raise_if_coverage_complete(recipe_id, state.admin_id)
        if state.verbose:
            print(f'  Link (overlay): no {recipe_id} for {state.admin_id}; skipping.')
        return state
    if state.verbose:
        print(
            f'  Link (overlay): {len(ref_raw):,d} {entity_type or "ref"} ({recipe_id})'
        )

    ref_polys = _prepare_reference(
        ref_raw,
        recipe_id,
        entity_type,
        state,
        aggregation_function=aggregation_function,
        sort_by=sort_by,
        list_columns=list_columns,
    )

    sidecar_path = None
    fingerprint = None
    footprints_on_ref = None
    if save_link:
        sidecar_path = get_entity_link_path(
            get_recipe_id(state.recipe), recipe_id, state.admin_id
        )
        fingerprint = _link_fingerprint(
            state,
            recipe_id,
            {
                'min_fraction_of_largest': min_fraction,
                'area_intersection_m2_min': area_min_m2,
                'sort_by': sort_by,
                'list_columns': list_columns,
                'aggregation_function': (
                    None if aggregation_function is None else str(aggregation_function)
                ),
            },
        )
        if not state.reprocess:
            footprints_on_ref = _load_link_sidecar(
                sidecar_path, fingerprint, spine_id_col, verbose=state.verbose
            )
    computed_fresh = footprints_on_ref is None

    if computed_fresh:
        footprints_on_ref = overlay_polygons(
            state.spine,
            ref_polys,
            suffixes=('_spine', '_ref'),
            how='identity',
            iou=True,
            geom=True,
        )
    if state.verbose:
        print(
            f'  Link (overlay): {len(footprints_on_ref):,d} '
            f'spine-{entity_type or "ref"} overlaps'
        )
    if state.timer:
        state.timer.mark('Link')

    crosswalk = _build_crosswalk(
        footprints_on_ref, spine_id_col, min_fraction, area_min_m2
    )

    snapped_links = None
    if thresholds.get('snap_chains'):
        crosswalk, snapped_links = snap_chained_links(
            crosswalk,
            spine_id_col,
            fraction_max=float(thresholds.get('chain_fraction_max', 0.75)),
        )
        if state.verbose and len(snapped_links):
            n_snapped = snapped_links.index.get_level_values(spine_id_col).nunique()
            print(
                f'  Link (overlay): snapped {n_snapped:,d} chain-displaced '
                'footprints to their dominant parcel'
            )

    # Written on the reload path too (not just computed_fresh): the link and
    # chain labels are rebuilt from the current crosswalk every run, so the
    # sidecar must be rewritten to keep its stored labels in sync (the raw
    # overlay rows and the fingerprint are carried over unchanged).
    if save_link:
        _write_link_sidecar(
            sidecar_path,
            footprints_on_ref,
            crosswalk,
            fingerprint,
            snapped=snapped_links,
            verbose=state.verbose,
        )

    state.references[recipe_id] = ref_polys
    state.crosswalks[recipe_id] = crosswalk
    state.overlays[recipe_id] = footprints_on_ref
    if entity_type:
        state.reference_types[recipe_id] = entity_type
    return state


def _superseded_by_consolidation(
    index,
    spine_id_col: str,
    superseded_spine_ids: set,
    consolidated_pids: set,
):
    """Rows a condo-cluster consolidation replaces.

    A consolidation supersedes every link to a parcel it absorbed, and
    every link from a spine row it replaced. The three copies of those
    links (the crosswalk, the overlay, and the sidecar on disk) have to
    agree on that. They did not: the crosswalk filtered by parcel while
    the other two filtered by spine id, so a parcel joining a cluster
    whose old footprint was not itself superseded kept its
    pre-consolidation overlap beside the new 'condo cluster' link, and
    those copies carried the pair twice.

    Parameters
    ----------
    index : pandas.MultiIndex
        A (spine id, parcel id) index.
    spine_id_col : str
        Name of the spine id level.
    superseded_spine_ids : set
        Spine rows the consolidation replaced.
    consolidated_pids : set
        Parcels the consolidation absorbed.

    Returns
    -------
    numpy.ndarray
        Boolean mask, True where the row is superseded.
    """
    return index.get_level_values(spine_id_col).isin(
        superseded_spine_ids
    ) | index.get_level_values('parcel_id').isin(consolidated_pids)


def _build_crosswalk(
    footprints_on_ref,
    spine_id_col: str,
    min_fraction: float,
    area_min_m2: float,
) -> pd.DataFrame:
    """Build the trimmed spine-reference crosswalk from the identity overlay.

    Pure function shared by the fresh-overlay and sidecar-reload paths, so
    the sliver trimming and link labeling can never diverge between them.
    Tolerates a geometry-free overlay (the reloaded sidecar).
    """
    # A repeated (spine id, reference id) pair is not a real overlap: a
    # genuine multi-overlap pairs one spine id with several *distinct*
    # reference ids, never twice with the same one. Left alone, the
    # copies go down the multi branch, compete on area, and one is
    # trimmed as a neighbor.
    #
    # Measured over the 2026-09-08 rebuild, every such pair was an exact
    # duplicate row, identical in every column including the
    # intersection area, in 2 to 32 copies (Wake NC: 386 pairs over
    # 1,886 rows; Harris TX: 3 over 6). Dropping exact copies is
    # therefore lossless, and it has to happen before the guard, which
    # otherwise fails a third of the counties on rows that carry no
    # information. Anything that survives this is a pair whose rows
    # actually disagree, which is the case worth refusing.
    footprints_on_ref = footprints_on_ref[~footprints_on_ref.duplicated(keep='first')]
    require_unique_index(
        footprints_on_ref.index, f'link_to_reference crosswalk on {spine_id_col}'
    )

    crosswalk_cols = [v for v in _CROSSWALK_COLS if v in footprints_on_ref.columns]
    mask_multi = footprints_on_ref.index.get_level_values(spine_id_col).duplicated(
        keep=False
    )

    footprints_single = (
        footprints_on_ref[~mask_multi]
        .reset_index()
        .set_index(spine_id_col)[['parcel_id'] + crosswalk_cols]
    )
    footprints_single.insert(
        1,
        'link',
        np.where(
            footprints_single['parcel_id'].notnull(),
            'unique parcel',
            'no parcel',
        ),
    )

    footprints_multi = footprints_on_ref[mask_multi].copy()
    footprints_multi['fraction_of_largest'] = (
        footprints_multi['area_intersection_m2']
        / footprints_multi.groupby(spine_id_col)['area_intersection_m2'].transform(
            'max'
        )
    ).round(3)

    mask_identified = footprints_multi.index.get_level_values('parcel_id').notnull()
    footprints_multi_identified = footprints_multi[mask_identified]
    footprints_multi_unidentified = footprints_multi[~mask_identified]

    footprints_multi_identified_trimmed = footprints_multi_identified.query(
        f'fraction_of_largest >= {min_fraction} '
        f'and area_intersection_m2 >= {area_min_m2}'
    )
    footprints_multi_trimmed = pd.concat(
        [footprints_multi_identified_trimmed, footprints_multi_unidentified]
    )
    mask_still_multi = footprints_multi_trimmed.index.get_level_values(
        spine_id_col
    ).duplicated(keep=False)

    multi_crosswalk_cols = [v for v in _CROSSWALK_COLS if v in footprints_multi.columns]
    footprints_single_from_multi = (
        footprints_multi_trimmed[~mask_still_multi]
        .reset_index()
        .set_index(spine_id_col)[['parcel_id'] + multi_crosswalk_cols]
    )
    footprints_single_from_multi.insert(
        1,
        'link',
        np.where(
            footprints_single_from_multi['parcel_id'].notnull(),
            'unique parcel (dropping small neighbor)',
            'no parcel',
        ),
    )

    footprints_single = pd.concat(
        [
            footprints_single.drop(
                set(footprints_single.index) & set(footprints_single_from_multi.index)
            ),
            footprints_single_from_multi,
        ]
    ).sort_index()

    split_cols = [v for v in _CROSSWALK_COLS if v in footprints_multi_trimmed.columns]
    footprints_to_split = footprints_multi_trimmed[mask_still_multi][
        split_cols + (['geometry'] if 'geometry' in footprints_multi_trimmed else [])
    ].copy()
    footprints_to_split.insert(0, 'link', 'multi-parcel footprint')

    return pd.concat(
        [
            footprints_single.reset_index().set_index([spine_id_col, 'parcel_id']),
            footprints_to_split.drop(columns='geometry', errors='ignore'),
        ]
    ).sort_index()


def snap_chained_links(
    crosswalk: pd.DataFrame,
    spine_id_col: str,
    fraction_max: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Snap chain-displaced multi-parcel links to their dominant parcel.

    A footprint layer displaced relative to the parcel layer makes each
    footprint straddle its own parcel and the next one over, chaining
    footprint-parcel-footprint-parcel down the block and inflating
    n_parcels_per_footprint for every home on it. Such a footprint is snapped
    to its dominant (largest-intersection) parcel when

    - every minor link's parcel is a *different* footprint's dominant or
      unique parcel — the neighbor demonstrably has its own building. A
      genuine shared row-house footprint never satisfies this: the
      neighboring parcels' only building is the shared footprint itself, so
      real multi-parcel buildings keep their multi links; and
    - every minor link's fraction_of_largest is at most *fraction_max* —
      a near-equal split leaves the dominant side genuinely ambiguous, so it
      is left alone.

    Ownership is computed from the pre-snap crosswalk (dominants never move),
    so one pass resolves whole chains deterministically regardless of row
    order. Uses no geometry, so it works identically on a reloaded
    geometry-free link sidecar.

    Returns
    -------
    tuple of (pandas.DataFrame, pandas.DataFrame)
        The adjusted crosswalk — each snapped footprint collapses to a single
        link relabeled ``'unique parcel (snapped from chain)'`` — and the
        removed minor rows (empty when nothing was snapped).
    """
    multi = crosswalk[crosswalk['link'] == 'multi-parcel footprint']
    if multi.empty:
        return crosswalk, crosswalk.iloc[0:0]

    flat = multi.reset_index().sort_values(
        'area_intersection_m2', ascending=False, kind='stable'
    )
    is_dominant = ~flat.duplicated(subset=spine_id_col).to_numpy()

    single_links = crosswalk['link'].astype('string').str.startswith('unique parcel')
    owned = set(
        crosswalk.index.get_level_values('parcel_id')[
            single_links.fillna(False).to_numpy()
        ].dropna()
    )
    owned |= set(flat.loc[is_dominant, 'parcel_id'].dropna())

    if 'fraction_of_largest' in flat.columns:
        fraction = pd.to_numeric(flat['fraction_of_largest'], errors='coerce')
    else:
        fraction = flat['area_intersection_m2'] / flat.groupby(spine_id_col)[
            'area_intersection_m2'
        ].transform('max')

    # A minor link's parcel in `owned` is necessarily another footprint's home:
    # a footprint is either a multi or a single link (never both), and within
    # one footprint the dominant and minor parcels are distinct index entries.
    # The dominant side must be a real parcel — a footprint whose largest
    # overlap is the unmatched (null-parcel) identity remainder has nothing to
    # snap to.
    minor_ok = flat['parcel_id'].isin(owned) & (fraction <= fraction_max)
    row_ok = pd.Series(
        np.where(is_dominant, flat['parcel_id'].notna(), minor_ok),
        index=flat.index,
    )
    snap_fp = row_ok.groupby(flat[spine_id_col]).transform('all').to_numpy()

    minor_pairs = pd.MultiIndex.from_frame(
        flat.loc[snap_fp & ~is_dominant, [spine_id_col, 'parcel_id']]
    )
    dominant_pairs = pd.MultiIndex.from_frame(
        flat.loc[snap_fp & is_dominant, [spine_id_col, 'parcel_id']]
    )
    snapped = crosswalk.loc[crosswalk.index.isin(minor_pairs)]
    out = crosswalk.drop(index=snapped.index)
    out.loc[out.index.isin(dominant_pairs), 'link'] = (
        'unique parcel (snapped from chain)'
    )
    return out, snapped
