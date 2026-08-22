"""
Pipeline steps that create and use relationships between the spine and
reference datasets:
- link_to_reference: load a reference and build a spine ↔ reference crosswalk
- infer_spine_additions: add new spine entries inferred from a reference
- resolve_overlaps: remove remaining geometry overlaps from the spine
"""

from __future__ import annotations

import json
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pandas.api.types import is_datetime64_any_dtype

from openplaces.core.attribute_registry import (
    get_agg_func,
    get_attributes,
    load_registry,
)
from openplaces.core.schema import AdminId, SourceGeometryType
from openplaces.diagnostics import find_recipes
from openplaces.geo.ids import (
    PARCEL_ID_ALNUM_KEYS,
    add_openlocationcode_index,
    add_parcel_id_alnum,
    get_geo_ids,
)
from openplaces.geo.link import get_entity_link_path
from openplaces.geo.polygon import (
    clean_polygons,
    get_areas,
    local_metric_crs,
    overlay_polygons,
    resolve_overlapping_polygons,
)
from openplaces.io import to_parquet
from openplaces.io.aggregate import _agg_func_for, aggregate_rows, read_file_metadata
from openplaces.io.cleanup import read_receipt
from openplaces.io.harmonizer import (
    _STEP_PHASES,
    HarmonizeState,
    _register,
    _rename_right_index,
    restrict_to_admin_by_name,
)
from openplaces.io.readers import get_entities
from openplaces.io.transform import make_index_unique, remap
from openplaces.recipe import (
    get_output_path,
    get_recipe_by_id,
    get_recipe_dependencies,
    get_recipe_id,
    get_save_admin_level,
    resolve_attribute_name,
    source_id_from_recipe_id,
)

# The key link_by_id joins on unless a recipe names another. Auto-discovery
# resolves its own key per match; anything else here is a caller override.
DEFAULT_LINK_KEY = 'parcel_id_local'

# Columns carried in spine-reference crosswalk tables (index levels excluded).
_CROSSWALK_COLS = [
    'area_intersection_m2',
    'iou',
    'area_intersection_m2_inner',
    'fraction_of_largest',
]

# Parquet footer key holding a link sidecar's validity fingerprint.
_LINK_METADATA_KEY = 'openplaces:link'
_LINK_INDEX_KEY = 'openplaces:link_index'

# Buffer applied to a condo cluster's own parcel footprint before testing
# whether a real footprint fragment touches it, to tolerate the usual
# building-outline/parcel-boundary digitization slack without admitting a
# genuinely disjoint fragment from an unrelated cluster.
_REAL_FOOTPRINT_TOUCH_TOLERANCE_M = 2.0

# Floor applied to a per-parcel coverage fraction before raising it to a
# negative power (the generalized-mean coverage score) -- avoids a literal
# 0 producing inf, while still driving the score for a fully-uncovered
# parcel down close to 0 (a tiny base raised to a negative power is huge,
# dominating the weighted sum).
_COVERAGE_SCORE_EPS = 1e-12


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
    the recipe_id of the one with the most-specific ``admin_id`` prefix that
    is a parent of (or equal to) *admin_id*.
    """
    df = find_recipes(entity_type, stage='ingest')
    if df.empty:
        return None
    admin_str = str(admin_id)
    candidates = []
    for _, row in df.iterrows():
        rid_str = row['admin_id']
        if rid_str == '' or admin_str.startswith(rid_str):
            level = rid_str.count('-') + 1 if rid_str else 0
            prefix = f'{rid_str}_' if rid_str else ''
            recipe_id = f'{prefix}{entity_type}-{row["source_id"]}-{row["version"]}'

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


def _fingerprint_safe_step(step_cfg: dict) -> dict:
    """A pipeline entry reduced to its fingerprint-relevant config.

    Drops ``save_link`` (toggling persistence must never force a
    recompute) and the chain-snapping thresholds (``snap_chains`` /
    ``chain_fraction_max`` are geometry-free relabeling applied after the
    overlay, deliberately excluded since format 1).
    """
    entry = {k: v for k, v in step_cfg.items() if k != 'save_link'}
    thresholds = entry.get('thresholds')
    if isinstance(thresholds, dict):
        entry['thresholds'] = {
            k: v
            for k, v in thresholds.items()
            if k not in ('snap_chains', 'chain_fraction_max')
        }
    return entry


def _link_fingerprint(
    state: HarmonizeState, ref_recipe_id: str, step_config: dict
) -> dict:
    """Validity fingerprint stored in (and checked against) a link sidecar.

    Records the step configuration, the ordered configs of every
    geometry-phase pipeline entry that ran before this step (format 2 --
    the spine reaching the join is shaped by those steps, so a changed
    spine threshold invalidates the sidecar even though the mid-pipeline
    spine itself is deliberately not fingerprinted), and the size/mtime
    of every resolvable ingest-stage input of the harmonize recipe for
    this admin unit (the reference parquet among them). A source that
    was deliberately deleted stays verifiable through its tombstone
    receipt's recorded size/mtime; a missing source with no receipt
    yields nulls, which no longer match once the file reappears (fail
    safe: recompute).
    """
    from openplaces.io.cleanup import _relative_posix
    from openplaces.io.harmonizer import _load_steps

    # Phase tags live on the @_register decorators, so every step module
    # must be imported before _STEP_PHASES is consulted -- a caller that
    # reached this function without going through the dispatch loop (the
    # geospine loader, a test) may not have triggered the module imports.
    _load_steps()

    prior_geometry_steps = []
    if state.step_index is not None:
        for prior in (state.recipe.get('pipeline') or [])[: state.step_index]:
            if not isinstance(prior, dict):
                continue
            if _STEP_PHASES.get(prior.get('step')) == 'geometry':
                prior_geometry_steps.append(_fingerprint_safe_step(prior))

    upstream_ids = {ref_recipe_id}
    try:
        edges = get_recipe_dependencies(state.recipe, admin_id=state.admin_id)
        upstream_ids |= {e.upstream_recipe_id for e in edges if e.upstream_recipe_id}
    except Exception:
        pass

    sources = []
    for upstream_id in sorted(upstream_ids):
        try:
            upstream = get_recipe_by_id(upstream_id)
            if upstream.get('stage', 'ingest') != 'ingest':
                continue
            source_admin = (
                state.admin_id.truncate_to_level(get_save_admin_level(upstream))
                if state.admin_id
                else None
            )
            path = get_output_path(upstream, admin_id=source_admin)
        except Exception:
            continue
        entry = {'path': _relative_posix(path), 'size': None, 'mtime': None}
        if path.exists():
            stat = path.stat()
            entry['size'] = stat.st_size
            entry['mtime'] = round(stat.st_mtime, 3)
        else:
            receipt = read_receipt(path)
            if receipt is not None:
                entry['size'] = receipt.get('source_size_bytes')
                mtime = receipt.get('source_mtime')
                entry['mtime'] = round(mtime, 3) if mtime is not None else None
        sources.append(entry)

    return {
        'format': 2,
        'spine_recipe_id': get_recipe_id(state.recipe),
        'ref_recipe_id': ref_recipe_id,
        'admin_id': str(state.admin_id) if state.admin_id is not None else None,
        'step_config': step_config,
        'prior_geometry_steps': prior_geometry_steps,
        'sources': sources,
    }


def _load_link_sidecar(
    sidecar_path, fingerprint: dict, spine_id_col: str, verbose: bool = False
):
    """Reload the persisted identity overlay iff its fingerprint matches.

    Returns the geometry-free overlay (MultiIndex [spine_id, parcel_id])
    or None when the sidecar is absent or invalid (recompute, fail safe).
    Only the footer is read for the validity check.
    """
    if sidecar_path is None or not sidecar_path.exists():
        return None
    stored_raw = read_file_metadata(sidecar_path).get(_LINK_METADATA_KEY)
    if stored_raw is None:
        return None
    try:
        stored = json.loads(stored_raw)
    except json.JSONDecodeError:
        return None
    if stored != fingerprint:
        if verbose:
            print(
                '  Link (overlay): sidecar fingerprint mismatch; recomputing overlay.'
            )
        return None
    overlay = pd.read_parquet(sidecar_path)
    overlay = overlay.set_index([spine_id_col, 'parcel_id'])
    overlay = overlay.drop(columns=['link', 'link_chain'], errors='ignore')
    if verbose:
        print(f'  Link (overlay): reloaded link sidecar {sidecar_path.name}')
    return overlay


def _write_link_sidecar(
    sidecar_path,
    footprints_on_ref,
    crosswalk: pd.DataFrame,
    fingerprint: dict,
    snapped: pd.DataFrame | None = None,
    verbose: bool = False,
) -> None:
    """Persist the geometry-free full identity overlay with link labels.

    The sidecar is a superset of the trimmed crosswalk: every raw overlay
    pair (including sub-threshold slivers and unmatched spine rows) with
    the crosswalk's link label left-joined on (null = trimmed-out pair).
    With chain snapping enabled (*snapped* not None, see
    :func:`snap_chained_links`), a ``link_chain`` column records the
    adjustment: ``'snapped minor'`` on the removed minor pairs (whose
    ``link`` is null, like any other pair excluded from attribution) and
    ``'snapped dominant'`` on the promoted 1-1 link, so the full physical
    overlap stays queryable even though it no longer drives attribution.
    """
    flat = pd.DataFrame(footprints_on_ref.drop(columns='geometry', errors='ignore'))
    if 'link' in crosswalk.columns:
        flat = flat.join(crosswalk['link'])
    if snapped is not None:
        chain = pd.Series(pd.NA, index=flat.index, dtype=object)
        chain[flat.index.isin(snapped.index)] = 'snapped minor'
        promoted = crosswalk.index[
            crosswalk['link'] == 'unique parcel (snapped from chain)'
        ]
        chain[flat.index.isin(promoted)] = 'snapped dominant'
        flat['link_chain'] = chain
    to_parquet(
        flat.reset_index(),
        sidecar_path,
        file_metadata={_LINK_METADATA_KEY: json.dumps(fingerprint)},
    )
    if verbose:
        print(f'  Link (overlay): wrote link sidecar {sidecar_path.name}')


def _load_point_link_sidecar(sidecar_path, fingerprint: dict, verbose: bool = False):
    """Reload a persisted point crosswalk iff its fingerprint matches.

    Returns the geometry-free flat crosswalk with the reference's native
    index restored (its name is stored beside the fingerprint, since a
    reference index may be unnamed), or None when the sidecar is absent
    or invalid (recompute, fail safe). Only the footer is read for the
    validity check.
    """
    if sidecar_path is None or not sidecar_path.exists():
        return None
    metadata = read_file_metadata(sidecar_path)
    stored_raw = metadata.get(_LINK_METADATA_KEY)
    if stored_raw is None:
        return None
    try:
        stored = json.loads(stored_raw)
    except json.JSONDecodeError:
        return None
    if stored != fingerprint:
        if verbose:
            print('  Link (point): sidecar fingerprint mismatch; recomputing links.')
        return None
    linked = pd.read_parquet(sidecar_path)
    index_name = metadata.get(_LINK_INDEX_KEY) or 'index'
    if index_name in linked.columns:
        linked = linked.set_index(index_name)
        if index_name == 'index':
            linked.index.name = None
    if verbose:
        print(f'  Link (point): reloaded link sidecar {sidecar_path.name}')
    return linked


def _write_point_link_sidecar(
    sidecar_path, linked: pd.DataFrame, fingerprint: dict, verbose: bool = False
) -> None:
    """Persist the flat point crosswalk, geometry-free.

    Unlike the overlay sidecar this is the *final* crosswalk (after every
    pass, filter, and aggregation), so nothing is recomputed on the
    reload path and the sidecar is only written when computed fresh.
    Proximity-pass rows can still carry the reference geometry; it is
    dropped here (no downstream consumer reads crosswalk geometry), so
    fresh and reloaded crosswalks differ only in that column.
    """
    flat = pd.DataFrame(linked.drop(columns='geometry', errors='ignore'))
    index_name = flat.index.name or 'index'
    to_parquet(
        flat.reset_index(),
        sidecar_path,
        file_metadata={
            _LINK_METADATA_KEY: json.dumps(fingerprint),
            _LINK_INDEX_KEY: index_name,
        },
    )
    if verbose:
        print(f'  Link (point): wrote link sidecar {sidecar_path.name}')


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
        sort_cols = [c for c in ['source', 'structure_value'] if c in group.columns]
        if sort_cols:
            group = group.sort_values(
                sort_cols, ascending=[True, False][: len(sort_cols)]
            )
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
            {
                'join': 'spatial_point',
                'thresholds': thresholds,
                'remap_id': remap_id,
            },
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
        sort_cols = [c for c in ['source', 'structure_value'] if c in linked.columns]
        if sort_cols:
            linked = linked.sort_values(
                sort_cols,
                ascending=[True, False][: len(sort_cols)],
            )
        if linked.index.duplicated().any():
            linked = linked[~linked.index.duplicated()].copy()

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
            if _pts_poly.index.duplicated().any():
                _pts_poly = _pts_poly[~_pts_poly.index.duplicated()].copy()
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
            if ref_on_poly.index.duplicated().any():
                ref_on_poly = ref_on_poly[~ref_on_poly.index.duplicated()].copy()
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


def _find_admin_scoped_recipe_ids(state: HarmonizeState, entity_type: str) -> list[str]:
    """Ingest recipes of *entity_type* whose admin scope covers ``state.admin_id``.

    One recipe id per (admin_id, source_id, filename_suffix), keeping the
    newest version when several exist for the same source (mirrors the
    specificity/version precedence ``find_entity_recipe_id`` uses,
    ``recipe.py:444-463``). ``filename_suffix`` keeps this a *competing-
    alternative* dedup, not a same-source-can-only-mean-one-recipe dedup:
    two recipe files sharing admin_id/source_id/version but distinguished
    by a filename suffix (e.g. a PACS roll's own APPRAISAL_INFO recipe
    alongside its ``_improvement-detail`` sibling, both ``source_id:
    victoriacad``) are genuinely different tables meant to coexist, not
    two versions of the same one competing to be "the" victoriacad
    recipe -- see :func:`_discover_link_sources`, which joins every one
    of them.

    Returned least-specific-admin-id-first, version ascending within a tier:
    :func:`link_by_id`'s auto-discover mode joins sources in this order, so
    the most admin-specific source's attributes are the ones applied last
    (see its column-priority rule) -- a county-scoped recipe's own values
    win over a statewide recipe's by default, not whichever happens to have
    the newer version string.

    Recipes with ``exclude_from_auto_discover: true`` are skipped -- for a
    parcel-entity ingest recipe that is a reference dataset consumed only via
    an explicit crosswalk (e.g. a legacy/external source's own attributes,
    not meant to auto-roll into the canonical spine).
    """
    if state.admin_id is None:
        return []
    best: dict[tuple[str, str, str], tuple[str, str, int]] = {}
    for _, row in find_recipes(entity_type, stage='ingest').iterrows():
        if row['exclude_from_auto_discover']:
            continue
        admin_id_str = row['admin_id']
        if not admin_id_str or not AdminId(admin_id_str).is_parent_or_equal_of(
            state.admin_id
        ):
            continue
        key = (admin_id_str, row['source_id'], row['filename_suffix'])
        recipe_id = row['recipe_id']
        specificity = admin_id_str.count('-') + 1
        if key not in best or row['version'] > best[key][0]:
            best[key] = (row['version'], recipe_id, specificity)
    return [
        recipe_id
        for _version, recipe_id, _specificity in sorted(
            best.values(), key=lambda vrs: (vrs[2], vrs[0])
        )
    ]


def _discover_link_sources(state: HarmonizeState, entity_type: str) -> list[dict]:
    """Find ingest sources covering ``state.admin_id`` and how to join each.

    A standalone roll (the candidate's own primary entity) joins on the
    standardized cross-source key ``parcel_id_local``. A bundled
    ``additional_layers`` entry joins on its declared ``layer_key`` if
    present (a same-source key shared with its primary entity, e.g.
    MassGIS's ``parcel_id_admin2``), else also falls back to
    ``parcel_id_local``.

    Ordered least-specific-admin-id-first, version ascending within a tier
    (see :func:`_find_admin_scoped_recipe_ids`): :func:`link_by_id` joins
    matches in this order and, for any column covered by more than one
    match, prefers whichever source is applied last as long as it covers a
    majority of parcels — so the most admin-specific source's attributes
    win by default, falling back to version only among equally-specific
    sources.

    Each match also carries its own ``aggregation_function`` (a recipe's or
    ``additional_layers`` entry's own top-level ``aggregation_function``
    key, or ``None``), letting one specific ingest recipe declare that its
    rows are structurally 1:many for a reason the attribute registry's
    global default does not anticipate -- e.g. a PACS
    ``APPRAISAL_IMPROVEMENT_DETAIL`` roll, one row per building component,
    where ``area_sqft`` needs summing per property rather than the
    registry's ``mean``. Scoped to the recipe that declares it: unlike a
    caller-supplied override on the :func:`link_by_id` step itself, it
    never reaches a sibling match's columns.
    """
    from openplaces.recipe import get_recipe_by_id

    matches = []
    for recipe_id in _find_admin_scoped_recipe_ids(state, entity_type):
        recipe = get_recipe_by_id(recipe_id)
        matches.append(
            {
                'recipe_id': recipe_id,
                'layer': None,
                'key': 'parcel_id_local',
                'aggregation_function': recipe.get('aggregation_function'),
            }
        )
        for layer_spec in recipe.get('additional_layers') or []:
            if 'entity' not in layer_spec:
                continue
            matches.append(
                {
                    'recipe_id': recipe_id,
                    'layer': str(layer_spec['entity'].entity_type),
                    'key': layer_spec.get('layer_key', 'parcel_id_local'),
                    'aggregation_function': layer_spec.get('aggregation_function'),
                }
            )
    return matches


def _apply_remap_csvs(state: HarmonizeState, recipe_id: str) -> HarmonizeState:
    """Auto-apply any ``{recipe_id}_*-remap.csv`` crosswalk found beside *recipe_id*.

    Each match's filename (the part between ``{recipe_id}_`` and
    ``-remap.csv``, dashes replaced by underscores) names the spine column it
    remaps; applied only when that column is present. Output columns are
    whichever non-key columns the crosswalk's own header defines (e.g.
    ``use_group``/``use_subgroup`` for a use-code crosswalk). The crosswalk's
    own key length determines how much to truncate codes before lookup
    (handles e.g. 3- vs 4-digit codes sharing one 3-digit crosswalk).
    """
    if state.spine is None:
        return state
    from openplaces.path import recipe_path
    from openplaces.recipe import get_recipe_by_id

    recipe = get_recipe_by_id(recipe_id)
    recipe_dir = recipe_path(recipe['admin_id'], recipe['entity'], as_dir=True)
    if not recipe_dir.exists():
        return state

    spine = state.spine
    prefix = f'{recipe_id}_'
    for csv_path in sorted(recipe_dir.glob(f'{prefix}*-remap.csv')):
        stem = csv_path.stem
        if not stem.endswith('-remap'):
            continue
        column = stem[len(prefix) : -len('-remap')].replace('-', '_')
        if column not in spine.columns:
            continue
        table = pd.read_csv(csv_path, dtype=str)
        key_col = table.columns[0]
        table = table.drop_duplicates(subset=[key_col]).set_index(key_col)
        key_lengths = table.index.to_series().astype(str).str.len()
        key_length = int(key_lengths.mode().iat[0])
        codes = spine[column].astype('string').str.slice(0, key_length)
        for target in table.columns:
            spine[target] = codes.map(table[target])
        if state.verbose:
            matched = codes.isin(table.index).sum()
            print(
                f'  link_by_id: applied {csv_path.name} '
                f'({matched:,d}/{len(spine):,d} {column!r} matched)'
            )

    state.spine = spine
    return state


def _align_for_combine(existing, incoming):
    """Make two columns safe to `combine_first`, or fall back to object.

    `Series.combine_first` re-infers a dtype for the union, and on two date
    columns that disagree about time zones it raises rather than choosing
    one ("Tz-aware datetime.datetime cannot be converted to datetime64").
    That is a real mix: one source stamps its sale dates UTC and another
    writes naive local dates, and nothing upstream reconciles them. It
    surfaced the first time a statewide layer's date columns reached a
    county whose own source is naive.

    Where both sides are datetimes, the tz-aware one is converted to UTC
    and the naive one localized to it, so the union is comparable rather
    than merely combinable. Anything else that cannot be reconciled falls
    back to object dtype: losing a dtype is recoverable, losing a county's
    whole harmonize run is not.
    """
    left_dt = is_datetime64_any_dtype(existing)
    right_dt = is_datetime64_any_dtype(incoming)
    if not (left_dt or right_dt):
        return existing, incoming
    if not (left_dt and right_dt):
        # One side is datetimes and the other is not (commonly object
        # holding datetimes, which is what `to_datetime` chokes on). There
        # is no timezone to reconcile because one side has no dtype to
        # reconcile it with, so hand both over as object and let
        # combine_first do the only thing that cannot raise.
        return existing.astype(object), incoming.astype(object)
    left_tz = getattr(existing.dtype, 'tz', None)
    right_tz = getattr(incoming.dtype, 'tz', None)
    if (left_tz is None) == (right_tz is None):
        return existing, incoming
    try:
        aware = 'UTC'
        existing = (
            existing.dt.tz_localize(aware)
            if left_tz is None
            else existing.dt.tz_convert(aware)
        )
        incoming = (
            incoming.dt.tz_localize(aware)
            if right_tz is None
            else incoming.dt.tz_convert(aware)
        )
    except (AttributeError, TypeError, ValueError):
        return existing.astype(object), incoming.astype(object)
    return existing, incoming


def _write_prioritized(
    spine: gpd.GeoDataFrame,
    name: str,
    new_vals: pd.Series,
    majority_coverage: float = 0.5,
    provenance_token: str | None = None,
) -> None:
    """Write *new_vals* into ``spine[name]``, applying the recency/coverage rule.

    A column already on the spine came from an earlier (less admin-specific,
    per :func:`_find_admin_scoped_recipe_ids`'s ordering) source. *new_vals*
    only overwrites it outright when *new_vals* itself covers at least
    *majority_coverage* of the spine; otherwise it only fills the existing
    column's gaps, so a sparse more-specific source can't blank out a more
    complete less-specific one.

    Parameters
    ----------
    provenance_token : str, optional
        When given, also updates the ``{name}_source`` categorical sidecar
        (via :func:`openplaces.io.harmonizer._record_source`) for exactly
        the cells this call actually changes -- a value written where none
        existed, or an existing value outright replaced on a
        majority-coverage overwrite. Cells left unchanged (including
        still-null cells) are never touched. Reserve this for a recipe's
        declared "key" columns (see ``link_by_id``'s ``track_provenance``)
        rather than every joined attribute, to avoid a provenance-sidecar
        column explosion.
    """
    from openplaces.io.harmonizer import _record_source

    if name not in spine.columns:
        spine[name] = new_vals
        if provenance_token:
            _record_source(spine, name, new_vals.notna(), provenance_token)
        return

    before = spine[name].copy() if provenance_token else None
    coverage = new_vals.notna().mean() if len(new_vals) else 0.0
    existing, incoming = _align_for_combine(spine[name], new_vals)
    if coverage >= majority_coverage:
        spine[name] = incoming.combine_first(existing)
    else:
        spine[name] = existing.combine_first(incoming)

    if provenance_token:
        after = spine[name]
        changed = after.notna() & (before.isna() | (before != after))
        _record_source(spine, name, changed, provenance_token)


# A join key value carried by this many rows, or by this share of the
# reference, is a placeholder rather than an identifier. Measured on the
# 2026 coastal-Texas and eastern-NC sources: a healthy file's most common
# `parcel_id_local` covers 5 rows (Kleberg) to 49 (Pender, a real
# multi-unit building), while Brazoria's most common value is '0' on
# 24,241 rows and its runner-up covers 2,443. The floor keeps a genuine
# condominium's units together; the share scales with file size.
DEGENERATE_KEY_MIN_ROWS = 100
DEGENERATE_KEY_MAX_SHARE = 0.001


def _placeholder_key_mask(key: pd.Series) -> pd.Series:
    """Values that cannot be identifiers whatever their frequency.

    Empty strings and all-zero codes ('0', '000', '000-00-0000'): assessors
    write these where a parcel number is unknown, and every row carrying one
    is a different parcel.
    """
    text = key.astype('string').str.strip()
    blank = text.isna() | text.eq('')
    zeros = text.str.replace(r'[^0-9A-Za-z]', '', regex=True).str.fullmatch('0+')
    return blank | zeros.fillna(False)


def _neutralize_degenerate_keys(
    ref: pd.DataFrame,
    ref_key: str,
    recipe_id: str | None = None,
) -> pd.DataFrame:
    """Blank out join-key values that are placeholders, not identifiers.

    A key shared by thousands of rows makes every mode of
    :func:`link_by_id` wrong, and silently: 'attributes' picks one row
    arbitrarily for all of them, 'count' reports the whole group's size on
    each, and 'aggregate' **sums** their value columns and writes that sum
    onto every one. Measured consequence before this guard: a quarter-acre
    Brazoria County lot carrying a $7.5 billion `total_value`, and 152
    eastern-NC footprints holding 62% of the region's entire delivered
    parcel value.

    Setting the key to missing (rather than dropping the rows) lets each
    mode's existing `dropna(subset=[ref_key])` skip them, so an unmatched
    parcel simply gains no reference attributes -- the same outcome as a
    parcel the reference never mentioned.

    Returns *ref* unchanged when nothing is degenerate.
    """
    if ref_key not in ref.columns or ref.empty:
        return ref
    key = ref[ref_key]
    bad = _placeholder_key_mask(key)

    counts = key.astype('string').value_counts()
    cutoff = max(DEGENERATE_KEY_MIN_ROWS, DEGENERATE_KEY_MAX_SHARE * len(ref))
    overused = set(counts[counts > cutoff].index)
    if overused:
        bad = bad | key.astype('string').isin(overused)

    # Only rows that actually carried a value are a change worth reporting;
    # an already-missing key was never going to join.
    bad = bad & key.notna()
    if not bad.any():
        return ref
    ref = ref.copy()
    ref.loc[bad, ref_key] = pd.NA
    where = f' in {recipe_id}' if recipe_id else ''
    sample = ', '.join(repr(v) for v in sorted(overused)[:3])
    warnings.warn(
        f'link_by_id: {int(bad.sum()):,} of {len(ref):,} reference rows'
        f'{where} carry a degenerate {ref_key!r} (a placeholder, or a '
        f'value shared by more than {cutoff:,.0f} rows'
        + (f'; e.g. {sample}' if overused else '')
        + '). They are left unjoined rather than aggregated together.',
        stacklevel=2,
    )
    return ref


def _warn_if_duplicate_key(
    key_series: pd.Series,
    key_name: str,
    context: str,
    is_own_identity_key: bool = True,
) -> None:
    """Warn when *key_series* has duplicate values.

    openplaces indices (e.g. ``parcel_id``, a geo_id) are unique by design.
    Joining on a column that isn't -- ``parcel_id_local`` is the common case --
    is fine as long as it's *known* to be a many-to-one key (and, for
    'aggregate'/'count' modes, is actually aggregated); this makes that
    non-uniqueness visible instead of a silent assumption.

    *is_own_identity_key* silences the warning when *key_name* is a foreign
    key borrowed from another entity (e.g. ``parcel_id_local`` on a
    transaction spine) rather than the checked side's own identity -- there,
    duplicates are structurally expected (many transactions share one
    parcel) rather than a same-entity collision.
    """
    if not is_own_identity_key:
        return
    valid = key_series.dropna()
    dup_mask = valid.duplicated(keep=False)
    n_dup_rows = int(dup_mask.sum())
    if n_dup_rows:
        n_dup_values = int(valid[dup_mask].nunique())
        warnings.warn(
            f'link_by_id ({context}): {key_name!r} is not unique -- '
            f'{n_dup_rows:,d} rows share a value with at least one other row, '
            f'across {n_dup_values:,d} duplicated values.'
        )


def _columns_as_pairs(
    columns: list[str] | dict[str, str] | None,
) -> list[tuple[str, str]]:
    """Normalize *columns* to ``(ref_column, output_name)`` pairs.

    A plain list keeps the reference column's own name as the output name; a
    dict (``{ref_column: output_name}``) renames on the way in, e.g. joining a
    transaction reference's ``price``/``recorded_date`` onto the canonical
    ``last_sale_price``/``last_sale_date`` parcel attributes.
    """
    if not columns:
        return []
    if isinstance(columns, dict):
        return list(columns.items())
    return [(c, c) for c in columns]


@_register('link_by_id')
def link_by_id(
    state: HarmonizeState,
    recipe_id: str | None = None,
    auto_discover: bool = False,
    entity_type: str = 'parcel',
    mode: str = 'attributes',
    spine_key: str = DEFAULT_LINK_KEY,
    ref_key: str = DEFAULT_LINK_KEY,
    columns: list[str] | dict[str, str] | None = None,
    aggregation_function: dict[str, str] | None = None,
    suffix: str | None = None,
    count_as: str | None = None,
    flag_as: str | None = 'is_transacted',
    layer: str | None = None,
    ref_sort_by: str | None = None,
    ref_sort_ascending: bool = True,
    track_provenance: list[str] | None = None,
    _protect_own_columns: set[str] | None = None,
) -> HarmonizeState:
    """Link a reference entity to the spine by a precomputed id key (non-spatial).

    Joins on the standardized matching key (``parcel_id_local``) that data
    ingestion already computed on both sides, so no re-conversion happens here.

    Column priority (``'attributes'`` and ``'aggregate'`` modes): when a
    column is already on the spine (from an earlier call, e.g. an earlier
    *auto_discover* match), the new source only overwrites it outright if the
    new source covers a majority of spine rows for that column; otherwise it
    only fills the existing column's gaps (see :func:`_write_prioritized`).
    Combined with *auto_discover*'s least-specific-to-most-specific join
    order, this makes the most admin-specific source the default winner for
    each column, without letting a sparse more-specific source blank out a
    more complete less-specific one.

    Parameters
    ----------
    recipe_id : str, optional
        Reference entity recipe (e.g. an assessment roll or a transaction
        table). Required unless *auto_discover* is set.
    auto_discover : bool
        When set, ignore *recipe_id* and instead discover every ingest
        recipe of *entity_type* (plus any bundled ``additional_layers``)
        whose admin scope covers the admin unit being processed
        (:func:`_discover_link_sources`), and recurse into this function once
        per match. Each match always joins via ``mode='aggregate'`` (a
        correct generalization of ``'attributes'`` for 1:1 data too) on its
        resolved key (``parcel_id_local`` for a standalone roll; a layer's
        own ``layer_key`` for a bundled ``additional_layers`` entry), with
        *columns* defaulting to the attribute registry's canonical columns
        for that entity type (:func:`~openplaces.core.attribute_registry.
        get_attributes`) — pass an explicit *columns* to override this
        default for every discovered match. *suffix* and *count_as* are
        forwarded to each discovered match unchanged (e.g. a fixed
        ``count_as='n_transactions'`` for every auto-discovered
        ``entity_type='transaction'`` source). Any ``*-remap.csv`` crosswalk
        found beside a matched source is applied automatically (see
        :func:`_apply_remap_csvs`). A standalone match that is also one of
        the spine's own geometry sources has its ``resolve_spine``
        ``keep_columns`` protected row-by-row (via ``_protect_own_columns``,
        internal-only) rather than dropped from the join outright: only the
        rows where *this* source's own geometry actually won
        (``geometry_source`` equals its label) are shielded, since those
        already carry a correct, un-aggregated value from ``resolve_spine``
        directly -- re-deriving one via this aggregate join would pool
        values across every row sharing a non-unique local key. Every other
        row (geometry won by a *different* source) is free to receive this
        source's value through the ordinary null-aware
        :func:`_write_prioritized` gap-fill/overwrite, so a more
        admin-specific source's richer keep_columns data can still fill a
        gap left by whichever source won geometry there. Falls back to the
        old, coarser "drop the column from this match entirely" guard when
        the spine has no ``geometry_source`` column at all (e.g. a
        :func:`union_spine_sources`-built, non-spatial spine).
    entity_type : str
        Entity type to discover when *auto_discover* is set (default
        ``'parcel'``).
    mode : {'attributes', 'count', 'aggregate'}
        ``'attributes'`` joins *columns* from the reference onto the spine
        (1:1 on the key, keeping the first reference row per key). ``'count'``
        aggregates a 1:many reference into a per-spine count (*count_as*) and a
        boolean presence flag (*flag_as*) — used to track which parcels have
        been transacted. ``'aggregate'`` reduces a 1:many reference onto the
        spine by grouping on the key and applying each column's attribute-
        registry aggregation (e.g. sum land_value/n_dwellings, max year_built),
        or *aggregation_function*'s override where one is given for that
        column, falling back to the first non-null value for columns without
        a usable rule; it also emits a per-key record count. Use it when several
        reference rows share one spine key, such as MassGIS L3_ASSESS condominium
        records stacked on one parcel polygon.
    spine_key, ref_key : str
        Key columns on the spine and reference (default ``parcel_id_local``).
        Unlike an entity's own index (unique by design), this key is not
        guaranteed unique; see :func:`_warn_if_duplicate_key`, which warns
        when either side turns out to have duplicates, so that risk is
        visible rather than silently assumed away.
    columns : list of str or dict of {str: str}, optional
        Reference columns to attach in ``'attributes'``/``'aggregate'`` mode.
        A dict renames each reference column to its value on write (e.g.
        ``{'price': 'last_sale_price', 'recorded_date': 'last_sale_date'}``),
        so the registry aggregation lookup and ``_write_prioritized`` gap-fill
        apply to the actual canonical output name rather than the reference's
        own column name.
    aggregation_function : dict of {str: str}, optional
        ``'aggregate'`` mode only. Per-output-column override of the
        attribute-registry aggregation, keyed by the same post-rename
        canonical output name as *columns*'s dict form (e.g.
        ``{'area_sqft': 'sum'}``). Exists because the registry default is a
        property-level default (``area_sqft`` is ``'mean'``, i.e. one value
        per property in most sources), while a reference where several rows
        share a key for a structural reason -- e.g. one row per building
        component in a PACS ``APPRAISAL_IMPROVEMENT_DETAIL`` roll -- needs
        that key's *sum* instead. Changing the registry default would
        corrupt every other recipe's one-row-per-property case, so the
        override lives here, per recipe, instead. A column absent from the
        dict keeps the registry default. Resolved through the same
        :func:`~openplaces.table._agg_func_for` alias lookup as the
        registry path, so ``'join_nonnull'`` works here too, not only the
        plain pandas reducer names.

        In ``auto_discover`` mode this explicit override still applies to
        every discovered match, so prefer letting the *reference recipe*
        declare its own top-level ``aggregation_function`` key instead
        (:func:`_discover_link_sources` reads it): that scopes the
        override to just that one recipe's columns, leaving every sibling
        match's registry default untouched. Pass this parameter in
        ``auto_discover`` mode only when the override genuinely belongs to
        the caller, not to one specific source.
    suffix : str, optional
        Suffix appended to attached column names (``'attributes'``/
        ``'aggregate'`` mode).
    count_as, flag_as : str, optional
        Output column names in ``'count'`` mode (default ``'n_transactions'``/
        ``'is_transacted'`` when unset). In ``'aggregate'`` mode, *count_as*
        names the per-key record count column (default ``'n_records_per_key'``
        when unset — deliberately not ``'n_transactions'`` unless requested,
        since ``'aggregate'`` is also used for non-transaction references like
        MassGIS condo unit stacks); pass ``flag_as=None`` to skip the
        ``'count'``-mode presence flag when it isn't needed (e.g. it's exactly
        ``count_as > 0`` and not worth persisting).
    layer : str, optional
        Secondary layer (entity type or full entity string) of an
        ``additional_layers`` entity to load from *recipe_id*, e.g. the
        ``property`` assessor table bundled inside a MassGIS parcel recipe.
    ref_sort_by : str, optional
        Reference column to sort by before ``'aggregate'``/``'attributes'``
        mode picks a row per key. The registry's ``'first'``/``'last'``
        aggregation (and ``'attributes'`` mode's own duplicate-key pick) take
        whichever row happens to come first in *ref*'s existing order, which
        is not necessarily meaningful order (e.g. a raw transaction table is
        not guaranteed sorted by date) -- set this to make ``'first'`` mean
        "most recent" for a column like ``last_sale_price``/``last_sale_date``
        derived from a ``recorded_date`` reference.
    ref_sort_ascending : bool, default True
        Sort direction for *ref_sort_by* (``False`` so ``'first'`` picks the
        most recent row when sorting by a date column).
    track_provenance : list of str, optional
        Output column names (post-rename, post-suffix base names) to record
        per-cell source provenance for via :func:`_write_prioritized`'s
        ``provenance_token`` (see there) -- writes a ``{column}_source``
        sidecar for exactly the columns named here, not every joined
        attribute. In ``auto_discover`` mode, forwarded unchanged to every
        discovered match, each stamping its own ``source_id`` (see
        :func:`~openplaces.recipe.source_id_from_recipe_id`) as the token.
    """
    if auto_discover:
        # A standalone roll that is also one of the spine's own geometry
        # sources (state.metadata['spine_source_recipe_ids'], set by
        # resolve_spine) would otherwise re-derive its keep_columns
        # attributes (e.g. use_group/use_subgroup) by aggregating across
        # every spine row sharing its join key -- overwriting an
        # already-correct per-geometry value with one pooled from unrelated
        # rows. Those columns are already on the spine directly from the
        # same source's own row wherever its geometry won; protect exactly
        # those rows (see _protect_own_columns below) rather than dropping
        # the column from the whole match, so this same source can still
        # fill a keep_columns gap on a row a *different* source's geometry
        # occupies.
        spine_source_ids = state.metadata.get('spine_source_recipe_ids', set())
        spine_keep_columns = state.metadata.get('spine_keep_columns', set())
        has_geometry_source = (
            state.spine is not None and 'geometry_source' in state.spine.columns
        )
        for match in _discover_link_sources(state, entity_type):
            match_columns = columns or list(
                get_attributes(match['layer'] or entity_type).index
            )
            protect_columns: set[str] | None = None
            if match['layer'] is None and match['recipe_id'] in spine_source_ids:
                keep_overlap = {c for c in match_columns if c in spine_keep_columns}
                if keep_overlap:
                    if has_geometry_source:
                        protect_columns = keep_overlap
                    else:
                        # No geometry_source to key row-level protection on
                        # (e.g. a union_spine_sources-built non-spatial
                        # spine) -- fall back to the coarser column drop
                        # rather than risk the pooled-duplicate-key
                        # corruption this guard exists to prevent.
                        match_columns = [
                            c for c in match_columns if c not in keep_overlap
                        ]
                        if not match_columns:
                            continue
            # A match's own declared override (e.g. the improvement-detail
            # sibling's area_sqft: sum) wins over the caller's for the
            # columns it names, but never reaches a sibling match with no
            # such declaration -- see _discover_link_sources.
            match_aggregation_function = {
                **(aggregation_function or {}),
                **(match['aggregation_function'] or {}),
            }
            # Auto-discovery normally picks the key per match. An
            # explicit key from the caller overrides it, which is what
            # lets a second pass re-run the same discovery on the
            # punctuation-free fallback key.
            state = link_by_id(
                state,
                recipe_id=match['recipe_id'],
                mode='aggregate',
                spine_key=(
                    spine_key if spine_key != DEFAULT_LINK_KEY else match['key']
                ),
                ref_key=(ref_key if ref_key != DEFAULT_LINK_KEY else match['key']),
                columns=match_columns,
                aggregation_function=match_aggregation_function or None,
                suffix=suffix,
                count_as=count_as,
                layer=match['layer'],
                ref_sort_by=ref_sort_by,
                ref_sort_ascending=ref_sort_ascending,
                track_provenance=track_provenance,
                _protect_own_columns=protect_columns,
            )
            state = _apply_remap_csvs(state, match['recipe_id'])
        return state

    if recipe_id is None:
        warnings.warn('link_by_id: no recipe_id and auto_discover is False; skipping.')
        return state

    if state.spine is None:
        warnings.warn('link_by_id: spine is None; skipping.')
        return state
    # The punctuation-free fallback key is derived on demand rather than
    # required from ingest. It is a pure function of id columns both sides
    # already carry, so deriving it here makes the fallback work against
    # everything already on disk instead of forcing a re-ingest of every
    # parcel source in the country.
    if spine_key in PARCEL_ID_ALNUM_KEYS:
        # Recomputed, never inherited. resolve_spine would otherwise carry
        # a copy from whichever source won the geometry, which for exactly
        # the counties this fallback exists to serve is the source that
        # has no usable id -- Pender arrived with 49 of 55,101 filled.
        state.spine = add_parcel_id_alnum(state.spine, key=spine_key)
    if spine_key not in state.spine.columns:
        warnings.warn(
            f'link_by_id: spine has no {spine_key!r}; skipping {recipe_id}. '
            'Was the spine source ingested with a parcel_id_local directive '
            'and resolve_spine keep_columns?'
        )
        return state

    try:
        ref = get_entities(recipe_id, state.admin_id, layer=layer)
    except (FileNotFoundError, OSError, KeyError, ValueError):
        # The reference is not available for this admin (e.g. a source-specific
        # roll listed in a shared pipeline that does not apply here, including
        # one scoped to a different state) or has not been ingested; skip.
        if state.verbose:
            print(f'  link_by_id: no {recipe_id} for {state.admin_id}; skipping.')
        return state
    if ref is not None and layer is None:
        # A reference scoped coarser than state.admin_id with no matching
        # admin-id column (e.g. a statewide transaction table keyed only by a
        # free-text county name) comes back unfiltered from get_entities --
        # restrict it here so a per-county aggregate isn't silently pooled
        # across the whole state.
        ref = restrict_to_admin_by_name(ref, recipe_id, state.admin_id)
    if ref is not None and ref_key in PARCEL_ID_ALNUM_KEYS:
        ref = add_parcel_id_alnum(ref, key=ref_key)
    if ref is None or ref_key not in ref.columns:
        if ref_key in PARCEL_ID_ALNUM_KEYS:
            # A derived fallback key simply cannot be built for a source
            # that carries none of its input columns -- a county roll has
            # no statewide PIN column, and that is the normal case rather
            # than a misconfigured recipe. Silent, or every extra pass
            # warns once per source in every county.
            if state.verbose:
                print(f'  link_by_id: {recipe_id} cannot derive {ref_key!r}; skipping.')
        else:
            warnings.warn(
                f'link_by_id: reference {recipe_id} has no {ref_key!r}; skipping.'
            )
        return state
    if ref_sort_by and ref_sort_by in ref.columns:
        ref = ref.sort_values(ref_sort_by, ascending=ref_sort_ascending, kind='stable')

    spine = state.spine
    skey = spine[spine_key].astype('string')
    # Before any mode reads the key: a placeholder shared by thousands of
    # rows is not an identifier, and every mode below would silently treat
    # it as one (see _neutralize_degenerate_keys).
    ref = _neutralize_degenerate_keys(ref, ref_key, recipe_id)
    rkey = ref[ref_key].astype('string')
    spine_entity = state.recipe.get('entity')
    spine_entity_type = (
        str(spine_entity.entity_type) if spine_entity is not None else None
    )
    is_own_key = spine_entity_type is None or spine_key.startswith(
        f'{spine_entity_type}_id'
    )
    _warn_if_duplicate_key(skey, spine_key, 'spine key', is_own_identity_key=is_own_key)

    if mode == 'attributes':
        pairs = [(c, o) for c, o in _columns_as_pairs(columns) if c in ref.columns]
        # 'attributes' keeps one arbitrary row per key (no aggregation, unlike
        # 'aggregate'/'count') -- a duplicate ref_key here is silently resolved
        # by drop_duplicates below, so flag it before that happens.
        _warn_if_duplicate_key(rkey, ref_key, 'attributes reference key')
        ref_unique = ref.dropna(subset=[ref_key]).drop_duplicates(ref_key).copy()
        ref_unique.index = ref_unique[ref_key].astype('string')
        provenance_cols = set(track_provenance or [])
        token = source_id_from_recipe_id(recipe_id) if provenance_cols else None
        for col, out_name in pairs:
            name = f'{out_name}{suffix}' if suffix else out_name
            ref_series = ref_unique[col]
            mapper = ref_series.to_dict() if ref_series.empty else ref_series
            _write_prioritized(
                spine,
                name,
                skey.map(mapper),
                provenance_token=token if out_name in provenance_cols else None,
            )
        if state.verbose:
            matched = skey.isin(set(rkey.dropna())).sum()
            print(
                f'  Link by id (attributes): {matched:,d}/{len(spine):,d} spine '
                f'rows matched {recipe_id} ({len(pairs)} columns)'
            )
    elif mode == 'count':
        count_as = count_as or 'n_transactions'
        counts = rkey.dropna().value_counts()
        mapper = counts.to_dict() if counts.empty else counts
        spine[count_as] = skey.map(mapper).fillna(0).astype('int64')
        if flag_as:
            spine[flag_as] = spine[count_as] > 0
        if state.verbose:
            linked = int((spine[count_as] > 0).sum())
            print(
                f'  Link by id (count): {linked:,d}/'
                f'{len(spine):,d} spine rows linked to {recipe_id} ({count_as})'
            )
    elif mode == 'aggregate':
        pairs = [
            (c, o)
            for c, o in _columns_as_pairs(columns)
            if c in ref.columns and c != ref_key
        ]
        ref_valid = ref.dropna(subset=[ref_key]).copy()
        ref_valid[ref_key] = ref_valid[ref_key].astype('string')
        grouped = ref_valid.groupby(ref_key, sort=False)

        # Registry-driven reduction (sum values/dwellings, mean year, etc.);
        # columns without a usable registry rule fall back to the first value.
        # Looked up by the *output* name -- the canonical slot being filled --
        # not the reference's own column name, so a rename (e.g. price ->
        # last_sale_price) still resolves the right aggregation. 'join_nonnull'
        # (e.g. address, use_group) is not a pandas groupby function on its
        # own -- routed through _agg_func_for (the same helper aggregate_rows
        # uses) to concatenate every distinct non-null value instead of
        # silently degrading to 'first' (an arbitrary row's value, discarding
        # every other row's -- the original multi-property-per-parcel
        # collapse bug). 'address' gets its own joiner there
        # (join_nonnull_addresses) rather than the generic one: a condo/
        # apartment building's per-unit property records typically differ
        # only by an APT/UNIT/# suffix, and the plain joiner's ' + '-
        # concatenation of every unit's full address corrupts downstream
        # address parsing (no parser can split a multi-address blob back
        # into one street/number). 'use_group' and other join_nonnull
        # columns keep the plain joiner -- their concatenated values are
        # consumed directly, never re-parsed.
        reducible = {
            'sum',
            'mean',
            'max',
            'min',
            'first',
            'last',
            'median',
            'join_nonnull',
        }
        own_geometry_mask = None
        if _protect_own_columns and 'geometry_source' in spine.columns:
            own_label = source_id_from_recipe_id(recipe_id)
            own_geometry_mask = spine['geometry_source'].astype('string') == own_label
        provenance_cols = set(track_provenance or [])
        token = source_id_from_recipe_id(recipe_id) if provenance_cols else None
        for col, out_name in pairs:
            canonical_name = resolve_attribute_name(out_name)
            fname = (aggregation_function or {}).get(out_name) or get_agg_func(
                canonical_name
            )
            func = (
                _agg_func_for(canonical_name, fname) if fname in reducible else 'first'
            )
            name = f'{out_name}{suffix}' if suffix else out_name
            col_series = ref_valid[col]
            # A registry-numeric column can still arrive here as pandas
            # 'string'/object dtype (e.g. a fixed-width ingest, which never
            # casts a mapped column's dtype -- see the PACS improvement-
            # detail recipe). 'sum'/'mean'/'median' on a string column
            # crashes outright; 'min'/'max' does not, but silently compares
            # lexicographically instead of numerically (e.g. '12' < '5'),
            # which is worse -- no crash to notice it by. Coerce first for
            # either failure mode.
            if fname in (
                'sum',
                'mean',
                'median',
                'min',
                'max',
            ) and not pd.api.types.is_numeric_dtype(col_series):
                grouped_col = pd.to_numeric(col_series, errors='coerce').groupby(
                    ref_valid[ref_key], sort=False
                )
            else:
                grouped_col = grouped[col]
            agg_series = grouped_col.agg(func)
            mapper = agg_series.to_dict() if agg_series.empty else agg_series
            new_vals = skey.map(mapper)
            if (
                _protect_own_columns
                and out_name in _protect_own_columns
                and own_geometry_mask is not None
            ):
                new_vals = new_vals.mask(own_geometry_mask)
            _write_prioritized(
                spine,
                name,
                new_vals,
                provenance_token=token if out_name in provenance_cols else None,
            )
        count_col = count_as or 'n_records_per_key'
        gsize = grouped.size()
        mapper = gsize.to_dict() if gsize.empty else gsize
        spine[count_col] = skey.map(mapper).fillna(0).astype('int64')
        if state.verbose:
            matched = skey.isin(set(rkey.dropna())).sum()
            print(
                f'  Link by id (aggregate): {matched:,d}/{len(spine):,d} spine '
                f'rows matched {recipe_id} ({len(pairs)} columns, {count_col})'
            )
    else:
        raise ValueError(
            f'link_by_id: unknown mode {mode!r}; expected '
            "'attributes', 'count', or 'aggregate'."
        )

    state.spine = spine
    return state


@_register('link_address_ranges')
def link_address_ranges(
    state: HarmonizeState,
    recipe_id: str,
    columns: list[str] | dict[str, str] | None = None,
    street_column: str = 'address_street',
    number_column: str = 'address_number',
    admin4_column: str = 'admin4_id',
    suffix: str | None = None,
) -> HarmonizeState:
    """Resolve multi-unit address ranges left unmatched by :func:`link_by_id`.

    A raw address number like ``'704-706'`` (a standard multi-unit/multi-
    family deed notation, preserved by
    :func:`~openplaces.geo.address.normalize_address_components` rather than
    squashed into ``'704706'``) never matches a parcel reference's own
    single-number ``address_id_local`` key via the ordinary
    :func:`link_by_id` steps -- no real parcel has that combined number. This
    step splits the range (:func:`~openplaces.geo.address.split_number_range`)
    and tries each individual number against *recipe_id*'s own
    ``admin4|street|number`` key components (the same construction
    :func:`~openplaces.io.harmonizer.addresses.derive_address_id_local` uses)
    instead. It also handles the mirror case: a *reference* row whose own
    number is a range (e.g. a parcel recorded as ``'20-22 Main St'``) while
    the spine lists a single plain number that's one of the two halves --
    for spine rows :func:`link_by_id` left unmatched, each reference range
    is likewise registered under both of its individual numbers.

    A multi-family range normally corresponds to one parcel: if exactly one
    distinct reference row resolves (from either side's range, or a plain
    match), link it (gap-fill only, via :func:`_write_prioritized`, so an
    already-linked row is never touched). If more than one distinct
    reference row resolves -- ambiguous, and not expected in the normal case
    -- the row is left unmatched rather than guessed at; every such row this
    call finds is reported in a single warning with a small sample, since
    arbitrarily picking one parcel over the other would be wrong until the
    ambiguity is looked at deliberately.

    Parameters mirror :func:`link_by_id`'s naming for the columns it shares
    (*street_column*/*number_column*/*admin4_column* also match
    :func:`~openplaces.io.harmonizer.addresses.derive_address_id_local`'s
    defaults, so a recipe using those defaults on both sides needs none of
    them repeated here).
    """
    if state.spine is None or number_column not in state.spine.columns:
        return state

    from openplaces.geo.address import canonicalize_for_match, split_number_range

    spine = state.spine
    split = (
        spine[number_column]
        .astype('string')
        .map(lambda v: split_number_range(v) if pd.notna(v) else None)
    )
    is_range = split.notna()

    try:
        ref = get_entities(recipe_id, state.admin_id)
    except (FileNotFoundError, OSError, KeyError, ValueError):
        if state.verbose:
            print(
                f'  link_address_ranges: no {recipe_id} for {state.admin_id}; skipping.'
            )
        return state
    if ref is not None:
        ref = restrict_to_admin_by_name(ref, recipe_id, state.admin_id)
    if (
        ref is None
        or street_column not in ref.columns
        or number_column not in ref.columns
    ):
        return state

    admin_str = str(state.admin_id) if state.admin_id else ''
    admin_levels = AdminId(admin_str).levels if admin_str else ()
    admin1_id = admin_levels[0] if admin_levels else None

    def _base_key(frame):
        street = frame[street_column].astype('string').fillna('')
        canon_map = {
            s: canonicalize_for_match(s, admin1_id) if s else ''
            for s in street.unique()
        }
        canon_street = street.map(canon_map)
        number = (
            frame[number_column].astype('string').str.strip().str.upper().fillna('')
        )
        admin4 = (
            frame[admin4_column].astype('string').fillna('')
            if admin4_column in frame.columns
            else pd.Series('', index=frame.index)
        )
        has_base = canon_street.ne('') & number.ne('')
        return (
            admin4 + '|' + canon_street + '|' + number,
            has_base,
            canon_street,
            admin4,
        )

    def _register_key(index_by_key: dict, key, row_id) -> None:
        bucket = index_by_key.setdefault(key, [])
        if row_id not in bucket:
            bucket.append(row_id)

    # Reference key index: each row's own combined key, plus -- for a row
    # whose own number is itself a range -- each half's key too, pointing
    # back to the same row. A key claimed by more than one distinct row
    # (e.g. a real plain parcel at '22 Main St' *and* a range parcel
    # '20-22 Main St' both registering '.../22') is kept as an explicit
    # multi-row bucket rather than silently picking one, so it surfaces as
    # an ambiguous match below like any other multi-parcel case.
    ref_key, ref_has_base, ref_canon_street, ref_admin4 = _base_key(ref)
    ref_key_to_rows: dict[str, list] = {}
    for row_id, key in ref_key[ref_has_base].items():
        _register_key(ref_key_to_rows, key, row_id)

    ref_range_split = (
        ref[number_column]
        .astype('string')
        .map(lambda v: split_number_range(v) if pd.notna(v) else None)
    )
    for row_id in ref.index[ref_range_split.notna() & ref_has_base]:
        num1, num2 = ref_range_split.loc[row_id]
        prefix = f'{ref_admin4.loc[row_id]}|{ref_canon_street.loc[row_id]}|'
        _register_key(ref_key_to_rows, prefix + num1, row_id)
        _register_key(ref_key_to_rows, prefix + num2, row_id)

    pairs = [(c, o) for c, o in _columns_as_pairs(columns) if c in ref.columns]
    output_names = [f'{o}{suffix}' if suffix else o for _, o in pairs]

    _, _, spine_canon_street, spine_admin4 = _base_key(spine)

    existing = [n for n in output_names if n in spine.columns]
    already_matched = (
        spine[existing].notna().any(axis=1)
        if existing
        else pd.Series(False, index=spine.index)
    )
    plain_unmatched = ~is_range & spine[number_column].notna() & ~already_matched
    attempt_rows = is_range | plain_unmatched
    if not attempt_rows.any():
        return state

    matched_index: dict = {}
    ambiguous_rows = []
    n_range_attempted = 0
    for idx in spine.index[attempt_rows]:
        if is_range.loc[idx]:
            n_range_attempted += 1
            num1, num2 = split.loc[idx]
            candidates = [num1, num2]
        else:
            candidates = [str(spine.loc[idx, number_column]).strip().upper()]
        prefix = f'{spine_admin4.loc[idx]}|{spine_canon_street.loc[idx]}|'

        distinct_rows: list = []
        matched_keys: list = []
        for key in dict.fromkeys(prefix + c for c in candidates):
            for row_id in ref_key_to_rows.get(key, []):
                if row_id not in distinct_rows:
                    distinct_rows.append(row_id)
                    matched_keys.append(key)

        if len(distinct_rows) == 1:
            matched_index[idx] = distinct_rows[0]
        elif len(distinct_rows) > 1:
            ambiguous_rows.append(
                {
                    'street': spine_canon_street.loc[idx],
                    'number': spine.loc[idx, number_column],
                    'matched_keys': matched_keys,
                }
            )

    if ambiguous_rows:
        warnings.warn(
            f'link_address_ranges: {len(ambiguous_rows)} row(s) have an '
            f'address number matching more than one distinct parcel in '
            f'{recipe_id!r} (via a range on either side); a multi-family '
            'range should normally resolve to a single parcel, so these '
            'rows are left unmatched until the ambiguity is resolved. '
            f'Sample: {ambiguous_rows[:5]}',
            stacklevel=2,
        )

    match_series = pd.Series(matched_index, dtype='object')
    for col, out_name in pairs:
        name = f'{out_name}{suffix}' if suffix else out_name
        new_vals = match_series.map(ref[col]).reindex(spine.index)
        _write_prioritized(spine, name, new_vals)

    state.spine = spine
    if state.verbose:
        print(
            f'  Link address ranges: {len(matched_index):,d}/'
            f'{int(attempt_rows.sum()):,d} rows matched {recipe_id} '
            f'({n_range_attempted:,d} range-shaped) ({len(pairs)} columns)'
        )
    return state


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
    superseded_pair_mask = pd.Series(False, index=crosswalk.index)

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
        linked_mask = crosswalk.index.get_level_values('parcel_id').isin(cluster_pids)
        superseded_spine_ids.update(ev['linked_spine_ids'])
        superseded_pair_mask |= linked_mask

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
    state.crosswalks[recipe_id] = pd.concat(
        [crosswalk[~superseded_pair_mask], additions_df]
    ).sort_index()

    overlay = state.overlays.get(recipe_id)
    if overlay is not None:
        overlay_superseded = overlay.index.get_level_values(spine_id_col).isin(
            superseded_spine_ids
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
            kept = existing.loc[
                ~existing.index.get_level_values(spine_id_col).isin(
                    superseded_spine_ids
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


@_register('resolve_overlaps', phase='geometry')
def resolve_overlaps(
    state: HarmonizeState,
    **_params,
) -> HarmonizeState:
    """Resolve remaining geometry overlaps in the spine.

    Calls :func:`~openplaces.geo.polygon.resolve_overlapping_polygons` on
    ``state.spine`` (with ``keep=False``).
    """
    if state.spine is None:
        return state

    state.spine = clean_polygons(state.spine)
    state.spine = resolve_overlapping_polygons(state.spine, keep=False)
    if state.verbose:
        print(f'  Resolve: {len(state.spine):,d} after resolving overlaps')
    if state.timer:
        state.timer.mark('Resolve')
    return state
