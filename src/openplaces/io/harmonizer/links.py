"""
Pipeline steps that create and use relationships between the spine and
reference datasets:
- link_to_reference: load a reference and build a spine ↔ reference crosswalk
- infer_spine_additions: add new spine entries inferred from a reference
- resolve_overlaps: remove remaining geometry overlaps from the spine
"""

from __future__ import annotations

import warnings

import geopandas as gpd
import numpy as np
import pandas as pd

from openplaces.core.attribute_registry import load_registry
from openplaces.core.schema import AdminId, SourceGeometryType
from openplaces.diagnostics import find_recipes
from openplaces.geo.ids import add_openlocationcode_index
from openplaces.geo.polygon import (
    clean_polygons,
    get_areas,
    overlay_polygons,
    resolve_overlapping_polygons,
)
from openplaces.io.aggregate import aggregate_rows
from openplaces.io.harmonizer import HarmonizeState, _register
from openplaces.io.readers import get_entities
from openplaces.io.transform import make_index_unique, remap

# Columns carried in spine-reference crosswalk tables (index levels excluded).
_CROSSWALK_COLS = [
    'area_intersection_m2',
    'iou',
    'area_intersection_m2_inner',
    'fraction_of_largest',
]


@_register('link_to_reference')
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
        such as ``classify_footprint_role``.
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
        )
    elif join == 'spatial_point':
        return _link_spatial_point(
            state,
            resolved_recipe_id,
            resolved_entity_type,
            remap_id,
            thresholds or {},
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
    best_level = -1
    best_row = None
    for _, row in df.iterrows():
        rid_str = row['admin_id']
        if rid_str == '' or admin_str.startswith(rid_str):
            level = rid_str.count('-') + 1 if rid_str else 0
            if level > best_level:
                best_level = level
                best_row = row
    if best_row is None:
        return None
    prefix = f'{best_row["admin_id"]}_' if best_row['admin_id'] else ''
    return (
        f'{prefix}{best_row["entity_type"]}'
        f'-{best_row["source_id"]}-{best_row["version"]}'
    )


def _link_spatial_overlay(
    state: HarmonizeState,
    recipe_id: str,
    entity_type: str | None,
    thresholds: dict,
    aggregation_function=None,
    sort_by: str | None = None,
    list_columns: list[str] | None = None,
) -> HarmonizeState:
    """Polygon-on-polygon identity overlay; builds spine-reference crosswalk."""
    min_fraction = thresholds.get('min_fraction_of_largest', 1 / 6)
    area_min_m2 = thresholds.get('area_intersection_m2_min', 10)
    spine_id_col = state.spine.index.name

    ref_raw = get_entities(recipe_id, state.admin_id, geom=True)
    if state.verbose:
        print(
            f'  Link (overlay): {len(ref_raw):,d} {entity_type or "ref"} ({recipe_id})'
        )

    ref = ref_raw.copy()

    registry = load_registry()
    numeric_attrs = set(registry.index[registry['data_type'].isin(['float', 'int'])])
    for numeric_col in numeric_attrs:
        if numeric_col in ref.columns:
            ref[numeric_col] = pd.to_numeric(ref[numeric_col], errors='coerce')

    ref['purpose_group_combined'] = (
        pd.Categorical(
            ref['purpose_group'].astype(str).fillna('n/a')
            + ' | '
            + ref['purpose_subgroup'].astype(str).fillna('n/a')
        )
        if 'purpose_group' in ref.columns
        else None
    )
    ref['has_duplicate_geometry'] = ref['geo_id'].duplicated(keep=False)
    ref['ha'] = get_areas(ref, 'ha')

    ref_polys = ref[~ref['geo_id'].duplicated()][
        [
            c
            for c in ['geometry', 'geo_id', 'ha', 'has_duplicate_geometry']
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
        ref_polys = ref_polys.join(ref_agg)
    if 'improvement_value' in ref_polys.columns and 'ha' in ref_polys.columns:
        ref_polys['improvement_value_per_ha'] = (
            ref_polys['improvement_value'] / ref_polys['ha']
        )

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
        split_cols + ['geometry']
    ].copy()
    footprints_to_split.insert(0, 'link', 'multi-parcel footprint')

    crosswalk = pd.concat(
        [
            footprints_single.reset_index().set_index([spine_id_col, 'parcel_id']),
            footprints_to_split.drop(columns='geometry'),
        ]
    ).sort_index()

    state.references[recipe_id] = ref_polys
    state.crosswalks[recipe_id] = crosswalk
    state.overlays[recipe_id] = footprints_on_ref
    if entity_type:
        state.reference_types[recipe_id] = entity_type
    return state


def _rename_right_index(
    gdf: pd.DataFrame,
    right_index_name: str | None,
    target: str,
) -> pd.DataFrame:
    """Rename the right-index column added by ``gpd.sjoin`` to *target*.

    geopandas names the right-index column ``'{name}_right'`` when there is a
    column-name conflict, ``'{name}'`` when there is no conflict (geopandas
    1.x), and ``'index_right'`` when the right index has no name.  This helper
    handles all three conventions.
    """
    candidates = (
        [f'{right_index_name}_right', right_index_name, 'index_right']
        if right_index_name
        else ['index_right']
    )
    for col in candidates:
        if col in gdf.columns:
            return gdf.rename(columns={col: target})
    return gdf


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
       tagged ``n_dwelling_units = 1``; downstream :func:`_aggregate_multipoint`
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

    if 'n_dwelling_units' not in ref.columns:
        ref['n_dwelling_units'] = 1.0
    else:
        ref.loc[ref['n_dwelling_units'].isna(), 'n_dwelling_units'] = 1.0

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
    ``n_dwelling_units`` across all matched points per footprint.

    The highest-quality representative row (by existing sort order) is kept as
    the output row; its ``purpose_subgroup`` and ``n_dwelling_units`` are
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
                    rep['n_dwelling_units'] = float(round(total))
            elif _is_addr:
                if 'n_dwelling_units' in group.columns:
                    total = (
                        pd.to_numeric(group['n_dwelling_units'], errors='coerce')
                        .fillna(1.0)
                        .sum()
                    )
                else:
                    # each single_dwelling_point represents one unit
                    total = float(len(group))
                if total > 0:
                    rep['n_dwelling_units'] = float(total)
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


def _link_spatial_point(
    state: HarmonizeState,
    recipe_id: str,
    entity_type: str | None,
    remap_id: str | None,
    thresholds: dict,
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

    Optional pre-processing and post-processing steps controlled via
    *thresholds*:

    ``dedup_addresses`` (bool)
        Run :func:`_dedup_address_points` before spatial linking.  Groups by
        base address (street + housenumber), keeps the unit-less record as the
        spatial representative, and counts unit siblings as ``n_dwelling_units``.
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
    """
    _EA_CRS = 'EPSG:6933'
    proximity_m: float = thresholds.get('proximity_m', 10.0)
    far_proximity_m: float = thresholds.get('far_proximity_m', 100.0)
    unbounded_m: float = thresholds.get('unbounded_proximity_m', 0.0)
    dedup_addresses: bool = bool(thresholds.get('dedup_addresses', False))
    use_size_limit: bool = bool(thresholds.get('use_size_limit', False))
    aggregate_mp: bool = bool(thresholds.get('aggregate_multipoint', False))

    ref = get_entities(recipe_id, state.admin_id, geom=True)
    if remap_id:
        ref = remap(ref, remap_id)
    if state.verbose:
        print(f'  Load: {len(ref):,d} {entity_type or "point"} ({recipe_id})')

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

    # Pass 1 — within (Lochhead Step 2)
    within = gpd.sjoin(ref, state.spine[['geometry']]).drop(columns='geometry')
    within = _rename_right_index(within, spine_id_col, spine_id_col)
    attributed_idx: set = set(within.index)
    n_pass1 = len(attributed_idx)

    # Build per-class size limits from Pass 1 for use in Passes 2–3
    size_limit_dict: dict[str, tuple[float, float]] = {}
    if use_size_limit:
        size_limit_dict = _build_size_limit_dict(within, state.spine, spine_id_col)

    # Pass 2 — inner proximity, default 10 m (Step 5)
    if proximity_m > 0:
        unlinked = ref[~ref.index.isin(attributed_idx)]
        if not unlinked.empty:
            spine_proj = state.spine[['geometry']].to_crs(_EA_CRS)
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
            pts_parcel = _rename_right_index(pts_parcel, poly_ref_id_col, '_pt_parcel')

            fp_parcel = (
                state.crosswalks[overlay_ids[0]]
                .reset_index()[[spine_id_col, poly_ref_id_col]]
                .drop_duplicates(spine_id_col)
                .set_index(spine_id_col)[poly_ref_id_col]
                .rename('_fp_parcel')
            )

            spine_proj = state.spine[['geometry']].to_crs(_EA_CRS)
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
            spine_proj_p4 = state.spine[['geometry']].to_crs(_EA_CRS)
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
        _fp_has_same = _is_same_parcel.groupby(linked[spine_id_col]).transform('any')
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
        linked = _aggregate_multipoint(linked, spine_id_col, sgt, verbose=state.verbose)

    # Attach polygon reference ID (e.g. parcel_id) to each linked point
    if poly_ref is None and overlay_ids:
        poly_ref = state.references.get(overlay_ids[0])
    if poly_ref is not None:
        poly_ref_id_col = poly_ref.index.name
        ref_on_poly = gpd.sjoin(ref, poly_ref[['geometry']], how='left').drop(
            columns='geometry'
        )
        ref_on_poly = _rename_right_index(ref_on_poly, poly_ref_id_col, poly_ref_id_col)
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
            first_source = linked.groupby(group_cols, sort=False)['source'].transform(
                'first'
            )
            mask_to_drop = (
                mask_dup
                & linked['source'].isin(low_quality)
                & first_source.eq('Parcel')
            )
            linked = linked[~mask_to_drop]

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
    if state.timer:
        state.timer.mark('Link (point)')

    state.references[recipe_id] = ref
    state.crosswalks[recipe_id] = linked
    if entity_type:
        state.reference_types[recipe_id] = entity_type
    return state


@_register('infer_spine_additions')
def infer_spine_additions(
    state: HarmonizeState,
    entity_type: str | None = None,
    recipe_id: str | None = None,
    thresholds: dict | None = None,
) -> HarmonizeState:
    """Add spine entries inferred from a reference crosswalk.

    For each reference polygon that has no existing spine coverage and
    exceeds the improvement-value threshold, creates a new spine geometry
    equal to the reference polygon geometry.

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
        per purpose group to be eligible for inference.
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

    ref_stat_cols = [
        c
        for c in [
            'purpose_group',
            'improvement_value_per_ha',
            'has_duplicate_geometry',
        ]
        if c in ref_polys.columns
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

    n_footprints_per_group = (
        (
            (
                footprint_ref_data['purpose_group'].value_counts()
                / ref_polys['purpose_group'].value_counts()
            )
            .fillna(0)
            .rename('n_footprints_mean')
        )
        if 'purpose_group' in footprint_ref_data.columns
        else pd.Series(dtype=float)
    )

    imp_val_q_col = f'improvement_value_per_ha_q{q}'
    has_imp = 'improvement_value_per_ha' in footprint_ref_data.columns
    if has_imp and 'purpose_group' in footprint_ref_data.columns:
        imp_val_q_by_group = (
            footprint_ref_data.sample(frac=1)
            .reset_index()
            .drop_duplicates('parcel_id', keep=False)
            .groupby('purpose_group')['improvement_value_per_ha']
            .quantile((q, 0.5))
            .unstack()
            .rename(
                columns={
                    q: imp_val_q_col,
                    0.5: 'improvement_value_per_ha_median',
                }
            )
        )
    else:
        imp_val_q_by_group = None

    parcel_ids_with_footprint = crosswalk.index.get_level_values('parcel_id').unique()
    mask_without_footprint = ~ref_polys.index.isin(parcel_ids_with_footprint)

    candidate_cols = [
        c
        for c in [
            'purpose_group',
            'improvement_value',
            'improvement_value_per_ha',
            'has_duplicate_geometry',
            'geometry',
        ]
        if c in ref_polys.columns
    ]
    ref_candidates = ref_polys[mask_without_footprint][candidate_cols].copy()

    if imp_val_q_by_group is not None:
        ref_candidates = ref_candidates.join(imp_val_q_by_group, on='purpose_group')
    if not n_footprints_per_group.empty:
        ref_candidates = ref_candidates.join(n_footprints_per_group, on='purpose_group')

    imp_val_floor = 0.0
    if has_imp:
        imp_val_floor = (
            ref_polys[ref_polys.index.isin(parcel_ids_with_footprint)]
            .query('improvement_value_per_ha > 0')['improvement_value_per_ha']
            .quantile(q)
        )
    if state.verbose:
        print(f'  Infer: footprint if improvement_value > $ {imp_val_floor:,.0f}/ha')

    n_col_ok = (
        ref_candidates['n_footprints_mean'].gt(n_min)
        if 'n_footprints_mean' in ref_candidates.columns
        else pd.Series(True, index=ref_candidates.index)
    )
    if imp_val_q_col in ref_candidates.columns and has_imp:
        val_col_ok = ref_candidates['improvement_value_per_ha'].gt(
            ref_candidates[imp_val_q_col].div(2).clip(lower=imp_val_floor)
        )
    else:
        val_col_ok = pd.Series(True, index=ref_candidates.index)

    mask_inferred = n_col_ok & val_col_ok
    ref_inferred = ref_candidates[mask_inferred]

    footprints_from_ref = add_openlocationcode_index(
        ref_inferred[['geometry']].reset_index(), name=state.spine.index.name
    )
    footprints_from_ref = footprints_from_ref[
        ~footprints_from_ref.index.isin(state.spine.index)
    ]
    _base = recipe_id.rsplit('_', 1)[-1]
    _parts = _base.split('-', 2)
    _source_id = _parts[1] if len(_parts) > 1 else _base
    _et = entity_type or (_parts[0] if _parts else 'reference')
    footprints_from_ref['source'] = f'{_et}.{_source_id}'

    state.spine = pd.concat(
        [state.spine, footprints_from_ref[['geometry', 'source']]]
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


@_register('resolve_overlaps')
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
