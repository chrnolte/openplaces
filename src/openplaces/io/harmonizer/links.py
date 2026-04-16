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

from openplaces.core.attribute_registry import get_agg_func, load_registry
from openplaces.core.schema import AdminId
from openplaces.diagnostics import find_recipes
from openplaces.geo.ids import add_openlocationcode_index
from openplaces.geo.polygon import (
    clean_polygons,
    get_areas,
    overlay_polygons,
    resolve_overlapping_polygons,
)
from openplaces.io.harmonizer import HarmonizeState, _register
from openplaces.io.readers import get_entities
from openplaces.io.transform import make_index_unique, remap

# Columns carried in spine-reference crosswalk tables.
_CROSSWALK_COLS = [
    'footprint_id',
    'parcel_id',
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
) -> HarmonizeState:
    """Load a reference dataset and build a spine ↔ reference crosswalk.

    Populates ``state.references[recipe_id]``,
    ``state.crosswalks[recipe_id]``, ``state.overlays[recipe_id]`` (for
    ``spatial_overlay`` joins), and ``state.reference_types[recipe_id]``.

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
    remap_id : str, optional
        Recipe ID for a column-remap table applied to the reference after
        loading (see :func:`openplaces.io.transform.remap`).
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

    if join == 'spatial_overlay':
        return _link_spatial_overlay(
            state,
            resolved_recipe_id,
            resolved_entity_type,
            thresholds or {},
        )
    elif join == 'spatial_point':
        return _link_spatial_point(
            state,
            resolved_recipe_id,
            resolved_entity_type,
            remap_id,
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
) -> HarmonizeState:
    """Polygon-on-polygon identity overlay; builds spine-reference crosswalk."""
    min_fraction = thresholds.get('min_fraction_of_largest', 1 / 6)
    area_min_m2 = thresholds.get('area_intersection_m2_min', 10)

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

    def _join_nonnull(x):
        parts = [v for v in x if v is not None and pd.notna(v)]
        return ' + '.join(parts) if parts else None

    agg_aliases = {'join_nonnull': _join_nonnull}
    agg_cols: dict = {}
    for col in ref.columns:
        fname = get_agg_func(col)
        if fname:
            agg_cols[col] = agg_aliases.get(fname, fname)
    if agg_cols and 'geo_id' in ref.columns:
        ref_agg = ref.groupby('geo_id').agg(agg_cols)
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
    mask_multi = footprints_on_ref.index.get_level_values('footprint_id').duplicated(
        keep=False
    )

    footprints_single = (
        footprints_on_ref[~mask_multi]
        .reset_index()
        .set_index('footprint_id')[['parcel_id'] + crosswalk_cols]
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
        / footprints_multi.groupby('footprint_id')['area_intersection_m2'].transform(
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
        'footprint_id'
    ).duplicated(keep=False)

    multi_crosswalk_cols = [v for v in _CROSSWALK_COLS if v in footprints_multi.columns]
    footprints_single_from_multi = (
        footprints_multi_trimmed[~mask_still_multi]
        .reset_index()
        .set_index('footprint_id')[['parcel_id'] + multi_crosswalk_cols]
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
            footprints_single.reset_index().set_index(['footprint_id', 'parcel_id']),
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

    geopandas names the right-index column ``'{name}_right'`` when the right
    GeoDataFrame has a named index, and ``'index_right'`` otherwise.  This
    helper handles both conventions.
    """
    candidates = (
        [f'{right_index_name}_right', 'index_right']
        if right_index_name
        else ['index_right']
    )
    for col in candidates:
        if col in gdf.columns:
            return gdf.rename(columns={col: target})
    return gdf


def _link_spatial_point(
    state: HarmonizeState,
    recipe_id: str,
    entity_type: str | None,
    remap_id: str | None,
) -> HarmonizeState:
    """Point-in-polygon join: reference points → spine entities.

    Joins reference points to the current spine geometries to obtain the
    spine entity ID, then to any polygon reference in state (via
    ``state.overlays``) to obtain the polygon reference ID.  Using the live
    spine geometry (post-``resolve_overlaps``) rather than the overlay ensures
    that all points inside an entity are captured regardless of reference
    coverage.
    """
    ref = get_entities(recipe_id, state.admin_id, geom=True)
    if remap_id:
        ref = remap(ref, remap_id)
    if state.verbose:
        print(f'  Load: {len(ref):,d} {entity_type or "point"} ({recipe_id})')

    spine_id_col = state.spine.index.name

    linked = gpd.sjoin(ref, state.spine[['geometry']]).drop(columns='geometry')
    linked = _rename_right_index(linked, state.spine.index.name, spine_id_col)

    sort_cols = [c for c in ['source', 'structure_value'] if c in linked.columns]
    if sort_cols:
        linked = linked.sort_values(
            sort_cols,
            ascending=[True, False][: len(sort_cols)],
        )
    if linked.index.duplicated().any():
        linked = linked[~linked.index.duplicated()].copy()

    overlay_ids = list(state.overlays.keys())
    if overlay_ids:
        poly_recipe_id = overlay_ids[0]
        poly_ref = state.references.get(poly_recipe_id)
        if poly_ref is not None:
            ref_on_poly = gpd.sjoin(ref, poly_ref[['geometry']], how='left').drop(
                columns='geometry'
            )
            poly_ref_id_col = poly_ref.index.name
            ref_on_poly = _rename_right_index(
                ref_on_poly, poly_ref_id_col, poly_ref_id_col
            )
            if ref_on_poly.index.duplicated().any():
                ref_on_poly = ref_on_poly[~ref_on_poly.index.duplicated()].copy()
            if poly_ref_id_col in ref_on_poly.columns:
                linked = linked.join(ref_on_poly[[poly_ref_id_col]])

    if 'source' in linked.columns:
        group_cols = [
            c
            for c in [spine_id_col, poly_ref.index.name if overlay_ids else None]
            if c and c in linked.columns
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
        print(f'  Link (point): {n_linked:,d} points linked')
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
        ref_inferred[['geometry']].reset_index(), name='footprint_id'
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
