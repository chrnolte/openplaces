"""
Pipeline steps that attach attributes from reference datasets to the spine:
  - reconcile_attributes: aggregate columns from established crosswalks
  - infer_attributes: compute derived columns (area, value ratios, etc.)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from openplaces.core.attribute_registry import get_categorical_attrs
from openplaces.geo.polygon import get_areas
from openplaces.io.harmonizer import HarmonizeState, _register


def _resolve_suffix(
    crosswalk_key: str,
    entity_type: str | None,
    state,
    default: str = '_ref',
) -> str:
    """Return the column suffix for reference attributes.

    When the reference entity type matches the spine entity type (e.g.,
    both are ``'building'``), the suffix is derived from the source_id in
    the recipe_id (e.g., ``'_nsi'``).  Otherwise the entity type is used
    (e.g., ``'_parcel'``).
    """
    spine_entity = state.recipe.get('entity')
    spine_entity_type = (
        str(spine_entity.entity_type) if spine_entity is not None else None
    )
    if entity_type and spine_entity_type and entity_type == spine_entity_type:
        base = crosswalk_key.rsplit('_', 1)[-1]
        parts = base.split('-', 2)
        source_id = parts[1] if len(parts) > 1 else base
        return f'_{source_id}'
    return f'_{entity_type}' if entity_type else default


def _point_suffix(
    crosswalk_key: str,
    entity_type: str | None = None,
    col: str | None = None,
) -> str:
    """Compose suffix ``_{entity_type}_{source_id}`` for point references.

    When *col* is supplied and *entity_type* already appears in the column
    name (e.g. ``'dwelling'`` in ``'n_dwelling_units'``), the entity_type
    part is dropped to avoid redundancy (``'_overture'`` instead of
    ``'_dwelling_overture'``).

    Examples: ``US_building-nsi-2022``, ``building`` -> ``_building_nsi``;
    ``dwelling-overture-2025``, ``dwelling`` -> ``_dwelling_overture``;
    ``dwelling-overture-2025``, ``dwelling``, col=``n_dwelling_units``
    -> ``_overture``.
    """
    base = crosswalk_key.rsplit('_', 1)[-1]
    parts = base.split('-', 2)
    source_id = parts[1] if len(parts) > 1 else base
    if entity_type:
        if col and entity_type in col:
            return f'_{source_id}'
        return f'_{entity_type}_{source_id}'
    return f'_{source_id}'


# Occupancy-class → expected dwelling-unit count (Lochhead et al. 2026, Table 3).
# Used as a fallback when n_dwelling_units is missing from parcel data.
_OCC_UNITS: dict[str, float] = {
    'Single Family': 1.0,
    'Manufactured': 1.0,
    'Multi-Family (2 units)': 2.0,
    'Multi-Family (3-4 units)': 3.5,
    'Multi-Family (5-9 units)': 7.0,
    'Multi-Family (10-19 units)': 14.5,
    'Multi-Family (20-50 units)': 35.0,
    'Multi-Family (50+ units)': 51.0,
}


def reverse_occ_units(total_units: float) -> str:
    """Re-classify a summed unit count to the nearest *purpose_subgroup* label.

    Mirrors the ``map_to_units`` logic from Lochhead et al. (2026).  Used when
    multiple point sources (e.g. NSI) link to the same footprint and their unit
    counts must be aggregated and re-classified.
    """
    n = round(float(total_units))
    if n <= 1:
        return 'Single Family'
    if n == 2:
        return 'Multi-Family (2 units)'
    if n <= 4:
        return 'Multi-Family (3-4 units)'
    if n <= 9:
        return 'Multi-Family (5-9 units)'
    if n <= 19:
        return 'Multi-Family (10-19 units)'
    if n <= 50:
        return 'Multi-Family (20-50 units)'
    return 'Multi-Family (50+ units)'


# Columns carried from a polygon reference (e.g. parcel) to the spine.
_POLYGON_REF_COLS = [
    'purpose_group',
    'purpose_subgroup',
    'purpose_group_combined',
    'improvement_value',
    'land_value',
    'year_built',
    'address',
]

# Columns carried from a point reference (e.g. NSI buildings) to the spine.
_POINT_REF_COLS = [
    'purpose_subgroup',
    'openplaces_group',
    'structure_value',
    'year_built_block_median',
    'source',
    'area_sqft',
    'n_stories',
    'n_dwelling_units',
    'address_street',
    'address_number',
    'postal_code',
    'city',
]


def _collect_dwelling_linked(state: HarmonizeState) -> set:
    """Return spine IDs linked to a dwelling point (single_dwelling_point source).

    Used by :func:`_attribute_polygon_reference` to implement Lochhead et al.
    (2026) Table 4: when distributing parcel attributes across multiple
    footprints on the same parcel, restrict attribution to dwelling-linked
    footprints when any footprint in the parcel has dwelling evidence.
    """
    from openplaces.core.schema import SourceGeometryType as _SGT

    if state.spine is None:
        return set()
    spine_id_col = state.spine.index.name
    ids: set = set()
    for rid, sgt in state.source_geometry_types.items():
        if sgt != _SGT.single_dwelling_point:
            continue
        cw = state.crosswalks.get(rid)
        if cw is not None and spine_id_col in cw.columns:
            ids.update(cw[spine_id_col].dropna().unique())
    return ids


@_register('reconcile_attributes')
def reconcile_attributes(
    state: HarmonizeState,
    sources: list[dict] | None = None,
    priority: dict[str, list[str]] | None = None,
) -> HarmonizeState:
    """Aggregate reference attributes to the spine via established crosswalks.

    For each source in *sources*, looks up the crosswalk in
    ``state.crosswalks`` (resolved via ``recipe_id`` or ``entity_type``) and
    aggregates the requested columns to the spine.

    Parameters
    ----------
    sources : list of dict
        Each dict describes one reference source and may contain:

        ``recipe_id`` (str, optional)
            Explicit crosswalk key in ``state.crosswalks``.
        ``entity_type`` (str, optional)
            Selects all matching crosswalks via ``state.reference_types``;
            used when ``recipe_id`` is absent.
        ``columns`` (list of str, optional)
            Columns to aggregate.  Defaults to all available columns from
            the corresponding default column list.
    priority : dict of {feature: [source_suffix, ...]}, optional
        Between-source priority for specific features (Lochhead et al. 2026,
        Step C).  Each key is a bare feature name (e.g. ``'year_built'``);
        the value is an ordered list of source suffixes (without leading ``_``)
        to try in order.  The first non-null suffixed column wins.

        Example::

            priority:
              purpose_subgroup: [nsi, parcel]
              year_built: [parcel, nsi]
    """
    if state.spine is None or not sources:
        return state

    dwelling_linked_ids = _collect_dwelling_linked(state)

    for src_cfg in sources:
        entity_type = src_cfg.get('entity_type')
        recipe_id = src_cfg.get('recipe_id')
        columns = src_cfg.get('columns')

        if recipe_id is not None:
            crosswalk_keys = [recipe_id] if recipe_id in state.crosswalks else []
        elif entity_type is not None:
            crosswalk_keys = list(state.get_crosswalks_by_type(entity_type).keys())
        else:
            warnings.warn(
                'reconcile_attributes: source entry has neither '
                "'recipe_id' nor 'entity_type'; skipping."
            )
            continue

        src_thresholds: dict = src_cfg.get('thresholds') or {}

        for crosswalk_key in crosswalk_keys:
            ref_entity_type = state.reference_types.get(crosswalk_key, entity_type)
            crosswalk = state.crosswalks.get(crosswalk_key)
            if crosswalk is None:
                warnings.warn(
                    f'reconcile_attributes: crosswalk for '
                    f'{crosswalk_key!r} not in state; skipping.'
                )
                continue

            if isinstance(crosswalk.index, pd.MultiIndex):
                ref_polys = state.references.get(crosswalk_key)
                state = _attribute_polygon_reference(
                    state,
                    crosswalk_key,
                    ref_entity_type,
                    ref_polys,
                    columns,
                    thresholds=src_thresholds,
                    dwelling_linked_ids=dwelling_linked_ids,
                )
            else:
                state = _attribute_point_reference(
                    state,
                    crosswalk_key,
                    ref_entity_type,
                    crosswalk,
                    columns,
                )

    if priority:
        for feature, source_order in priority.items():
            cols = [
                f'{feature}_{s}'
                for s in source_order
                if f'{feature}_{s}' in state.spine.columns
            ]
            if cols:
                state.spine[feature] = state.spine[cols].bfill(axis=1).iloc[:, 0]

    if state.timer:
        state.timer.mark('Attribute')
    return state


def _attribute_polygon_reference(
    state: HarmonizeState,
    crosswalk_key: str,
    entity_type: str | None,
    ref_polys,
    columns: list[str] | None,
    thresholds: dict | None = None,
    dwelling_linked_ids: set | None = None,
) -> HarmonizeState:
    """Attribute a polygon reference (e.g. parcels) to the spine.

    Parameters
    ----------
    thresholds : dict, optional
        ``use_volume_weight`` (bool, default False) — weight ``improvement_value``
        and ``n_dwelling_units`` distribution by ``area × n_stories`` instead of
        area alone (Lochhead et al. 2026).  Requires a ``n_stories*`` column in
        the spine (populated by a prior NSI ``link_to_reference`` step).
    dwelling_linked_ids : set, optional
        Spine IDs linked to a dwelling point
        (:class:`~openplaces.core.schema.SourceGeometryType.single_dwelling_point`
        source).  When provided, parcel values (``improvement_value``,
        ``land_value``, ``n_dwelling_units``) are distributed only to
        dwelling-linked footprints within parcels that have dwelling evidence
        (Lochhead et al. 2026, Table 4, Cases 1–2 vs. Case 3).
    """
    spine = state.spine
    spine_id_col = spine.index.name
    overlay = state.overlays.get(crosswalk_key)
    if overlay is None or ref_polys is None:
        return state

    thresholds = thresholds or {}
    use_volume_weight: bool = bool(thresholds.get('use_volume_weight', False))

    suffix = _resolve_suffix(crosswalk_key, entity_type, state, default='_ref')
    spine_entity = state.recipe.get('entity')
    spine_entity_type = (
        str(spine_entity.entity_type) if spine_entity is not None else 'entity'
    )
    n_other_col = f'n_other_{spine_entity_type}s{suffix}'
    avail_cols = [c for c in (columns or _POLYGON_REF_COLS) if c in ref_polys.columns]

    overlay = overlay.copy()
    overlay['area_fraction'] = overlay['area_intersection_m2'] / overlay.groupby(
        'parcel_id'
    )['area_intersection_m2'].transform('sum')

    if use_volume_weight:
        stories_col = next(
            (c for c in spine.columns if c.startswith('n_stories')), None
        )
        if stories_col is not None:
            fp_idx = overlay.index.get_level_values(spine_id_col)
            n_eff = (
                spine[stories_col].reindex(fp_idx).fillna(1.0).clip(lower=1.0).values
            )
            overlay['_w'] = overlay['area_intersection_m2'] * n_eff
            overlay['area_fraction'] = overlay['_w'] / overlay.groupby('parcel_id')[
                '_w'
            ].transform('sum')
            overlay = overlay.drop(columns='_w')

    mask_has_ref = overlay.index.get_level_values('parcel_id').notnull()

    # Prefer dwelling-linked footprints when distributing parcel values across
    # multiple footprints on the same parcel (Lochhead et al. 2026, Table 4).
    # For parcels where ≥1 footprint has dwelling evidence, zero out
    # area_fraction for footprints WITHOUT evidence so they receive no parcel
    # value (improvement_value, n_dwelling_units). Suppressed IDs are tracked
    # to apply the same rule to land_value below.
    suppressed_ids: set = set()
    if dwelling_linked_ids:
        overlay_ref = overlay[mask_has_ref]
        fp_ids_ref = overlay_ref.index.get_level_values(spine_id_col)
        has_dwelling = pd.Series(
            [fid in dwelling_linked_ids for fid in fp_ids_ref],
            index=overlay_ref.index,
        )
        parcel_has_dwelling = has_dwelling.groupby('parcel_id').transform('any')
        mask_suppress = parcel_has_dwelling & ~has_dwelling
        if mask_suppress.any():
            suppressed_ids = set(fp_ids_ref[mask_suppress])
            new_frac = overlay_ref['area_fraction'].copy()
            new_frac.loc[mask_suppress] = 0.0
            frac_by_parcel = new_frac.groupby('parcel_id').transform('sum')
            new_frac = (new_frac / frac_by_parcel.replace(0, float('nan'))).fillna(0.0)
            overlay.loc[mask_has_ref, 'area_fraction'] = new_frac

    footprint_ref_attrs = (
        overlay[mask_has_ref][['area_intersection_m2', 'area_fraction']]
        .reset_index()
        .set_index('parcel_id')
        .join(ref_polys[avail_cols])
        .reset_index()
        .set_index(spine_id_col)
    )

    spine['n_footprint_parcel'] = (
        footprint_ref_attrs.groupby(spine_id_col)
        .size()
        .reindex(spine.index, fill_value=0)
    )

    if 'purpose_group_combined' in footprint_ref_attrs.columns:
        footprint_purpose_group_areas = (
            footprint_ref_attrs.groupby([spine_id_col, 'purpose_group_combined'])[
                'area_intersection_m2'
            ]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        spine[f'purpose_group_combined{suffix}'] = (
            footprint_purpose_group_areas.drop_duplicates(spine_id_col).set_index(
                spine_id_col
            )['purpose_group_combined']
        )
        spine[f'purpose_group_combined{suffix}_all'] = (
            footprint_purpose_group_areas.groupby(spine_id_col)[
                'purpose_group_combined'
            ].apply(' + '.join)
        )

    footprint_parcel_areas = footprint_ref_attrs.reset_index()[
        [spine_id_col, 'parcel_id', 'area_intersection_m2']
    ]
    parcel_area_stats = footprint_parcel_areas.groupby('parcel_id')[
        'area_intersection_m2'
    ].agg(max_area='max', n_fp='count')
    footprint_parcel_areas = footprint_parcel_areas.join(
        parcel_area_stats, on='parcel_id'
    )
    footprint_parcel_areas['_n_col'] = footprint_parcel_areas['n_fp'] - 1
    primary_footprints = (
        footprint_parcel_areas.sort_values('area_intersection_m2', ascending=False)
        .drop_duplicates(spine_id_col)
        .set_index(spine_id_col)
    )
    spine[n_other_col] = primary_footprints['_n_col'].reindex(spine.index)

    if 'area_spine_m2' in overlay.columns:
        identified_area = (
            overlay.loc[mask_has_ref, 'area_intersection_m2']
            .groupby(level=spine_id_col)
            .sum()
        )
        spine_area = (
            overlay['area_spine_m2']
            .groupby(level=spine_id_col)
            .first()
            .replace(0, float('nan'))
        )
        spine[f'overlap_fraction{suffix}'] = (identified_area / spine_area).reindex(
            spine.index, fill_value=0.0
        )
    else:
        spine[f'overlap_fraction{suffix}'] = np.nan

    if (
        'address' in footprint_ref_attrs.columns
        or 'land_value' in footprint_ref_attrs.columns
    ):
        footprint_primary_row = (
            footprint_ref_attrs.reset_index()
            .sort_values('area_intersection_m2', ascending=False)
            .drop_duplicates(spine_id_col)
            .set_index(spine_id_col)
        )
        if 'address' in footprint_ref_attrs.columns:
            spine[f'address{suffix}'] = footprint_primary_row['address'].reindex(
                spine.index
            )
        if 'land_value' in footprint_ref_attrs.columns:
            if 'footprint_role' in spine.columns:
                is_principal = spine['footprint_role'].eq('primary')
            else:
                is_principal = spine[n_other_col].eq(0)
            spine[f'land_value{suffix}'] = (
                footprint_primary_row['land_value']
                .reindex(spine.index)
                .where(is_principal)
            )

    # Suppress land_value for footprints that have no dwelling evidence in
    # parcels where other footprints do (Lochhead et al. 2026, Table 4).
    if suppressed_ids and f'land_value{suffix}' in spine.columns:
        spine.loc[spine.index.isin(suppressed_ids), f'land_value{suffix}'] = np.nan

    footprint_ref_attrs = footprint_ref_attrs.copy()

    # Zero out improvement_value and n_dwelling_units for secondary footprints so
    # that financial value is only allocated to primary structures on the parcel.
    _value_cols = [
        c
        for c in ('improvement_value', 'n_dwelling_units')
        if c in footprint_ref_attrs.columns
    ]
    if _value_cols and 'footprint_role' in spine.columns:
        secondary_ids = spine.index[spine['footprint_role'].eq('secondary')]
        footprint_ref_attrs.loc[
            footprint_ref_attrs.index.isin(secondary_ids), _value_cols
        ] = 0.0

    if 'improvement_value' in footprint_ref_attrs.columns:
        footprint_ref_attrs['improvement_value'] = (
            footprint_ref_attrs['improvement_value']
            .mul(footprint_ref_attrs['area_fraction'])
            .round(2)
        )

    if 'year_built' in footprint_ref_attrs.columns:
        footprint_ref_attrs['year_built'] = footprint_ref_attrs['year_built'].replace(
            0, np.nan
        )

    if 'n_dwelling_units' in footprint_ref_attrs.columns:
        footprint_ref_attrs['n_dwelling_units'] = (
            footprint_ref_attrs['n_dwelling_units']
            .mul(footprint_ref_attrs['area_fraction'])
            .round(2)
        )

    numeric_agg: dict[str, str] = {}
    if 'improvement_value' in footprint_ref_attrs.columns:
        numeric_agg['improvement_value'] = 'sum'
    if 'n_dwelling_units' in footprint_ref_attrs.columns:
        numeric_agg['n_dwelling_units'] = 'sum'
    if 'year_built' in footprint_ref_attrs.columns:
        numeric_agg['year_built'] = 'mean'
    if numeric_agg:
        rename_map = {
            'improvement_value': f'improvement_value{suffix}',
            'n_dwelling_units': f'n_dwelling_units{suffix}',
            'year_built': f'year_built{suffix}',
        }
        spine = spine.join(
            footprint_ref_attrs.groupby(spine_id_col)
            .agg(numeric_agg)
            .rename(columns={k: v for k, v in rename_map.items() if k in numeric_agg})
        )

    footprints_from_ref = state.metadata.get(f'inferred_from_{crosswalk_key}')
    mask_ref_src = spine['source'].str.contains(r'\.', regex=True, na=False)
    if (
        mask_ref_src.any()
        and footprints_from_ref is not None
        and 'parcel_id' in footprints_from_ref.columns
    ):
        ref_attr_cols = [
            c for c in (columns or _POLYGON_REF_COLS) if c in ref_polys.columns
        ]
        inferred_ref_attrs = (
            footprints_from_ref[['parcel_id']]
            .join(ref_polys[ref_attr_cols], on='parcel_id')
            .rename(
                columns={
                    'purpose_group_combined': f'purpose_group_combined{suffix}',
                    'improvement_value': f'improvement_value{suffix}',
                    'n_dwelling_units': f'n_dwelling_units{suffix}',
                    'land_value': f'land_value{suffix}',
                    'year_built': f'year_built{suffix}',
                    'address': f'address{suffix}',
                }
            )
        )
        year_built_col = f'year_built{suffix}'
        if year_built_col in inferred_ref_attrs.columns:
            inferred_ref_attrs[year_built_col] = inferred_ref_attrs[
                year_built_col
            ].replace(0, np.nan)
        inferred_ref_attrs['n_footprint_parcel'] = 1
        inferred_ref_attrs[n_other_col] = 0
        inferred_ref_attrs[f'overlap_fraction{suffix}'] = 1.0
        overlap_cols = [c for c in spine.columns if c in inferred_ref_attrs.columns]
        spine.loc[mask_ref_src, overlap_cols] = inferred_ref_attrs[overlap_cols]

    state.spine = spine
    return state


def _attribute_point_reference(
    state: HarmonizeState,
    crosswalk_key: str,
    entity_type: str | None,
    crosswalk: pd.DataFrame,
    columns: list[str] | None,
) -> HarmonizeState:
    """Attribute a point reference (e.g. NSI) to the spine."""
    spine = state.spine
    suffix = _point_suffix(crosswalk_key, entity_type)

    avail_cols = columns or [c for c in _POINT_REF_COLS if c in crosswalk.columns]
    renamed: dict[str, str] = {
        c: f'{c}{_point_suffix(crosswalk_key, entity_type, col=c)}'
        for c in avail_cols
        if c in crosswalk.columns
    }

    spine_id_col = state.spine.index.name
    if spine_id_col not in crosswalk.columns:
        warnings.warn(
            f'_attribute_point_reference: {crosswalk_key!r} crosswalk has no '
            f"'{spine_id_col}' column; skipping."
        )
        return state

    group_sizes = crosswalk.groupby(spine_id_col).size()
    if group_sizes.max() > 1:
        spine[f'n_point{suffix}'] = group_sizes.reindex(spine.index, fill_value=0)

    purpose_group_col = next(
        (c for c in avail_cols if 'purpose_subgroup' in c and c in crosswalk.columns),
        None,
    )
    structure_value_col = next(
        (c for c in avail_cols if 'structure_value' in c and c in crosswalk.columns),
        None,
    )
    if purpose_group_col:
        purpose_group_out = renamed.get(purpose_group_col, purpose_group_col)
        if structure_value_col:
            footprint_purpose_group_areas = (
                crosswalk.groupby([spine_id_col, purpose_group_col])[
                    structure_value_col
                ]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )
            spine[f'{purpose_group_out}_all'] = footprint_purpose_group_areas.groupby(
                spine_id_col
            )[purpose_group_col].apply(' + '.join)
        else:
            footprint_purpose_group_areas = (
                crosswalk.groupby(spine_id_col)[purpose_group_col].first().reset_index()
            )
        spine[purpose_group_out] = footprint_purpose_group_areas.drop_duplicates(
            spine_id_col
        ).set_index(spine_id_col)[purpose_group_col]

    openplaces_group_col = next(
        (c for c in avail_cols if 'openplaces_group' in c and c in crosswalk.columns),
        None,
    )
    if openplaces_group_col and structure_value_col:
        openplaces_group_out = renamed.get(openplaces_group_col, openplaces_group_col)
        footprint_openplaces_group_areas = (
            crosswalk.groupby([spine_id_col, openplaces_group_col])[structure_value_col]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        spine[openplaces_group_out] = footprint_openplaces_group_areas.drop_duplicates(
            spine_id_col
        ).set_index(spine_id_col)[openplaces_group_col]
        spine[f'{openplaces_group_out}_all'] = footprint_openplaces_group_areas.groupby(
            spine_id_col
        )[openplaces_group_col].apply(' + '.join)

    numeric_agg: dict[str, str] = {}
    if structure_value_col:
        numeric_agg[structure_value_col] = 'sum'
    year_built_col = next(
        (c for c in avail_cols if 'year_built' in c and c in crosswalk.columns),
        None,
    )
    if year_built_col:
        numeric_agg[year_built_col] = 'mean'
    if 'n_dwelling_units' in avail_cols and 'n_dwelling_units' in crosswalk.columns:
        numeric_agg['n_dwelling_units'] = 'sum'
    if numeric_agg:
        spine = spine.join(
            crosswalk.groupby(spine_id_col).agg(numeric_agg).rename(columns=renamed)
        )

    handled = set(numeric_agg) | {'purpose_subgroup', 'openplaces_group'}
    str_cols = [
        c
        for c in avail_cols
        if c in crosswalk.columns
        and c not in handled
        and not pd.api.types.is_numeric_dtype(crosswalk[c])
        and not pd.api.types.is_bool_dtype(crosswalk[c])
        and not pd.api.types.is_datetime64_any_dtype(crosswalk[c])
    ]
    if str_cols:

        def _join_unique(s: pd.Series) -> str | None:
            seen = dict.fromkeys(str(v) for v in s if pd.notna(v))
            return '; '.join(seen) if seen else None

        spine = spine.join(
            crosswalk.groupby(spine_id_col)[str_cols]
            .agg(_join_unique)
            .rename(columns={c: renamed[c] for c in str_cols if c in renamed})
        )

    if openplaces_group_col:
        poly_overlay_keys = list(state.overlays.keys())
        poly_refs = {
            rid: state.references[rid]
            for rid in poly_overlay_keys
            if rid in state.references
        }
        if poly_refs:
            poly_recipe_id = next(iter(poly_refs))
            poly_ref = poly_refs[poly_recipe_id]
            poly_suffix = _resolve_suffix(
                poly_recipe_id,
                state.reference_types.get(poly_recipe_id),
                state,
            )
            purpose_group_spine_col = next(
                (c for c in spine.columns if c.startswith('purpose_group_combined')),
                None,
            )
            openplaces_group_out = renamed.get(
                openplaces_group_col, openplaces_group_col
            )
            poly_attr_cols = [c for c in _POLYGON_REF_COLS if c in poly_ref.columns]
            poly_ref_id_col = poly_ref.index.name
            if (
                purpose_group_spine_col
                and openplaces_group_col in crosswalk.columns
                and poly_attr_cols
            ):
                points_with_poly_ref = crosswalk.join(
                    poly_ref[poly_attr_cols].rename(
                        columns={c: f'{c}{poly_suffix}' for c in poly_attr_cols}
                    ),
                    on=poly_ref_id_col,
                    lsuffix='_point',
                )
                purpose_group_join_col = f'purpose_group_combined{poly_suffix}'
                if (
                    purpose_group_join_col in points_with_poly_ref.columns
                    and openplaces_group_col in points_with_poly_ref.columns
                ):
                    counts = points_with_poly_ref[
                        [purpose_group_join_col, openplaces_group_col]
                    ].value_counts()
                    fractions = (
                        counts.div(counts.groupby(purpose_group_join_col).sum())
                        .rename('fraction')
                        .round(3)
                    )
                    inferred_groups = (
                        pd.concat([counts, fractions], axis=1)
                        .reset_index()
                        .sort_values(
                            [purpose_group_join_col, 'count'],
                            ascending=[True, False],
                        )
                        .drop_duplicates(purpose_group_join_col)
                        .set_index(purpose_group_join_col)
                    )
                    inferred_col = f'openplaces_group{poly_suffix}'
                    spine = spine.join(
                        inferred_groups[openplaces_group_col].rename(inferred_col),
                        on=purpose_group_spine_col,
                    )

    state.spine = spine
    return state


@_register('classify_footprint_role')
def classify_footprint_role(
    state: HarmonizeState,
    entity_type: str | None = None,
    thresholds: dict | None = None,
    **_params,
) -> HarmonizeState:
    """Classify spine footprints as ``'primary'``, ``'secondary'``, or ``'unknown'``.

    Uses dwelling-point and building-point evidence to assign roles within each
    parcel (Lochhead et al. 2026, Table 4):

    1. If any footprint on the parcel has dwelling-point evidence
       (``SourceGeometryType.single_dwelling_point``), those footprints are
       ``'primary'``; all others on the same parcel are ``'secondary'``.
    2. Else if any footprint has single-building-point evidence
       (``SourceGeometryType.single_building_point``, e.g. NSI), those are
       ``'primary'``; all others are ``'secondary'``.
    3. If no footprint on a multi-footprint parcel has evidence, all are
       ``'secondary'``.
    4. Footprints that are the sole geometry on their parcel are always
       ``'primary'``.
    5. Footprints not linked to any parcel are ``'unknown'``, unless they
       carry dwelling-point evidence — those are promoted to ``'primary'``.

    Parameters
    ----------
    entity_type : str, optional
        Entity type used to locate the parcel crosswalk in ``state.crosswalks``.
        Defaults to ``'parcel'``.
    thresholds : dict, optional
        Not currently used; retained for recipe compatibility.
    """
    from openplaces.core.schema import SourceGeometryType as _SGT

    if state.spine is None:
        return state

    spine_id_col = state.spine.index.name
    parcel_type = entity_type or 'parcel'

    parcel_crosswalks = state.get_crosswalks_by_type(parcel_type)
    if not parcel_crosswalks:
        if state.verbose:
            print(f'  classify_footprint_role: no {parcel_type} crosswalk; skipping.')
        return state

    parcel_recipe_id = next(iter(parcel_crosswalks))
    crosswalk = parcel_crosswalks[parcel_recipe_id]

    fp_parcel = crosswalk.reset_index()[[spine_id_col, 'parcel_id']].dropna(
        subset=['parcel_id']
    )
    fp_parcel = fp_parcel[fp_parcel[spine_id_col].isin(state.spine.index)]

    parcel_fp_count = fp_parcel.groupby('parcel_id')[spine_id_col].transform('count')
    multi_fp = fp_parcel[parcel_fp_count > 1].copy()

    # Seed: parcel-linked → 'primary', unlinked → 'unknown'.
    # Single-footprint parcels keep 'primary' and never enter the loop below.
    role = pd.Series('unknown', index=state.spine.index, dtype=object)
    role.loc[role.index.isin(set(fp_parcel[spine_id_col]))] = 'primary'

    if multi_fp.empty:
        state.spine['footprint_role'] = pd.Categorical(
            role, categories=['primary', 'secondary', 'unknown']
        )
        return state

    # Collect point-evidence sets.
    address_evidence: set = set()
    building_point_evidence: set = set()
    for rid, sgt in state.source_geometry_types.items():
        if sgt not in {_SGT.single_dwelling_point, _SGT.single_building_point}:
            continue
        cw = state.crosswalks.get(rid)
        if cw is None:
            continue
        linked_ids = (
            cw[spine_id_col].dropna().unique()
            if spine_id_col in cw.columns
            else cw.index[cw.index.isin(state.spine.index)]
        )
        if sgt == _SGT.single_dwelling_point:
            address_evidence.update(linked_ids)
        elif sgt == _SGT.single_building_point:
            building_point_evidence.update(linked_ids)

    for parcel_id, group in multi_fp.groupby('parcel_id'):
        fp_ids = set(group[spine_id_col])
        has_addr = fp_ids & address_evidence
        has_bldg = fp_ids & building_point_evidence

        if has_addr:
            # Dwelling evidence wins: dwelling-linked footprints are primary;
            # everything else on this parcel is secondary.
            for fp_id in fp_ids - has_addr:
                role[fp_id] = 'secondary'
        elif has_bldg:
            # NSI evidence: NSI-linked footprints are primary, rest secondary.
            for fp_id in fp_ids - has_bldg:
                role[fp_id] = 'secondary'
        else:
            # No point evidence on this parcel — all footprints are secondary.
            # (Primary requires evidence, or sole occupancy of the parcel.)
            for fp_id in fp_ids:
                role[fp_id] = 'secondary'

    # Promote unlinked footprints with dwelling evidence from 'unknown' to 'primary'.
    for fp_id in address_evidence:
        if fp_id in state.spine.index and role[fp_id] == 'unknown':
            role[fp_id] = 'primary'

    state.spine['footprint_role'] = pd.Categorical(
        role, categories=['primary', 'secondary', 'unknown']
    )
    if state.verbose:
        counts = state.spine['footprint_role'].value_counts()
        print(
            '  classify_footprint_role: '
            + ', '.join(f'{k}={v:,d}' for k, v in counts.items())
        )
    if state.timer:
        state.timer.mark('Classify')
    return state


@_register('infer_attributes')
def infer_attributes(
    state: HarmonizeState,
    derived: list[str] | None = None,
    **_params,
) -> HarmonizeState:
    """Compute derived columns on the spine.

    Parameters
    ----------
    derived : list of str, optional
        Names of derived columns to compute.  Supported values:

        ``'area'`` / ``'m2'``
            Footprint area in square metres (stored as ``'m2'``).
        ``'value_per_sqft'`` / ``'value_per_area'``
            ``improvement_value{suffix} / m2`` and
            ``structure_value{suffix} / m2``.
        ``'openplaces_group_combined'``
            Combined group label reconciling polygon and point reference
            sources.
        ``'n_dwelling_units'``
            Fill null ``n_dwelling_units`` values from occupancy-class mapping
            when a ``purpose_subgroup`` column is present on the spine.

        When *derived* is ``None`` or empty, all of the above are attempted.
    """
    if state.spine is None:
        return state

    spine = state.spine
    compute_all = not derived

    def _want(name: str) -> bool:
        if compute_all:
            return True
        return any(d in (name, name.split('_')[0]) for d in (derived or []))

    if _want('area') or _want('m2'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            spine['m2'] = get_areas(spine, unit='m2')

    if _want('value_per_sqft') or _want('value_per_area'):
        if 'm2' in spine.columns:
            for col in list(spine.columns):
                if col.endswith('_per_area'):
                    continue
                if col.startswith('improvement_value') or col.startswith(
                    'structure_value'
                ):
                    spine[f'{col}_per_area'] = spine[col] / spine['m2']

    if _want('openplaces_group_combined') or _want('openplaces_group'):
        poly_suffix = next(
            (
                _resolve_suffix(rid, state.reference_types.get(rid), state)
                for rid in state.overlays
            ),
            None,
        )
        point_suffix = next(
            (
                _point_suffix(rid, state.reference_types.get(rid))
                for rid in state.crosswalks
                if rid not in state.overlays
            ),
            None,
        )
        if poly_suffix and point_suffix:
            out_col = f'openplaces_group{poly_suffix}{point_suffix}'
            a = spine.get(out_col) or spine.get(f'openplaces_group{poly_suffix}')
            b = spine.get(f'openplaces_group{point_suffix}')
            poly_label = poly_suffix.lstrip('_')
            point_label = point_suffix.lstrip('_')
            if a is not None and b is not None:
                both = a.notna() & b.notna()
                same = both & (a == b)
                spine[out_col] = np.select(
                    [same, both, a.notna(), b.notna()],
                    [
                        a,
                        f'{poly_label}: ' + a + f' | {point_label}: ' + b,
                        a,
                        b,
                    ],
                    default='',
                )

    if _want('n_dwelling_units'):
        if 'n_dwelling_units' not in spine.columns:
            spine['n_dwelling_units'] = np.nan
        null_mask = spine['n_dwelling_units'].isna()
        if null_mask.any():
            subgroup_col = next(
                (c for c in spine.columns if c.startswith('purpose_subgroup')), None
            )
            if subgroup_col is not None:
                inferred = spine.loc[null_mask, subgroup_col].map(_OCC_UNITS)
                spine.loc[null_mask, 'n_dwelling_units'] = inferred

    cat_attrs = get_categorical_attrs()
    cat_sorted = sorted(cat_attrs, key=len, reverse=True)
    for col in spine.columns:
        base = next((a for a in cat_sorted if col.startswith(a)), None)
        if base is not None and spine[col].dtype != 'category':
            spine[col] = pd.Categorical(spine[col])

    state.spine = spine
    if state.verbose:
        print(f'  Save: {len(spine):,d} spine entries for {state.admin_id}')
    if state.timer:
        state.timer.mark('Save')
    return state
