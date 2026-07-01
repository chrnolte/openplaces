"""
Pipeline steps that attach reference-dataset evidence to the spine:
  - reconcile_attributes: aggregate source columns from established crosswalks
  - classify_footprint_priority: assign each footprint's priority on its parcel

Value selection, gap-filling, and occupancy inference run in the curation stage
(see ``openplaces.io.curator``), not here.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from openplaces.io.harmonizer import HarmonizeState, _register

__all__ = [
    'classify_footprint_priority',
    'reconcile_attributes',
    'reverse_occ_units',
]


def _resolve_suffix(
    crosswalk_key: str,
    entity_type: str | None,
    state,
    default: str = '_ref',
) -> str:
    """Return the column suffix for reference attributes.

    Follows the naming convention (see the attribute-registry notes):

    - Same entity type as the spine (e.g. both ``'building'``): the source
      disambiguates, so the suffix is the source id (e.g. ``'_nsi'``).
    - A cross-entity ``parcel`` reference: parcels are interchangeable, so the
      suffix is the entity type only (``'_parcel'``).
    - Any other cross-entity reference: the source is not interchangeable, so the
      suffix carries entity type and source (e.g. ``'_footprint_fema'`` for FEMA
      footprints attributed to a parcel spine).
    """
    spine_entity = state.recipe.get('entity')
    spine_entity_type = (
        str(spine_entity.entity_type) if spine_entity is not None else None
    )
    base = crosswalk_key.rsplit('_', 1)[-1]
    parts = base.split('-', 2)
    source_id = parts[1] if len(parts) > 1 else base
    if entity_type and spine_entity_type and entity_type == spine_entity_type:
        return f'_{source_id}'
    if not entity_type:
        return default
    if entity_type == 'parcel':
        return f'_{entity_type}'
    return f'_{entity_type}_{source_id}'


def _point_suffix(
    crosswalk_key: str,
    entity_type: str | None = None,
    col: str | None = None,
) -> str:
    """Compose suffix ``_{entity_type}_{source_id}`` for point references.

    When *col* is supplied and *entity_type* already appears in the column
    name (e.g. ``'dwelling'`` in ``'n_dwellings'``), the entity_type
    part is dropped to avoid redundancy (``'_overture'`` instead of
    ``'_dwelling_overture'``).

    Examples: ``US_building-nsi-2022``, ``building`` -> ``_building_nsi``;
    ``dwelling-overture-2025``, ``dwelling`` -> ``_dwelling_overture``;
    ``dwelling-overture-2025``, ``dwelling``, col=``n_dwellings``
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
# Used as a fallback when n_dwellings is missing from parcel data.
_OCC_UNITS: dict[str, float] = {
    'Single Family': 1.0,
    'Single Family, 1 story, no basement': 1.0,
    'Single Family, 1 story, with basement': 1.0,
    'Single Family, 2 story, no basement': 1.0,
    'Single Family, 2 story, with basement': 1.0,
    'Single Family, 3 story, no basement': 1.0,
    'Single Family, 3 story, with basement': 1.0,
    'Single Family, split-level, no basement': 1.0,
    'Single Family, split-level, with basement': 1.0,
    'Manufactured Home': 1.0,
    'Multi-Family, 2 units': 2.0,
    'Multi-Family, 3-4 units': 3.5,
    'Multi-Family, 5-10 units': 7.0,
    'Multi-Family, 10-19 units': 14.5,
    'Multi-Family, 20-50 units': 35.0,
    'Multi-Family, 50 plus units': 51.0,
    'Multi-Family (2 units)': 2.0,
    'Multi-Family (3-4 units)': 3.5,
    'Multi-Family (5-9 units)': 7.0,
    'Multi-Family (5-10 units)': 7.0,
    'Multi-Family (10-19 units)': 14.5,
    'Multi-Family (20-50 units)': 35.0,
    'Multi-Family (50+ units)': 51.0,
    'Multi-Family (50 plus units)': 51.0,
}


def reverse_occ_units(total_units: float) -> str:
    """Re-classify a summed unit count to the nearest *occupancy_type* label.

    Mirrors the ``map_to_units`` logic from Lochhead et al. (2026).  Used when
    multiple NSI points link to the same footprint and their unit counts must be
    aggregated and re-classified.
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


# Columns carried from a polygon reference (e.g. parcel) to the spine. Parcels
# carry the use_* vocabulary ("what it is used for"); the purpose_* entries keep
# a building/footprint polygon reference working. Only columns present on the
# reference are carried (see the `if c in ref.columns` filters at the call sites).
_POLYGON_REF_COLS = [
    'use_group',
    'use_subgroup',
    'use_group_combined',
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
    'occupancy_type',
    'group',
    'structure_value',
    'year_built_block_median',
    'source',
    'area_sqft',
    'n_stories',
    'n_dwellings',
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
) -> HarmonizeState:
    """Aggregate reference attributes to the spine via established crosswalks.

    For each source in *sources*, looks up the crosswalk in
    ``state.crosswalks`` (resolved via ``recipe_id`` or ``entity_type``) and
    aggregates the requested columns to the spine as source-suffixed evidence
    columns (e.g. ``improvement_value_parcel``, ``occupancy_type_building_nsi``).

    This step only attributes evidence; between-source value selection,
    gap-filling, and occupancy inference now run in the curation stage
    (see ``openplaces.io.curator``).

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
        ``remap_id`` (str, optional)
            Recipe id of a two-column value crosswalk (raw -> canonical) applied
            in place to the matching reference column before aggregation (e.g.
            canonicalizing FEMA ``occupancy_type`` via its occupancy-type-remap).
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
        remap_id: str | None = src_cfg.get('remap_id')

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
                    remap_id=remap_id,
                )
            else:
                state = _attribute_point_reference(
                    state,
                    crosswalk_key,
                    ref_entity_type,
                    crosswalk,
                    columns,
                    collect_ids=src_cfg.get('collect_ids', False),
                )

    if state.timer:
        state.timer.mark('Attribute')
    return state


def _dominant_by_area(attrs, spine_id_col: str, col: str, area_col: str):
    """Dominant categorical value per spine entity, weighted by overlap area.

    Returns ``(dominant, joined)``: *dominant* is the value with the largest total
    *area_col* within each spine entity; *joined* lists every value for that entity
    ordered by descending area, joined with ``' + '``. Both are indexed by spine id.
    """
    areas = (
        attrs.groupby([spine_id_col, col])[area_col]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    dominant = areas.drop_duplicates(spine_id_col).set_index(spine_id_col)[col]
    joined = areas.groupby(spine_id_col)[col].apply(' + '.join)
    return dominant, joined


def _attribute_polygon_reference(
    state: HarmonizeState,
    crosswalk_key: str,
    entity_type: str | None,
    ref_polys,
    columns: list[str] | None,
    thresholds: dict | None = None,
    dwelling_linked_ids: set | None = None,
    remap_id: str | None = None,
) -> HarmonizeState:
    """Attribute a polygon reference (e.g. parcels) to the spine.

    Works in either direction: parcels attributed to a footprint spine, or
    footprint polygons attributed to a parcel spine (the FEMA-occupancy case).
    Categorical columns are attributed as the dominant value by overlap area; the
    numeric value-distribution blocks are column-guarded, so they simply do not
    run when the requested columns do not include them.

    Parameters
    ----------
    thresholds : dict, optional
        ``use_volume_weight`` (bool, default False) — weight ``improvement_value``
        and ``n_dwellings`` distribution by ``area × n_stories`` instead of
        area alone (Lochhead et al. 2026).  Requires a ``n_stories*`` column in
        the spine (populated by a prior NSI ``link_to_reference`` step).
    dwelling_linked_ids : set, optional
        Spine IDs linked to a dwelling point
        (:class:`~openplaces.core.schema.SourceGeometryType.single_dwelling_point`
        source).  When provided, parcel values (``improvement_value``,
        ``land_value``, ``n_dwellings``) are distributed only to
        dwelling-linked footprints within parcels that have dwelling evidence
        (Lochhead et al. 2026, Table 4, Cases 1–2 vs. Case 3).
    remap_id : str, optional
        Recipe id of a two-column value crosswalk (raw -> canonical). Applied in
        place to the matching reference column (the one named like the crosswalk
        key) before aggregation, e.g. canonicalizing FEMA ``occupancy_type``.
    """
    spine = state.spine
    spine_id_col = spine.index.name
    overlay = state.overlays.get(crosswalk_key)
    if overlay is None or ref_polys is None:
        return state

    if remap_id:
        from openplaces.io.transform import get_crosswalk

        crosswalk = get_crosswalk({'recipe_id': remap_id})
        remap_col = crosswalk.index.name
        if remap_col in ref_polys.columns:
            ref_polys = ref_polys.copy()
            mapped = ref_polys[remap_col].map(crosswalk)
            ref_polys[remap_col] = mapped.where(mapped.notna(), ref_polys[remap_col])

    thresholds = thresholds or {}
    use_volume_weight: bool = bool(thresholds.get('use_volume_weight', False))

    suffix = _resolve_suffix(crosswalk_key, entity_type, state, default='_ref')
    spine_entity = state.recipe.get('entity')
    spine_entity_type = (
        str(spine_entity.entity_type) if spine_entity is not None else 'entity'
    )
    # Relational counts read as n_{counted}s_per_{grouping}. Both are totals
    # (include the footprint/parcel itself), so the two directions are symmetric.
    ref_label = suffix.lstrip('_')
    n_ref_per_spine_col = f'n_{ref_label}s_per_{spine_entity_type}'
    n_spine_per_ref_col = f'n_{spine_entity_type}s_per_{ref_label}'
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
    # value (improvement_value, n_dwellings). Suppressed IDs are tracked
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

    spine[n_ref_per_spine_col] = (
        footprint_ref_attrs.groupby(spine_id_col)
        .size()
        .reindex(spine.index, fill_value=0)
    )

    # Attribute categorical columns as the dominant value (and an `_all` summary)
    # by overlap area. The combined land-use label is attributed this way; so is
    # any other categorical column the recipe requests (e.g. FEMA occupancy_type
    # on a parcel spine). The raw use_*/purpose_* components are folded into the
    # combined label, and address/value columns are handled separately below, so
    # those are skipped here.
    _skip_generic = {
        'use_group',
        'use_subgroup',
        'purpose_group',
        'purpose_subgroup',
        'address',
    }
    combined_col = next(
        (
            c
            for c in ('use_group_combined', 'purpose_group_combined')
            if c in footprint_ref_attrs.columns
        ),
        None,
    )
    categorical_cols = [
        c
        for c in (combined_col, *[c for c in (columns or []) if c not in _skip_generic])
        if c is not None
        and c in footprint_ref_attrs.columns
        and not pd.api.types.is_numeric_dtype(footprint_ref_attrs[c])
    ]
    for col in dict.fromkeys(categorical_cols):
        dominant, joined = _dominant_by_area(
            footprint_ref_attrs, spine_id_col, col, 'area_intersection_m2'
        )
        spine[f'{col}{suffix}'] = dominant
        spine[f'{col}{suffix}_all'] = joined

    footprint_parcel_areas = footprint_ref_attrs.reset_index()[
        [spine_id_col, 'parcel_id', 'area_intersection_m2']
    ]
    parcel_area_stats = footprint_parcel_areas.groupby('parcel_id')[
        'area_intersection_m2'
    ].agg(max_area='max', n_fp='count')
    footprint_parcel_areas = footprint_parcel_areas.join(
        parcel_area_stats, on='parcel_id'
    )
    footprint_parcel_areas['_n_col'] = footprint_parcel_areas['n_fp']
    primary_footprints = (
        footprint_parcel_areas.sort_values('area_intersection_m2', ascending=False)
        .drop_duplicates(spine_id_col)
        .set_index(spine_id_col)
    )
    spine[n_spine_per_ref_col] = primary_footprints['_n_col'].reindex(spine.index)

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
        spine[f'overlap_fraction{suffix}'] = (
            (identified_area / spine_area).round(4).reindex(spine.index, fill_value=0.0)
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
            if 'priority_on_parcel' in spine.columns:
                is_principal = spine['priority_on_parcel'].eq('primary')
            else:
                is_principal = spine[n_spine_per_ref_col].eq(1)
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

    # Zero n_dwellings for secondary footprints: accessory structures contain no
    # dwellings. improvement_value is left missing after aggregation instead, to
    # match land_value (see below).
    if (
        'n_dwellings' in footprint_ref_attrs.columns
        and 'priority_on_parcel' in spine.columns
    ):
        secondary_ids = spine.index[spine['priority_on_parcel'].eq('secondary')]
        footprint_ref_attrs.loc[
            footprint_ref_attrs.index.isin(secondary_ids), 'n_dwellings'
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

    if 'n_dwellings' in footprint_ref_attrs.columns:
        footprint_ref_attrs['n_dwellings'] = (
            footprint_ref_attrs['n_dwellings']
            .mul(footprint_ref_attrs['area_fraction'])
            .round(2)
        )

    numeric_agg: dict[str, str] = {}
    if 'improvement_value' in footprint_ref_attrs.columns:
        numeric_agg['improvement_value'] = 'sum'
    if 'n_dwellings' in footprint_ref_attrs.columns:
        numeric_agg['n_dwellings'] = 'sum'
    if 'year_built' in footprint_ref_attrs.columns:
        numeric_agg['year_built'] = 'mean'
    if numeric_agg:
        rename_map = {
            'improvement_value': f'improvement_value{suffix}',
            'n_dwellings': f'n_dwellings{suffix}',
            'year_built': f'year_built{suffix}',
        }
        spine = spine.join(
            footprint_ref_attrs.groupby(spine_id_col)
            .agg(numeric_agg)
            .rename(columns={k: v for k, v in rename_map.items() if k in numeric_agg})
        )

    # Secondary footprints carry no parcel improvement value, mirroring land_value
    # (which is restricted to primary footprints). Set to NaN rather than 0 so the
    # absence is explicit (Lochhead et al. 2026, Table 4).
    imp_col = f'improvement_value{suffix}'
    if imp_col in spine.columns and 'priority_on_parcel' in spine.columns:
        spine.loc[spine['priority_on_parcel'].eq('secondary'), imp_col] = np.nan

    footprints_from_ref = state.metadata.get(f'inferred_from_{crosswalk_key}')
    mask_ref_src = spine['geometry_source'].str.contains(r'\.', regex=True, na=False)
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
                    'use_group_combined': f'use_group_combined{suffix}',
                    'purpose_group_combined': f'purpose_group_combined{suffix}',
                    'improvement_value': f'improvement_value{suffix}',
                    'n_dwellings': f'n_dwellings{suffix}',
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
        inferred_ref_attrs[n_ref_per_spine_col] = 1
        inferred_ref_attrs[n_spine_per_ref_col] = 1
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
    collect_ids: bool = False,
) -> HarmonizeState:
    """Attribute a point reference (e.g. NSI) to the spine."""
    spine = state.spine
    suffix = _point_suffix(crosswalk_key, entity_type)
    base = crosswalk_key.rsplit('_', 1)[-1]
    source_id = base.split('-', 2)[1] if '-' in base else base

    avail_cols = columns or [c for c in _POINT_REF_COLS if c in crosswalk.columns]
    renamed: dict[str, str] = {
        c: f'{c}{_point_suffix(crosswalk_key, entity_type, col=c)}'
        for c in avail_cols
        if c in crosswalk.columns
    }
    # Read naturally as n_<plural-entity>_<source> (e.g. n_dwellings_overture)
    # rather than carrying the verbose source attribute name.
    if 'n_dwellings' in renamed:
        renamed['n_dwellings'] = f'n_dwellings_{source_id}'

    spine_id_col = state.spine.index.name
    if spine_id_col not in crosswalk.columns:
        warnings.warn(
            f'_attribute_point_reference: {crosswalk_key!r} crosswalk has no '
            f"'{spine_id_col}' column; skipping."
        )
        return state

    count_col = f'n_{entity_type}s_{source_id}' if entity_type else f'n_point{suffix}'
    group_sizes = crosswalk.groupby(spine_id_col).size()
    if group_sizes.max() > 1:
        spine[count_col] = group_sizes.reindex(spine.index, fill_value=0)

    purpose_group_col = next(
        (
            c
            for c in avail_cols
            if ('purpose_subgroup' in c or 'occupancy_type' in c)
            and c in crosswalk.columns
        ),
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
        else:
            footprint_purpose_group_areas = (
                crosswalk.groupby(spine_id_col)[purpose_group_col].first().reset_index()
            )
        # Emit the single reconciled value before the multi-link `_all` summary.
        spine[purpose_group_out] = footprint_purpose_group_areas.drop_duplicates(
            spine_id_col
        ).set_index(spine_id_col)[purpose_group_col]
        if structure_value_col:
            spine[f'{purpose_group_out}_all'] = footprint_purpose_group_areas.groupby(
                spine_id_col
            )[purpose_group_col].apply(' + '.join)

    group_col = next(
        (c for c in avail_cols if c == 'group' and c in crosswalk.columns),
        None,
    )
    if group_col and structure_value_col:
        group_out = renamed.get(group_col, group_col)
        footprint_group_areas = (
            crosswalk.groupby([spine_id_col, group_col])[structure_value_col]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        spine[group_out] = footprint_group_areas.drop_duplicates(
            spine_id_col
        ).set_index(spine_id_col)[group_col]
        spine[f'{group_out}_all'] = footprint_group_areas.groupby(spine_id_col)[
            group_col
        ].apply(' + '.join)

    numeric_agg: dict[str, str] = {}
    if structure_value_col:
        numeric_agg[structure_value_col] = 'sum'
    year_built_col = next(
        (c for c in avail_cols if 'year_built' in c and c in crosswalk.columns),
        None,
    )
    if year_built_col:
        numeric_agg[year_built_col] = 'mean'
    if 'n_dwellings' in avail_cols and 'n_dwellings' in crosswalk.columns:
        numeric_agg['n_dwellings'] = 'sum'
    if numeric_agg:
        spine = spine.join(
            crosswalk.groupby(spine_id_col).agg(numeric_agg).rename(columns=renamed)
        )

    handled = set(numeric_agg) | {
        'purpose_subgroup',
        'occupancy_type',
        'group',
    }
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

    if collect_ids and crosswalk.index.name:
        index_name = crosswalk.index.name
        grouped_ids = (
            crosswalk.reset_index()
            .groupby(spine_id_col)[index_name]
            .agg(lambda s: '|'.join(s.dropna().astype(str).unique()))
        )
        spine[index_name] = grouped_ids.where(grouped_ids != '').reindex(spine.index)

    state.spine = spine
    return state


@_register('classify_footprint_priority')
def classify_footprint_priority(
    state: HarmonizeState,
    entity_type: str | None = None,
    thresholds: dict | None = None,
    **_params,
) -> HarmonizeState:
    """Classify each footprint's priority on its parcel.

    Assigns ``priority_on_parcel`` as ``'primary'``, ``'secondary'``, or
    ``'unknown'``.

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
            print(
                f'  classify_footprint_priority: no {parcel_type} crosswalk; skipping.'
            )
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
        state.spine['priority_on_parcel'] = pd.Categorical(
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

    state.spine['priority_on_parcel'] = pd.Categorical(
        role, categories=['primary', 'secondary', 'unknown']
    )
    if state.verbose:
        counts = state.spine['priority_on_parcel'].value_counts()
        print(
            '  classify_footprint_priority: '
            + ', '.join(f'{k}={v:,d}' for k, v in counts.items())
        )
    if state.timer:
        state.timer.mark('Classify')
    return state


@_register('derive_use_classes')
def derive_use_classes(
    state: HarmonizeState,
    combined_column: str = 'use_group_combined',
) -> HarmonizeState:
    """Build the combined use_group_combined label from use_group / use_subgroup.

    Source-specific raw use codes are mapped to the openplaces use_group and
    use_subgroup vocabulary upstream, via a ``*-remap.csv`` crosswalk that
    ``link_by_id``'s auto-discovery applies automatically when joining a
    source that ships one (see
    :func:`~openplaces.io.harmonizer.links._apply_remap_csvs`); this step only
    combines the two into the label the parcel land-use classifier groups and
    votes on, so it holds no code vocabulary of its own.

    No-op when the spine lacks ``use_group`` or ``use_subgroup`` (e.g. an
    admin whose source never produced a use code, or had none to crosswalk),
    so it is safe to leave in a shared pipeline.

    Parameters
    ----------
    combined_column : str, optional
        Output combined-label column (default ``use_group_combined``).
    """
    if (
        state.spine is None
        or 'use_group' not in state.spine.columns
        or 'use_subgroup' not in state.spine.columns
    ):
        if state.verbose:
            print('  derive_use_classes: no use_group/use_subgroup on spine; skipping.')
        return state

    spine = state.spine
    label = (
        spine['use_group'].astype(str).fillna('n/a')
        + ' | '
        + spine['use_subgroup'].astype(str).fillna('n/a')
    )
    spine[combined_column] = pd.Categorical(label)

    state.spine = spine
    if state.verbose:
        mapped = int(spine['use_group'].notna().sum())
        print(f'  derive_use_classes: combined {mapped:,d}/{len(spine):,d} parcels')
    return state


@_register('summarize_footprint_morphology')
def summarize_footprint_morphology(
    state: HarmonizeState,
    footprint_recipe_id: str,
    small_area_max_m2: float = 185.0,
    elongated_aspect_min: float = 2.0,
    on: str | None = 'parcel_id_local',
) -> HarmonizeState:
    """Attach per-parcel footprint morphology aggregates to the parcel spine.

    Footprint-to-parcel linkage is the harmonizer's job; this reads the footprint
    entity, assigns each footprint to a parcel (by a shared id column when *on* is
    present on both sides, else by spatial containment of the footprint's
    representative point), and writes the per-parcel counts the parcel land-use
    classifier consumes downstream: ``n_footprints_per_parcel``,
    ``n_small_elongated_footprints_per_parcel`` (manufactured-home-shaped), and
    ``max_footprint_area_m2``. The classification itself is parcel-curate work.

    Parameters
    ----------
    footprint_recipe_id : str
        Footprint entity recipe to read (geometry required).
    small_area_max_m2, elongated_aspect_min : float, optional
        A footprint counts as small-and-elongated (manufactured-home morphology)
        when its area is at most *small_area_max_m2* and its oriented aspect ratio
        is at least *elongated_aspect_min*.
    on : str, optional
        Shared id column for an id-based assignment (default ``parcel_id_local``);
        when absent on either side, a spatial within-join is used instead.
    """
    import geopandas as gpd

    from openplaces.geo.polygon import local_metric_crs
    from openplaces.io.harmonizer.spine import get_oriented_dims
    from openplaces.io.readers import get_entities

    if state.spine is None:
        return state
    spine = state.spine

    footprints = get_entities(footprint_recipe_id, state.admin_id, geom=True)
    if footprints is None or len(footprints) == 0:
        if state.verbose:
            print('  summarize_footprint_morphology: no footprints; skipping.')
        return state

    geom_m = footprints.geometry.to_crs(local_metric_crs(footprints))
    area = geom_m.area.to_numpy()
    dims = geom_m.map(get_oriented_dims)
    length = np.array([d[1] for d in dims])
    width = np.clip(np.array([d[2] for d in dims]), 1e-6, None)
    aspect = length / width
    small_elong = (area <= small_area_max_m2) & (aspect >= elongated_aspect_min)

    if on and on in footprints.columns and on in spine.columns:
        per_fp = pd.DataFrame(
            {
                '_pid': footprints[on].astype('string').to_numpy(),
                '_a': area,
                '_se': small_elong,
            }
        ).dropna(subset=['_pid'])
        grp = per_fp.groupby('_pid')
        key = spine[on].astype('string')
        n_fp = key.map(grp.size())
        n_se = key.map(grp['_se'].sum())
        max_a = key.map(grp['_a'].max())
    else:
        reps = gpd.GeoDataFrame(
            {'_a': area, '_se': small_elong},
            geometry=footprints.geometry.representative_point(),
            crs=footprints.crs,
        ).to_crs(spine.crs)
        # Carry the spine id as an explicit column so the join does not depend on
        # the sjoin right-index column name (which varies with the index name).
        spine_geom = spine[[spine.geometry.name]].copy()
        spine_geom['_spine_id'] = spine.index
        joined = gpd.sjoin(reps, spine_geom, predicate='within', how='left').dropna(
            subset=['_spine_id']
        )
        grp = joined.groupby('_spine_id')
        n_fp = grp.size().reindex(spine.index)
        n_se = grp['_se'].sum().reindex(spine.index)
        max_a = grp['_a'].max().reindex(spine.index)

    spine['n_footprints_per_parcel'] = n_fp.fillna(0).astype('int64')
    spine['n_small_elongated_footprints_per_parcel'] = n_se.fillna(0).astype('int64')
    spine['max_footprint_area_m2'] = max_a
    state.spine = spine

    if state.verbose:
        print(
            '  summarize_footprint_morphology: '
            f'{int((spine["n_small_elongated_footprints_per_parcel"] > 0).sum()):,} '
            f'parcels with manufactured-home-shaped footprints.'
        )
    if state.timer:
        state.timer.mark('Summarize')
    return state


@_register('classify_occupancy')
def classify_occupancy(state: HarmonizeState, **kwargs) -> HarmonizeState:
    """Predict building occupancy type from street-view imagery.

    Reads the image metadata parquet written by the ``google_streetview``
    ingest recipe, builds an ImageSet, and runs the local OccupancyClassifier.
    Predictions (``'Residential'`` or ``'Other'``) are joined onto
    ``state.spine`` as the column ``purpose_group``.

    Model weights are downloaded from Zenodo on first use.

    Parameters
    ----------
    state
        Current harmonize state.  ``state.recipe`` must contain
        ``image_recipe`` pointing to the streetview ingest recipe ID.
    **kwargs
        Forwarded from the harmonizer dispatch; unused.

    Returns
    -------
    HarmonizeState
        Updated state with ``purpose_group`` column added to ``state.spine``.
    """
    from pathlib import Path

    from openplaces.io.enricher.detectors.occupancy import OccupancyClassifier
    from openplaces.io.readers import get_entities
    from openplaces.io.scrapers.types import Image, ImageSet
    from openplaces.recipe import get_recipe_by_id

    image_recipe_id = state.recipe.get('image_recipe')
    if not image_recipe_id:
        raise ValueError("classify_occupancy requires 'image_recipe' in recipe.")
    image_recipe = get_recipe_by_id(image_recipe_id)
    meta_df = get_entities(image_recipe, state.admin_id)
    if meta_df is None or 'image_path' not in meta_df.columns:
        return state

    image_set = ImageSet()
    image_set.dir_path = ''
    for idx, row in meta_df.iterrows():
        p = Path(row['image_path'])
        if p.exists():
            image_set.dir_path = str(p.parent)
            image_set.add_image(idx, Image(p.name))

    if not image_set.images:
        return state

    predictions = OccupancyClassifier().predict(image_set)
    pred_series = pd.Series(predictions, name='purpose_group')
    pred_series.index.name = state.spine.index.name
    state.spine = state.spine.join(pred_series, how='left')
    return state


@_register('classify_roof_shape')
def classify_roof_shape(state: HarmonizeState, **kwargs) -> HarmonizeState:
    """Predict roof shape per building from satellite imagery.

    Reads the image metadata parquet written by the ``google_satellite``
    ingest recipe, builds an ImageSet, and runs the local RoofShapeClassifier.
    Predictions (``'Flat'``, ``'Gable'``, or ``'Hip'``) are joined onto
    ``state.spine`` as the column ``roof_shape``.

    Model weights are downloaded from Zenodo on first use.

    Parameters
    ----------
    state
        Current harmonize state.  ``state.recipe`` must contain
        ``image_recipe`` pointing to the satellite ingest recipe ID.
    **kwargs
        Forwarded from the harmonizer dispatch; unused.

    Returns
    -------
    HarmonizeState
        Updated state with ``roof_shape`` column added to ``state.spine``.
    """
    from pathlib import Path

    from openplaces.io.enricher.detectors.roof_shape import RoofShapeClassifier
    from openplaces.io.readers import get_entities
    from openplaces.io.scrapers.types import Image, ImageSet
    from openplaces.recipe import get_recipe_by_id

    image_recipe_id = state.recipe.get('image_recipe')
    if not image_recipe_id:
        raise ValueError("classify_roof_shape requires 'image_recipe' in recipe.")
    image_recipe = get_recipe_by_id(image_recipe_id)
    meta_df = get_entities(image_recipe, state.admin_id)
    if meta_df is None or 'image_path' not in meta_df.columns:
        return state

    image_set = ImageSet()
    image_set.dir_path = ''
    for idx, row in meta_df.iterrows():
        p = Path(row['image_path'])
        if p.exists():
            image_set.dir_path = str(p.parent)
            image_set.add_image(idx, Image(p.name))

    if not image_set.images:
        return state

    predictions = RoofShapeClassifier().predict(image_set)
    pred_series = pd.Series(predictions, name='roof_shape')
    pred_series.index.name = state.spine.index.name
    state.spine = state.spine.join(pred_series, how='left')
    return state
