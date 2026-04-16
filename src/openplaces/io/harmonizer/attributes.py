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
]


@_register('reconcile_attributes')
def reconcile_attributes(
    state: HarmonizeState,
    sources: list[dict] | None = None,
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
    """
    if state.spine is None or not sources:
        return state

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
                )
            else:
                state = _attribute_point_reference(
                    state,
                    crosswalk_key,
                    ref_entity_type,
                    crosswalk,
                    columns,
                )

    if state.timer:
        state.timer.mark('Attribute')
    return state


def _attribute_polygon_reference(
    state: HarmonizeState,
    crosswalk_key: str,
    entity_type: str | None,
    ref_polys,
    columns: list[str] | None,
) -> HarmonizeState:
    """Attribute a polygon reference (e.g. parcels) to the spine."""
    spine = state.spine
    overlay = state.overlays.get(crosswalk_key)
    if overlay is None or ref_polys is None:
        return state

    suffix = _resolve_suffix(crosswalk_key, entity_type, state, default='_ref')
    avail_cols = [c for c in (columns or _POLYGON_REF_COLS) if c in ref_polys.columns]

    overlay = overlay.copy()
    overlay['area_fraction'] = overlay['area_intersection_m2'] / overlay.groupby(
        'parcel_id'
    )['area_intersection_m2'].transform('sum')

    mask_has_ref = overlay.index.get_level_values('parcel_id').notnull()

    footprint_ref_attrs = (
        overlay[mask_has_ref][['area_intersection_m2', 'area_fraction']]
        .reset_index()
        .set_index('parcel_id')
        .join(ref_polys[avail_cols])
        .reset_index()
        .set_index('footprint_id')
    )

    spine['n_footprint_parcel'] = (
        footprint_ref_attrs.groupby('footprint_id')
        .size()
        .reindex(spine.index, fill_value=0)
    )

    if 'purpose_group_combined' in footprint_ref_attrs.columns:
        footprint_purpose_group_areas = (
            footprint_ref_attrs.groupby(['footprint_id', 'purpose_group_combined'])[
                'area_intersection_m2'
            ]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        spine[f'purpose_group_combined{suffix}'] = (
            footprint_purpose_group_areas.drop_duplicates('footprint_id').set_index(
                'footprint_id'
            )['purpose_group_combined']
        )
        spine[f'purpose_group_combined{suffix}_all'] = (
            footprint_purpose_group_areas.groupby('footprint_id')[
                'purpose_group_combined'
            ].apply(' + '.join)
        )

    footprint_parcel_areas = footprint_ref_attrs.reset_index()[
        ['footprint_id', 'parcel_id', 'area_intersection_m2']
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
        .drop_duplicates('footprint_id')
        .set_index('footprint_id')
    )
    spine[f'n_colocated{suffix}'] = primary_footprints['_n_col'].reindex(spine.index)

    if 'area_spine_m2' in overlay.columns:
        identified_area = (
            overlay.loc[mask_has_ref, 'area_intersection_m2']
            .groupby(level='footprint_id')
            .sum()
        )
        spine_area = (
            overlay['area_spine_m2']
            .groupby(level='footprint_id')
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
            .drop_duplicates('footprint_id')
            .set_index('footprint_id')
        )
        if 'address' in footprint_ref_attrs.columns:
            spine[f'address{suffix}'] = footprint_primary_row['address'].reindex(
                spine.index
            )
        if 'land_value' in footprint_ref_attrs.columns:
            is_principal = spine[f'overlap_fraction{suffix}'] == 1.0
            spine[f'land_value{suffix}'] = (
                footprint_primary_row['land_value']
                .reindex(spine.index)
                .where(is_principal)
            )

    footprint_ref_attrs = footprint_ref_attrs.copy()
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

    numeric_agg: dict[str, str] = {}
    if 'improvement_value' in footprint_ref_attrs.columns:
        numeric_agg['improvement_value'] = 'sum'
    if 'year_built' in footprint_ref_attrs.columns:
        numeric_agg['year_built'] = 'mean'
    if numeric_agg:
        spine = spine.join(
            footprint_ref_attrs.groupby('footprint_id')
            .agg(numeric_agg)
            .rename(
                columns={
                    'improvement_value': f'improvement_value{suffix}',
                    'year_built': f'year_built{suffix}',
                }
            )
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
        inferred_ref_attrs[f'n_colocated{suffix}'] = 0
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
    suffix = _resolve_suffix(crosswalk_key, entity_type, state, default='_point')

    avail_cols = columns or [c for c in _POINT_REF_COLS if c in crosswalk.columns]
    renamed: dict[str, str] = {
        c: f'{c}{suffix}' for c in avail_cols if c in crosswalk.columns
    }

    spine_id_col = state.spine.index.name
    if spine_id_col not in crosswalk.columns:
        warnings.warn(
            f'_attribute_point_reference: {crosswalk_key!r} crosswalk has no '
            f"'{spine_id_col}' column; skipping."
        )
        return state

    spine['n_point'] = (
        crosswalk.groupby(spine_id_col).size().reindex(spine.index, fill_value=0)
    )

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
    if numeric_agg:
        spine = spine.join(
            crosswalk.groupby(spine_id_col).agg(numeric_agg).rename(columns=renamed)
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

        When *derived* is ``None`` or empty, all of the above are attempted.
    """
    import warnings as _w

    if state.spine is None:
        return state

    spine = state.spine
    compute_all = not derived

    def _want(name: str) -> bool:
        if compute_all:
            return True
        return any(d in (name, name.split('_')[0]) for d in (derived or []))

    if _want('area') or _want('m2'):
        with _w.catch_warnings():
            _w.simplefilter('ignore')
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
                _resolve_suffix(rid, state.reference_types.get(rid), state)
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
