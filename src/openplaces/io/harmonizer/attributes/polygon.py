"""Attribution from a polygon reference through the overlay
crosswalk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from openplaces.core.attribute_registry import get_agg_func
from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer.apportion import (
    APPORTIONED_VALUE_COLUMNS,
    apportion_reference_values,
)
from openplaces.io.harmonizer.attributes._shared import (
    _POLYGON_REF_COLS,
    _attributed_name,
    _dominant_by_area,
    _resolve_suffix,
)
from openplaces.recipe import resolve_attribute_name


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
    trimmed_crosswalk = state.crosswalks.get(crosswalk_key)
    if overlay is None or trimmed_crosswalk is None or ref_polys is None:
        return state
    # Snapshot the spine's own columns before this crosswalk attributes
    # anything, so every id-column naming decision in this call (including the
    # later synthetic-fallback block) treats the spine's pre-existing native
    # columns — not columns this same call already wrote — as the collision
    # to guard against. Also reserve the spine's own restored index name
    # (e.g. a parcel spine's 'parcel_id', renamed to a working name during
    # processing — see resolve_spine): that name isn't in spine.columns yet,
    # but a same-named bare id column would collide with it once the
    # harmonizer's save step restores the index.
    reserved_cols = set(spine.columns) | {state.metadata.get('spine_index_name')}

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
    # ref_label may be a compound "{entity_type}_{source_id}" (e.g. 'footprint_fema');
    # pluralize just the entity-type word and keep the source suffix intact, so this
    # reads 'footprints_fema', not the ungrammatical 'footprint_femas'.
    if entity_type and ref_label.startswith(entity_type):
        ref_label_plural = entity_type + 's' + ref_label[len(entity_type) :]
    else:
        ref_label_plural = f'{ref_label}s'
    n_ref_per_spine_col = f'n_{ref_label_plural}_per_{spine_entity_type}'
    n_spine_per_ref_col = f'n_{spine_entity_type}s_per_{ref_label}'
    avail_cols = [c for c in (columns or _POLYGON_REF_COLS) if c in ref_polys.columns]

    mask_has_ref = overlay.index.get_level_values('parcel_id').notnull()
    mask_has_ref_trimmed = trimmed_crosswalk.index.get_level_values(
        'parcel_id'
    ).notnull()

    volume_weight = None
    if use_volume_weight:
        stories_col = next(
            (c for c in spine.columns if c.startswith('n_stories')), None
        )
        if stories_col is not None:
            volume_weight = spine[stories_col]

    # Value apportionment (improvement_value, n_dwellings, year_built,
    # land_value, address — overlap-fraction shares, dwelling-linked
    # suppression, primary-only and secondary rules) is delegated to the
    # shared implementation the curate stage also uses on the persisted link
    # sidecar; joined back onto the spine, suffixed, below.
    value_result = apportion_reference_values(
        overlay[mask_has_ref].reset_index()[
            [spine_id_col, 'parcel_id', 'area_intersection_m2']
        ],
        ref_polys[[c for c in avail_cols if c in APPORTIONED_VALUE_COLUMNS]],
        spine_id_col=spine_id_col,
        priority=spine.get('priority_on_parcel'),
        dwelling_linked_ids=dwelling_linked_ids,
        volume_weight=volume_weight,
    )

    # Categorical/numeric attribution and the relational counts below read the
    # *trimmed* crosswalk (sub-threshold sliver overlaps already dropped by
    # _build_crosswalk's fraction_of_largest/area_intersection_m2_min floors), not
    # the raw identity overlay -- a footprint that merely clips a sliver of a
    # neighboring parcel should not count as evidence of anything. Value
    # apportionment above intentionally still reads the raw overlay, unchanged.
    footprint_ref_attrs = (
        trimmed_crosswalk[mask_has_ref_trimmed][['area_intersection_m2']]
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

    # Every reference this spine row links to in the trimmed crosswalk, not
    # just the single dominant one a plain id column (e.g. 'parcel_id') would
    # give -- lets a downstream consumer credit a minor link too (e.g.
    # summarize_footprint_morphology crediting a parcel a footprint spans but
    # doesn't dominate). Harmonize-stage and scoped to the trimmed crosswalk
    # (post fraction_of_largest/area_intersection_m2_min); distinct from the
    # curate-stage collect_link_ids/'parcel_id_all' (reads the raw, untrimmed
    # overlay sidecar with its own threshold options) even where the name
    # coincides -- the two run on different recipes' outputs and are never
    # compared directly.
    spine[f'{ref_label}_id_all'] = (
        footprint_ref_attrs.reset_index()
        .groupby(spine_id_col)['parcel_id']
        .agg(lambda s: '|'.join(s.dropna().astype(str).unique()))
        .reindex(spine.index)
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
    footprint_parcel_areas['_n_col'] = footprint_parcel_areas['n_fp']
    primary_footprints = (
        footprint_parcel_areas.sort_values('area_intersection_m2', ascending=False)
        .drop_duplicates(spine_id_col)
        .set_index(spine_id_col)
    )
    spine[n_spine_per_ref_col] = (
        primary_footprints['_n_col'].reindex(spine.index).fillna(0).astype('int64')
    )

    # The dominant parcel's own globally-unique id (the overlay's true join
    # key), so the curate stage can join the curated parcel lane back onto
    # each footprint (link_curated_entity — see its entity_key/ref_key docs
    # for why the globally-unique id, not a locally-scoped one). Only
    # meaningful for a parcel reference: the overlay's reference-side index
    # is always internally named 'parcel_id' regardless of the actual
    # reference entity_type (e.g. the reverse FEMA-footprint-onto-parcel-spine
    # crosswalk), so writing it unconditionally here would attribute a
    # *footprint's* id under the name 'parcel_id' -- colliding with a parcel
    # spine's own 'parcel_id' index in that reverse case.
    # Written before the categorical-column loop below so parcel_id sorts
    # ahead of its locally-scoped counterparts in the curated output
    # (order_columns falls back to creation order when registry sort ranks
    # tie, which they do for these bare id columns).
    if entity_type == 'parcel':
        spine[_attributed_name('parcel_id', suffix, reserved_cols)] = (
            primary_footprints['parcel_id'].reindex(spine.index)
        )
        # The dominant link's own raw overlap area, so a later step
        # (summarize_footprint_morphology) can apply a minimum-overlap floor
        # without redoing the spatial overlay -- the number is already sitting
        # in primary_footprints from the sort above.
        spine[f'area_intersection_m2{suffix}'] = primary_footprints[
            'area_intersection_m2'
        ].reindex(spine.index)

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
        out_col = _attributed_name(col, suffix, reserved_cols)
        spine[out_col] = dominant
        spine[f'{out_col}_all'] = joined

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

    # Join the shared apportionment's value columns (computed above), suffixed.
    if len(value_result.columns):
        spine = spine.join(
            value_result.rename(
                columns={c: f'{c}{suffix}' for c in value_result.columns}
            )
        )

    # Any other requested numeric column (e.g. FEMA height) not covered by the
    # shared apportionment above: aggregate with the attribute registry's
    # default function (fallback 'mean'), unweighted by overlap fraction.
    remaining_numeric_cols = [
        c
        for c in avail_cols
        if c in footprint_ref_attrs.columns
        and c not in APPORTIONED_VALUE_COLUMNS
        and pd.api.types.is_numeric_dtype(footprint_ref_attrs[c])
    ]
    if remaining_numeric_cols:
        registry_agg = {
            c: get_agg_func(resolve_attribute_name(c)) or 'mean'
            for c in remaining_numeric_cols
        }
        remaining_rename = {
            c: _attributed_name(c, suffix, reserved_cols)
            for c in remaining_numeric_cols
        }
        spine = spine.join(
            footprint_ref_attrs.groupby(spine_id_col)
            .agg(registry_agg)
            .rename(columns=remaining_rename)
        )

    footprints_from_ref = state.metadata.get(f'inferred_from_{crosswalk_key}')
    # A spine built by union_spine_sources carries 'source', not
    # 'geometry_source', and has no reference-inferred rows at all, so
    # the column is read only once the block is known to apply.
    has_inferred = (
        footprints_from_ref is not None
        and 'parcel_id' in footprints_from_ref.columns
        and 'geometry_source' in spine.columns
    )
    mask_ref_src = (
        spine['geometry_source'].str.contains(r'\.', regex=True, na=False)
        if has_inferred
        else None
    )
    if has_inferred and mask_ref_src.any():
        ref_attr_cols = [
            c for c in (columns or _POLYGON_REF_COLS) if c in ref_polys.columns
        ]
        # As above, the bare/generic 'parcel_id' here is only meaningful when
        # the reference truly is a parcel; the reverse (non-parcel) direction
        # never produces a synthetic-fallback row anyway (a parcel spine has
        # no parcel-boundary fallback of its own), but stay consistent.
        id_assignment = (
            {
                _attributed_name('parcel_id', suffix, reserved_cols): (
                    footprints_from_ref['parcel_id']
                )
            }
            if entity_type == 'parcel'
            else {}
        )
        inferred_ref_attrs = (
            footprints_from_ref[['parcel_id']]
            .join(ref_polys[ref_attr_cols], on='parcel_id')
            .assign(**id_assignment)
            .rename(
                columns={
                    'parcel_id_local': _attributed_name(
                        'parcel_id_local', suffix, reserved_cols
                    ),
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
