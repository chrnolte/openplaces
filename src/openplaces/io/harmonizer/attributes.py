"""
Pipeline steps that attach reference-dataset evidence to the spine:
  - reconcile_attributes: aggregate source columns from established crosswalks
  - classify_footprint_priority: assign each footprint's priority on its parcel

Value selection, gap-filling, and occupancy inference run in the curation stage
(see ``openplaces.io.curator``), not here.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

from openplaces.core.attribute_registry import get_agg_func
from openplaces.io.harmonizer import HarmonizeState, _register
from openplaces.io.harmonizer.apportion import (
    APPORTIONED_VALUE_COLUMNS,
    apportion_reference_values,
)
from openplaces.recipe import resolve_attribute_name, source_id_from_recipe_id

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
    source_id = source_id_from_recipe_id(crosswalk_key)
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
    source_id = source_id_from_recipe_id(crosswalk_key)
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


@_register('rename_columns')
def rename_columns(state: HarmonizeState, columns: dict[str, str]) -> HarmonizeState:
    """Rename spine columns.

    A small, generic escape hatch for the rare case a later step would
    otherwise silently overwrite a column in place -- e.g. the parcel
    spine renames its raw, as-ingested ``address``/``city`` (attached bare
    by ``link_by_id``'s auto-discovered assessor join, the same convention
    ``resolve_spine``'s own ``keep_columns`` uses for a parcel's other
    native attributes) to ``address_original``/``city_original`` before
    ``reconcile_addresses`` runs, since that step's own output defaults to
    those same bare names.

    Parameters
    ----------
    columns : dict of {old name: new name}
        Missing source columns are skipped, the same "missing evidence is
        tolerated" convention used throughout this codebase.
    """
    if state.spine is None:
        return state
    present = {old: new for old, new in columns.items() if old in state.spine.columns}
    state.spine = state.spine.rename(columns=present)
    return state


def _join_distinct(areas: pd.DataFrame, spine_id_col: str, col: str) -> pd.Series:
    """Join every distinct *col* value per spine entity with ``' + '``.

    Left missing for a spine entity whose group has only one distinct value —
    joining it with itself would just repeat the single reconciled value
    already stored separately, adding no information.
    """
    grouped = areas.groupby(spine_id_col)[col]
    joined = grouped.apply(' + '.join)
    return joined.where(grouped.nunique() > 1)


_ID_COLUMN = re.compile(r'.+_id(_.+)?$')


def _attributed_name(col: str, suffix: str, reserved_cols: set[str]) -> str:
    """Output column name for a cross-attributed *col* from a reference entity.

    Unlike other columns, an id column (``{entity}_id*``, e.g. ``parcel_id``)
    does not carry the ``_{source}``/``_{entity}_{source}`` provenance suffix
    other attributed columns get: it already names the row it identifies, so
    the suffix would be redundant. Falls back to the suffixed name when the
    bare id would collide with a column the spine already had natively,
    before this crosswalk started attributing anything (e.g. a locally-scoped
    id column kept by ``resolve_spine``'s ``keep_columns``) — *reserved_cols*
    should be a snapshot taken once at the top of the calling function, not
    the live, growing column set, so this stays consistent across every call
    within one crosswalk attribution.
    """
    if _ID_COLUMN.match(col) and col not in reserved_cols:
        return col
    return f'{col}{suffix}'


def _dominant_by_area(attrs, spine_id_col: str, col: str, area_col: str):
    """Dominant categorical value per spine entity, weighted by overlap area.

    Returns ``(dominant, joined)``: *dominant* is the value with the largest total
    *area_col* within each spine entity; *joined* lists every value for that entity
    ordered by descending area, joined with ``' + '`` — missing where there is only
    one distinct value (see :func:`_join_distinct`). Both are indexed by spine id.
    """
    areas = (
        attrs.groupby([spine_id_col, col])[area_col]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    dominant = areas.drop_duplicates(spine_id_col).set_index(spine_id_col)[col]
    joined = _join_distinct(areas, spine_id_col, col)
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
    mask_ref_src = spine['geometry_source'].str.contains(r'\.', regex=True, na=False)
    if (
        mask_ref_src.any()
        and footprints_from_ref is not None
        and 'parcel_id' in footprints_from_ref.columns
    ):
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


def _attribute_point_reference(
    state: HarmonizeState,
    crosswalk_key: str,
    entity_type: str | None,
    crosswalk: pd.DataFrame,
    columns: list[str] | None,
    collect_ids: bool = False,
) -> HarmonizeState:
    """Attribute a point reference (e.g. NSI) to the spine.

    Points flagged by the link step's recipe-chosen duplicate resolution
    (``duplicate_resolution`` non-null, see
    :func:`~openplaces.io.harmonizer.links.flag_duplicate_points`) are
    excluded here — the merge point — from every aggregate: the match count,
    the value-weighted occupancy/group picks and their ``_all`` summaries,
    the numeric sums/means, and the collected ids. The flagged rows stay on
    ``state.crosswalks`` untouched.
    """
    spine = state.spine
    if 'duplicate_resolution' in crosswalk.columns:
        crosswalk = crosswalk[crosswalk['duplicate_resolution'].isna()]
    suffix = _point_suffix(crosswalk_key, entity_type)
    source_id = source_id_from_recipe_id(crosswalk_key)

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

    # Always write the match count, zeros included, so its presence in the
    # curated output doesn't vary by admin unit (some counties never have a
    # footprint matched to >1 point). Skipped only where the name collides
    # with a renamed attribute output (the Overture case: n_dwellings is
    # itself renamed to n_dwellings_overture, identical to count_col here) --
    # without this check, group_sizes.max() > 1 for that combination would
    # make the assignment below collide with the later n_dwellings sum join.
    count_col = f'n_{entity_type}s_{source_id}' if entity_type else f'n_point{suffix}'
    group_sizes = crosswalk.groupby(spine_id_col).size()
    if count_col not in renamed.values():
        spine[count_col] = group_sizes.reindex(spine.index, fill_value=0).astype(
            'int64'
        )

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
            spine[f'{purpose_group_out}_all'] = _join_distinct(
                footprint_purpose_group_areas, spine_id_col, purpose_group_col
            )

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
        spine[f'{group_out}_all'] = _join_distinct(
            footprint_group_areas, spine_id_col, group_col
        )

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

    # n_dwellings is summed separately (not folded into numeric_agg above) so a
    # point flagged `exclude_from_upward_correction` -- an ESRI record
    # duplicating another source at the same location, e.g. a home-office
    # artifact -- can be excluded from just this sum, without affecting
    # structure_value/year_built, which aren't upward-correction pathways.
    if 'n_dwellings' in avail_cols and 'n_dwellings' in crosswalk.columns:
        dwelling_rows = crosswalk
        if 'exclude_from_upward_correction' in crosswalk.columns:
            dwelling_rows = crosswalk[
                ~crosswalk['exclude_from_upward_correction'].fillna(False)
            ]
        n_dwellings_sum = dwelling_rows.groupby(spine_id_col)['n_dwellings'].sum()
        spine = spine.join(
            n_dwellings_sum.rename(renamed.get('n_dwellings', 'n_dwellings'))
        )

    # Any other requested numeric column (e.g. n_stories, area_sqft) not covered
    # by a special case above: aggregate with the attribute registry's default
    # function (fallback 'mean') so it still reaches the spine.
    remaining_numeric_cols = [
        c
        for c in avail_cols
        if c in crosswalk.columns
        and c not in numeric_agg
        and c != 'n_dwellings'
        and pd.api.types.is_numeric_dtype(crosswalk[c])
    ]
    if remaining_numeric_cols:
        registry_agg = {
            c: get_agg_func(resolve_attribute_name(c)) or 'mean'
            for c in remaining_numeric_cols
        }
        spine = spine.join(
            crosswalk.groupby(spine_id_col).agg(registry_agg).rename(columns=renamed)
        )

    handled = (
        set(numeric_agg)
        | set(remaining_numeric_cols)
        | {
            'purpose_subgroup',
            'occupancy_type',
            'group',
        }
    )
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
    6. A synthetic, parcel-derived fallback geometry (``geometry_source``
       starting with ``'{entity_type}.'``, set by
       :func:`~openplaces.io.harmonizer.links.infer_spine_additions`) is
       always ``'primary'``, overriding the above: it stands in for the
       parcel's one inferred building and was never eligible for the
       crosswalk-seeded evidence rules (it postdates the footprint-parcel
       crosswalk that seeds them).

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
    entity_type = entity_type or 'parcel'

    # A synthetic, reference-derived fallback row (added by
    # infer_spine_additions after the crosswalk below was built, so it can
    # never appear in it) stands in for the reference entity's one inferred
    # building and is always 'primary', regardless of what the
    # crosswalk/evidence rules below would otherwise assign it. geometry_source
    # is prefixed with the entity_type infer_spine_additions was called with
    # (e.g. 'parcel.spine'); matching on entity_type specifically (not just
    # any '.') avoids misclassifying a fallback synthesized from a different
    # reference entity_type.
    is_synthetic = (
        state.spine['geometry_source']
        .astype('string')
        .str.startswith(f'{entity_type}.', na=False)
        if 'geometry_source' in state.spine.columns
        else pd.Series(False, index=state.spine.index)
    )

    parcel_crosswalks = state.get_crosswalks_by_type(entity_type)
    if not parcel_crosswalks:
        if state.verbose:
            print(
                f'  classify_footprint_priority: no {entity_type} crosswalk; skipping.'
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
    role.loc[is_synthetic] = 'primary'

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

    role.loc[is_synthetic] = 'primary'

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
    columns: list[str] | None = None,
) -> HarmonizeState:
    """Build the combined use_group_combined label from use_group / use_subgroup.

    Source-specific raw use codes are mapped to the openplaces use_group and
    use_subgroup vocabulary upstream, via a ``*-remap.csv`` crosswalk that
    ``link_by_id``'s auto-discovery applies automatically when joining a
    source that ships one (see
    :func:`~openplaces.io.harmonizer.links._apply_remap_csvs`); this step only
    combines the two into the label the parcel land-use classifier groups and
    votes on, so it holds no code vocabulary of its own.

    Falls back to whichever of ``use_group`` / ``use_subgroup`` reached the
    spine when only one did (e.g. a source, like Florida's DOR use code, with
    no subgroup taxonomy to crosswalk) rather than skipping the whole column
    -- per-row as well as per-column, so a row with one field blank (empty or
    whitespace-only, not just ``NaN``) falls back to the other alone instead
    of producing a degenerate ``' | '`` label.

    Parameters
    ----------
    combined_column : str, optional
        Output combined-label column (default ``use_group_combined``).
    columns : list, optional
        The label's parts, joined in order. Each entry is either a column
        name, or a list of alternative column names coalesced per row --
        the first of them that has a value wins, and the rest are ignored
        for that row. Defaults to
        ``[['use_group', 'use_group_code'], ['use_subgroup', 'use_subgroup_code']]``.

        The alternative groups exist because land use arrives in whichever
        column the source happens to populate. A source that ships a raw
        code has it crosswalked into the ``use_group`` / ``use_subgroup``
        vocabulary upstream, so the code contributes nothing extra and is
        skipped; but a source whose code has *no* crosswalk (or whose
        crosswalk does not cover that code) would otherwise contribute
        nothing at all, even though the code is perfectly good grouping
        and voting evidence. Galveston County, TX is the measured case:
        no state category code and no crosswalk for its local codes, but
        98% coverage of ``use_subgroup_code``. Coalescing rather than
        appending is what keeps this from fragmenting cohorts -- where the
        crosswalk did fire, the code adds no new label values.

        Listing ``building_style`` alongside them lets counties whose
        land-use text carries no occupancy signal contribute one from
        their structure description -- Sampson County, NC is the
        motivating case: its land-use column is a *land segment* type
        (homesite / cropland / woodland) that matches no land-use keyword,
        while its style column names the structure and identifies
        thousands of manufactured homes the classifier would otherwise
        never see. Note that the combined label is also the grouping key
        for cohort statistics such as ``footprint_area_log_zscore``, so
        adding a high-cardinality column fragments those cohorts; add one
        only where it earns its place.
    """
    if state.spine is None:
        return state
    spine = state.spine
    groups = [
        [entry] if isinstance(entry, str) else list(entry)
        for entry in (
            columns
            or [
                ['use_group', 'use_group_code'],
                ['use_subgroup', 'use_subgroup_code'],
            ]
        )
    ]
    present = [[c for c in group if c in spine.columns] for group in groups]
    present = [group for group in present if group]
    if not present:
        if state.verbose:
            flat = [c for group in groups for c in group]
            print(f'  derive_use_classes: none of {flat} on spine; skipping.')
        return state

    def _clean(column: str) -> pd.Series:
        cleaned = spine[column].astype('string').str.strip()
        return cleaned.mask(cleaned.eq(''))

    filled: dict[str, int] = {}

    def _coalesce(group: list[str]) -> pd.Series:
        """First column of *group* with a value, per row."""
        part = _clean(group[0])
        for column in group[1:]:
            alternative = _clean(column)
            gap = part.isna() & alternative.notna()
            if gap.any():
                filled[column] = int(gap.sum())
            part = part.mask(gap, alternative)
        return part

    # Join whichever parts a row actually has, so a row missing one field
    # falls back to the others alone rather than producing a degenerate
    # ' | ' label -- per row, not just per column.
    label = None
    for group in present:
        part = _coalesce(group)
        if label is None:
            label = part
            continue
        both = label.notna() & part.notna()
        only_part = label.isna() & part.notna()
        label = label.mask(both, label.fillna('') + ' | ' + part.fillna(''))
        label = label.mask(only_part, part)

    spine[combined_column] = pd.Categorical(label)
    state.spine = spine
    if state.verbose:
        mapped = int(label.notna().sum())
        print(f'  derive_use_classes: combined {mapped:,d}/{len(spine):,d} parcels')
        for column, count in filled.items():
            print(
                f'    fell back to {column} for {count:,d} row(s) with no '
                f'crosswalked value'
            )
    return state


@_register('summarize_footprint_morphology')
def summarize_footprint_morphology(
    state: HarmonizeState,
    footprint_recipe_id: str,
    small_area_max_m2: float = 185.0,
    elongated_aspect_min: float = 2.0,
    on: str | None = 'parcel_id',
    min_overlap_m2: float = 10.0,
    overlap_column: str = 'area_intersection_m2_parcel',
    priority_column: str = 'priority_on_parcel',
    dwelling_column: str = 'n_dwellings_overture',
    span_column: str = 'n_parcels_per_footprint',
) -> HarmonizeState:
    """Attach per-parcel footprint morphology aggregates to the parcel spine.

    Footprint-to-parcel linkage is the harmonizer's job; this reads the footprint
    entity, assigns each footprint to a parcel (by a shared id column when *on* is
    present on both sides, else by spatial containment of the footprint's
    representative point), and writes the per-parcel counts the parcel land-use
    classifier consumes downstream: ``n_footprints_per_parcel``,
    ``n_small_elongated_footprints_per_parcel`` (manufactured-home-shaped),
    ``max_footprint_area_m2``, ``footprint_area_m2_dominant``,
    ``footprint_area_m2_in_parcel``, ``n_primary_footprints_per_parcel``,
    ``footprint_area_m2_primary``, ``max_parcels_per_footprint``, and
    ``max_dwellings_per_footprint``. The classification itself is parcel-curate
    work.

    ``max_parcels_per_footprint`` and ``max_dwellings_per_footprint`` are scoped
    to *dwelling-confirmed* footprints only (*is_primary_candidate* — see below —
    AND a positive *dwelling_column*): ``priority_on_parcel == 'primary'`` alone
    is not evidence of a real dwelling — it's also assigned to a sole footprint
    on its own parcel, an NSI-only-evidence footprint, or a synthetic fallback
    row, none of which confirm occupancy. ``max_parcels_per_footprint`` is the
    largest *span_column* value among those confirmed footprints (does any of
    this parcel's confirmed footprints span multiple parcels — a real,
    non-FEMA-geometry building-shape signal); ``max_dwellings_per_footprint`` is
    the largest confirmed dwelling count on any single one of them (a footprint
    itself holding multiple dwellings, independent of parcel boundaries).

    ``on`` defaults to the globally-unique ``parcel_id`` rather than
    ``parcel_id_local``: the latter is only locally cross-comparable and can
    collide across genuinely distinct parcels within one admin unit (e.g. every
    unit of a condo complex sharing one assessor PIN), which would silently
    merge their footprint counts together.

    ``n_footprints_per_parcel`` counts every footprint row that clears
    *min_overlap_m2* (its overlap with this parcel, via *overlap_column*, when
    present), including a parcel-derived synthetic fallback geometry
    (``geometry_source`` containing ``.``, set by :func:`infer_spine_additions`)
    regardless of overlap size when that is the parcel's only footprint — a
    fallback's geometry is the parcel boundary, so its "overlap" is trivially the
    whole parcel and it needs no floor. ``n_small_elongated_footprints_per_parcel``,
    ``max_footprint_area_m2``, and ``footprint_area_m2_dominant`` additionally
    exclude synthetic rows entirely: a fallback's area/aspect ratio are not
    meaningful size/shape evidence. Like ``max_footprint_area_m2``,
    ``footprint_area_m2_dominant`` stays ``NaN`` (not ``0``) for a parcel with no
    real, non-synthetic footprint at all.
    ``n_primary_footprints_per_parcel`` additionally excludes footprints whose
    *priority_column* value (when present) is ``'secondary'`` — a real, but
    accessory, structure (garage, shed) that clears the overlap floor but isn't a
    distinct home; synthetic fallback rows still count here too.
    ``footprint_area_m2_primary`` sums real footprint area over that same
    non-secondary subset (excluding synthetic rows, like
    ``footprint_area_m2_dominant``).

    ``footprint_area_m2_dominant`` and ``footprint_area_m2_primary`` both sum
    each contributing footprint's full, unclipped polygon area — even when
    part of that footprint's geometry actually lies outside the parcel,
    because the footprint straddles a boundary. **Never divide either by
    parcel area to estimate footprint coverage share — the ratio can exceed
    1.** ``footprint_area_m2_in_parcel`` is the coverage-safe alternative:
    for every footprint that clears *min_overlap_m2* against *this* parcel
    specifically (including footprints whose dominant parcel is a different,
    neighboring one but that still spill a real sliver onto this parcel), it
    sums that footprint's geometry clipped to this parcel's own boundary (a
    real geometric intersection, via :func:`openplaces.geo.polygon.overlay_polygons`).
    It is bounded by construction to at most this parcel's own area, and
    (like the two sums above) excludes synthetic rows and stays ``NaN`` for a
    parcel with no real footprint coverage at all.

    Parameters
    ----------
    footprint_recipe_id : str
        Footprint entity recipe to read (geometry required).
    small_area_max_m2, elongated_aspect_min : float, optional
        A footprint counts as small-and-elongated (manufactured-home morphology)
        when its area is at most *small_area_max_m2* and its oriented aspect ratio
        is at least *elongated_aspect_min*.
    on : str, optional
        Shared id for an id-based assignment (default ``parcel_id``): a footprint
        entity column matched against either a same-named spine column or (for a
        parcel spine, whose own true id lives on the index during this step --
        see ``resolve_spine``) the spine's index. When absent on either side, a
        spatial within-join is used instead (which has no overlap-area or
        priority data to apply *min_overlap_m2*/*priority_column* with, so every
        contained footprint counts toward all four outputs). When a
        ``{on}_all`` column is present (e.g. ``parcel_id_all``, written by
        :func:`_attribute_polygon_reference`), every pipe-joined id counts
        toward this parcel's outputs, not just *on*'s single dominant one --
        the fix for a footprint spanning many parcels (e.g. one real building
        over many tiny condo-unit parcels) that would otherwise only ever
        credit whichever parcel it overlaps most.
    min_overlap_m2 : float, optional
        Minimum overlap (m²) with this parcel, from *overlap_column*, for a
        non-synthetic footprint to count toward any of the four outputs (default
        10, matching this codebase's ``area_intersection_m2_min`` convention).
        Excludes a sliver footprint whose only detected parcel candidate happens
        to be this one.
    overlap_column : str, optional
        Footprint-entity column holding each footprint's overlap area (m²) with
        its dominant parcel (default ``area_intersection_m2_parcel``, written by
        :func:`_attribute_polygon_reference`). Ignored (no floor applied) if
        absent from the footprint entity.
    priority_column : str, optional
        Footprint-entity column holding each footprint's structural role on its
        parcel (default ``priority_on_parcel``, written by
        :func:`classify_footprint_priority`). Ignored (no exclusion applied) if
        absent from the footprint entity.
    dwelling_column : str, optional
        Footprint-entity column holding each footprint's confirmed dwelling
        count (default ``n_dwellings_overture``). A footprint counts toward
        ``max_parcels_per_footprint``/``max_dwellings_per_footprint`` only when
        this is positive. Ignored (neither output is set) if absent from the
        footprint entity.
    span_column : str, optional
        Footprint-entity column holding how many parcels that footprint spans
        (default ``n_parcels_per_footprint``, written by
        :func:`_attribute_polygon_reference` when the footprint spine attributes
        the parcel reference). Ignored (``max_parcels_per_footprint`` not set) if
        absent from the footprint entity.
    """
    import geopandas as gpd

    from openplaces.geo.polygon import local_metric_crs, overlay_polygons
    from openplaces.io.harmonizer.spine import get_oriented_dims
    from openplaces.io.readers import get_entities

    if state.spine is None:
        return state
    spine = state.spine

    # missing='ignore': an admin unit can genuinely have no footprint
    # coverage at all (no OBM/Microsoft source ingested there), the same
    # tolerated case this function's own empty check below already expects.
    footprints = get_entities(
        footprint_recipe_id, state.admin_id, geom=True, missing='ignore'
    )
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

    # Parcel-derived synthetic fallback geometries (geometry_source containing
    # '.', set by infer_spine_additions) are the parcel boundary, not a building
    # outline, so their area/aspect are excluded from the size/shape aggregates
    # (max_footprint_area_m2, n_small_elongated_footprints_per_parcel). They
    # still count toward n_footprints_per_parcel below: that count exists
    # because independent value evidence already implied a building is there.
    if 'geometry_source' in footprints.columns:
        is_synthetic = (
            footprints['geometry_source']
            .astype('string')
            .str.contains(r'\.', regex=True, na=False)
            .to_numpy()
        )
    else:
        is_synthetic = np.zeros(len(footprints), dtype=bool)
    real_area = np.where(is_synthetic, np.nan, area)
    real_small_elong = small_elong & ~is_synthetic

    # A non-synthetic footprint must clear the overlap floor to count toward any
    # output (a sliver footprint whose only detected parcel candidate is this one
    # is not real evidence of a structure on it); synthetic fallback rows bypass
    # this, same reasoning as real_area/real_small_elong above. Missing
    # overlap_column (e.g. a not-yet-regenerated footprint entity) disables the
    # floor entirely, matching the function's prior behavior.
    if overlap_column in footprints.columns:
        overlap = pd.to_numeric(footprints[overlap_column], errors='coerce').to_numpy()
        meets_floor = is_synthetic | (overlap >= min_overlap_m2)
    else:
        meets_floor = np.ones(len(footprints), dtype=bool)

    # A footprint marked 'secondary' (an accessory structure, per
    # classify_footprint_priority's dwelling/NSI-point evidence) clears the floor
    # but isn't a distinct home; excluded only from the stricter
    # n_primary_footprints_per_parcel below. Synthetic rows bypass this too.
    if priority_column in footprints.columns:
        not_secondary = (
            is_synthetic
            | (footprints[priority_column].astype('string') != 'secondary').to_numpy()
        )
    else:
        not_secondary = np.ones(len(footprints), dtype=bool)
    is_primary_candidate = meets_floor & not_secondary

    # Confirmed-dwelling subset for max_parcels_per_footprint/
    # max_dwellings_per_footprint: is_primary_candidate alone lets through
    # sole-footprint-on-parcel, NSI-only, and synthetic rows with no dwelling
    # evidence at all, so a positive dwelling_column is required on top of it.
    if dwelling_column in footprints.columns:
        dwellings = (
            pd.to_numeric(footprints[dwelling_column], errors='coerce')
            .fillna(0)
            .to_numpy()
        )
        dwelling_confirmed = is_primary_candidate & (dwellings > 0)
    else:
        dwellings = np.zeros(len(footprints))
        dwelling_confirmed = np.zeros(len(footprints), dtype=bool)

    if span_column in footprints.columns:
        n_parcels_span = pd.to_numeric(
            footprints[span_column], errors='coerce'
        ).to_numpy()
    else:
        n_parcels_span = np.full(len(footprints), np.nan)

    # A parcel spine's own true id (parcel_id) lives on the index during this
    # step under a renamed, working index name (e.g. 'spine_id') --
    # resolve_spine records the *original* name in
    # state.metadata['spine_index_name'] and only restores it at save time --
    # so 'on' may match a plain column, the spine's current index name, or
    # that recorded original name, not just a column.
    on_in_spine = on and (
        on in spine.columns
        or on == spine.index.name
        or on == state.metadata.get('spine_index_name')
    )
    if on and on in footprints.columns and on_in_spine:
        # {on}_all (written by _attribute_polygon_reference, e.g.
        # 'parcel_id_all') pipe-joins every reference this footprint links to
        # in the trimmed crosswalk, not just its single dominant one -- a
        # footprint spanning several parcels (e.g. one real building over
        # many tiny condo-unit parcels) would otherwise only ever credit
        # whichever parcel it overlaps most, leaving every other linked
        # parcel silently uncounted by every output below. Falls back to the
        # plain dominant-only column when the reference recipe hasn't been
        # regenerated with it yet.
        on_all_col = f'{on}_all'
        pid_source = (
            footprints[on_all_col]
            if on_all_col in footprints.columns
            else footprints[on]
        )
        per_fp = pd.DataFrame(
            {
                '_pid': pid_source.astype('string').to_numpy(),
                '_a': real_area,
                '_se': real_small_elong,
                '_meets_floor': meets_floor,
                '_primary': is_primary_candidate,
                '_dwellings': dwellings,
                '_confirmed': dwelling_confirmed,
                '_span': n_parcels_span,
            }
        ).dropna(subset=['_pid'])
        if on_all_col in footprints.columns:
            per_fp = per_fp.assign(_pid=per_fp['_pid'].str.split('|')).explode('_pid')
        key = (
            spine[on].astype('string')
            if on in spine.columns
            else spine.index.to_series().astype('string')
        )

        grp = per_fp[per_fp['_meets_floor']].groupby('_pid')
        n_fp = key.map(grp.size())
        n_se = key.map(grp['_se'].sum())
        max_a = key.map(grp['_a'].max())
        # min_count=1: an all-synthetic (all-NaN) or footprint-less group must
        # stay NaN here too, matching max_a's "no real footprint" semantics,
        # not silently become 0 (pandas' default sum-of-nothing).
        sum_a = key.map(grp['_a'].sum(min_count=1))

        grp_primary = per_fp[per_fp['_primary']].groupby('_pid')
        n_primary = key.map(grp_primary.size())
        sum_a_primary = key.map(grp_primary['_a'].sum(min_count=1))

        grp_confirmed = per_fp[per_fp['_confirmed']].groupby('_pid')
        max_dwellings = key.map(grp_confirmed['_dwellings'].max())
        max_span = key.map(grp_confirmed['_span'].max())
    else:
        reps = gpd.GeoDataFrame(
            {
                '_a': real_area,
                '_se': real_small_elong,
                '_meets_floor': meets_floor,
                '_primary': is_primary_candidate,
                '_dwellings': dwellings,
                '_confirmed': dwelling_confirmed,
                '_span': n_parcels_span,
            },
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

        grp = joined[joined['_meets_floor']].groupby('_spine_id')
        n_fp = grp.size().reindex(spine.index)
        n_se = grp['_se'].sum().reindex(spine.index)
        max_a = grp['_a'].max().reindex(spine.index)
        sum_a = grp['_a'].sum(min_count=1).reindex(spine.index)

        grp_primary = joined[joined['_primary']].groupby('_spine_id')
        n_primary = grp_primary.size().reindex(spine.index)
        sum_a_primary = grp_primary['_a'].sum(min_count=1).reindex(spine.index)

        grp_confirmed = joined[joined['_confirmed']].groupby('_spine_id')
        max_dwellings = grp_confirmed['_dwellings'].max().reindex(spine.index)
        max_span = grp_confirmed['_span'].max().reindex(spine.index)

    # footprint_area_m2_in_parcel: real, clipped footprint area actually
    # inside this parcel's own boundary, across every non-synthetic footprint
    # that touches it above min_overlap_m2 -- including footprints whose
    # *dominant* parcel is a different, neighboring one but that still spill
    # a real sliver onto this parcel. Unlike footprint_area_m2_dominant/
    # footprint_area_m2_primary (full, unclipped footprint area, credited
    # only to the one dominant parcel), this is bounded by construction to at
    # most this parcel's own area -- computed independently of the id-join
    # vs. spatial-fallback branch above, since it needs every touching
    # parcel, not just each footprint's single assigned one.
    non_synth = footprints.loc[~is_synthetic]
    if len(non_synth) == 0:
        in_parcel_sum = pd.Series(np.nan, index=spine.index)
    else:
        pairs = overlay_polygons(
            spine,
            non_synth.to_crs(spine.crs),
            how='intersection',
            area_intersection=True,
            geom=False,
            suffixes=('_spine', '_footprint'),
        )
        if len(pairs) == 0:
            in_parcel_sum = pd.Series(np.nan, index=spine.index)
        else:
            clipped = pairs['area_intersection_m2']
            clipped = clipped[clipped >= min_overlap_m2]
            in_parcel_sum = clipped.groupby(level=0).sum().reindex(spine.index)

    spine['n_footprints_per_parcel'] = n_fp.fillna(0).astype('int64')
    spine['n_small_elongated_footprints_per_parcel'] = n_se.fillna(0).astype('int64')
    spine['max_footprint_area_m2'] = max_a
    spine['footprint_area_m2_dominant'] = sum_a
    spine['footprint_area_m2_in_parcel'] = in_parcel_sum
    spine['n_primary_footprints_per_parcel'] = n_primary.fillna(0).astype('int64')
    spine['footprint_area_m2_primary'] = sum_a_primary
    spine['max_dwellings_per_footprint'] = max_dwellings.fillna(0).astype('int64')
    spine['max_parcels_per_footprint'] = max_span.fillna(0).astype('int64')
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


@_register('detect_shared_land_groups')
def detect_shared_land_groups(
    state: HarmonizeState,
    land_value_column: str = 'land_value',
    improvement_value_column: str = 'improvement_value',
    max_land_area_ha: float = 2.0,
    max_land_aspect_ratio: float = 2.5,
    min_group_size: int = 2,
    max_group_size: int = 15,
    group_id_column: str = 'shared_land_parcel_id',
    property_count_column: str = 'n_properties_per_parcel',
    source_column: str = 'property_source',
) -> HarmonizeState:
    """Detect shared-land parcel groups (horizontally-separated townhomes).

    A real, MassGIS-condo-style per-unit source (:func:`link_by_id`'s
    property-spine count join) represents one land parcel plus several
    *virtual* sub-records with no geometry of their own. This step detects
    the horizontally-separated mirror image, observed in Carteret County,
    NC: two or more separately-parceled, real unit polygons (each with its
    own improvement value -- e.g. a townhome) clustered around a
    *different*, real "land" parcel that carries no value of its own at all
    (the common land's value having been rolled into the units, exactly
    like a MassGIS condo's land parcel).

    Real cadastral parcels tile the plane rather than overlapping, so a
    shared-land parcel is *adjacent* to its units, not literally overlapping
    them -- this uses a ``'touches'`` spatial join (shared boundary), not
    area intersection. Adjacency alone massively overcounts: it also matches
    every ordinary lot fronting a road or water body, whose GIS parcel can
    be a single record touching hundreds of unrelated lots county-wide.
    *max_land_area_ha* and *max_land_aspect_ratio* (oriented length/width,
    via :func:`openplaces.io.harmonizer.spine.get_oriented_dims`) filter a
    land candidate down to what a real, compact shared common area looks
    like, rejecting the long thin shape of a road/right-of-way/waterway
    parcel; *max_group_size* additionally rejects an implausibly large
    cluster (a subdivision-wide common area or HOA amenity parcel, not a
    handful of townhome units sharing a strip of land). This is a heuristic,
    not a proof of enclosure -- validated empirically against Carteret
    County, NC (e.g. an 8-unit group at "111 New Bern St, Atlantic Beach,
    NC", the land parcel itself carrying $0 of both values); the thresholds
    are tunable per state if they prove too permissive or too strict
    elsewhere.

    A parcel is a land candidate when both *land_value_column* and
    *improvement_value_column* are 0 (or missing), its own area is at most
    *max_land_area_ha*, and its oriented aspect ratio is at most
    *max_land_aspect_ratio*. A (non-candidate) parcel with positive
    *improvement_value_column* qualifies as a unit when it touches a land
    candidate; a unit touching more than one land candidate is assigned to
    whichever it shares the most touching pairs with (ties broken
    arbitrarily). Land candidates whose qualifying unit count falls outside
    ``[min_group_size, max_group_size]`` are discarded.

    Writes *property_count_column* / *source_column* (``'shared_land_group'``)
    on each qualifying land parcel (the group's unit count) and each
    enclosed unit parcel (count 1 -- it is itself one property), and
    *group_id_column* on each unit parcel (its land parcel's own index
    value) -- run before :func:`estimate_property_counts`, which never
    overwrites a *source_column* value this step already set.

    Parameters
    ----------
    land_value_column, improvement_value_column : str, optional
        Bare parcel value columns (defaults ``'land_value'``,
        ``'improvement_value'``).
    max_land_area_ha : float, optional
        Maximum area for a land candidate (default 2.0 ha).
    max_land_aspect_ratio : float, optional
        Maximum oriented length/width ratio for a land candidate (default
        2.5) -- excludes elongated road/right-of-way/waterway parcels.
    min_group_size, max_group_size : int, optional
        Qualifying unit count window for a land candidate's group to count
        (default 2-15) -- excludes a single ambiguous enclosed parcel
        (could be a driveway easement) and an implausibly large cluster.
    group_id_column : str, optional
        Output column on each unit parcel, holding its land parcel's index
        value (default ``'shared_land_parcel_id'``).
    property_count_column, source_column : str, optional
        Output rollup columns (defaults ``'n_properties_per_parcel'``,
        ``'property_source'``) -- shared with :func:`estimate_property_counts`
        and the property-spine count join.
    """
    if state.spine is None:
        return state
    spine = state.spine
    required = {land_value_column, improvement_value_column, 'area_ha'}
    if not required.issubset(spine.columns):
        if state.verbose:
            print(
                f'  detect_shared_land_groups: {sorted(required - set(spine.columns))} '
                'missing; skipping.'
            )
        return state

    import geopandas as gpd

    from openplaces.geo.polygon import local_metric_crs
    from openplaces.io.harmonizer.spine import get_oriented_dims

    land_value = pd.to_numeric(spine[land_value_column], errors='coerce').fillna(0)
    improvement_value = pd.to_numeric(
        spine[improvement_value_column], errors='coerce'
    ).fillna(0)
    area_ha = pd.to_numeric(spine['area_ha'], errors='coerce')

    is_land_candidate = (
        (land_value <= 0)
        & (improvement_value <= 0)
        & area_ha.notna()
        & (area_ha > 0)
        & (area_ha <= max_land_area_ha)
    )
    is_unit_candidate = improvement_value > 0

    if not is_land_candidate.any() or not is_unit_candidate.any():
        if state.verbose:
            print('  detect_shared_land_groups: no candidates found.')
        return state

    land_sub = spine.loc[is_land_candidate]
    geom_m = land_sub.geometry.to_crs(local_metric_crs(land_sub))
    dims = geom_m.map(get_oriented_dims)
    length = np.array([d[1] for d in dims])
    width = np.clip(np.array([d[2] for d in dims]), 1e-6, None)
    is_land_candidate.loc[land_sub.index] = (length / width) <= max_land_aspect_ratio

    if not is_land_candidate.any():
        if state.verbose:
            print('  detect_shared_land_groups: no compact candidates found.')
        return state

    idx_name = spine.index.name or 'index'
    land_gdf = spine.loc[is_land_candidate, ['geometry']].reset_index()
    unit_gdf = spine.loc[is_unit_candidate, ['geometry']].reset_index()
    land_col, unit_col = f'{idx_name}_land', f'{idx_name}_unit'
    touching = gpd.sjoin(
        unit_gdf,
        land_gdf,
        predicate='touches',
        how='inner',
        lsuffix='_unit',
        rsuffix='_land',
    ).rename(columns={f'{idx_name}__unit': unit_col, f'{idx_name}__land': land_col})
    if len(touching) == 0:
        if state.verbose:
            print('  detect_shared_land_groups: no touching units found.')
        return state

    # A unit touching more than one land candidate goes to whichever it
    # shares the most touching pairs with (a compact land candidate normally
    # touches a unit along one boundary segment, so this is rarely a tie).
    pair_counts = (
        touching.groupby([unit_col, land_col]).size().rename('_n').reset_index()
    )
    pair_counts = pair_counts.sort_values('_n', ascending=False)
    assigned = pair_counts.drop_duplicates(subset=[unit_col], keep='first')

    group_sizes = assigned.groupby(land_col).size()
    qualifying = group_sizes[
        (group_sizes >= min_group_size) & (group_sizes <= max_group_size)
    ]
    if qualifying.empty:
        if state.verbose:
            print(
                f'  detect_shared_land_groups: no group with '
                f'{min_group_size}-{max_group_size} units found.'
            )
        return state
    assigned = assigned[assigned[land_col].isin(qualifying.index)]

    source = (
        spine[source_column].astype(object)
        if source_column in spine.columns
        else pd.Series(pd.NA, index=spine.index, dtype=object)
    )
    count = (
        pd.to_numeric(spine[property_count_column], errors='coerce')
        if property_count_column in spine.columns
        else pd.Series(np.nan, index=spine.index)
    )

    is_qualifying_land = spine.index.to_series().isin(qualifying.index)
    source = source.mask(is_qualifying_land, 'shared_land_group')
    count = count.mask(is_qualifying_land, spine.index.to_series().map(qualifying))

    unit_to_land = assigned.set_index(unit_col)[land_col]
    is_unit = spine.index.to_series().isin(unit_to_land.index)
    source = source.mask(is_unit, 'shared_land_group')
    count = count.mask(is_unit, 1)

    spine[property_count_column] = count
    spine[source_column] = source
    spine[group_id_column] = spine.index.to_series().map(unit_to_land)
    state.spine = spine

    if state.verbose:
        print(
            f'  detect_shared_land_groups: {len(qualifying):,} shared-land groups '
            f'covering {len(unit_to_land):,} units.'
        )
    return state


@_register('detect_condo_building_clusters')
def detect_condo_building_clusters(
    state: HarmonizeState,
    land_value_column: str = 'land_value',
    improvement_value_column: str = 'improvement_value',
    area_column: str = 'area_ha',
    use_subgroup_column: str = 'use_subgroup',
    max_unit_area_ha: float = 0.05,
    max_hub_area_ha: float = 3.0,
    max_hub_aspect_ratio: float = 5.0,
    min_group_size: int = 2,
    max_group_size: int = 200,
    group_id_column: str = 'building_cluster_id',
) -> HarmonizeState:
    """Detect stacked-condo-unit parcel clusters sharing one physical building.

    A real, vertically-stacked condo building (unlike the horizontally-
    separated townhome pattern :func:`detect_shared_land_groups` targets) is
    platted as one legal parcel per unit, each a tiny sliver of the
    building's own footprint (tens to low hundreds of m², well below the
    thousands of m² typical of an ordinary house lot), often alongside one
    or more shared "common area" parcels (land held in common -- $0 land
    AND improvement value, e.g. Carteret County, NC's condo-recording
    convention) that the units touch. Confirmed on real data (a 24-99 unit
    oceanfront complex in Carteret County): a single building's real
    footprint can end up split across the parcel/footprint linking pipeline
    as one or two partial real footprints (dominant-linked only to the
    shared common-area parcel, per-unit overlap shares too small to survive
    :func:`~openplaces.io.harmonizer.links.link_to_reference`'s
    ``fraction_of_largest`` trim) plus a scatter of per-unit synthetic
    fallback footprints for whichever units the real footprint(s) don't
    happen to dominate -- this step exists to recognize the underlying
    one-building reality *before* that footprint-side mess, from parcel-side
    evidence alone, so a later footprint-spine step can consolidate it back
    into one coherent shape.

    This reuses :func:`detect_shared_land_groups`'s ``'touches'``-adjacency
    pattern but is a distinct step with a distinct trigger, not a parameter
    variant of it: raising that step's own ``max_group_size`` to cover a
    30-unit condo building would re-admit the county-wide `WATER`/`ROW`
    placeholder-parcel false positives it excludes by design. The
    discriminator here is *unit parcel size* (``max_unit_area_ha``) rather
    than that step's near-zero-*hub*-value-only test, which alone cannot
    tell a condo common area from an HOA park serving ordinary
    single-family lots.

    A unit candidate has area at most *max_unit_area_ha* and positive
    *improvement_value_column*. A hub candidate (optional -- a cluster with
    no separate common-area parcel still forms from mutually-touching units
    alone) has near-zero land and improvement value, area at most
    *max_hub_area_ha*, and oriented aspect ratio at most
    *max_hub_aspect_ratio*. Either candidacy is denied outright to a parcel
    whose *use_subgroup_column* (falling back to ``'use_group_combined'``)
    text matches a small blocklist of known single-family/mobile-home
    labels -- a best-effort, defense-in-depth safety net, since source
    ``use_subgroup`` text is not standardized across recipes (confirmed:
    one county's own single-family label is ``'RESIDENTIAL PRIMARY'``, not
    ``'single family'``), applied only where that column is present and
    non-null; see :func:`_cluster_condo_parcels` for the exact blocklist.

    Clusters are the connected components of the ``'touches'`` adjacency
    graph restricted to unit and hub candidates, with one refinement: a
    shared hub can attach to more than one unit sub-cluster (a common area
    legitimately serving several buildings) but never merges those
    sub-clusters into each other -- unit-unit edges alone determine which
    units belong together (confirmed on real data, Carteret County, NC: a
    0.56 ha shared common-area parcel otherwise chains 3 physically
    separate townhouse-unit groups, 30-130 m apart, into one bogus
    component). A component's *group_id_column* is one of its own unit
    parcel index values (the smallest, for a stable choice), so both a
    hub-anchored and a hub-less cluster resolve the same way. Components
    outside ``[min_group_size, max_group_size]`` are dropped. Because a hub
    can now belong to more than one qualifying cluster, this column can
    only record one of them per hub row (picked deterministically, not
    meaningfully); the full multi-cluster membership is only available via
    :func:`_cluster_condo_parcels`'s own return value.

    Parameters
    ----------
    land_value_column, improvement_value_column : str, optional
        Bare parcel value columns (defaults ``'land_value'``,
        ``'improvement_value'``).
    area_column : str, optional
        Parcel area in hectares (default ``'area_ha'``).
    use_subgroup_column : str, optional
        Land-use subgroup column checked against the single-family/mobile-
        home blocklist (default ``'use_subgroup'``); falls back to
        ``'use_group_combined'`` when absent, and is skipped entirely when
        neither is present.
    max_unit_area_ha : float, optional
        Maximum area for a unit candidate (default 0.05 ha = 500 m² --
        captures ~93% of Carteret County, NC's own condo/townhouse unit
        parcels by area while only 5.8% of ordinary single-family lots
        there fall under it, with *use_subgroup_column* as backstop for
        that residual risk).
    max_hub_area_ha, max_hub_aspect_ratio : float, optional
        Hub candidate filters, same semantics as
        :func:`detect_shared_land_groups` (defaults 3.0 ha, 5.0). The area
        default covers the 95th percentile of Carteret County, NC's own
        zero-value-candidate parcels (2.63 ha) and mainly guards against an
        oversized-but-compact false hub (e.g. a golf course) admitting
        unrelated ordinary lots as "units" -- a risk already reduced by
        *max_unit_area_ha* and the use-subgroup blocklist above, so this
        cap is a secondary backstop. The aspect-ratio default is what
        actually rejects an elongated `ROW`/`WATER` shape regardless of
        area; raised from an earlier, stricter 2.5 because real
        `'COMMON AREA'`-labeled parcels are routinely elongated by their
        own function (beach walkways, drainage strips, canal-front
        access) -- 2.5 excluded roughly a third of them (Carteret County,
        NC: 75th percentile aspect ratio 3.25, 90th 5.15) -- while the
        residual `ROW`/`WATER` false-positive risk this guards against is
        narrower than when this filter was first set, now that
        *max_unit_area_ha* and the use-subgroup blocklist independently
        keep most ordinary lots from ever becoming unit candidates in the
        first place.
    min_group_size, max_group_size : int, optional
        Component size window (default 2-200) -- wide enough for a large
        real resort/motel complex's full unit count (confirmed on real
        data, Carteret County, NC: a legitimate 96-parcel building cluster
        at "1505 Salter Path Rd"), still finite as a backstop against a
        long chain of directly-touching unit candidates with no hub at all
        (e.g. a dense block of many separate small rowhouses) -- the
        *other* runaway-component risk this window guards against, beyond
        the shared-hub risk the two-pass union-find above (see "Clusters
        are..." further up) already handles structurally regardless of
        this cap.
    group_id_column : str, optional
        Output column on every cluster member, hub included (default
        ``'building_cluster_id'``).
    """
    if state.spine is None:
        return state
    spine = state.spine
    result = _cluster_condo_parcels(
        spine,
        land_value_column=land_value_column,
        improvement_value_column=improvement_value_column,
        area_column=area_column,
        use_subgroup_column=use_subgroup_column,
        max_unit_area_ha=max_unit_area_ha,
        max_hub_area_ha=max_hub_area_ha,
        max_hub_aspect_ratio=max_hub_aspect_ratio,
        min_group_size=min_group_size,
        max_group_size=max_group_size,
        verbose=state.verbose,
    )
    if result is None:
        return state
    # A hub parcel is tagged with the same building_cluster_id as its
    # units here regardless -- the geometry-side hub/unit distinction
    # only matters to consolidate_condo_cluster_footprints.
    component, _hub_ids = result
    # A hub attached to multiple qualifying clusters (change 3) has a
    # duplicate index entry in `component`; a scalar column can only hold
    # one, so keep the first (deterministic, not otherwise meaningful).
    component_for_column = component[~component.index.duplicated(keep='first')]

    spine[group_id_column] = spine.index.to_series().map(component_for_column)
    state.spine = spine
    return state


# Case-insensitive substrings of use_subgroup/use_group_combined text that
# indicate a parcel is a known single-family or mobile/manufactured-home
# unit -- never a condo/townhome unit or a common-area hub. Best-effort
# only: source use_subgroup text is not standardized across recipes (one
# county's own single-family label is 'RESIDENTIAL PRIMARY', not 'single
# family'), so this is a defense-in-depth layer on top of the area-based
# candidacy test and the hub-bridging fix below, not a substitute for
# either.
_NON_CONDO_USE_SUBGROUP_TERMS = (
    'single family',
    'single-family',
    'mobile home',
    'manufactured home',
    'residential primary',
)


def _cluster_condo_parcels(
    parcels,
    land_value_column: str = 'land_value',
    improvement_value_column: str = 'improvement_value',
    area_column: str = 'area_ha',
    use_subgroup_column: str = 'use_subgroup',
    max_unit_area_ha: float = 0.05,
    max_hub_area_ha: float = 3.0,
    max_hub_aspect_ratio: float = 5.0,
    min_group_size: int = 2,
    max_group_size: int = 200,
    verbose: bool = False,
) -> tuple[pd.Series, set] | None:
    """Core clustering logic behind :func:`detect_condo_building_clusters`.

    Factored out (state-free, operating on any parcel GeoDataFrame) so
    :func:`~openplaces.io.harmonizer.links.consolidate_condo_cluster_footprints`
    can run the identical detection against the raw parcel reference a
    footprint-spine pipeline already has loaded, not just against an already-
    harmonized parcel spine -- the two recipes' pipelines run in an order
    (footprint spine before parcel spine, since the parcel spine's own
    ``summarize_footprint_morphology`` step reads the harmonized footprint
    spine) that makes a footprint-side step reading the parcel spine's own
    *output* impossible without a circular dependency.

    Parameters are exactly the *state*-independent subset of
    :func:`detect_condo_building_clusters`'s -- see there for the full
    algorithm description.

    Returns
    -------
    tuple of (pandas.Series, set) or None
        ``(component, hub_ids)``: *component* is the cluster id per
        qualifying parcel, indexed like *parcels* (only member rows
        present, not the full index) -- a hub parcel attached to more than
        one qualifying sub-cluster (see the two-pass union-find below)
        appears once per cluster it belongs to, so the index may contain
        duplicate values for such a hub; *hub_ids* is the subset of that
        same index classified as a hub/common-area candidate (real,
        positive ``improvement_value_column`` decides a unit; near-zero
        land *and* improvement value plus a compact shape decides a hub --
        see the ``is_hub_candidate`` test below) -- callers that fall back
        to a cluster's own parcel geometry (e.g.
        :func:`~openplaces.io.harmonizer.links.consolidate_condo_cluster_footprints`)
        must exclude *hub_ids* from that union: a hub's own polygon is the
        surrounding lot/common area, not part of the building, and is
        typically an order of magnitude larger than a real unit parcel.
        ``None`` if no candidates, no touching pairs, or no qualifying
        cluster was found.
    """
    required = {land_value_column, improvement_value_column, area_column}
    if not required.issubset(parcels.columns):
        if verbose:
            print(
                f'  detect_condo_building_clusters: '
                f'{sorted(required - set(parcels.columns))} missing; skipping.'
            )
        return None

    import geopandas as gpd

    from openplaces.geo.polygon import local_metric_crs
    from openplaces.io.harmonizer.spine import get_oriented_dims

    land_value = pd.to_numeric(parcels[land_value_column], errors='coerce').fillna(0)
    improvement_value = pd.to_numeric(
        parcels[improvement_value_column], errors='coerce'
    ).fillna(0)
    area_ha = pd.to_numeric(parcels[area_column], errors='coerce')

    is_unit_candidate = (
        area_ha.notna() & (area_ha <= max_unit_area_ha) & (improvement_value > 0)
    )
    is_hub_candidate = (
        (land_value <= 0)
        & (improvement_value <= 0)
        & area_ha.notna()
        & (area_ha > 0)
        & (area_ha <= max_hub_area_ha)
    )
    if is_hub_candidate.any():
        hub_sub = parcels.loc[is_hub_candidate]
        geom_m = hub_sub.geometry.to_crs(local_metric_crs(hub_sub))
        dims = geom_m.map(get_oriented_dims)
        length = np.array([d[1] for d in dims])
        width = np.clip(np.array([d[2] for d in dims]), 1e-6, None)
        is_hub_candidate.loc[hub_sub.index] = (length / width) <= max_hub_aspect_ratio

    use_subgroup_col = (
        use_subgroup_column
        if use_subgroup_column in parcels.columns
        else 'use_group_combined'
        if 'use_group_combined' in parcels.columns
        else None
    )
    if use_subgroup_col is not None:
        use_subgroup = parcels[use_subgroup_col].astype('string').str.lower()
        is_non_condo_use = use_subgroup.str.contains(
            '|'.join(_NON_CONDO_USE_SUBGROUP_TERMS), na=False, regex=True
        )
        is_unit_candidate &= ~is_non_condo_use
        is_hub_candidate &= ~is_non_condo_use

    is_candidate = is_unit_candidate | is_hub_candidate
    if is_candidate.sum() < min_group_size:
        if verbose:
            print('  detect_condo_building_clusters: no candidates found.')
        return None

    idx_name = parcels.index.name or 'index'
    cand_gdf = parcels.loc[is_candidate, ['geometry']].reset_index()
    a_col, b_col = f'{idx_name}_a', f'{idx_name}_b'
    touching = gpd.sjoin(
        cand_gdf, cand_gdf, predicate='touches', how='inner', lsuffix='_a', rsuffix='_b'
    ).rename(columns={f'{idx_name}__a': a_col, f'{idx_name}__b': b_col})
    if len(touching) == 0:
        if verbose:
            print('  detect_condo_building_clusters: no touching candidates found.')
        return None

    # Connected components over the candidate 'touches' graph (a plain
    # union-find over a small edge list -- one building's worth of parcels at
    # a time, not a large general graph, so no need for a library dependency).
    parent: dict[str, str] = {}

    def _find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def _union(x: str, y: str) -> None:
        rx, ry = _find(x), _find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    hub_ids = set(parcels.index[is_hub_candidate])
    for pid in cand_gdf[idx_name]:
        parent.setdefault(pid, pid)

    # Pass 1: union unit-unit edges only. A hub is deliberately never
    # unioned into this graph -- if it were, a single shared hub could
    # transitively chain multiple, geometrically separate unit groups
    # (e.g. one HOA common-area parcel bordering several distinct
    # townhouse buildings) into a single bogus component. Confirmed on
    # real data (Carteret County, NC): a 0.56 ha COMMON AREA parcel
    # otherwise bridges 3 physically disjoint unit groups, 30-130 m apart,
    # into one. Two adjacent hubs (e.g. neighboring walkway segments)
    # never union with each other either, matching the prior hub-hub
    # guard.
    hub_unit_edges: list[tuple[str, str]] = []
    for a, b in zip(touching[a_col], touching[b_col]):
        a_hub, b_hub = a in hub_ids, b in hub_ids
        if a_hub or b_hub:
            if a_hub != b_hub:
                hub_unit_edges.append((a, b) if a_hub else (b, a))
            continue
        _union(a, b)

    # Pass 2: attach each hub to every unit base-component it directly
    # touches, without merging those base components together -- a hub
    # may legitimately serve multiple separate buildings, but those
    # buildings' unit groups must never become one output row just
    # because they share a hub.
    unit_pids = [pid for pid in cand_gdf[idx_name] if pid not in hub_ids]
    component_index = list(unit_pids)
    component_values = [_find(pid) for pid in unit_pids]
    hub_component_ids: dict[str, set[str]] = {}
    for hub, unit in hub_unit_edges:
        hub_component_ids.setdefault(hub, set()).add(_find(unit))
    for hub, comp_ids in hub_component_ids.items():
        for comp_id in comp_ids:
            component_index.append(hub)
            component_values.append(comp_id)

    component = pd.Series(component_values, index=pd.Index(component_index))
    sizes = component.value_counts()
    qualifying = sizes[(sizes >= min_group_size) & (sizes <= max_group_size)]
    if qualifying.empty:
        if verbose:
            print(
                f'  detect_condo_building_clusters: no cluster with '
                f'{min_group_size}-{max_group_size} parcels found.'
            )
        return None
    component = component[component.isin(qualifying.index)]
    hub_ids = hub_ids & set(component.index)

    if verbose:
        print(
            f'  detect_condo_building_clusters: {len(qualifying):,} building clusters '
            f'covering {component.index.nunique():,} parcels.'
        )
    return component, hub_ids


@_register('estimate_property_counts')
def estimate_property_counts(
    state: HarmonizeState,
    property_count_column: str = 'n_properties_per_parcel',
    source_column: str = 'property_source',
    dwelling_column: str = 'n_dwellings',
    footprint_dwelling_column: str = 'max_dwellings_per_footprint',
) -> HarmonizeState:
    """Fill in ``property_source``/``n_properties_per_parcel`` where still unset.

    A real per-unit property source (a bundled ``additional_layers`` table,
    e.g. MassGIS's L3_ASSESS) or a geometrically inferred shared-land group
    (see the parcel spine's own shared-land step) each write
    *property_count_column*/*source_column* themselves, with
    ``source_column`` set to ``'source'``/``'shared_land_group'``
    respectively -- this step never overwrites either. Everywhere else, it
    estimates a count from evidence that already implies internal
    multiplicity even though no per-unit row exists to count directly:
    *dwelling_column* (the parcel's own recorded dwelling-unit count) or
    *footprint_dwelling_column* (confirmed Overture dwelling points on a
    single footprint on the parcel, from
    :func:`summarize_footprint_morphology` -- run this step after that one).
    A parcel with neither a real source nor multiplicity evidence gets
    ``source_column = 'none'`` and *property_count_column* left at 0/missing.

    This estimate is deliberately not a fabricated 1:1 property row (that
    would misrepresent a state with no per-unit source as having verified
    unit-level data) -- it only ever raises the count, and only when the
    parcel's own evidence already shows multiplicity; ``'estimated'`` marks
    that distinction so downstream consumers (e.g. transaction linking) know
    not to trust it as a real, addressable list of units the way a
    ``'source'``/``'shared_land_group'`` parcel's units are.

    Parameters
    ----------
    property_count_column : str, optional
        Per-parcel property/unit count (default ``'n_properties_per_parcel'``).
    source_column : str, optional
        Provenance of *property_count_column* (default ``'property_source'``).
    dwelling_column : str, optional
        Parcel's own recorded dwelling-unit count (default ``'n_dwellings'``).
    footprint_dwelling_column : str, optional
        Confirmed dwelling count on a single footprint on the parcel (default
        ``'max_dwellings_per_footprint'``).
    """
    if state.spine is None:
        return state
    spine = state.spine

    # Built as fresh Series and assigned back wholesale (never a partial
    # `.loc[mask, col] = ...` mutation of an existing column) -- a column
    # freshly written by an upstream `.map(mapper).astype(...)` call (e.g.
    # link_by_id's count mode) can carry a read-only backing array in some
    # pandas/geopandas dtype combinations, which a partial in-place
    # assignment then fails on.
    source = (
        spine[source_column].astype(object)
        if source_column in spine.columns
        else pd.Series(pd.NA, index=spine.index, dtype=object)
    )
    count = (
        pd.to_numeric(spine[property_count_column], errors='coerce')
        if property_count_column in spine.columns
        else pd.Series(np.nan, index=spine.index)
    )

    unset = source.isna()
    has_real_source = unset & (count.fillna(0) > 0)
    source = source.mask(has_real_source, 'source')

    n_dwellings = (
        pd.to_numeric(spine[dwelling_column], errors='coerce').fillna(0)
        if dwelling_column in spine.columns
        else pd.Series(0.0, index=spine.index)
    )
    n_footprint_dwellings = (
        pd.to_numeric(spine[footprint_dwelling_column], errors='coerce').fillna(0)
        if footprint_dwelling_column in spine.columns
        else pd.Series(0.0, index=spine.index)
    )
    estimate = pd.concat([n_dwellings, n_footprint_dwellings], axis=1).max(axis=1)

    needs_estimate = source.isna() & (estimate >= 2)
    count = count.mask(needs_estimate, estimate)
    source = source.mask(needs_estimate, 'estimated')
    source = source.mask(source.isna(), 'none')

    spine[property_count_column] = count
    spine[source_column] = source.astype('category')
    state.spine = spine

    if state.verbose:
        counts = spine[source_column].value_counts()
        print(f'  estimate_property_counts: {counts.to_dict()}')
    return state


@_register('attribute_dwelling_address')
def attribute_dwelling_address(
    state: HarmonizeState,
    footprint_recipe_id: str,
    on: str = 'parcel_id',
    priority_column: str = 'priority_on_parcel',
    overlap_column: str = 'area_intersection_m2_parcel',
    columns: dict[str, str] | None = None,
) -> HarmonizeState:
    """Relay each parcel's primary footprint's dwelling-point address evidence.

    Dwelling points (e.g. dwelling-overture-2025) are only ever spatially
    linked to footprints, not parcels, so a parcel has no direct access to
    that evidence on its own -- but every parcel's *primary* footprint (per
    :func:`classify_footprint_priority`, the one(s) carrying dwelling/
    building-point evidence) does. This reads *footprint_recipe_id*'s
    harmonized spine, keeps each parcel's primary footprint(s) (largest
    *overlap_column* wins when a parcel has more than one, e.g. a multi-unit
    property with several dwelling-linked buildings), and copies the
    requested *columns* onto the matching parcel row -- so a parcel spine's
    own ``reconcile_addresses`` can declare a source built from this relayed
    evidence. This alone misses a dwelling point whose footprint was never
    detected at all; pair it with a direct parcel<->dwelling spatial link
    (``link_to_reference``/``reconcile_attributes``, same as the footprint
    spine's own dwelling link) as a lower-priority fallback source for that
    case.

    A no-op if ``state.spine`` is ``None``, *footprint_recipe_id* has no
    saved output yet, *on* is not shared between the two spines, or
    *priority_column* is absent from the footprint entity (there would be no
    way to tell which footprint should represent the parcel) -- the same
    "missing evidence is tolerated" convention used throughout this codebase
    (``reconcile_addresses``/``reconcile_attributes``/``reconcile_values``).

    Parameters
    ----------
    footprint_recipe_id : str
        Footprint entity recipe to read (the harmonized spine).
    on : str, optional
        Shared parcel id (default ``'parcel_id'``): a footprint-entity
        column matched against either the parcel spine's current index name
        or its original name recorded in
        ``state.metadata['spine_index_name']`` -- same resolution
        :func:`summarize_footprint_morphology` uses, since a parcel spine's
        own true id lives on the index at this point in the pipeline.
    priority_column : str, optional
        Footprint-entity column marking each footprint's structural role on
        its parcel (default ``'priority_on_parcel'``, written by
        :func:`classify_footprint_priority`). Only ``'primary'`` rows are
        used.
    overlap_column : str, optional
        Footprint-entity column holding each footprint's overlap area (m2)
        with its dominant parcel (default ``'area_intersection_m2_parcel'``).
        Breaks ties among multiple primary footprints on one parcel; ignored
        (first row kept) if absent.
    columns : dict of {footprint column: parcel column}, optional
        Columns to copy (default: the four raw dwelling-overture evidence
        columns ``reconcile_attributes`` writes on the footprint spine,
        mapped to the same names with ``_overture`` swapped for
        ``_footprint`` -- e.g. ``address_street_dwelling_overture`` ->
        ``address_street_dwelling_footprint``, distinct names so a direct
        parcel<->dwelling link's own ``_dwelling_overture`` columns don't
        collide with this relay). Missing source columns are skipped.
    """
    from openplaces.io.readers import get_entities

    if state.spine is None:
        return state
    spine = state.spine

    columns = columns or {
        f'{comp}_dwelling_overture': f'{comp}_dwelling_footprint'
        for comp in ('address_street', 'address_number', 'city', 'postal_code')
    }

    # missing='ignore': footprint_recipe_id genuinely has no output yet for
    # some admin units (not yet harmonized, or itself skipped a reference
    # with zero coverage) -- documented as a tolerated no-op above, not an
    # error; this function already prints its own message below.
    footprints = get_entities(
        footprint_recipe_id, state.admin_id, geom=False, missing='ignore'
    )
    if footprints is None or len(footprints) == 0:
        if state.verbose:
            print('  attribute_dwelling_address: no footprints; skipping.')
        return state

    on_in_spine = on in spine.columns or on in (
        spine.index.name,
        state.metadata.get('spine_index_name'),
    )
    if not (on in footprints.columns and on_in_spine):
        if state.verbose:
            print(f'  attribute_dwelling_address: {on!r} not shared; skipping.')
        return state

    if priority_column not in footprints.columns:
        if state.verbose:
            print(
                f'  attribute_dwelling_address: no {priority_column!r} on '
                f'{footprint_recipe_id}; skipping.'
            )
        return state

    source_cols = [c for c in columns if c in footprints.columns]
    if not source_cols:
        if state.verbose:
            print('  attribute_dwelling_address: no evidence columns found; skipping.')
        return state

    is_primary = footprints[priority_column].astype('string') == 'primary'
    per_fp = footprints[is_primary].copy()
    if per_fp.empty:
        return state
    per_fp['_pid'] = per_fp[on].astype('string')
    per_fp = per_fp.dropna(subset=['_pid'])
    if overlap_column in per_fp.columns:
        per_fp = per_fp.sort_values(overlap_column, ascending=False)
    per_fp = per_fp.drop_duplicates(subset='_pid', keep='first').set_index('_pid')

    key = (
        spine[on].astype('string')
        if on in spine.columns
        else spine.index.to_series().astype('string')
    )

    for src_col in source_cols:
        spine[columns[src_col]] = key.map(per_fp[src_col])

    state.spine = spine
    if state.verbose:
        print(
            f'  attribute_dwelling_address: relayed evidence from '
            f'{len(per_fp):,} primary footprints.'
        )
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
