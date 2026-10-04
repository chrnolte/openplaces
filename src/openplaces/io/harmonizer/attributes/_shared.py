"""Helpers the attribution modules share: provenance suffixes,
the reference column lists, reverse_occ_units and the
per-column reducers.
"""

from __future__ import annotations

import re

import pandas as pd

from openplaces.recipe import source_id_from_recipe_id


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
    'gross_floor_area_sqft',
    'n_stories',
    'n_dwellings',
    'address_street',
    'address_number',
    'postal_code',
    'city',
]


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
