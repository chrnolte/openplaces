"""Attribution from a point reference through the point link."""

from __future__ import annotations

import warnings

import pandas as pd

from openplaces.core.attribute_registry import get_agg_func
from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer.attributes._shared import (
    _POINT_REF_COLS,
    _join_distinct,
    _point_suffix,
)
from openplaces.recipe import resolve_attribute_name, source_id_from_recipe_id


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
    # Named for every declared column, not only the ones the crosswalk
    # carries: a column this reference had nothing to say about is still
    # written (null) at the end, so one admin unit's spine cannot ship a
    # narrower schema than its neighbors'.
    renamed: dict[str, str] = {
        c: f'{c}{_point_suffix(crosswalk_key, entity_type, col=c)}' for c in avail_cols
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

    # Any other requested numeric column (e.g. n_stories,
    # gross_floor_area_sqft) not covered by a special case above:
    # aggregate with the attribute registry's default function
    # (fallback 'mean') so it still reaches the spine.
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

    # Every declared column ends up present, null where this reference
    # carried it for no row (or carried it not at all). A unit the
    # reference does not cover then differs from a covered one by the
    # values, never by the schema.
    for out_name in renamed.values():
        if out_name not in spine.columns:
            spine[out_name] = pd.NA

    state.spine = spine
    return state
