"""Shared-land group detection (detect_shared_land_groups)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from openplaces.io.harmonizer import HarmonizeState, _register


@_register('detect_shared_land_groups', phase='geometry')
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
