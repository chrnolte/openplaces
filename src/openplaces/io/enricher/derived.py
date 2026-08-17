"""
Enrichment step for statistics derived from spine geometry and
existing spine columns alone -- no external raster/dataset involved.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np

from openplaces.geo.polygon import get_areas
from openplaces.io.enricher import EnrichState, _register

_SUPPORTED_COLUMNS = [
    'area_ha',
    'area_ha_log',
    'centroid_x',
    'centroid_y',
    'n_buildings_per_ha',
]

_AREA_DEPENDENTS = {'area_ha', 'area_ha_log', 'n_buildings_per_ha'}


@_register('derive_from_spine')
def derive_from_spine(
    state: EnrichState,
    columns: list[str] | None = None,
) -> EnrichState:
    """Derive simple per-parcel variables from spine geometry/columns.

    Requires spine geometry (set ``spine_geom: true`` on the enrich recipe).

    Overlaps the harmonize step ``derive_geometry_attributes``, which writes
    ``area_ha`` and centroid lat/long onto a spine that runs it. This step
    exists for spines that do not, and shares the same area measurement via
    :func:`~openplaces.geo.polygon.get_areas` so the two agree. A spine that
    already carries these columns should prefer the harmonize step.

    Parameters
    ----------
    columns : list of str, optional
        Which derived variables to compute. Defaults to all of:
        ``area_ha`` (parcel area in hectares), ``area_ha_log`` (its natural
        log), ``centroid_x``/``centroid_y`` (centroid coordinates in the
        spine's own CRS), and ``n_buildings_per_ha`` (``n_buildings`` per
        hectare; missing where the spine has no ``n_buildings`` column, as
        for IGAC-covered parcels).
    """
    columns = columns or _SUPPORTED_COLUMNS
    unknown = set(columns) - set(_SUPPORTED_COLUMNS)
    if unknown:
        raise ValueError(f'Unsupported derive_from_spine columns: {sorted(unknown)}')

    spine = state.spine
    if not isinstance(spine, gpd.GeoDataFrame):
        raise ValueError(
            "derive_from_spine requires spine geometry; set 'spine_geom: "
            "true' on the enrich recipe."
        )

    area_ha = None
    if _AREA_DEPENDENTS & set(columns):
        # Shared with derive_geometry_attributes rather than reprojecting
        # here, so both routes report the same area for the same parcel.
        area_ha = get_areas(spine, unit='ha')

    if 'area_ha' in columns:
        state.evidence['area_ha'] = area_ha
    if 'area_ha_log' in columns:
        state.evidence['area_ha_log'] = np.log(area_ha)
    if 'centroid_x' in columns or 'centroid_y' in columns:
        centroids = spine.geometry.centroid
        if 'centroid_x' in columns:
            state.evidence['centroid_x'] = centroids.x
        if 'centroid_y' in columns:
            state.evidence['centroid_y'] = centroids.y
    if 'n_buildings_per_ha' in columns:
        if 'n_buildings' in spine.columns:
            state.evidence['n_buildings_per_ha'] = spine['n_buildings'].div(area_ha)
        else:
            state.evidence['n_buildings_per_ha'] = np.nan

    return state
