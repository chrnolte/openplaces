"""
Enrichment step for statistics derived from spine geometry and
existing spine columns alone -- no external raster/dataset involved.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np

from openplaces.io.enricher import EnrichState, _register

# Equal-area projection used to compute parcel area.
_EQUAL_AREA_CRS = 'EPSG:6933'

_SUPPORTED_COLUMNS = ['ha', 'ln_ha', 'x', 'y', 'n_bld_fp_per_ha']


@_register('derive_from_spine')
def derive_from_spine(
    state: EnrichState,
    columns: list[str] | None = None,
) -> EnrichState:
    """Derive simple per-parcel variables from spine geometry/columns.

    Requires spine geometry (set ``spine_geom: true`` on the enrich recipe)
    for every supported column except a bare pass-through.

    Parameters
    ----------
    columns : list of str, optional
        Which derived variables to compute. Defaults to all of:
        ``ha`` (parcel area, hectares), ``ln_ha`` (log of ``ha``),
        ``x``/``y`` (centroid coordinates, in the spine's own CRS),
        ``n_bld_fp_per_ha`` (``n_buildings / ha``; missing where the spine
        has no ``n_buildings`` column, e.g. IGAC-covered parcels).
    """
    columns = columns or _SUPPORTED_COLUMNS
    unknown = set(columns) - set(_SUPPORTED_COLUMNS)
    if unknown:
        raise ValueError(f'Unsupported derive_from_spine columns: {sorted(unknown)}')

    spine = state.spine
    needs_geometry = bool({'ha', 'ln_ha', 'x', 'y', 'n_bld_fp_per_ha'} & set(columns))
    if needs_geometry and not isinstance(spine, gpd.GeoDataFrame):
        raise ValueError(
            "derive_from_spine requires spine geometry; set 'spine_geom: "
            "true' on the enrich recipe."
        )

    ha = None
    if {'ha', 'ln_ha', 'n_bld_fp_per_ha'} & set(columns):
        ha = spine.geometry.to_crs(_EQUAL_AREA_CRS).area.div(10_000)

    if 'ha' in columns:
        state.evidence['ha'] = ha
    if 'ln_ha' in columns:
        state.evidence['ln_ha'] = np.log(ha)
    if 'x' in columns or 'y' in columns:
        centroids = spine.geometry.centroid
        if 'x' in columns:
            state.evidence['x'] = centroids.x
        if 'y' in columns:
            state.evidence['y'] = centroids.y
    if 'n_bld_fp_per_ha' in columns:
        if 'n_buildings' in spine.columns:
            state.evidence['n_bld_fp_per_ha'] = spine['n_buildings'].div(ha)
        else:
            state.evidence['n_bld_fp_per_ha'] = np.nan

    return state
