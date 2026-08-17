"""
Enrichment steps for zonal statistics.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import box

from openplaces.geo.raster import (
    sample_exactextract,
    sample_rasterized,
    sample_rasterstats,
)
from openplaces.io.enricher import EnrichState, _register
from openplaces.path import resolve_raster_path

_METHODS = {
    'exactextract': sample_exactextract,
    'rasterstats': sample_rasterstats,
    'rasterized': sample_rasterized,
}


def sample_raster(
    state: EnrichState,
    raster_path: str,
    raster_key: str,
    stat: str,
    column: str | None = None,
    method: str = 'exactextract',
) -> EnrichState:
    """Sample *raster_path* over the spine's parcel polygons and write *column*.

    Parameters
    ----------
    raster_path : str
        Path to the raster file.
    raster_key : str
        Label used to build the default column name and (for
        ``method='exactextract'``) passed to
        :func:`~openplaces.geo.raster.zonal_stats_with_exactextract` as the
        output column prefix.
    stat : str
        Statistic to compute (e.g. ``'mean'``, ``'max'``). ``'exactextract'``
        and ``'rasterstats'`` accept whatever their respective libraries
        support; ``'rasterized'`` supports ``mean``, ``max``, ``min``,
        ``sum``, ``std``, ``count``.
    column : str, optional
        Evidence column name. Defaults to ``f'{raster_key}_{stat}'``.
    method : {'exactextract', 'rasterstats', 'rasterized'}
        Zonal-statistics implementation (see
        :mod:`openplaces.geo.raster`). ``'exactextract'`` (default) is the
        most accurate (fractional pixel-area weighting). ``'rasterstats'``
        uses the ``rasterstats`` library. ``'rasterized'`` burns parcels to
        the raster grid and aggregates with numpy.
    """
    # Recipes name rasters relative to the configured raster root, so
    # the same recipe runs on any machine (absolute paths pass through).
    raster_path = resolve_raster_path(raster_path)

    if method not in _METHODS:
        raise ValueError(f'method must be one of {list(_METHODS)}, got {method!r}')
    column = column or f'{raster_key}_{stat}'

    spine = state.spine
    if not isinstance(spine, gpd.GeoDataFrame):
        raise ValueError(
            "This step requires spine geometry; set 'spine_geom: true' on "
            'the enrich recipe.'
        )

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        raster_extent = box(*src.bounds)

    parcels_r = spine.to_crs(raster_crs)
    parcels_r = parcels_r[
        parcels_r.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])
    ]
    parcels_r = parcels_r[parcels_r.intersects(raster_extent)]

    if parcels_r.empty:
        state.evidence[column] = pd.NA
        return state

    values = _METHODS[method](parcels_r, raster_path, raster_key, stat)
    state.evidence[column] = values.reindex(state.evidence.index)
    return state


@_register('zonal_stats')
def zonal_stats(
    state: EnrichState,
    raster_path: str,
    raster_key: str,
    stat: str = 'mean',
    column: str | None = None,
    method: str = 'exactextract',
) -> EnrichState:
    """Compute per-parcel zonal statistics from a raster (elevation, slope, ...).

    See :func:`sample_raster` for parameter details.
    """
    return sample_raster(state, raster_path, raster_key, stat, column, method)
