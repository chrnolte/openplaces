"""
Enrichment steps for zonal statistics.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import box

from openplaces.core.schema import AdminId
from openplaces.geo.raster import (
    sample_exactextract,
    sample_rasterized,
    sample_rasterstats,
)
from openplaces.io.enricher import EnrichState, _register
from openplaces.io.readers import get_dataset
from openplaces.path import resolve_raster_path
from openplaces.recipe import get_recipe_by_id, get_save_admin_level

_METHODS = {
    'exactextract': sample_exactextract,
    'rasterstats': sample_rasterstats,
    'rasterized': sample_rasterized,
}


def resolve_raster_reference(
    state: EnrichState,
    raster_path: str | None = None,
    raster_recipe_id: str | None = None,
    layer: str | None = None,
    year: int | str | None = None,
    raster_admin_id=None,
):
    """Return the raster file a step should sample, by path or by recipe.

    A recipe names a raster in one of two ways. *raster_path* is a
    location under the configured raster root (or absolute), for products
    placed by hand. *raster_recipe_id* names a raster dataset recipe,
    optionally with a derived *layer* and *year* (a ``{layer}_{year}``
    partition written by the Ingester's ``derived_layers``); the file is
    looked up at the admin unit of that recipe's save level that contains
    the unit being enriched, so a department-wide raster serves every
    municipality inside it. Naming a recipe rather than a path is what
    lets the DAG see the dependency.

    Exactly one of *raster_path* and *raster_recipe_id* is required.
    *raster_admin_id* names the raster's own admin unit outright, for a
    spine processed at a coarser level than the raster is saved at (a
    national transaction spine against a departmental product), where
    truncation cannot find it.
    """
    if (raster_path is None) == (raster_recipe_id is None):
        raise ValueError('Pass exactly one of raster_path and raster_recipe_id.')
    if raster_path is not None:
        return resolve_raster_path(raster_path)

    recipe = get_recipe_by_id(raster_recipe_id)
    save_level = get_save_admin_level(recipe)
    if raster_admin_id is not None:
        admin_id = AdminId(raster_admin_id)
    else:
        admin_id = state.admin_id
        if admin_id is not None and admin_id.get_level() > save_level:
            admin_id = admin_id.truncate_to_level(save_level)
        elif admin_id is not None and admin_id.get_level() < save_level:
            raise ValueError(
                f'{raster_recipe_id} saves at admin level {save_level}, coarser '
                f'units like {admin_id} cannot be resolved to one; pass '
                'raster_admin_id.'
            )
    partition_id = None
    if layer is not None or year is not None:
        if layer is None or year is None:
            raise ValueError('layer and year go together for a derived raster.')
        partition_id = f'{layer}_{year}'
    return get_dataset(recipe, admin_id, partition_id=partition_id)


def sample_raster(
    state: EnrichState,
    raster_path: str | None,
    raster_key: str,
    stat: str,
    column: str | None = None,
    method: str = 'exactextract',
    raster_recipe_id: str | None = None,
    layer: str | None = None,
    year: int | str | None = None,
    raster_admin_id=None,
) -> EnrichState:
    """Sample a raster over the spine's polygons and write *column*.

    Parameters
    ----------
    raster_path : str, optional
        Path to the raster file, relative to the raster root or absolute.
        Alternatively name a recipe through *raster_recipe_id*.
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
    raster_recipe_id : str, optional
        Raster dataset recipe to sample instead of a path; with *layer*
        and *year*, one of its derived layers. See
        :func:`resolve_raster_reference`.
    layer, year : optional
        Derived layer name and year under *raster_recipe_id*.
    raster_admin_id : str, optional
        Admin unit the raster is saved for, when it cannot be inferred
        from the unit being enriched.
    """
    raster_path = resolve_raster_reference(
        state, raster_path, raster_recipe_id, layer, year, raster_admin_id
    )

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
    raster_key: str,
    raster_path: str | None = None,
    stat: str = 'mean',
    column: str | None = None,
    method: str = 'exactextract',
    raster_recipe_id: str | None = None,
    layer: str | None = None,
    year: int | str | None = None,
    raster_admin_id=None,
) -> EnrichState:
    """Compute per-parcel zonal statistics from a raster (elevation, slope, ...).

    See :func:`sample_raster` for parameter details.
    """
    return sample_raster(
        state,
        raster_path,
        raster_key,
        stat,
        column,
        method,
        raster_recipe_id=raster_recipe_id,
        layer=layer,
        year=year,
        raster_admin_id=raster_admin_id,
    )
