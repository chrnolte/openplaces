"""
Terrain-elevation sourcing for the 3D land/building viz.

Samples the ingested USGS 3DEP DEM (see
``recipes/US/_all/land/elevation/usgs/3dep``) to ground `viz.terrain`'s
extruded polygons in real-world elevation: per-vertex draping for land
polygons (`drape_parcel_elevation`), and a single flat zonal-mean elevation
per building footprint (`get_building_elevation`). Results are cached to
disk under `cfg.cache_dir`, keyed by each row's own index (e.g.
``parcel_id`` / ``footprint_id``) -- since that index already encodes a
geometry-shape hash (see `geo.ids.get_geo_ids`), an unchanged shape reuses
its cached value and only new/changed rows touch the raster again. If an
admin unit's DEM hasn't been ingested yet, it's ingested automatically
on first use (see `_ingest_missing_dem`).
"""

import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
import shapely

from openplaces.config import cfg
from openplaces.core.schema import AdminId
from openplaces.geo.raster import sample_raster_at_points, zonal_stats_with_exactextract
from openplaces.io.readers import get_dataset

__all__ = [
    'drape_parcel_elevation',
    'get_building_elevation',
    'add_z_offset',
    'scale_z',
]


def drape_parcel_elevation(
    gdf: gpd.GeoDataFrame,
    elevation_recipe,
    admin_id_column: str = 'admin3_id',
    cache: bool = True,
    silent: bool = False,
):
    """Drape land/parcel polygons onto the ground via per-vertex elevation.

    Extracts every unique vertex position across `gdf` (deduplicating
    shared corners between adjacent parcels), samples the DEM at each
    (vectorized -- see `geo.raster.sample_raster_at_points`), and rebuilds
    each polygon with that elevation as its per-vertex Z coordinate, so the
    resulting geometry follows the real terrain along its whole boundary
    rather than sitting at one flat elevation. Buildings should use
    `get_building_elevation` instead -- a building's base must stay flat.

    Results are cached to disk per admin unit under `cfg.cache_dir`, keyed
    by `gdf`'s own index (e.g. ``parcel_id``); an id already present in the
    cache is reused without resampling the raster (see module docstring).

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Land/parcel polygons to drape.
    elevation_recipe : str or dict
        DEM dataset recipe (e.g. ``'US_land-elevation-usgs-3dep'``), passed
        to `io.readers.get_dataset` per admin unit. Ingested automatically
        (via `io.ingester.ingest`) for any admin unit whose DEM doesn't
        exist on disk yet -- a one-time cost per admin unit.
    admin_id_column : str
        Column on `gdf` naming each row's admin unit at the DEM's own
        ingestion granularity (default ``'admin3_id'``, matching both the
        DEM recipe's and the curated parcel/footprint entities' county
        level) -- used to resolve which DEM tile covers each row and to
        scope the on-disk cache file.
    cache : bool
        If True (default), read/write the on-disk elevation cache. Set
        False to always resample from the raster.
    silent : bool
        If True, suppress the out-of-extent/nodata-vertex warning.

    Returns
    -------
    geometry : geopandas.GeoSeries
        `gdf`'s geometry, unchanged in x/y, with each vertex's Z set to
        its sampled elevation.
    mean_elevation : numpy.ndarray
        Each row's own vertex-elevation mean -- a single representative
        ground elevation per row, for ``total_elevation`` bookkeeping
        (`viz.terrain`) even though the geometry itself varies per vertex.
    """

    def compute_miss(miss_gdf, dem_path):
        return _drape_miss(miss_gdf, dem_path, silent=silent)

    combined = _grouped_cache_compute(
        gdf,
        elevation_recipe,
        admin_id_column,
        cache,
        kind='parcel_elevation',
        geom=True,
        compute_miss=compute_miss,
        silent=silent,
    )
    return (
        gpd.GeoSeries(combined['geometry'], crs=gdf.crs),
        combined['mean_elevation'].to_numpy(dtype=float),
    )


def get_building_elevation(
    gdf: gpd.GeoDataFrame,
    elevation_recipe,
    admin_id_column: str = 'admin3_id',
    cache: bool = True,
    silent: bool = False,
):
    """Average ground elevation under each building footprint.

    Buildings render as flat extrusions (a real building's base is flat),
    so unlike `drape_parcel_elevation` this returns one scalar per row: the
    DEM's zonal mean over each footprint polygon
    (`geo.raster.zonal_stats_with_exactextract`), independent of whichever
    parcel the footprint is later stacked on via `viz.terrain`'s `stack_on`.

    Caching and parameters otherwise match `drape_parcel_elevation`.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Building/footprint polygons.
    elevation_recipe : str or dict
        DEM dataset recipe, passed to `io.readers.get_dataset` per admin
        unit.
    admin_id_column : str
        Column naming each row's admin unit (default ``'admin3_id'``).
    cache : bool
        If True (default), read/write the on-disk elevation cache.
    silent : bool
        Unused directly (zonal stats raise no out-of-extent warning of
        their own); kept for signature parity with `drape_parcel_elevation`.

    Returns
    -------
    numpy.ndarray
        One elevation value (meters) per row, row-aligned to `gdf`. NaN for
        any row `zonal_stats_with_exactextract` drops as an invalid
        geometry (mirrors `viz.terrain`'s own missing-value handling).
    """

    def compute_miss(miss_gdf, dem_path):
        stats = zonal_stats_with_exactextract(miss_gdf, dem_path, stats='mean')
        elevation = stats['mean'].reindex(miss_gdf.index)
        return pd.DataFrame(
            {'elevation': elevation.to_numpy(dtype=float)}, index=elevation.index
        )

    combined = _grouped_cache_compute(
        gdf,
        elevation_recipe,
        admin_id_column,
        cache,
        kind='building_elevation',
        geom=False,
        compute_miss=compute_miss,
        silent=silent,
    )
    return combined['elevation'].to_numpy(dtype=float)


def add_z_offset(geometry, offset_per_row):
    """Add a per-row scalar Z offset to every vertex of each geometry.

    Unlike `shapely.force_3d` (which sets an absolute Z on 2D input but
    leaves an already-3D geometry's existing Z untouched), this *adds*
    `offset_per_row[i]` to whatever Z geometry `i` already has -- 0 for
    still-2D geometries, or each vertex's own already-draped elevation
    otherwise. Used in `viz.terrain` to layer a `stack_on`/`elevation_column`
    scalar offset on top of `drape_parcel_elevation`'s per-vertex terrain
    without flattening it back to a single Z per row.

    Parameters
    ----------
    geometry : array-like of shapely geometries
        2D or 3D polygon geometries.
    offset_per_row : array-like of float
        One Z offset per geometry, added to every one of that geometry's
        own vertices.

    Returns
    -------
    numpy.ndarray of shapely geometries
    """
    geom3d = shapely.force_3d(np.asarray(geometry), z=0.0)
    xyz, row_index = shapely.get_coordinates(geom3d, include_z=True, return_index=True)
    xyz[:, 2] = xyz[:, 2] + np.asarray(offset_per_row, dtype=float)[row_index]
    return shapely.set_coordinates(geom3d, xyz)


def scale_z(geometry, factor: float):
    """Multiply every vertex's own Z by `factor`.

    Used by `viz.terrain`'s `terrain_exaggeration` to visually exaggerate
    real-world terrain relief -- e.g. `drape_parcel_elevation`'s per-vertex
    Z, which on a typical parcel spans only a few meters and can otherwise
    be nearly imperceptible next to the value-based extrusion height. x/y
    are untouched; a still-2D input geometry is unaffected (its implicit
    Z of 0 stays 0 regardless of `factor`).

    Parameters
    ----------
    geometry : array-like of shapely geometries
        2D or 3D polygon geometries.
    factor : float
        Multiplier applied to each vertex's own Z value.

    Returns
    -------
    numpy.ndarray of shapely geometries
    """
    geom3d = shapely.force_3d(np.asarray(geometry), z=0.0)
    xyz = np.asarray(shapely.get_coordinates(geom3d, include_z=True))
    xyz[:, 2] = xyz[:, 2] * factor
    return shapely.set_coordinates(geom3d, xyz)


def _drape_miss(miss_gdf: gpd.GeoDataFrame, dem_path, silent: bool):
    """Per-vertex draped geometry + per-row mean elevation for uncached rows."""
    geom3d = shapely.force_3d(miss_gdf.geometry.to_numpy(), z=0.0)
    xyz, row_index = shapely.get_coordinates(geom3d, include_z=True, return_index=True)

    unique_xy, inverse = np.unique(xyz[:, :2], axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)

    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
    sample_x, sample_y = _reproject_points(
        unique_xy[:, 0], unique_xy[:, 1], miss_gdf.crs, dem_crs
    )
    unique_z = sample_raster_at_points(dem_path, sample_x, sample_y)

    n_missing = int(np.isnan(unique_z).sum())
    if n_missing and not silent:
        warnings.warn(
            f'{n_missing} of {len(unique_z)} sampled vertex(es) fell outside the '
            'DEM extent or on a nodata pixel; setting their elevation to 0.',
            stacklevel=3,
        )
    unique_z = np.nan_to_num(unique_z, nan=0.0)

    xyz[:, 2] = unique_z[inverse]
    draped = shapely.set_coordinates(geom3d, xyz)

    mean_elevation = (
        pd.Series(xyz[:, 2])
        .groupby(row_index)
        .mean()
        .reindex(range(len(miss_gdf)))
        .to_numpy(dtype=float)
    )
    return gpd.GeoDataFrame(
        {'geometry': draped, 'mean_elevation': mean_elevation},
        index=miss_gdf.index,
        crs=miss_gdf.crs,
    )


def _reproject_points(x, y, src_crs, dst_crs):
    """Reproject point coordinate arrays; no-op when CRSs already match."""
    if src_crs is None or dst_crs is None or pyproj.CRS(src_crs) == pyproj.CRS(dst_crs):
        return x, y
    transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return transformer.transform(x, y)


def _cache_path(admin_id, kind: str) -> Path:
    """Cache file path for one admin unit's derived elevation values.

    Not a recipe-defined output (this is regenerable derived data, not a
    canonical entity/dataset) -- lives directly under `cfg.cache_dir`,
    mirroring the admin-id directory structure `path.py` uses elsewhere.
    """
    admin_id = admin_id if isinstance(admin_id, AdminId) else AdminId(admin_id)
    return (
        cfg.cache_dir
        / admin_id.to_path()
        / 'viz_elevation'
        / f'{admin_id}_{kind}.parquet'
    )


def _load_cache(path: Path, geom: bool):
    if not path.exists():
        return None
    return gpd.read_parquet(path) if geom else pd.read_parquet(path)


def _save_cache(existing, fresh, path: Path, geom: bool) -> None:
    combined = (
        fresh
        if existing is None
        else pd.concat([existing[~existing.index.isin(fresh.index)], fresh])
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if geom:
        gpd.GeoDataFrame(combined).to_parquet(path)
    else:
        combined.to_parquet(path)


def _grouped_cache_compute(
    gdf: gpd.GeoDataFrame,
    elevation_recipe,
    admin_id_column: str,
    cache: bool,
    kind: str,
    geom: bool,
    compute_miss,
    silent: bool = False,
):
    """Shared cache + per-admin-unit-DEM orchestration.

    Groups `gdf` by `admin_id_column` (matching the DEM's own per-admin-unit
    tiling), reuses cached rows, and calls `compute_miss(miss_gdf, dem_path)`
    only for rows not already cached for that admin unit. Returns a frame
    row-aligned to `gdf` (its original order/index), combining cache hits
    and freshly computed rows.
    """
    if gdf.empty:
        return (
            gpd.GeoDataFrame(index=gdf.index) if geom else pd.DataFrame(index=gdf.index)
        )

    pieces = []
    for admin_id, group in gdf.groupby(admin_id_column, sort=False):
        cache_file = _cache_path(admin_id, kind)
        cached = _load_cache(cache_file, geom) if cache else None

        is_hit = (
            group.index.isin(cached.index)
            if cached is not None
            else np.zeros(len(group), dtype=bool)
        )
        if is_hit.any():
            pieces.append(cached.loc[group.index[is_hit]])

        miss = group[~is_hit]
        if len(miss):
            dem_path = get_dataset(elevation_recipe, admin_id=admin_id)
            if not Path(dem_path).exists():
                _ingest_missing_dem(elevation_recipe, admin_id, dem_path, silent)
            fresh = compute_miss(miss, dem_path)
            pieces.append(fresh)
            if cache:
                _save_cache(cached, fresh, cache_file, geom)

    combined = pd.concat(pieces) if pieces else gdf.iloc[:0]
    return combined.reindex(gdf.index)


def _ingest_missing_dem(
    elevation_recipe, admin_id, dem_path: Path, silent: bool
) -> None:
    """Ingest `elevation_recipe` for `admin_id` when its DEM hasn't been fetched yet.

    Imported lazily (not at module level) since `io.ingester` pulls in the
    full ingestion dependency stack (~1.4s), which every other caller of
    this module -- i.e. every `viz.terrain` call, even ones never touching
    `elevation_recipe` -- would otherwise pay for unconditionally.
    """
    from openplaces.io.ingester import ingest

    if not silent:
        warnings.warn(
            f'No DEM ingested yet for {admin_id!s} -- ingesting {elevation_recipe!r} '
            'now (one-time; cached to disk afterward).',
            stacklevel=4,
        )
    ingest(elevation_recipe, admin_ids=str(admin_id), verbose=not silent)

    if not dem_path.exists():
        raise FileNotFoundError(
            f'Ingesting {elevation_recipe!r} for {admin_id!s} did not produce the '
            f'expected DEM at {dem_path} -- this admin unit may have no 3DEP coverage.'
        )
