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
    'get_elevation_datum',
    'resolve_dem_admin_ids',
    'add_z_offset',
    'clamp_z',
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


def resolve_dem_admin_ids(
    gdf: gpd.GeoDataFrame,
    elevation_recipe,
    admin_id_column: str | None = None,
) -> pd.Series:
    """Each row's admin id truncated to the DEM recipe's own save level.

    A DEM is tiled per admin unit, and `io.readers.get_dataset` refuses an
    id at any other level, so a frame keyed by a finer unit has to be mapped
    up: a level-4 town is covered by its level-3 county's DEM. The level is
    read off the recipe rather than assumed, so this keeps working if the
    DEM is ever re-tiled at a different granularity.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Frame whose rows carry admin ids, in `admin_id_column` or the index.
    elevation_recipe : str or dict
        DEM dataset recipe.
    admin_id_column : str, optional
        Column holding each row's admin id. When None (default), the ids are
        read from `gdf`'s index.

    Returns
    -------
    pandas.Series of str
        Admin ids at the DEM's own level, indexed like `gdf`.
    """
    from openplaces.recipe import get_recipe_by_id, get_save_admin_level

    dem_recipe = (
        get_recipe_by_id(elevation_recipe)
        if isinstance(elevation_recipe, str)
        else elevation_recipe
    )
    dem_level = get_save_admin_level(dem_recipe)

    raw = gdf[admin_id_column] if admin_id_column is not None else gdf.index
    try:
        parsed = [AdminId(*str(value).split('-')) for value in raw]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            'Cannot resolve a DEM for these rows: each needs an admin id to '
            'find the raster covering it, and this frame is not indexed by '
            'admin id. Pass a frame from `get_admin`, or name the column '
            'holding the ids.',
        ) from exc

    resolved = []
    for admin, original in zip(parsed, raw, strict=True):
        parent = admin.truncate_to_level(dem_level)
        if parent is None:
            raise ValueError(
                f'Admin unit {original!r} is coarser than the level-{dem_level} '
                'tiling the DEM recipe is saved at, so no single DEM covers '
                'it. Use a finer admin level.',
            )
        resolved.append(str(parent))
    return pd.Series(resolved, index=gdf.index)


def get_elevation_datum(
    gdf: gpd.GeoDataFrame,
    elevation_recipe,
    quantile: float = 0.0,
    admin_id_column: str | None = None,
):
    """Ground elevation to treat as z=0 for a scene, in meters.

    The 3D viz extrudes from sea level by default, which puts the whole
    scene as far above the flat basemap as the land happens to be above the
    ocean — a few hundred meters here, more once `terrain_exaggeration`
    multiplies it. That offset is not harmless: with the camera tilted to
    pitch `p`, anything `h` meters up appears shifted from its own
    basemap position by `h * tan(p)`, so at pitch 75 a scene 345 m up
    (Lancaster at 3x) reads about 1.3 km away from the streets it sits on.

    Referencing the scene to the ground beneath it removes that whole term.
    What is left is only the relief *within* the extent, which is what the
    terrain is actually meant to show.

    Defaults to the **minimum** rather than the mean deliberately. Using a
    mean would put half the terrain below z=0, i.e. underneath a flat
    basemap, hiding it. A minimum guarantees every sampled elevation lands
    at or above the ground plane. `quantile` can trade that guarantee for a
    tighter reference; callers that do should clamp (see
    `viz.terrain.show_value_terrain_layer`'s `elevation_datum`, which
    clamps regardless).

    Compute this **once per scene** and pass the same value to every layer
    — parcels, buildings, boundaries, the draped basemap. A datum that
    differs between layers slides them vertically relative to each other,
    the same failure mode as a mismatched `terrain_exaggeration`.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Geometries defining the scene's extent. Only their bounds and admin
        ids are used, so any layer of the scene gives the same answer as
        long as they cover the same extent.
    elevation_recipe : str or dict
        DEM dataset recipe, passed to `io.readers.get_dataset`.
    quantile : float
        Quantile of the sampled elevations to use, in [0, 1]. Defaults to 0
        (the minimum). Raise it to reference a scene whose extent dips into
        a valley or offshore that would otherwise drag the datum down.
    admin_id_column : str, optional
        Column naming each row's admin unit at the DEM's own tiling. When
        None (default), the ids are read from `gdf`'s index.

    Returns
    -------
    float
        The reference elevation in meters. 0.0 if the DEM has no data over
        the extent, which keeps the sea-level behavior rather than raising.
    """
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f'quantile must be in [0, 1], got {quantile!r}.')

    ids = resolve_dem_admin_ids(gdf, elevation_recipe, admin_id_column).unique()

    values = []
    for admin_id in ids:
        dem_path = Path(get_dataset(elevation_recipe, admin_id=admin_id))
        if not dem_path.exists():
            _ingest_missing_dem(elevation_recipe, admin_id, dem_path, silent=True)
        with rasterio.open(dem_path) as src:
            bounds = gdf.to_crs(src.crs).total_bounds
            window = rasterio.windows.from_bounds(*bounds, transform=src.transform)
            band = src.read(1, window=window, masked=True, boundless=True)
        finite = np.asarray(band.compressed(), dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            values.append(np.quantile(finite, quantile))

    return float(min(values)) if values else 0.0


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


def clamp_z(geometry, lower: float = 0.0):
    """Raise every vertex Z below `lower` up to it, leaving x/y alone.

    Used by `viz.terrain`'s `elevation_datum` to guarantee that referenced
    ground never sinks below the basemap plane, where a flat basemap would
    simply hide it. Clamping (rather than shifting the whole scene down to
    fit) keeps the datum meaning what it says for the rest of the extent;
    only the part that would have gone under is flattened onto the plane.

    Parameters
    ----------
    geometry : array-like of shapely geometries
        2D or 3D geometries. A still-2D geometry has an implicit Z of 0 and
        is unaffected by the default `lower`.
    lower : float
        Floor applied to every vertex's Z. Defaults to 0.

    Returns
    -------
    numpy.ndarray of shapely geometries
    """
    geom3d = shapely.force_3d(np.asarray(geometry), z=0.0)
    xyz = np.asarray(shapely.get_coordinates(geom3d, include_z=True))
    xyz[:, 2] = np.maximum(xyz[:, 2], lower)
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
