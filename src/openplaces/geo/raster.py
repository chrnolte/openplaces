#!/usr/bin/env python

"""
raster.py

Functions for processing raster data
"""

# Uncomment if needed:
# !pip install exactextract geopandas rasterio numpy

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Union

import geopandas as gpd
import numpy as np
import rasterio
from exactextract import exact_extract


def _clean_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Drop or repair geometries that would cause exactextract to fail.

    Uses openplaces.geo.vector.fix_geometries (buffer(0)) for geometry repair,
    then applies additional checks needed specifically for exactextract.

    Steps applied in order:
    1. Drop null geometries
    2. Drop empty geometries
    3. Fix invalid geometries via fix_geometries() (buffer(0))
    4. Drop anything still invalid after repair
    5. Keep only Polygon / MultiPolygon types
    6. Cast problematic dtypes to string (exactextract only handles bool/int/float/str)
    7. Reset index so exactextract sees a clean 0-based integer index

    Returns a cleaned copy; the original is not modified.
    Issues a warning summarising how many features were removed.
    """
    from openplaces.geo.vector import fix_geometries

    original_len = len(gdf)
    gdf = gdf.copy()

    # 1. Null geometries
    gdf = gdf[gdf.geometry.notna()]

    # 2. Empty geometries
    gdf = gdf[~gdf.geometry.is_empty]

    # 3. Fix invalid geometries — try fix_geometries (buffer(0)) first,
    # then fall back to shapely make_valid for anything still broken
    if (~gdf.geometry.is_valid).any():
        gdf = fix_geometries(gdf)

    # 4. For anything still invalid, try shapely make_valid as fallback
    still_invalid = ~gdf.geometry.is_valid
    if still_invalid.any():
        from shapely.validation import make_valid
        gdf.loc[still_invalid, "geometry"] = gdf.loc[still_invalid, "geometry"].apply(make_valid)

    # 5a. Drop anything still invalid after both repair attempts
    gdf = gdf[gdf.geometry.is_valid]

    # 5b. Drop empty geometries again — make_valid can produce empty geometries,
    # and empty geometries pass is_valid but cause exactextract to fail
    gdf = gdf[~gdf.geometry.is_empty]

    # 5c. Keep only polygon types
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]

    # 6. Cast problematic dtypes to string
    # exactextract only handles bool, int, float, and str field types.
    # Anything else (object, nullable Int64, mixed, categorical, etc.) is cast to str.
    _safe_dtypes = (np.bool_, np.integer, np.floating)
    for col in gdf.columns:
        if col == gdf.geometry.name:
            continue
        if not any(np.issubdtype(gdf[col].dtype, t) for t in _safe_dtypes):
            gdf[col] = gdf[col].astype(str)

    # 7. Reset index
    gdf = gdf.reset_index(drop=True)

    removed = original_len - len(gdf)
    if removed > 0:
        warnings.warn(
            f"clean_geometry: removed {removed} of {original_len} features "
            f"(null, empty, invalid, or non-polygon geometries). "
            f"{len(gdf)} features remain.",
            stacklevel=3,
        )

    return gdf


def zonal_stats_with_exactextract(
    vector: VectorInput,
    raster: RasterInput,
    stats,
    *,
    weights: RasterInput | None = None,
    include_cols: str | list[str] | None = None,
    col_prefix: str | None = None,
    nodata: float | None = None,
    reproject: bool = False,
    clean_geometry: bool = True,
    strategy: str = "feature-sequential",
    progress: bool = False,
) -> gpd.GeoDataFrame:
    """
    Compute zonal statistics for polygons over a raster using exactextract.

    exactextract weights each raster pixel by the fraction of its area covered
    by the polygon, making results accurate even for small polygons or coarse
    rasters where many cells are only partially inside a feature.

    Parameters
    ----------
    vector : str, Path, or GeoDataFrame
        Polygon vector data. Any GDAL/OGR-readable path (.shp, .gpkg, .geojson,
        .fgb, ...) or an in-memory GeoDataFrame are both accepted.
    raster : str, Path, or rasterio.DatasetReader
        Raster to summarise. Any path that GDAL, rasterio, or xarray can open,
        or an already-open rasterio DatasetReader. Multi-band rasters are
        supported; each band produces its own output columns named
        ``{stat}_band_{n}`` (single-band rasters just use ``{stat}``).
    stats : str, list, or Operation
        Statistics to compute. Passed directly to ``exactextract.exact_extract``
        as the ``ops`` argument — no restrictions. Examples:

        - Single string:              ``"mean"``
        - List of strings:            ``["mean", "min", "max", "count"]``
        - Parameterised string:       ``"quantile(q=0.9)"``
        - Coverage-weighted sum:      ``"sum(coverage_weight=area_spherical_km2)"``
        - Weighted operations:        ``"weighted_mean"`` (requires ``weights``)
        - Categorical/pixel arrays:   ``"majority"``, ``"pixels"``, ``"frac"``
        - Custom Python callable taking ``(values, coverage)`` or
          ``(values, coverage, weights)`` and returning a scalar
        - ``exactextract.Operation`` objects

        Full list: https://isciences.github.io/exactextract/operations.html

    weights : str, Path, or rasterio.DatasetReader, optional
        A second raster used as pixel weights for ``weighted_mean``,
        ``weighted_sum``, and related operations — e.g. a population raster
        when computing a population-weighted mean temperature.
    include_cols : str or list of str, optional
        Attribute columns from the vector to carry through to the output
        alongside the computed stats.
    col_prefix : str, optional
        Prefix prepended to every stat column name. For example,
        ``col_prefix="ndvi_"`` turns ``"mean"`` into ``"ndvi_mean"``.
        Useful when calling this function multiple times with different rasters
        and merging results into one GeoDataFrame.
    nodata : float, optional
        Override the nodata value stored in the raster file. Pixels equal to
        this value are excluded from all computations. Uses
        ``RasterioRasterSource`` internally.
    reproject : bool, default False
        If True and the vector CRS differs from the raster CRS, reproject
        the vector before computing stats. The output GeoDataFrame will have
        the raster CRS. If False, exactextract will issue its own CRS warning
        but proceed (which may give incorrect results for large CRS mismatches).
    clean_geometry : bool, default True
        If True, automatically drop null/empty geometries, repair invalid
        geometries with buffer(0), and remove any non-polygon geometry types
        before passing to exactextract. A warning is issued listing how many
        features were removed. Set to False to skip cleaning (e.g. if you have
        already cleaned upstream and want to avoid the overhead).
    strategy : {"feature-sequential", "raster-sequential"}, default "feature-sequential"
        Processing strategy passed to ``exact_extract``.
        ``"feature-sequential"`` (default) iterates over features and reads
        only the pixels that overlap each feature — good for most use cases.
        ``"raster-sequential"`` iterates over raster chunks and is faster
        when features are large or numerous, at the cost of higher memory use.
    progress : bool or callable, default False
        If True, display a tqdm progress bar. A callable may also be provided;
        it will be called with ``(fraction_complete, status_message)``.

    Returns
    -------
    GeoDataFrame
        One row per input polygon. Columns: ``geometry``, any
        ``include_cols``, then one column per requested statistic
        (optionally prefixed by ``col_prefix``). Index matches the input.

    Raises
    ------
    TypeError
        If ``vector`` or ``raster`` are not a recognised type.
    ValueError
        If ``stats`` is empty or ``strategy`` is not a valid option.

    Examples
    --------
    Mean elevation per parcel (file paths)::

        result = zonal_stats("parcels.shp", "elevation.tif", "mean")

    Multiple stats, GeoDataFrame input, column prefix::

        result = zonal_stats(
            parcels_gdf,
            "ndvi.tif",
            ["mean", "min", "max", "quantile(q=0.9)"],
            include_cols="parcel_id",
            col_prefix="ndvi_",
        )

    Population-weighted mean temperature::

        result = zonal_stats(
            "counties.gpkg",
            "temperature.tif",
            "weighted_mean",
            weights="population.tif",
        )

    Population sum from a density raster (area-weighted)::

        result = zonal_stats(
            "counties.gpkg",
            "pop_density.tif",
            "sum(coverage_weight=area_spherical_km2)",
        )

    Custom callable — 90th percentile::

        def p90(values, coverage):
            return float(np.percentile(values.compressed(), 90))

        result = zonal_stats("parcels.gpkg", "slope.tif", ["mean", p90])
    """
    # ── 1. Validate strategy ──────────────────────────────────────────────────
    valid_strategies = {"feature-sequential", "raster-sequential"}
    if strategy not in valid_strategies:
        raise ValueError(f"'strategy' must be one of {valid_strategies}, got {strategy!r}.")

    # ── 2. Validate stats ─────────────────────────────────────────────────────
    if stats is None or (
        hasattr(stats, "__len__") and not isinstance(stats, str) and len(stats) == 0
    ):
        raise ValueError("'stats' must not be empty.")

    # ── 3. Normalise vector input ─────────────────────────────────────────────
    if isinstance(vector, (str, Path)):
        gdf = gpd.read_file(vector)
    elif isinstance(vector, gpd.GeoDataFrame):
        gdf = vector.copy()
    else:
        raise TypeError(
            f"'vector' must be a file path (str/Path) or GeoDataFrame, "
            f"got {type(vector).__name__}."
        )

    if gdf.empty:
        warnings.warn("Input vector has no features; returning empty GeoDataFrame.", stacklevel=2)
        return gdf

    # ── 4. Geometry cleaning ──────────────────────────────────────────────────
    if clean_geometry:
        gdf = _clean_geometries(gdf)
        if gdf.empty:
            raise ValueError("No valid polygon geometries remain after cleaning.")

    # ── 5. Normalise raster path ──────────────────────────────────────────────
    if isinstance(raster, (str, Path)):
        raster_path = str(raster)
        raster_arg = raster_path
    elif isinstance(raster, rasterio.DatasetReader):
        raster_path = raster.name
        raster_arg = raster
    else:
        raise TypeError(
            f"'raster' must be a file path (str/Path) or open rasterio DatasetReader, "
            f"got {type(raster).__name__}."
        )

    # ── 6. Nodata override ────────────────────────────────────────────────────
    if nodata is not None:
        from exactextract import RasterioRasterSource
        _src = rasterio.open(raster_path)
        raster_arg = RasterioRasterSource(_src, nodata=nodata)

    # ── 7. CRS check & optional reprojection ──────────────────────────────────
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs

    if reproject and raster_crs and gdf.crs and not gdf.crs.equals(raster_crs):
        gdf = gdf.to_crs(raster_crs)

    # ── 8. Validate include_cols ──────────────────────────────────────────────
    # exactextract throws an opaque IndexError if include_cols references
    # columns that don't exist — catch this early with a clear error.
    if include_cols is not None:
        cols_requested = [include_cols] if isinstance(include_cols, str) else list(include_cols)
        missing = [c for c in cols_requested if c not in gdf.columns]
        if missing:
            raise ValueError(
                f"include_cols references columns not found in vector: {missing}. "
                f"Available columns: {list(gdf.columns)}"
            )

    # ── 9. Normalise weights ──────────────────────────────────────────────────
    weights_arg = str(weights) if isinstance(weights, Path) else weights

    # ── 10. Run exactextract ──────────────────────────────────────────────────
    result = exact_extract(
        rast=raster_arg,
        vec=gdf,
        ops=stats,
        weights=weights_arg,
        include_cols=include_cols,
        include_geom=True,
        output="pandas",
        strategy=strategy,
        progress=progress,
    )

    # ── 11. Optional column prefix ────────────────────────────────────────────
    if col_prefix:
        passthrough = {"geometry"}
        if include_cols:
            passthrough |= set([include_cols] if isinstance(include_cols, str) else include_cols)
        result = result.rename(
            columns={
                c: f"{col_prefix}{c}"
                for c in result.columns
                if c not in passthrough
            }
        )

    # ── 12. Restore original index ────────────────────────────────────────────
    result.index = gdf.index

    return gpd.GeoDataFrame(result, geometry="geometry", crs=gdf.crs)
