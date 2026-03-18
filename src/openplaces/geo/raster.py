import warnings
from pathlib import Path

import geopandas as gpd
import rasterio
from exactextract import exact_extract

from openplaces.geo.vector import clean_polygons


def zonal_stats_with_exactextract(
    vector,
    raster,
    stats,
    *,
    weights=None,
    include_cols=None,
    col_prefix=None,
    nodata=None,
    reproject=True,
    clean_geometry=True,
    strategy='raster-sequential',
    progress=False,
):
    """
    Compute zonal statistics for polygons over a raster using exactextract.
    raster-sequential strategy processes the raster in chuncks, significantly
    speeding up processing time

    Parameters
    ----------
    vector : str, Path, or GeoDataFrame
        Polygon vector data
    raster : str, Path, or rasterio.DatasetReader
        Raster to summarise
    stats : str, list, or Operation
        Statistics to compute. Passed directly to `exactextract.exact_extract`
        Ex. mean, min, max, count, mahority, quantile, sum, weighted mean
    weights : str, Path, or rasterio.DatasetReader, optional
        A second raster used as pixel weights
        when computing a population-weighted mean temperature.
    include_cols : str or list of str, optional
        Attribute columns from the vector to carry through to the output
        alongside the computed stats
    col_prefix : str, optional
        Prefix prepended to every stat column name
    nodata : float, optional
        Override the nodata value stored in the raster file
    reproject : bool, default False
        If True and the vector CRS differs from the raster CRS, reproject
        the vector before computing stats.
    clean_geometry : bool, default True
        If True, automatically drop null/empty geometries, repair invalid
        geometries
    strategy : {"feature-sequential", "raster-sequential"}, default "raster-sequential"
        raster-sequential iterates over raster chunks and is faster
    """

    # Validate strategy
    valid_strategies = {'feature-sequential', 'raster-sequential'}
    if strategy not in valid_strategies:
        raise ValueError(
            f"'strategy' must be one of {valid_strategies}, got {strategy!r}."
        )

    # Validate stats
    if stats is None or (
        hasattr(stats, '__len__') and not isinstance(stats, str) and len(stats) == 0
    ):
        raise ValueError("'stats' must not be empty.")

    # Normalise vector input
    if isinstance(vector, str | Path):
        gdf = gpd.read_file(vector)
    elif isinstance(vector, gpd.GeoDataFrame):
        gdf = vector.copy()
    else:
        raise TypeError(
            f"'vector' must be a file path (str/Path) or GeoDataFrame, "
            f'got {type(vector).__name__}.'
        )

    if gdf.empty:
        warnings.warn(
            'Input vector has no features; returning empty GeoDataFrame.', stacklevel=2
        )
        return gdf

    if clean_geometry:
        gdf = clean_polygons(gdf)
        if gdf.empty:
            raise ValueError('No valid polygon geometries remain after cleaning.')

    # Normalize raster input
    if isinstance(raster, str | Path):
        raster_path = str(raster)
        raster_arg = raster_path
    elif isinstance(raster, rasterio.DatasetReader):
        raster_path = raster.name
        raster_arg = raster
    else:
        raise TypeError(
            f"'raster' must be a file path (str/Path) or open rasterio DatasetReader, "
            f'got {type(raster).__name__}.'
        )

    # Nodata override
    if nodata is not None:
        from exactextract import RasterioRasterSource

        _src = rasterio.open(raster_path)
        raster_arg = RasterioRasterSource(_src, nodata=nodata)
        print('Nodata overide occured')

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs

    if reproject and raster_crs and gdf.crs and not gdf.crs.equals(raster_crs):
        gdf = gdf.to_crs(raster_crs)
        print(f'Shapefile reprojected to {raster_crs}')

    # Normalise weights
    weights_arg = str(weights) if isinstance(weights, Path) else weights

    # Create a GeoDataFrame with only geometry to avoid dtype issues
    gdf_geom = gpd.GeoDataFrame(geometry=gdf.geometry, crs=gdf.crs)

    # add parcel ID as a regular column to allow for future join back to gpd
    index_name = gdf.index.name or 'index'
    gdf_geom[index_name] = gdf.index

    # Run exactextract
    result = exact_extract(
        rast=raster_arg,
        vec=gdf_geom,
        ops=stats,
        weights=weights_arg,
        include_cols=[index_name],
        include_geom=True,
        output='pandas',
        strategy=strategy,
        progress=progress,
    )

    # Optional column prefix
    if col_prefix:
        passthrough = {'geometry', index_name}
        if include_cols:
            passthrough |= set(
                [include_cols] if isinstance(include_cols, str) else include_cols
            )
        result = result.rename(
            columns={
                c: f'{col_prefix}{c}' for c in result.columns if c not in passthrough
            }
        )

    # Set parcel ID as index (aligns correctly regardless of row order)
    result = result.set_index(index_name)
    # Drop geometry from result, join all original gdf columns back
    result = result.drop(columns=['geometry']).join(gdf)
    return gpd.GeoDataFrame(result, geometry='geometry', crs=gdf.crs)
