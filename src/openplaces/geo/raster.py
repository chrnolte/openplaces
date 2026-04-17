import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.mask import mask
from shapely.geometry import box

from exactextract import exact_extract

from openplaces.geo.polygon import clean_polygons

def zonal_stats_with_exactextract(
    vector,
    raster,
    stats,
    *,
    weights=None,
    include_cols=None,
    col_prefix=None,
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
        Statistics to compute. Passed directly to ``exactextract.exact_extract``
        Ex. mean, min, max, count, majority, quantile, sum, weighted mean
    weights : str, Path, or rasterio.DatasetReader, optional
        A second raster used as pixel weights
        when computing a population-weighted mean temperature.
    include_cols : str or list of str, optional
        Attribute columns from the vector to carry through to the output
        alongside the computed stats
    col_prefix : str, optional
        Prefix prepended to every stat column name
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


def clip(from_filepath, to_filepath, clip_filepath=None, res=None,
         buffer=None, snap=True, crs=None, dtype=None, nodata=None):
    """
    Clip raster to vector extent using rasterio.

    """

    # process vector clipping geometry
    if isinstance(clip_filepath, gpd.GeoDataFrame):
        clip_shp = clip_filepath.copy()
    else:
        clip_shp = gpd.read_file(clip_filepath)

    with rasterio.open(from_filepath) as src:
        src_crs = src.crs if crs is None else crs
        transform = src.transform
        src_nodata = src.nodata
        src_dtype = src.dtypes[0]

        # ensure CRS match
        if clip_shp.crs != src_crs:
            clip_shp = clip_shp.to_crs(src_crs)

        # apply buffer
        if buffer is not None:
            clip_shp['geometry'] = clip_shp.buffer(buffer)

        # calculate bounds and snapping
        xmin, ymin, xmax, ymax = clip_shp.total_bounds
        if snap:
            res_x = res if res else abs(transform[0])
            res_y = res if res else abs(transform[4])
            xmin = np.floor((xmin - transform[2]) / res_x) * res_x + transform[2]
            ymin = np.floor((ymin - transform[5]) / res_y) * res_y + transform[5]
            xmax = np.ceil((xmax - transform[2]) / res_x) * res_x + transform[2]
            ymax = np.ceil((ymax - transform[5]) / res_y) * res_y + transform[5]
            snapped_geometry = [box(xmin, ymin, xmax, ymax)]
        else:
            snapped_geometry = clip_shp.geometry.values

        out_image, out_transform = mask(
            src,
            snapped_geometry,
            crop=True,
            all_touched=True,
            filled=False
        )

        out_meta = src.meta.copy()

    # mask pixels not inside or touching the actual clip geometry
    poly_mask = geometry_mask(
        clip_shp.geometry,
        transform=out_transform,
        invert=False,
        out_shape=(out_image.shape[1], out_image.shape[2]),
        all_touched=True,
    )
    out_image = np.ma.masked_where(poly_mask[np.newaxis, :, :], out_image)

    # determine output dtype
    out_dtype = dtype if dtype is not None else src_dtype

    # determine nodata value
    if nodata is not None:
        nodata_val = nodata
    elif src_nodata is not None:
        nodata_val = src_nodata
    else:
        np_dtype = np.dtype(out_dtype)
        if np.issubdtype(np_dtype, np.unsignedinteger):
            nodata_val = int(np.iinfo(np_dtype).max)
        elif np.issubdtype(np_dtype, np.signedinteger):
            nodata_val = int(np.iinfo(np_dtype).min)
        else:
            nodata_val = float('nan')

    # convert to output dtype and fill masked pixels with nodata sentinel
    out_image_filled = out_image.astype(out_dtype).filled(nodata_val)

    # update metadata — use deflate compression to limit output size
    out_meta.update({
        "driver": "GTiff",
        "height": out_image_filled.shape[1],
        "width": out_image_filled.shape[2],
        "transform": out_transform,
        "nodata": nodata_val,
        "dtype": out_dtype,
        "compress": "deflate",
        "zlevel": 6,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    })

    # save to disk
    with rasterio.open(to_filepath, "w", **out_meta) as dest:
        dest.write(out_image_filled)

    return True