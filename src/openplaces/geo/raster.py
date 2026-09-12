import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from exactextract import exact_extract
from rasterio import windows
from rasterio.features import geometry_mask, rasterize
from rasterio.mask import mask
from rasterio.windows import Window, from_bounds
from scipy.signal import fftconvolve
from shapely.geometry import box

from openplaces.geo.polygon import clean_polygons


def get_circular_footprint(radius: int) -> np.ndarray:
    """Circular kernel of given radius (in pixels); shape (2r+1, 2r+1)."""
    size = 2 * radius
    footprint = np.zeros((size + 1, size + 1))
    for x in range(size + 1):
        for y in range(size + 1):
            footprint[x, y] = (x - radius) ** 2 + (y - radius) ** 2 <= radius**2
    return footprint


def compute_vicinity_coverage(
    raster_path,
    bounds,
    bounds_crs=None,
    px_radius: int = 60,
):
    """Compute vicinity coverage for the window covering *bounds*.

    For every pixel in the (cropped) output window, the % of the surrounding
    ``px_radius``-pixel circular neighborhood that is 1 (vs. 0) in the source
    raster scoped to one bounds window instead of tiling the full national raster.

    To get correct values at the edges of *bounds* (e.g. an admin unit's own
    bounding box), the read window is padded by ``px_radius`` pixels on every
    side before convolving, so neighboring pixels outside *bounds* in an
    adjoining admin unit are still counted.

    Parameters
    ----------
    raster_path : str or Path
        Path to the source boolean (0/1) raster.
    bounds : tuple of float
        ``(minx, miny, maxx, maxy)`` to compute coverage for.
    bounds_crs : str or CRS, optional
        CRS of *bounds*. Defaults to the raster's own CRS (no reprojection).
    px_radius : int
        Neighborhood radius, in source-raster pixels.

    Pixels carrying the source raster's nodata value are excluded from
    both the numerator and the denominator, so the percentage is a share
    of the observed neighborhood rather than of its full pixel count.

    Returns
    -------
    array : numpy.ndarray
        uint8 array (percent coverage, 0-100; 255 = nodata) for the part
        of the *bounds* window that lies inside the raster (halo already
        cropped off, and the window clipped to the raster's own extent).
    transform : affine.Affine
        Transform for `array`.
    crs
        The source raster's CRS.
    """
    kernel = get_circular_footprint(px_radius)

    with rasterio.open(raster_path) as src:
        if bounds_crs is not None and str(bounds_crs) != str(src.crs):
            bounds = rasterio.warp.transform_bounds(bounds_crs, src.crs, *bounds)

        core_window = from_bounds(*bounds, transform=src.transform)
        core_window = core_window.round_lengths().round_offsets()

        # Bounds can reach past the raster (a coastal or border admin
        # unit whose bounding box includes open water or another
        # country). Clip the core window to the raster before padding:
        # left uncliped, its offsets go negative, the crop below reads
        # from the end of the array, and the output transform would
        # place that truncated array outside the raster.
        raster_window = Window(0, 0, src.width, src.height)
        if not windows.intersect([core_window, raster_window]):
            raise ValueError(f'bounds {bounds} do not intersect raster {raster_path}.')
        core_window = core_window.intersection(raster_window)

        padded_window = Window(
            core_window.col_off - px_radius,
            core_window.row_off - px_radius,
            core_window.width + 2 * px_radius,
            core_window.height + 2 * px_radius,
        )
        padded_window = padded_window.intersection(raster_window)

        arr = src.read(1, window=padded_window).astype(np.float64)
        out_transform = src.window_transform(core_window)
        nodata = src.nodata

        # Where the halo was clipped by the raster's own extent (e.g. a
        # coastal/border admin unit), the padded window is smaller than
        # `px_radius` on that side; track the offset so the core crop below
        # still lines up with `core_window`.
        row_offset = round(core_window.row_off - padded_window.row_off)
        col_offset = round(core_window.col_off - padded_window.col_off)

        crs = src.crs

    # The source is a boolean 0/1 raster, so anything that is not 0 or 1
    # carries no coverage information: the declared nodata sentinel (255
    # in the uint8 layers this runs on, which `arr >= 0` read as a
    # covered pixel and pushed the percentage past 100), NaN, and any
    # stray negative. Excluding them from both sums leaves the
    # percentage a share of the neighborhood that was observed.
    valid_mask = ~np.isnan(arr) & (arr >= 0)
    if nodata is not None and not np.isnan(nodata):
        valid_mask &= arr != nodata
    valid = valid_mask.astype(np.float64)
    filled = np.where(valid_mask, arr, 0)

    # Both convolutions are integer counts (0/1 inputs, a 0/1 kernel),
    # and the FFT returns them with floating-point noise of either sign.
    # Rounded, a neighborhood with nothing observed has a denominator of
    # exactly zero rather than 1e-13, and a numerator of exactly zero
    # rather than -1e-13; unrounded, the ratio could reach any magnitude
    # and a negative one wrapped through the unsigned cast below (248 on
    # the Linux CI runner, where the noise fell the other way than on
    # the machine the test was written on).
    num = np.rint(fftconvolve(filled, kernel, mode='same'))
    den = np.rint(fftconvolve(valid, kernel, mode='same'))
    coverage = np.full(num.shape, np.nan)
    np.divide(num, den, out=coverage, where=den > 0)
    coverage = np.clip(coverage * 100, 0, 100)

    core = coverage[
        row_offset : row_offset + round(core_window.height),
        col_offset : col_offset + round(core_window.width),
    ]
    out = np.where(np.isnan(core), 255, core).astype('uint8')

    return out, out_transform, crs


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

    # Carry a synthetic row position, not the index label, as the
    # key back to gdf: a repeated label would multiply the join below
    # into the product of its occurrences.
    position_col = '_row_position'
    gdf_geom[position_col] = np.arange(len(gdf))

    # Run exactextract
    result = exact_extract(
        rast=raster_arg,
        vec=gdf_geom,
        ops=stats,
        weights=weights_arg,
        include_cols=[position_col],
        include_geom=True,
        output='pandas',
        strategy=strategy,
        progress=progress,
    )

    # Optional column prefix
    if col_prefix:
        passthrough = {'geometry', position_col}
        if include_cols:
            passthrough |= set(
                [include_cols] if isinstance(include_cols, str) else include_cols
            )
        result = result.rename(
            columns={
                c: f'{col_prefix}{c}' for c in result.columns if c not in passthrough
            }
        )

    # Put the stats back on the input rows by position (exactextract
    # may return them in any order), then rebuild the frame column by
    # column: neither a label join nor a concat can align an index
    # whose labels repeat.
    stats_frame = (
        result.drop(columns=['geometry'], errors='ignore')
        .set_index(position_col)
        .reindex(np.arange(len(gdf)))
    )
    geometry_col = gdf.geometry.name
    columns = {col: stats_frame[col].to_numpy() for col in stats_frame.columns}
    columns.update({col: gdf[col].values for col in gdf.columns})
    return gpd.GeoDataFrame(
        columns, index=gdf.index, geometry=geometry_col, crs=gdf.crs
    )


def sample_raster_at_points(raster_path, x, y):
    """Sample a single-band raster at arbitrary points, vectorized.

    Reads the full band into memory once, then resolves every point's
    pixel via the raster's inverse affine transform and fancy-indexes the
    array in bulk. This is much faster than iterating points one at a time
    (e.g. ``rasterio``'s own ``sample()`` generator) at the scale of a
    county's worth of parcel vertices (hundreds of thousands to millions of
    points) -- appropriate for the resolution/extent of a single ingested
    COG (e.g. a county-level 10m DEM), not for rasters too large to fit in
    memory.

    Parameters
    ----------
    raster_path : str or Path
        Path to a single-band raster (e.g. a COG written by
        `io.ingester.raster_ingester.fetch_raster_from_vrt`), in whatever
        CRS it was ingested in. `x`/`y` must already be in that CRS.
    x, y : array-like of float
        Point coordinates, in the raster's own CRS.

    Returns
    -------
    numpy.ndarray
        One value per point, `float`. NaN for points outside the raster's
        extent or landing on a nodata pixel.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    with rasterio.open(raster_path) as src:
        band = src.read(1, masked=True)
        cols, rows = ~src.transform * (x, y)

    cols = np.floor(cols).astype(np.int64)
    rows = np.floor(rows).astype(np.int64)
    in_bounds = (
        (rows >= 0) & (rows < band.shape[0]) & (cols >= 0) & (cols < band.shape[1])
    )

    values = np.full(len(x), np.nan)
    sampled = band[rows[in_bounds], cols[in_bounds]]
    values[in_bounds] = np.where(
        np.ma.getmaskarray(sampled), np.nan, np.ma.getdata(sampled)
    )
    return values


_RASTERIZED_STAT_FN = {
    'mean': np.nanmean,
    'max': np.nanmax,
    'min': np.nanmin,
    'sum': np.nansum,
    'std': np.nanstd,
    'count': lambda x: np.sum(~np.isnan(x)),
}


def sample_exactextract(parcels_r, raster_path, raster_key, stat) -> pd.Series:
    """Per-parcel zonal statistic via exactextract (fractional-area weighting)."""
    result = zonal_stats_with_exactextract(
        parcels_r,
        raster_path,
        stats=[stat],
        col_prefix=f'{raster_key}_',
        reproject=False,
        clean_geometry=False,
    )
    return result[f'{raster_key}_{stat}']


def sample_rasterstats(parcels_r, raster_path, raster_key, stat) -> pd.Series:
    """Per-parcel zonal statistic via the ``rasterstats`` library."""
    from rasterstats import zonal_stats as rasterstats_zonal_stats

    with rasterio.open(raster_path) as src:
        nodata = src.nodata

    raw = rasterstats_zonal_stats(
        vectors=parcels_r.geometry,
        raster=str(raster_path),
        stats=[stat],
        all_touched=False,
        nodata=nodata,
    )
    result = pd.DataFrame(raw, index=parcels_r.index)[stat]
    return result


def sample_rasterized(parcels_r, raster_path, raster_key, stat) -> pd.Series:
    """Burn polygons to the raster grid, then aggregate with a numpy groupby.

    No fractional-overlap weighting (unlike ``sample_exactextract``), but no
    extra dependency either.
    """
    if stat not in _RASTERIZED_STAT_FN:
        raise ValueError(
            f"method='rasterized' does not support stat={stat!r}. Choose "
            f'from: {list(_RASTERIZED_STAT_FN)}'
        )
    agg_fn = _RASTERIZED_STAT_FN[stat]

    with rasterio.open(raster_path) as src:
        bounds = parcels_r.total_bounds
        win = src.window(*bounds)
        win = win.intersection(Window(0, 0, src.width, src.height))
        win_transform = src.window_transform(win)
        raster_data = src.read(1, window=win)
        nodata = src.nodata

    parcel_ids = parcels_r.index
    int_indices = range(1, len(parcel_ids) + 1)

    shapes = (
        (geom, idx) for geom, idx in zip(parcels_r.geometry, int_indices, strict=True)
    )
    id_grid = rasterize(
        shapes,
        out_shape=raster_data.shape,
        transform=win_transform,
        fill=0,
        dtype='int32',
    )

    raster_f = raster_data.astype('float32')
    if nodata is not None:
        raster_f[raster_data == nodata] = np.nan

    flat_ids = id_grid.ravel()
    flat_vals = raster_f.ravel()
    mask = flat_ids > 0

    df = pd.DataFrame({'idx': flat_ids[mask], 'val': flat_vals[mask]})
    grouped = df.groupby('idx')['val'].agg(agg_fn)
    # Reindex on the burned row positions, not on the parcel ids:
    # keying a dict by the id collapses two rows sharing a label into
    # one entry, and both then read back the same statistic.
    values = grouped.reindex(np.arange(1, len(parcel_ids) + 1)).to_numpy(
        dtype='float64'
    )
    return pd.Series(values, index=parcel_ids, dtype='float64')


def clip(
    from_filepath,
    to_filepath,
    clip_filepath=None,
    res=None,
    buffer=None,
    snap=True,
    crs=None,
    dtype=None,
    nodata=None,
):
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
            src, snapped_geometry, crop=True, all_touched=True, filled=False
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
    out_meta.update(
        {
            'driver': 'GTiff',
            'height': out_image_filled.shape[1],
            'width': out_image_filled.shape[2],
            'transform': out_transform,
            'nodata': nodata_val,
            'dtype': out_dtype,
            'compress': 'deflate',
            'zlevel': 6,
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
        }
    )

    # save to disk
    with rasterio.open(to_filepath, 'w', **out_meta) as dest:
        dest.write(out_image_filled)

    return True
