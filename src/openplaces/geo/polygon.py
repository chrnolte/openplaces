"""
polygon.py

Core geometry and polygon operations on GeoDataFrames and Shapely geometries.
No recipe, admin, or heavy I/O dependencies.
"""

import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely
import shapely.ops
from polylabel import polylabel
from shapely.geometry import MultiPolygon, Point, Polygon

from openplaces.core.constants import AC_TO_HA, M2_TO_SQFT

PROJ4 = {
    'ortho': '+proj=ortho +lat_0={LAT} +lon_0={LON} +x_0=0 +y_0=0 '
    '+ellps=WGS84 +units=m +no_defs',
    'moon': '+proj=nsper +h=384400000 +lon_0={LON} +lat_0={LAT} +ellps=WGS84',
    'landsat': '+proj=nsper +h=705000 +lon_0={LON} +lat_0={LAT} +ellps=WGS84',
    'eck': '+proj=eck4 +ellps=WGS84',
}


def fix_polygons(gdf):
    """Fix invalid geometries in a GeoDataFrame using make_valid.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame that may have invalid geometries.
    """
    gdf = gdf.copy()
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        gdf.loc[invalid, 'geometry'] = shapely.make_valid(
            gdf.loc[invalid, 'geometry'].values
        )
    return gdf


def has_geometry(gdf):
    """Get boolean Series identifying entries with valid geometries

    Returns True for entries with non-empty and non-null geometries.

    Parameters
    ----------
    gdf : GeoDataFrame or GeoSeries
        Input Geodataframe or Geoseries
    """

    # Silence the warnings about the behavior of .notna()
    # filterwarnings('ignore', 'GeoSeries.notna', UserWarning)

    if isinstance(gdf, gpd.GeoDataFrame):
        warnings.filterwarnings('ignore', 'GeoSeries.notna', UserWarning)
        return ~gdf['geometry'].is_empty & gdf['geometry'].notna()
    elif isinstance(gdf, gpd.GeoSeries):
        return ~gdf.is_empty & gdf.notna()
    else:
        raise ValueError('Not a Geodataframe or Geoseries: ' + str(gdf))


def clean_polygons(gdf):
    """Return GeoDataFrame with only clean and valid polygons

    Attempts to fix (zero-buffer) invalid polygons and drops empty
    polygons (and those with unfixable errors).
    """

    original_len = len(gdf)
    gdf = gdf.copy()

    # 1. Remove null geometries and fix any invalid geometries
    gdf = gdf[has_geometry(gdf)]

    if (~gdf.geometry.is_valid).any():
        gdf = fix_polygons(gdf)

    # Drop invalid geometries
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.is_valid]

    removed = original_len - len(gdf)
    if removed > 0:
        warnings.warn(
            f'clean_geometry: removed {removed} of {original_len} features '
            f'(null, empty, invalid, or non-polygon geometries). '
            f'{len(gdf)} features remain.',
            stacklevel=3,
        )

    return gdf


def get_areas(gdf, unit='ha', crs='epsg:6933'):
    """Compute areas of polygons in a GeoDataFrame

    Parameters
    ----------
    gdf : GeoDataFrame
        Geodataframe with polygons
    unit : str
        Area unit.
        Currently interpreted: 'm2', 'ha', 'km2', 'ac', 'ft2', 'sqft'
    crs : coordinate reference system
        Coordinate system in which computation takes places
        Has to be equal-area and use meters as its unit.
    """

    if not crs_is_mea(crs):
        raise Exception(
            'Areas can currently only be computed in equal-'
            'area projections that use meters as units. '
            'Requested CRS is: ' + str(crs)
        )

    gdf = gdf[['geometry']].copy()

    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    if unit == 'm2':
        return gdf.area.rename('m2')
    elif unit == 'ha':
        return gdf.area.div(1e4).rename('ha')
    elif unit == 'ac':
        return gdf.area.div(1e4 * AC_TO_HA).rename('ac')
    elif unit == 'km2':
        return gdf.area.div(1e6).rename(unit)
    elif unit in ['ft2', 'sqft']:
        return gdf.area.mul(M2_TO_SQFT).rename(unit)
    else:
        raise Exception('Unit not yet interpreted:' + str(unit))


def crs_is_mea(crs):
    """Check if projection is an equal-area projection and uses meters.

    Detects Albers Equal Area and Cylindric Equal Area projections.

    Parameters
    ----------
    crs : pyproj CRS
        Coordinate Reference System
    """

    if not isinstance(crs, pyproj.CRS):
        if isinstance(crs, str):
            crs = pyproj.CRS(crs)
        else:
            raise Exception(
                'CRS type not yet implemented in `openplaces.geo.vector.crs_is_mea`: '
                + str(type(crs))
                + ': '
                + str(crs)
            )

    # Still testing whether this makes the warning disappear
    warnings.filterwarnings('ignore', category=UserWarning)
    crs_dict = crs.to_dict()
    warnings.filterwarnings('default', category=UserWarning)

    return (crs_dict['proj'] in ['cea', 'aea']) and (crs_dict['units'] == 'm')


def get_lat_long_centroids(gdf, crs='epsg:4326', geom=False):
    """Get centroids (lat, long, geometry) in geographic projection.

    Parameters
    ----------
    gdf : GeoDataFrame
        Geodataframe
    crs : projection
        Projection in which centroids will be computed.
        Should be geographic, as 'lat' and 'long' columns are returned.
    geom : bool
        If True, returns GeoDataFrame with point geometries in same
        projection as `gdf`.
    """

    crs_orig = gdf.crs

    gdf = gdf[['geometry']].copy()

    # Project geodataframe to requested projection
    if crs != crs_orig:
        gdf = gdf.to_crs(crs)

    # Suppress warnings if centroids are in geographic CRS
    warnings.filterwarnings('ignore', category=UserWarning)
    gdf['geometry'] = gdf['geometry'].centroid
    warnings.filterwarnings('default', category=UserWarning)

    gdf['lat'] = gdf.geometry.y
    gdf['long'] = gdf.geometry.x

    if geom and crs != crs_orig:
        gdf = gdf.to_crs(crs_orig)

    return gdf if geom else gdf.drop(columns='geometry')


def get_pois(
    d,
    how='points',
    precision_ratio=0.001,
    prec_min=0.5,
    crs='epsg:3395',
    orthogonal=False,
):
    """Get the Poles of Inaccessibility (PoI) for a polygon geodataframe

    Parameters
    ----------
    d : GeoDataFrame or GeoSeries
        Geodataframe or GeoSeries containing polygons
    how : str
        Determines how POIs will be returned. Options include:
        'tuples': Series of tuples: (x, y, radius)
        'dataframe': DataFrame with columns: ('x_poi', 'y_poi', 'r_poi')
        'points': Geodataframe with points
        'points_only': Geoseries of points
        'circles': Geodataframe with circles / ellipses
        'circles_only': Geoseries of circles / ellipses
    precision_ratio : float
        Precision ratio used to define the precisions for:
        1. the polygon simplification algorithm
        2. the algorithm which finds the largest inscribed circle
        The precisions for both algorithms will be computed as:
        precision_ratio * square root of polygon area
    prec_min : float
        Minimum tolerance for both algorithms, defined in CRS units.
    crs : CRS
        Coordinate reference system (CRS) for computation of PoIs.
        This argument will be ignored if orthogonal is set to True.
        Mercator ('epsg:3395') is good for labels (more weight on width)
    orthogonal : bool
        If True, uses orthogonal projection around centroid to find PoI.
        Slower, but closer to real POI than using any single projection.
    """

    if isinstance(d, gpd.GeoDataFrame):
        dg = d['geometry'].copy()
    elif isinstance(d, gpd.GeoSeries):
        dg = d.copy()
    else:
        raise Exception('d must be GeoDataFrame or GeoSeries.')

    if orthogonal:
        if how in ['points', 'points_only']:
            geom = 'point'
        elif how in ['circles', 'circles_only']:
            geom = 'circle'
        else:
            geom = None
        pois = dg.apply(
            lambda x: get_poi_ortho(x, precision_ratio, prec_min, geom=geom)
        )
        if how == 'tuples':
            return pois.apply(tuple, 1)
        if how == 'df':
            return pois
        pois['geometry'] = gpd.GeoSeries(pois['geometry'], crs=dg.crs)
        if how in ['points_only', 'circles_only']:
            return pois['geometry']
        if how in ['points', 'circles']:
            return pois[['x_poi', 'y_poi', 'r_poi', 'geometry']]
    else:
        if dg.crs.is_geographic and crs is None:
            warnings.warn(
                'Geographic CRS passed to lib.gis.get_pois(). '
                'Consider using a projected CRS for the PoI computation.'
            )

        reproject = dg.crs != crs
        if reproject:
            crs_orig = dg.crs
            dg = dg.to_crs(crs)

        pois = (
            dg.apply(lambda x: get_poi(x, precision_ratio, prec_min))
            .apply(pd.Series)
            .rename(columns={0: 'x_poi', 1: 'y_poi', 2: 'r_poi'})
        )
        points = [Point(x, y) for x, y in zip(pois['x_poi'], pois['y_poi'])]
        poi_points = gpd.GeoDataFrame(pois, crs=dg.crs, geometry=points)

        if reproject:
            poi_points_orig = poi_points.to_crs(crs_orig)
            poi_points_orig['x_poi'] = poi_points_orig['geometry'].x
            poi_points_orig['y_poi'] = poi_points_orig['geometry'].y
            poi_points['x_poi'] = poi_points_orig['x_poi']
            poi_points['y_poi'] = poi_points_orig['y_poi']

        if how in ['circles', 'circles_only']:
            poi_circles = poi_points.copy()
            poi_circles['geometry'] = poi_circles.apply(
                lambda x: x['geometry'].buffer(x['r_poi']), 1
            )
            if reproject:
                poi_circles = poi_circles.to_crs(crs_orig)

        if reproject:
            poi_points = poi_points_orig

        if how == 'tuples':
            return poi_points.apply(tuple, 1)
        if how == 'dataframe':
            return poi_points[[v for v in poi_points if v != 'geometry']]
        if how == 'points':
            return poi_points
        if how == 'points_only':
            return poi_points['geometry']
        if how == 'circles':
            return poi_circles
        if how == 'circles_only':
            return poi_circles['geometry']

        raise Exception("Cannot interpret how='" + how + "'")


def get_poi(geom, precision_ratio=0.001, prec_min=0.5):
    """Get Pole of Inaccessibility (PoI) for a polygon geometry

    Returns the PoI as a pandas Series of (x, y, radius).

    Parameters
    ----------
    geom : Polygon or MultiPolygon
        Polygon for which largest inscribed circle is to be found
    precision_ratio : float
        Precision ratio used to define the precisions for:
        1. the polygon simplification algorithm
        2. the algorithm which finds the largest inscribed circle
        The precisions for both algorithms will be computed as:
        precision_ratio * square root of polygon area
    prec_min : float
        Minimum tolerance for both algorithms, defined in CRS units.

    Notes
    -----
    The precision has a major influence on performance. For geohashing,
    a precision_ratio of 0.05 appears to strike a good balance between
    uniqueness (0.1 is already unique), computation speed (0.01 is much
    slower) and correctness (0.1 can be quite off in some locations).

    No PoI will be computed for polygons whose area is smaller than the
    square of the polygon-specific tolerance. Function will return
    (None, None, None).
    """

    if isinstance(geom, Polygon):
        prec = max(geom.area**0.5 * precision_ratio, prec_min)
        if geom.area < prec * prec / 4:
            return None, None, None
        (x, y), r = polylabel(
            get_polygon_xy(geom.simplify(prec)),
            precision=prec,
            with_distance=True,
        )
    elif isinstance(geom, MultiPolygon):
        x, y, r = None, None, None
        for poly in geom.geoms:
            prec = max(poly.area**0.5 * precision_ratio, prec_min)
            # Skip processing polygons that are too small
            if poly.area < prec * prec / 4:
                continue
            (xi, yi), ri = polylabel(
                get_polygon_xy(poly.simplify(prec)),
                precision=prec,
                with_distance=True,
            )
            if r is None or ri > r:
                x, y, r = xi, yi, ri
    else:
        raise TypeError(
            'Not an accepted geometry type for GID computation: ' + str(type(geom))
        )
    return (x, y, r)


def get_poi_ortho(
    poly, precision_ratio=0.001, prec_min=0.5, geom=None, crs_orig='epsg:4326'
):
    """Get POI with local ortho projection

    Parameters
    ----------
    poly : Polygon
        Polygon geometry
    precision_ratio : float
        Precision ratio used to define the precisions for:
        1. the polygon simplification algorithm
        2. the algorithm which finds the largest inscribed circle
        The precisions for both algorithms will be computed as:
        precision_ratio * square root of polygon area
    prec_min : float
        Minimum tolerance for both algorithms, defined in CRS units.
    geom : str
        If 'point', adds point geometry
        If 'circle', adds circle geometry
    crs_orig : str
        Original projection (of Polygon).
        Defaults to WGS84
    """

    # Get lat / long from centroid
    poly_c = poly.centroid
    lat, lon = poly_c.y, poly_c.x

    # Prepare transformers
    orig = pyproj.CRS(crs_orig)
    ortho = pyproj.CRS(get_proj4('ortho', lat, lon))
    to_ortho = pyproj.Transformer.from_crs(orig, ortho, always_xy=True).transform
    to_orig = pyproj.Transformer.from_crs(ortho, orig, always_xy=True).transform

    # Identify Point of Inaccessibility in orthogonal projection
    x_ortho, y_ortho, r = get_poi(
        shapely.ops.transform(to_ortho, poly), precision_ratio, prec_min
    )

    # Create geometries and reproject them
    point_ortho = Point(x_ortho, y_ortho)
    point_orig = shapely.ops.transform(to_orig, point_ortho)

    if geom is None:
        return pd.Series({'x_poi': point_orig.x, 'y_poi': point_orig.y, 'r_poi': r})
    elif geom == 'point':
        return pd.Series(
            {
                'x_poi': point_orig.x,
                'y_poi': point_orig.y,
                'r_poi': r,
                'geometry': point_orig,
            }
        )
    elif geom == 'circle':
        circle_ortho = point_ortho.buffer(r)
        circle_orig = shapely.ops.transform(to_orig, circle_ortho)
        return pd.Series(
            {
                'x_poi': point_orig.x,
                'y_poi': point_orig.y,
                'r_poi': r,
                'geometry': circle_orig,
            }
        )


def get_polygon_xy(geom):
    """Get a list of x/y coordinates from a shapely Polygon

    Returns a list of lists, one for each ring (exterior or interiors),
    each containing lists of points (x, y).

    MultiPolygons are not accepted.

    Parameters
    ----------
    geom : Polygon
        Polygon
    """

    xy_ext = [[x, y] for x, y in zip(geom.exterior.xy[0], geom.exterior.xy[1])]

    if len(geom.interiors) == 0:
        return [xy_ext]

    xy_int_list = []
    for interior in geom.interiors:
        xy = zip(interior.xy[0], interior.xy[1])
        xy_int_list += [[[x, y] for x, y in xy]]
    return [xy_ext] + xy_int_list


def add_geometry_derivatives(gdf, timer, **kwargs):
    """Add standardized geometry derivatives to the geodataframe

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame
    timer : openplaces.timing.Timer
        Timer for data processing
    kwargs : dict
        Dictionary of arguments will be assumed as coming from an
        openplaces.recipes.recipe
    """

    # Get latitude and longitude of centroids (for fast plotting)
    gdf = gdf.join(get_lat_long_centroids(gdf))
    timer.mark('Get latitude and longitude at WGS84 centroid')

    # Get latitude and longitude of centroids (for plotting)
    gdf = gdf.join(get_areas(gdf))
    timer.mark('Get centroids')

    # Get coordinates of poles of inaccessibility (66s - for labeling)
    if kwargs and 'compute_poi' in kwargs:
        warnings.warn('`compute_poi` is not part of ingestion recipes anymore')

    return gdf


def points_from_coords(
    df: pd.DataFrame,
    x: str = 'long',
    y: str = 'lat',
    crs='epsg:4326',
    keep_columns: bool = False,
) -> gpd.GeoDataFrame:
    """Convert a DataFrame with coordinate columns to a GeoDataFrame of points.

    Parameters
    ----------
    df:
        DataFrame with x and y coordinate columns.
    x:
        Name of the x (longitude) column.
    y:
        Name of the y (latitude) column.
    crs:
        Coordinate reference system of the x/y coordinates.
    keep_columns:
        If False (default), drop the x and y columns after conversion.
    """

    # Try 'lon' if 'long' is asked for, but missing
    if x == 'long' and 'long' not in df and 'lon' in df:
        x = 'lon'

    if isinstance(df, gpd.GeoDataFrame):
        df = df.drop(columns='geometry')

    if 'geometry' in df.columns:
        raise ValueError("Column 'geometry' already exists in DataFrame.")

    df = df.copy()

    gdf = gpd.GeoDataFrame(
        df.drop(columns=[x, y]) if not keep_columns else df,
        geometry=gpd.points_from_xy(df[x], df[y], crs=crs),
        crs=crs,
    )
    return gdf


def get_simplified_geometries(gdf, tolerance):
    """Returns a GeoDataFrame with simplified polygon geometries

    Uses `.simplify_coverage()` from `geopandas` to preserve topology

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame
    tolerance : float
        Simplification tolerance in CRS units
    """
    gdf = gdf.copy()

    # Catching argument errors: enforce positive float for tolerance
    if not isinstance(tolerance, float):
        raise ValueError(f'`tolerance` is not a float: {str(tolerance)}')
    if not tolerance > 0:
        raise ValueError(f'`tolerance` is not positive: {str(tolerance)}')

    gdf['geometry'] = gdf['geometry'].simplify_coverage(tolerance)
    return gdf


def get_proj4(proj, lat=0, lon=0, ellps='WGS84'):
    """Get proj4 string for a projection from lat/long.

    Parameters
    ----------
    proj: str
        Projection type. Currently: 'ortho' or 'nsper'.
    lat : numeric
        Latitude
    lon : numeric
        Longitude
    ellps : str
        Ellipsoid
    """
    proj4 = PROJ4[proj].replace('{LAT}', str(lat)).replace('{LON}', str(lon))
    if ellps != 'WGS84':
        proj4.replace('WGS84', ellps)
    return proj4


def find_overlaps(
    gdf: gpd.GeoDataFrame,
    min_overlap_m2: float = 1.0,
    area_crs: str = 'EPSG:6933',
    iou: bool = False,
) -> pd.DataFrame:
    """Return pairs of row indices whose polygons overlap by more than a sliver.

    Uses an STRtree spatial index for fast candidate detection, then computes
    exact intersection areas with vectorised Shapely only for candidate pairs.

    Slivers (shared edges, floating-point artefacts) are excluded via
    ``min_overlap_m2``. Both overlapping and fully-contained pairs are detected
    (i.e., the test is intersection area > threshold, not shapely 'overlaps').

    Parameters
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame with polygon geometries.
    min_overlap_m2 : float
        Minimum intersection area in m² to count as a real overlap.
        Default 1 m² filters boundary slivers.
    area_crs : str
        Equal-area CRS for area computation. Default: EPSG:6933.
    iou : bool
        If True, also return ``area_left_m2``, ``area_right_m2``, ``iou``
        (intersection-over-union), and ``overlap_ratio``
        (overlap as a fraction of the smaller polygon's area) columns.

    Returns
    -------
    pd.DataFrame
        One row per overlapping pair with columns
        ``{index_name}_left``, ``{index_name}_right``, and ``overlap_m2``.
        If ``iou=True``, also includes ``area_left_m2``, ``area_right_m2``,
        ``iou``, and ``overlap_ratio``.
        Returns an empty DataFrame if no overlaps exceed the threshold.
    """
    idx_name = gdf.index.name or 'index'
    left_col = f'{idx_name}_left'
    right_col = f'{idx_name}_right'
    base_cols = [left_col, right_col, 'overlap_m2']
    empty = pd.DataFrame(
        columns=base_cols
        + (['area_left_m2', 'area_right_m2', 'iou', 'overlap_ratio'] if iou else [])
    )

    geom = (
        gdf.geometry
        if gdf.crs and gdf.crs == pyproj.CRS(area_crs)
        else gdf.geometry.to_crs(area_crs)
    )
    geom_arr = geom.to_numpy()

    tree = shapely.STRtree(geom_arr)
    left_pos, right_pos = tree.query(geom_arr, predicate='intersects')

    # Upper triangle only: no self-matches, no duplicate pairs
    mask = left_pos < right_pos
    left_pos = left_pos[mask]
    right_pos = right_pos[mask]

    if len(left_pos) == 0:
        return empty

    areas = shapely.area(shapely.intersection(geom_arr[left_pos], geom_arr[right_pos]))
    real = areas >= min_overlap_m2

    if not real.any():
        return empty

    idx = gdf.index.to_numpy()
    result = pd.DataFrame(
        {
            left_col: idx[left_pos[real]],
            right_col: idx[right_pos[real]],
            'overlap_m2': areas[real],
        }
    )

    if iou:
        area_left = shapely.area(geom_arr[left_pos[real]])
        area_right = shapely.area(geom_arr[right_pos[real]])
        result['area_left_m2'] = area_left
        result['area_right_m2'] = area_right
        result['iou'] = areas[real] / (area_left + area_right - areas[real])
        result['overlap_ratio'] = areas[real] / np.minimum(area_left, area_right)

    return result.set_index([left_col, right_col])


def resolve_overlapping_polygons(
    df: gpd.GeoDataFrame,
    keep=None,
    overlap_ratio_threshold: float = 0.5,
    iou_threshold: float | None = None,
    compare_cols: list | None = None,
    snippet_cols: list | None = None,
) -> gpd.GeoDataFrame:
    """Resolve substantially overlapping polygon pairs in a GeoDataFrame.

    For each overlapping pair, non-ID and non-geometry columns are compared:

    - If identical: the second polygon is dropped (likely a data duplicate).
    - If different: behaviour is controlled by `keep`:

      - None (default): warn and keep both polygons.
      - True: keep both polygons silently (suppresses warning).
      - False: drop the smaller polygon of each pair.
      - 'fewest_nulls': drop the polygon with more null values (None, NaN,
        '') per pair; fall back to area if tied.
      - {'prefer_higher': '<column>'}: drop the polygon with the lower value
        in the named column per pair; fall back to area if tied. Works with
        ordered categoricals, numerics, or any comparable type.

    Parameters
    ----------
    df : GeoDataFrame
        Input GeoDataFrame with polygon geometries.
    keep : bool or str or dict
        How to resolve overlapping pairs with differing attributes.
        See above.
    overlap_ratio_threshold : float or None
        A pair counts as a substantial overlap when the intersection area
        is at least this fraction of *either* polygon's area
        (i.e. ``overlap / min(area_left, area_right) >= threshold``).
        This ensures that a small polygon largely covered by a larger one
        is flagged as an overlap problem. Default 0.5.
        Set to None to disable.
    iou_threshold : float or None
        Minimum intersection-over-union to treat two polygons as overlapping
        (i.e. ``overlap / (area_left + area_right - overlap) >= threshold``).
        IoU is symmetric and size-agnostic, making it well suited for
        establishing identity between two polygon datasets (e.g. matching
        a predicted footprint to a reference). Not applied by default (None).
    compare_cols : list of str, optional
        Columns used to detect identical vs differing pairs. If None,
        all columns except geometry and columns containing '_id' are used.
    snippet_cols : list of str, optional
        Columns shown in the warning data snippet (first 5). If None,
        the first 5 of `compare_cols` are used.

    Returns
    -------
    GeoDataFrame
        DataFrame with overlapping duplicates removed (when applicable).
    """
    conditions = []
    if overlap_ratio_threshold is not None:
        conditions.append('overlap_ratio >= @overlap_ratio_threshold')
    if iou_threshold is not None:
        conditions.append('iou > @iou_threshold')
    if not conditions:
        raise ValueError(
            'At least one of overlap_ratio_threshold or iou_threshold must be set.'
        )
    overlaps = find_overlaps(df, iou=True).query(
        ' | '.join(f'({c})' for c in conditions)
    )
    if overlaps.empty:
        return df

    if compare_cols is None:
        # Exclude ID columns and geometry from comparison: we want to detect
        # polygons that are spatial duplicates but differ in attribute content.
        skip = {c for c in df.columns if '_id' in c} | {'geometry'}
        compare_cols = [c for c in df.columns if c not in skip]

    if snippet_cols is None:
        snippet_cols = compare_cols[:5]

    # Unpack dict form {'prefer_higher': col} into a flat keep=False + prefer_col.
    prefer_col = None
    if isinstance(keep, dict):
        prefer_col = keep.get('prefer_higher')
        keep = False

    # Sort so that MultiIndex .loc lookups in the loop below don't trigger
    # PerformanceWarning about indexing past lexsort depth.
    if isinstance(df.index, pd.MultiIndex) and not df.index.is_monotonic_increasing:
        df = df.sort_index()

    # First pass: classify each overlapping pair as an exact duplicate
    # (identical non-ID attributes → safe to drop) or ambiguous (differing
    # attributes → requires a decision).
    dupes_to_drop = set()
    ambiguous = []
    for left_idx, right_idx in overlaps.index:
        # Skip if one side was already resolved in a prior iteration.
        if left_idx in dupes_to_drop or right_idx in dupes_to_drop:
            continue
        if compare_cols and df.loc[left_idx, compare_cols].equals(
            df.loc[right_idx, compare_cols]
        ):
            dupes_to_drop.add(right_idx)
        else:
            ambiguous.append((left_idx, right_idx))

    if dupes_to_drop:
        df = df.drop(index=list(dupes_to_drop))

    # Second pass: resolve ambiguous pairs according to `keep`.
    # Note: an index can appear in both `ambiguous` and `dupes_to_drop` if it
    # was the right side of a "differing" pair before being the right side of
    # an "identical" pair. Guard against accessing a dropped index with the
    # `dupes_to_drop` check below.
    if ambiguous and keep not in (True, None):
        to_drop = set()
        for left_idx, right_idx in ambiguous:
            if (
                left_idx in to_drop
                or right_idx in to_drop
                or left_idx in dupes_to_drop
                or right_idx in dupes_to_drop
            ):
                continue
            if prefer_col is not None and prefer_col in df.columns:
                # Keep the polygon with the higher value in prefer_col.
                # Works with ordered categoricals, numerics, or any comparable
                # type. Falls through to area tiebreak if values are equal or
                # either is NA.
                left_val = df.loc[left_idx, prefer_col]
                right_val = df.loc[right_idx, prefer_col]
                try:
                    if (
                        pd.notna(left_val)
                        and pd.notna(right_val)
                        and left_val != right_val
                    ):
                        to_drop.add(right_idx if left_val > right_val else left_idx)
                        continue
                except TypeError:
                    pass
            elif keep == 'fewest_nulls':
                # Keep the polygon with fewer missing values (None, NaN, '').
                # Falls through to area tiebreak if counts are equal.
                def _null_count(idx):
                    row = df.loc[idx, compare_cols]
                    return row.isna().sum() + (row == '').sum()

                nulls_left, nulls_right = _null_count(left_idx), _null_count(right_idx)
                if nulls_left != nulls_right:
                    to_drop.add(left_idx if nulls_left > nulls_right else right_idx)
                    continue
            # Fallback for keep=False, tied fewest_nulls, or tied prefer_higher:
            # drop the smaller polygon.
            area_left = overlaps.at[(left_idx, right_idx), 'area_left_m2']
            area_right = overlaps.at[(left_idx, right_idx), 'area_right_m2']
            to_drop.add(right_idx if area_left >= area_right else left_idx)
        df = df.drop(index=list(to_drop))

    msg_parts = ['\n']
    if dupes_to_drop:
        msg_parts.append(
            f'Dropped {len(dupes_to_drop)} polygon(s) with identical '
            'non-ID, non-geometry attributes.'
        )
    if ambiguous and keep is None:
        pair_str = '\n'.join(f'  {left} and {right}' for left, right in ambiguous[:5])
        # Exclude any indices already dropped as exact duplicates to avoid
        # a KeyError when building the snippet.
        snippet_idx = [
            idx for pair in ambiguous[:2] for idx in pair if idx not in dupes_to_drop
        ]
        snippet = (
            '\n' + df.loc[snippet_idx, snippet_cols].to_string() if snippet_cols else ''
        )
        msg_parts.append(
            f'Found {len(ambiguous)} polygon pair(s) with \033[1mdiffering\033[0m '
            'non-ID, non-geometry attributes.\n'
            'Set argument `keep` in resolve_overlapping_polygons:\n'
            '- `True` keeps duplicates\n'
            '- `False` drops the smaller polygon\n'
            "- 'fewest_nulls' prefers the more complete attribute data\n"
            "- {'prefer_higher': '<column>'} keeps top value in specific column.\n\n"
            'In recipes, use the parameter `keep_overlapping_polygons: ...`.\n\n'
            'Indices of overlaps (first 5):\n\n' + pair_str + '\n' + snippet
        )
    if len(msg_parts) > 1:
        threshold_desc = ' | '.join(
            filter(
                None,
                [
                    f'overlap_ratio >= {overlap_ratio_threshold}'
                    if overlap_ratio_threshold is not None
                    else None,
                    f'IoU > {iou_threshold}' if iou_threshold is not None else None,
                ],
            )
        )
        warnings.warn(
            f'\n\nOverlapping polygons detected ({threshold_desc}):'
            + '\n'.join(msg_parts)
        )

    return df


def _coverage_fractions(piece_intersection, index_name, gdf):
    """Return fraction of each polygon in gdf covered by piece_intersection."""
    frag_area = shapely.area(piece_intersection.geometry.values)
    covered = (
        pd.Series(frag_area, name='_fa')
        .groupby(piece_intersection[index_name].values)
        .sum()
    )
    native = pd.Series(
        shapely.area(gdf.geometry.values),
        index=gdf[index_name].values,
        name='_na',
    )
    native.index.name = index_name
    return covered / native


def _leftover_fragments(
    gdf_self, self_idx, gdf_other, other_idx, partial_ids, piece_intersection
):
    """ST_Difference of partially-covered self polygons minus union of
    matching other geometries."""
    partial_self = gdf_self[gdf_self[self_idx].isin(partial_ids)].set_index(self_idx)
    other_geom = gdf_other.set_index(other_idx)[['geometry']]
    pairs = (
        piece_intersection[piece_intersection[self_idx].isin(partial_ids)][
            [self_idx, other_idx]
        ]
        .copy()
        .join(other_geom.rename(columns={'geometry': '_og'}), on=other_idx)
    )
    other_union = pairs.groupby(self_idx)['_og'].agg(shapely.union_all)
    leftover = shapely.difference(
        partial_self.geometry.reindex(other_union.index).values,
        other_union.values,
    )
    keep = ~shapely.is_empty(leftover)
    return other_union.index[keep], leftover[keep]


def _unmatched_piece(gdf_self, self_idx, other_idx, other_data_cols, ids):
    """Rows from gdf_self for ids, with other-side columns filled as NA/NaN."""
    mask = gdf_self[self_idx].isin(ids)
    piece = gdf_self[mask].copy()
    piece[other_idx] = pd.array([pd.NA] * mask.sum(), dtype=object)
    for col in other_data_cols:
        piece[col] = float('nan')
    return piece


def _identity_overlay(gdf_left, gdf_right, left_index_name, right_index_name):
    """Fast identity overlay: intersection, unmatched left, leftover left fragments."""
    piece_intersection = gpd.overlay(
        gdf_left, gdf_right, how='intersection', keep_geom_type=False
    )
    piece_intersection = piece_intersection[
        piece_intersection.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])
    ]

    frac = _coverage_fractions(piece_intersection, left_index_name, gdf_left)
    matched_ids = set(piece_intersection[left_index_name].unique())
    fully_unmatched_ids = set(gdf_left[left_index_name].values) - matched_ids
    _TOL = 1e-6
    partial_ids = set(frac.index[frac < 1 - _TOL])

    right_data_cols = [
        c for c in gdf_right.columns if c not in ('geometry', right_index_name)
    ]

    pieces = [piece_intersection]

    if fully_unmatched_ids:
        pieces.append(
            _unmatched_piece(
                gdf_left,
                left_index_name,
                right_index_name,
                right_data_cols,
                fully_unmatched_ids,
            )
        )

    if partial_ids:
        leftover_index, leftover_geoms = _leftover_fragments(
            gdf_left,
            left_index_name,
            gdf_right,
            right_index_name,
            partial_ids,
            piece_intersection,
        )
        left_data_cols = [
            c for c in gdf_left.columns if c not in ('geometry', left_index_name)
        ]
        partial_left = gdf_left[gdf_left[left_index_name].isin(partial_ids)].set_index(
            left_index_name
        )
        pieces.append(
            gpd.GeoDataFrame(
                {
                    left_index_name: leftover_index,
                    **{
                        col: partial_left[col].reindex(leftover_index).values
                        for col in left_data_cols
                    },
                    right_index_name: pd.array(
                        [pd.NA] * len(leftover_index), dtype=object
                    ),
                    **{col: float('nan') for col in right_data_cols},
                    'geometry': leftover_geoms,
                },
                geometry='geometry',
                crs=gdf_left.crs,
            )
        )

    result = pd.concat(pieces, ignore_index=True)
    return gpd.GeoDataFrame(result, geometry='geometry', crs=gdf_left.crs)


def _union_overlay(gdf_left, gdf_right, left_index_name, right_index_name):
    """Fast union overlay: intersection + unmatched both sides + leftover both sides."""
    piece_intersection = gpd.overlay(
        gdf_left, gdf_right, how='intersection', keep_geom_type=False
    )
    piece_intersection = piece_intersection[
        piece_intersection.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])
    ]

    _TOL = 1e-6

    frac_left = _coverage_fractions(piece_intersection, left_index_name, gdf_left)
    matched_left = set(piece_intersection[left_index_name].unique())
    unmatched_left_ids = set(gdf_left[left_index_name].values) - matched_left
    partial_left_ids = set(frac_left.index[frac_left < 1 - _TOL])

    frac_right = _coverage_fractions(piece_intersection, right_index_name, gdf_right)
    matched_right = set(piece_intersection[right_index_name].unique())
    unmatched_right_ids = set(gdf_right[right_index_name].values) - matched_right
    partial_right_ids = set(frac_right.index[frac_right < 1 - _TOL])

    left_data_cols = [
        c for c in gdf_left.columns if c not in ('geometry', left_index_name)
    ]
    right_data_cols = [
        c for c in gdf_right.columns if c not in ('geometry', right_index_name)
    ]

    pieces = [piece_intersection]

    if unmatched_left_ids:
        pieces.append(
            _unmatched_piece(
                gdf_left,
                left_index_name,
                right_index_name,
                right_data_cols,
                unmatched_left_ids,
            )
        )
    if partial_left_ids:
        leftover_index, leftover_geoms = _leftover_fragments(
            gdf_left,
            left_index_name,
            gdf_right,
            right_index_name,
            partial_left_ids,
            piece_intersection,
        )
        partial_left = gdf_left[
            gdf_left[left_index_name].isin(partial_left_ids)
        ].set_index(left_index_name)
        pieces.append(
            gpd.GeoDataFrame(
                {
                    left_index_name: leftover_index,
                    **{
                        col: partial_left[col].reindex(leftover_index).values
                        for col in left_data_cols
                    },
                    right_index_name: pd.array(
                        [pd.NA] * len(leftover_index), dtype=object
                    ),
                    **{col: float('nan') for col in right_data_cols},
                    'geometry': leftover_geoms,
                },
                geometry='geometry',
                crs=gdf_left.crs,
            )
        )

    if unmatched_right_ids:
        pieces.append(
            _unmatched_piece(
                gdf_right,
                right_index_name,
                left_index_name,
                left_data_cols,
                unmatched_right_ids,
            )
        )
    if partial_right_ids:
        leftover_index, leftover_geoms = _leftover_fragments(
            gdf_right,
            right_index_name,
            gdf_left,
            left_index_name,
            partial_right_ids,
            piece_intersection,
        )
        partial_right = gdf_right[
            gdf_right[right_index_name].isin(partial_right_ids)
        ].set_index(right_index_name)
        pieces.append(
            gpd.GeoDataFrame(
                {
                    right_index_name: leftover_index,
                    **{
                        col: partial_right[col].reindex(leftover_index).values
                        for col in right_data_cols
                    },
                    left_index_name: pd.array(
                        [pd.NA] * len(leftover_index), dtype=object
                    ),
                    **{col: float('nan') for col in left_data_cols},
                    'geometry': leftover_geoms,
                },
                geometry='geometry',
                crs=gdf_right.crs,
            )
        )

    result = pd.concat(pieces, ignore_index=True)
    return gpd.GeoDataFrame(result, geometry='geometry', crs=gdf_left.crs)


def overlay_polygons(
    layer1,
    layer2,
    columns: list[str] | None = None,
    geom: bool = False,
    iou: bool = False,
    suffixes: tuple[str, str] | None = None,
    how: str = 'intersection',
) -> 'pd.DataFrame | gpd.GeoDataFrame':
    """Intersect two polygon datasets in memory using geopandas.

    Parameters
    ----------
    layer1, layer2 :
        GeoDataFrame or path to an attribute parquet file (read with
        ``openplaces.io.read_parquet``).
    columns :
        Extra columns to carry from the attribute tables into the result.
    geom :
        If True, return intersection geometry.
    iou :
        If True, compute intersection-over-union. Areas are in m²
        (EPSG:6933). Unmatched/leftover rows get NaN.
    suffixes :
        Required when both tables share the same index name, or when a
        requested column exists in both tables.
    how : {'intersection', 'union', 'identity'}
        Overlay type.
    """
    from pathlib import Path

    from openplaces.io import read_parquet

    _SUPPORTED_HOW = {'intersection', 'union', 'identity'}
    if how not in _SUPPORTED_HOW:
        raise ValueError(
            f'how={how!r} not supported. Choose from {sorted(_SUPPORTED_HOW)}.'
        )

    if isinstance(layer1, Path):
        layer1 = read_parquet(layer1, geom=True)
    if isinstance(layer2, Path):
        layer2 = read_parquet(layer2, geom=True)

    # Resolve index aliases
    idx1 = layer1.index.name or 'index'
    idx2 = layer2.index.name or 'index'
    if idx1 == idx2:
        if suffixes is None:
            raise ValueError(f'Both tables share index name {idx1!r}. Pass suffixes.')
        alias1 = f'{idx1}{suffixes[0]}'
        alias2 = f'{idx2}{suffixes[1]}'
    else:
        alias1, alias2 = idx1, idx2

    data1 = set(layer1.columns) - {'geometry'}
    data2 = set(layer2.columns) - {'geometry'}

    if columns is not None:
        ambiguous = [c for c in columns if c in data1 and c in data2]
        if ambiguous and suffixes is None:
            raise ValueError(
                f'Columns {ambiguous} exist in both tables. Pass suffixes.'
            )
        missing = [c for c in columns if c not in data1 and c not in data2]
        if missing:
            raise ValueError(f'Columns {missing} not found in either table.')
        if ambiguous:
            layer1 = layer1.rename(columns={c: f'{c}{suffixes[0]}' for c in ambiguous})
            layer2 = layer2.rename(columns={c: f'{c}{suffixes[1]}' for c in ambiguous})
        left_keep = [
            f'{c}{suffixes[0]}' if c in ambiguous else c for c in columns if c in data1
        ]
        right_keep = [
            f'{c}{suffixes[1]}' if c in ambiguous else c for c in columns if c in data2
        ]
    else:
        left_keep, right_keep = [], []

    # Internal area column names (prefixed to avoid user column conflicts)
    _A1, _A2 = '_area1', '_area2'

    def _slim(gdf, alias, orig_idx, area_col, extra):
        """Select columns, optionally add area, reset index."""
        keep = extra + ['geometry']
        if iou:
            if area_col not in gdf.columns:
                gdf = gdf.copy()
                gdf[area_col] = get_areas(gdf, 'm2')
            keep = extra + [area_col, 'geometry']
        out = gdf[keep].reset_index()
        if orig_idx != alias:
            out = out.rename(columns={orig_idx: alias})
        return out

    _gdf1 = _slim(layer1, alias1, idx1, _A1, left_keep)
    _gdf2 = _slim(layer2, alias2, idx2, _A2, right_keep)

    if how == 'intersection':
        result = gpd.overlay(_gdf1, _gdf2, how='intersection', keep_geom_type=False)
        result = result[result.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])]
    elif how == 'identity':
        result = _identity_overlay(_gdf1, _gdf2, alias1, alias2)
    else:
        result = _union_overlay(_gdf1, _gdf2, alias1, alias2)

    result = result.set_index([alias1, alias2])

    if iou:
        _sfx1 = suffixes[0] if suffixes is not None else '_left'
        _sfx2 = suffixes[1] if suffixes is not None else '_right'
        aint = get_areas(result, 'm2')
        has_both = result[_A1].notna() & result[_A2].notna()
        aint_matched = aint.where(has_both)  # NaN for unmatched/leftover rows
        denom = result[_A1] + result[_A2] - aint_matched
        result = result.rename(
            columns={
                _A1: f'area{_sfx1}_m2',
                _A2: f'area{_sfx2}_m2',
            }
        )
        result['area_intersection_m2'] = aint_matched
        result['iou'] = aint_matched / denom.replace(0, float('nan'))

    if not geom:
        return result.drop(columns=['geometry'])

    cols = [c for c in result if c != 'geometry'] + ['geometry']
    return gpd.GeoDataFrame(result[cols], geometry='geometry', crs=layer1.crs)
