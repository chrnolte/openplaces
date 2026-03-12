#!/usr/bin/env python

"""
vector.py

Functions for processing vector data (GeoDataFrames, geometries)
"""

import json
import warnings
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import pyogrio
import pyproj
import shapely
from polylabel import polylabel
from shapely.geometry import MultiPolygon, Point, Polygon

from openplaces.api import get_admin
from openplaces.core.constants import (
    AC_TO_HA,
    M2_TO_SQFT,
    STRING_SEPARATOR_BETWEEN_IDS,
)
from openplaces.recipe import get_recipe
from openplaces.timing import get_timer

PROJ4 = {
    'ortho': '+proj=ortho +lat_0={LAT} +lon_0={LON} +x_0=0 +y_0=0 '
    '+ellps=WGS84 +units=m +no_defs',
    'moon': '+proj=nsper +h=384400000 +lon_0={LON} +lat_0={LAT} +ellps=WGS84',
    'landsat': '+proj=nsper +h=705000 +lon_0={LON} +lat_0={LAT} +ellps=WGS84',
    'eck': '+proj=eck4 +ellps=WGS84',
}


def fix_polygons(gdf):
    """Fix polygons of a GeoDataFrame by adding a zero buffer.

    This fixes most invalid geometry issues found in parcel data.

    Parameters
    ----------
    gdf : GeoDataFrame
        Geodataframe that is suspected to have invalid geometries
    """
    gdf['geometry'] = gdf['geometry'].buffer(0)
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


def get_intersection_over_union(
    gdf_left,
    gdf_right,
    suffixes=('left', 'right'),
    area_unit='m2',
    how='intersection',
    drop_geometries=True,
):
    """Compute interaction over union of two GeoDataFrames"""

    # Make sure area column exists in both GeoDataFrames
    if area_unit not in gdf_left:
        gdf_left = gdf_left.copy()
        gdf_left[area_unit] = get_areas(gdf_left, area_unit)
    if area_unit not in gdf_right:
        gdf_right = gdf_right.copy()
        gdf_right[area_unit] = get_areas(gdf_right, area_unit)

    left_area_col = f'{area_unit}_{suffixes[0]}'
    right_area_col = f'{area_unit}_{suffixes[1]}'
    intersection_area_col = f'{area_unit}_intersection'

    # Overlay
    _gdf_left = (
        gdf_left[[area_unit, 'geometry']]
        .rename(columns={area_unit: left_area_col})
        .reset_index()
    )
    _gdf_right = (
        gdf_right[[area_unit, 'geometry']]
        .rename(columns={area_unit: right_area_col})
        .reset_index()
    )
    left_index_name = gdf_left.index.name
    right_index_name = gdf_right.index.name
    if left_index_name == right_index_name:
        left_index_name += '_' + suffixes[0]
        right_index_name += '_' + suffixes[1]
        _gdf_left = _gdf_left.rename(columns={gdf_left.index.name: left_index_name})
        _gdf_right = _gdf_right.rename(columns={gdf_right.index.name: right_index_name})
    gdf_overlay = gpd.overlay(
        _gdf_left,
        _gdf_right,
        how=how,
    )

    gdf_overlay = gdf_overlay.set_index([left_index_name, right_index_name])

    # Calculate area of overlap
    gdf_overlay[intersection_area_col] = get_areas(gdf_overlay, area_unit)

    # Calculate interaction over union
    gdf_overlay['iou'] = gdf_overlay[intersection_area_col] / (
        gdf_overlay[left_area_col]
        + gdf_overlay[right_area_col]
        - gdf_overlay[intersection_area_col]
    )

    if drop_geometries:
        gdf_overlay = gdf_overlay.drop(columns=['geometry'])
    else:
        # Move 'geometry' to the end
        gdf_overlay = gdf_overlay[
            [x for x in gdf_overlay if x != 'geometry'] + ['geometry']
        ]
    return gdf_overlay


def get_crs(filepath):
    """Get the CRS from the metadata of a file using `pyogrio`"""
    geo_metadata = pyogrio.read_info(filepath)
    if 'crs' not in geo_metadata:
        warnings.warn('No CRS found in input data.')
        return None

    return geo_metadata['crs']


def overlay_admin_ids(
    gdf,
    admin_geometries=None,
    admin_level=2,
    admin_id=None,
    admin_recipe=None,
    include_overlays=False,
    timer=None,
):
    """Add administrative unit IDs to GeoDataFrame using spatial joins

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame to join admin IDs to
    admin_geometries : gpd.GeoSeries
        GeoSeries of admin geometries with AdminID index.
        Pass this or `admin_level`
    admin_level : int
        Administrative level for which administrative IDs are to be
        joined. Typically a lower level (larger number) than the level
        of `admin_id`.
    admin_id : str
        Administrative unit of the GeoDataFrame.
        Determines which administrative units to consider.
    admin_recipe : dict or str
        Recipe of the administrative unit dataset to be used.
        String identifier or resolved recipe (dictionary).
        If None, the default recipe for administrations is used.
    include_overlays : bool
        If True, attempt a spatial polygon overlay for polygons for
        which the spatial join (centroids) returned no results.
    timer : openplaces.timing.Timer or None
        Timer
    """
    if timer is None:
        timer = get_timer('overlay_admin_ids', verbose=True)

    if isinstance(admin_recipe, str):
        # Assume recipe contained the complete filename, infer parts
        # {admin_id}_{entity}_{filename.ext}
        recipe_admin_id, entity, filename = admin_recipe.split(
            STRING_SEPARATOR_BETWEEN_IDS
        )
        admin_recipe = get_recipe(recipe_admin_id, entity, filename=filename)

    if admin_geometries is None:
        admin = get_admin(
            admin_id, admin_level, recipe=admin_recipe, columns=[], geom=True
        )
    else:
        admin = admin_geometries.to_frame()

    # Cast admin index to pd.Categorical to later save space in the joined column
    admin.index = pd.Index(pd.Categorical(admin.index), name=admin.index.name)
    timer.mark('Admin overlay: get admin layer')

    # Get centroid points with range index (for quicker processing)
    gdf_centroids = get_lat_long_centroids(gdf.reset_index()[['geometry']], geom=True)[
        ['geometry']
    ]
    timer.mark('Admin overlay: get centroids')

    gdf_sjoin = gpd.sjoin(gdf_centroids, admin[['geometry']], how='left')
    gdf[admin.index.name] = gdf_sjoin[admin.index.name].values
    del gdf_sjoin
    timer.mark('Admin overlay: spatial join')

    if include_overlays:
        mask = gdf[admin.index.name].isnull()
        if mask.any():
            gdf_overlay = (
                gpd.overlay(
                    gdf[mask][['geometry']].reset_index(),
                    admin[['geometry']].reset_index(),
                )
                .drop(columns='geometry')
                .set_index(gdf.index.name)
            )
            gdf.loc[mask, admin.index.name] = gdf_overlay[admin.index.name]
            del gdf_overlay
            timer.mark('Admin overlay: spatial overlay')

    return gdf


def _detect(attr_path: Path, geo_path: Path) -> tuple[str, str, list[str], str]:
    """Return (index_name, join_key, data_columns, crs) from parquet metadata."""
    schema = pq.read_schema(attr_path)
    index_name = schema.pandas_metadata['index_columns'][0]
    data_cols = [
        c['name'] for c in schema.pandas_metadata['columns'] if c['name'] != index_name
    ]
    geo_schema = pq.read_schema(geo_path)
    join_key = 'geo_id' if 'geo_id' in geo_schema.names else '_join_id'
    geo_meta = json.loads(geo_schema.metadata[b'geo'])
    crs = geo_meta['columns']['geometry']['crs']['id']['code']
    return index_name, join_key, data_cols, f'EPSG:{crs}'


def overlay_polygons(
    path1: Path,
    path2: Path,
    columns: list[str] | None = None,
    geom: bool = False,
    iou: bool = False,
    suffixes: tuple[str, str] | None = None,
    how: str = 'intersection',
) -> pd.DataFrame | gpd.GeoDataFrame:
    """Intersect two polygon datasets stored as GeoParquet.

    Each path should point to an attribute parquet file saved with
    `save_parquet()`. The corresponding `_geo.parquet` file is imputed
    automatically.

    Parameters
    ----------
    path1 :
        Path to the first attribute parquet file.
    path2 :
        Path to the second attribute parquet file.
    columns :
        Columns to return from the attribute tables. Columns are auto-detected
        from both tables. If a column exists in both tables, `suffixes` must
        be provided.
    geom :
        If True, return the clipped intersection geometry as a GeoDataFrame.
    iou :
        If True, compute intersection area, union area, and
        intersection-over-union ratio. Areas are computed in EPSG:6933 (m²).
        Only meaningful for matched pairs; unmatched rows get NaN.
    suffixes :
        Required when any requested column exists in both attribute tables,
        or when both tables share the same index name. Tuple of two strings,
        e.g. ('_tiles', '_admin'), appended to disambiguate column names,
        analogous to gpd.sjoin suffixes.
    how : {'intersection', 'union', 'identity'}
        Type of overlay operation:
        - 'intersection': only overlapping pairs (inner join)
        - 'union': all polygons from both sides; unmatched rows retain
          original geometry and get NaN for the missing index level
        - 'identity': all polygons from path1; unmatched path1 polygons
          retain original geometry and get NaN for the path2 index level

    Returns
    -------
    pd.DataFrame or gpd.GeoDataFrame
        MultiIndex of (index1, index2) detected from parquet metadata.
        Columns: those requested via `columns`, plus iou columns if iou=True,
        plus geometry if geom=True.

    Raises
    ------
    FileNotFoundError
        If any of the expected parquet files are missing.
    ValueError
        If any requested column exists in both tables and suffixes is None,
        or if both tables share the same index name and suffixes is None,
        or if how is not one of the supported operations.
    """
    _SUPPORTED_HOW = {'intersection', 'union', 'identity'}
    if how not in _SUPPORTED_HOW:
        raise ValueError(
            f'how={how!r} is not supported. Choose from {sorted(_SUPPORTED_HOW)}.'
        )

    path1 = Path(path1)
    path2 = Path(path2)
    geo_path1 = path1.with_stem(path1.stem + '_geo')
    geo_path2 = path2.with_stem(path2.stem + '_geo')

    for p in [path1, path2, geo_path1, geo_path2]:
        if not p.exists():
            raise FileNotFoundError(p)

    # Detect index names, join keys, and data columns
    index1, join_key1, data_cols1, crs1 = _detect(path1, geo_path1)
    index2, join_key2, data_cols2, crs2 = _detect(path2, geo_path2)

    # Resolve MultiIndex name conflict

    if index1 == index2:
        if suffixes is None:
            raise ValueError(
                f'Both tables have index name {index1!r}. '
                f'Pass suffixes to disambiguate.'
            )
        alias1 = f'{index1}{suffixes[0]}'
        alias2 = f'{index2}{suffixes[1]}'
    else:
        alias1 = index1
        alias2 = index2

    # Resolve requested columns

    col_select: list[str] = []

    if columns:
        set1 = set(data_cols1)
        set2 = set(data_cols2)

        ambiguous = [c for c in columns if c in set1 and c in set2]
        if ambiguous and suffixes is None:
            raise ValueError(
                f'Columns {ambiguous} exist in both tables. '
                f'Pass suffixes to disambiguate.'
            )

        missing = [c for c in columns if c not in set1 and c not in set2]
        if missing:
            raise ValueError(f'Columns {missing} not found in either attribute table.')

        for col in columns:
            in1 = col in set1
            in2 = col in set2
            if in1 and in2:
                col_select.append(f'a1.{col} AS {col}{suffixes[0]}')
                col_select.append(f'a2.{col} AS {col}{suffixes[1]}')
            elif in1:
                col_select.append(f'a1.{col}')
            else:
                col_select.append(f'a2.{col}')

    # Build SELECT clause

    def _area_m2(geom_expr: str, src_crs: str) -> str:
        """Reproject to EPSG:6933 and compute area in m²."""
        return f"ST_Area(ST_Transform({geom_expr}, '{src_crs}', 'EPSG:6933', true))"

    # SQL JOIN type and geometry expression depend on how
    if how == 'intersection':
        join_type = 'JOIN'
        geom_expr = 'ST_Intersection(g1.geometry::GEOMETRY, g2.geometry::GEOMETRY)'
    elif how == 'identity':
        join_type = 'LEFT JOIN'
        geom_expr = (
            f'CASE WHEN g2.{join_key2} IS NULL '
            'THEN g1.geometry::GEOMETRY '
            'ELSE ST_Intersection(g1.geometry::GEOMETRY, g2.geometry::GEOMETRY) '
            'END'
        )
    elif how == 'union':
        join_type = 'FULL OUTER JOIN'
        geom_expr = (
            f'CASE WHEN g2.{join_key2} IS NULL '
            'THEN g1.geometry::GEOMETRY '
            f'WHEN g1.{join_key1} IS NULL '
            'THEN g2.geometry::GEOMETRY '
            'ELSE ST_Intersection(g1.geometry::GEOMETRY, g2.geometry::GEOMETRY) '
            'END'
        )

    select_parts = [
        f'a1.{index1} AS {alias1}',
        f'a2.{index2} AS {alias2}',
    ]

    if iou:
        intersection_expr = (
            'ST_Intersection(g1.geometry::GEOMETRY, g2.geometry::GEOMETRY)'
        )
        select_parts += [
            f'{_area_m2("g1.geometry::GEOMETRY", crs1)} AS area1_m2',
            f'{_area_m2("g2.geometry::GEOMETRY", crs2)} AS area2_m2',
            f'{_area_m2(intersection_expr, crs1)} AS area_intersection_m2',
        ]

    select_parts += col_select

    if geom:
        select_parts.append(f'ST_AsWKB({geom_expr})::BLOB AS geometry')

    select_clause = ',\n            '.join(select_parts)

    # Execute query
    con = duckdb.connect()
    con.execute('INSTALL spatial; LOAD spatial;')
    con.execute('SET enable_progress_bar = false;')

    overlaps = con.execute(
        f"""
        SELECT
            {select_clause}
        FROM read_parquet('{geo_path1}') AS g1
        JOIN read_parquet('{path1}') AS a1
            ON g1.{join_key1} = a1.{join_key1}
        {join_type} read_parquet('{geo_path2}') AS g2
            ON ST_Intersects(
                g1.geometry::GEOMETRY,
                g2.geometry::GEOMETRY
            )
        LEFT JOIN read_parquet('{path2}') AS a2
            ON g2.{join_key2} = a2.{join_key2}
    """
    ).df()

    con.close()

    # Post-process
    if iou:
        area_union = (
            overlaps['area1_m2']
            + overlaps['area2_m2']
            - overlaps['area_intersection_m2']
        )
        overlaps['iou'] = overlaps['area_intersection_m2'] / area_union.replace(
            0, float('nan')
        )

    overlaps = overlaps.set_index([alias1, alias2])

    if geom:
        overlaps['geometry'] = shapely.from_wkb(overlaps['geometry'].apply(bytes))
        overlaps = overlaps[[v for v in overlaps if v != 'geometry'] + ['geometry']]
        return gpd.GeoDataFrame(overlaps, geometry='geometry', crs=crs1)

    return overlaps
