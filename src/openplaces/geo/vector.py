#!/usr/bin/env python

"""
vector.py

Functions for processing vector data (GeoDataFrames, geometries)

"""

import hashlib
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
from polylabel import polylabel
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import transform

from openplaces.api import get_admin_by_level
from openplaces.core.constants import (
    AC_TO_HA,
    M2_TO_SQFT,
    STRING_SEPARATOR_BETWEEN_IDS,
)
from openplaces.recipe import get_recipe
from openplaces.timing import get_timer
from openplaces.io.transform import add_unique_suffix

PROJ4 = {
    'ortho': '+proj=ortho +lat_0={LAT} +lon_0={LON} +x_0=0 +y_0=0 '
    '+ellps=WGS84 +units=m +no_defs',
    'moon': '+proj=nsper +h=384400000 +lon_0={LON} +lat_0={LAT} ' '+ellps=WGS84',
    'landsat': '+proj=nsper +h=705000 +lon_0={LON} +lat_0={LAT} ' '+ellps=WGS84',
    'eck': '+proj=eck4 +ellps=WGS84',
}


def fix_geometries(gdf):
    """Fix the geometries of a GeoDataFrame by adding a zero buffer.

    This fixes most invalid geometry issues found in parcel data.

    Parameters
    ----------
    gdf : GeoDataFrame
        Geodataframe that is suspected to have invalid geometries
    """
    gdf['geometry'] = gdf['geometry'].buffer(0)
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
    x_ortho, y_ortho, r = get_poi(transform(to_ortho, poly), precision_ratio, prec_min)

    # Create geometries and reproject them
    point_ortho = Point(x_ortho, y_ortho)
    point_orig = transform(to_orig, point_ortho)

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
        circle_orig = transform(to_orig, circle_ortho)
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


def get_geo_ids(
    gdf,
    grid_degrees=0.00003,  # ~3m at equator, ~2m at 45°N, ~1.5m at 60°N
    hash_length=24,
    handle_duplicates=True,
    verbose=False,
):
    """Generate stable, unique parcel IDs from polygon geometry.

    Uses fixed degree grid in EPSG:4326 for full Earth coverage.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame with parcel geometries
    grid_degrees : float
        Grid size in degrees (default 0.00003)
    hash_length : int
        Number of hex characters in output (default 18 = 72 bits)
    handle_duplicates : bool
        If True, adds numeric suffix to duplicate GIDs (default True)
    verbose: bool
        If True, prints information on duplicates

    Returns
    -------
    pd.Series
        Series of geo_ids with same index as input GeoDataFrame

    Notes
    -----
    Why degrees instead of projected CRS:
    - No projection covers entire Earth without distortion/singularities
    - Degree grid is globally consistent (same grid cell = same x/y)
    - Simple, fast (no reprojection needed)
    - Works everywhere including poles

    Trade-off:
    - Grid "size" in meters varies by latitude (larger at equator)
    - But parcels at same location always use same grid
    - This guarantees non-overlapping parcels get different IDs
    """

    # Ensure EPSG:4326
    if gdf.crs != 'epsg:4326':
        print('Reprojecting vector data to `epsg:4326` to compute `geo_ids`.')
        gdf = gdf.to_crs('epsg:4326')

    # Get bounds for each parcel
    # (using fillna(0) to avoid checking for empty geometries)
    bounds = gdf.bounds.fillna(0)

    # Quantize bbox corners (consistent grid for all parcels)
    minx_q = (bounds['minx'] / grid_degrees).round().astype(int)
    miny_q = (bounds['miny'] / grid_degrees).round().astype(int)
    maxx_q = (bounds['maxx'] / grid_degrees).round().astype(int)
    maxy_q = (bounds['maxy'] / grid_degrees).round().astype(int)

    # Area in square degrees (log scale)
    # Note: Area in degrees² varies with latitude, but that's okay
    # because we're comparing relative sizes at similar locations
    warnings.filterwarnings('ignore', 'Geometry is in a geographic CRS')
    area_deg2 = gdf.area.fillna(0)
    warnings.filterwarnings('default', 'Geometry is in a geographic CRS')
    area_q = (
        (np.log10(area_deg2 * 1e10 + 1) * 100).round().fillna(0).astype(int)
    )  # Scale up for precision

    # Compactness: perimeter²/area (dimensionless, so units don't matter)
    warnings.filterwarnings('ignore', 'Geometry is in a geographic CRS')
    compactness = (gdf.length**2) / (area_deg2 + 1e-10)
    warnings.filterwarnings('default', 'Geometry is in a geographic CRS')
    compact_q = (compactness * 10).round().fillna(0).astype(int)

    # Create hash inputs
    hash_inputs = (
        minx_q.astype(str)
        + ','
        + miny_q.astype(str)
        + ','
        + maxx_q.astype(str)
        + ','
        + maxy_q.astype(str)
        + ','
        + area_q.astype(str)
        + ','
        + compact_q.astype(str)
    )

    # Generate hash
    geo_ids = hash_inputs.apply(
        lambda s: hashlib.sha256(s.encode()).hexdigest()[:hash_length]
    )

    # Check for duplicates
    duplicates = geo_ids.duplicated(keep=False)

    geo_ids.loc[gdf['geometry'].is_empty] = 'no-geometry'

    if duplicates.any():
        n_dupl = duplicates.sum()
        if verbose:
            print(
                f"Warning: {n_dupl} polygons with duplicate GIDs "
                f"({n_dupl/len(geo_ids)*100:.2g}%)"
            )

            # Show some examples
            print('\nExample duplicates:')
            dup_examples = (
                geo_ids[duplicates].sort_values().head(min(10, duplicates.sum())).index
            )
            print(
                pd.concat(
                    [
                        geo_ids[dup_examples].rename('geo_id'),
                        hash_inputs[dup_examples].rename('hash_inputs'),
                        gdf.loc[dup_examples],
                    ],
                    axis=1,
                )
            )

        if handle_duplicates:
            # Handle collisions with suffix
            if verbose:
                print('Adding suffixes...')

            geo_ids.loc[duplicates] = add_unique_suffix(geo_ids[duplicates])

    return geo_ids.rename('geo_id')


def add_geo_id_index(gdf, handle_duplicates=True, verbose=False):
    """Return the GeoDataFrame using `geo_id` as the index

    Parameters
    ----------
    gdf : GeoDataFrame
        Polygon data
    handle_duplicates : bool
        If True, adds numeric suffix to duplicate GIDs (default True)
    verbose: bool
        If True, prints information on duplicates
    """

    gdf = gdf.copy()
    gdf.index = pd.Index(
        get_geo_ids(gdf, handle_duplicates=handle_duplicates, verbose=verbose),
        name='geo_id',
    )
    if gdf.index.duplicated().any():
        raise ValueError(
            'Unhandled duplicates found in `geo_id` index. '
            'Set `handle_duplicates=True` or pick a different indexing method.'
        )
    return gdf


def overlay_admin_ids(
    gdf, admin_id, admin_level=2, admin_recipe=None, include_overlays=False, timer=None
):
    """Add administrative unit IDs to GeoDataFrame using spatial joins

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame to join admin IDs to
    admin_id : str
        Administrative unit of the GeoDataFrame.
        Determines which administrative units to consider.
    admin_level : int
        Administrative level for which administrative IDs are to be
        joined. Typically a lower level (larger number) than the level
        of `admin_id`.
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

    admin = get_admin_by_level(admin_level, admin_id, recipe=admin_recipe, geom=True)

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
        mask = gdf['admin2_id'].isnull()
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
