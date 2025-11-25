import numpy as np
from pyproj import CRS


# DIRECTORY STRUCTURE

# Escape directory (to signal end of administrative folder depth)

ESCAPE_DIR = '_all'


# FILE HANDLING

# Extensions for compressed files (can be uncompressed with `zipfile`)
ZIP_EXTENSIONS = {'.zip', '.jar', '.kmz'}


# STRING

# String separators

# Standard separator for string aggregations (concatenations)
# This string must be unique enough to not occur in any interpreted
# string values in the database (including identifiers used by data
# contributors, e.g. for parcel numbers), so the aggregation can be
# reversed (string splitting) without loss of information. It should
# ideally also be free of special characters used in regular expressions
# (some string splitting functions interpret the separator as a regular
# expression).

STRING_SEPARATOR_AGGREGATION = ' ~~ '

# Schema-related stings

STRING_SEPARATOR_WITHIN_IDS = '-'

STRING_SEPARATOR_BETWEEN_IDS = '_'

# Regex patterns
# Admin1: ISO and HASC codes
RE_ADMIN1_IDS_AA_AA = '^[A-Z]{2}\\-[A-Z]{2}$'
RE_ADMIN1_IDS_AA_AA_EXTRACT = '^([A-Z]{2})\\-([A-Z]{2})$'
RE_ADMIN1_IDS_HASC = '^[A-Z]{2}\\.[A-Z]{2}$'
RE_ADMIN2_IDS_HASC = '^[A-Z]{2}\\.[A-Z]{2}\\.[A-Z]{2}$'

# Countries using HASC1 code for Admin-2 level
ADMIN0_ID_HASC1_A2 = ['AZ', 'BE', 'FR', 'GB', 'GN', 'IT', 'LV']

# Essential unit conversions

# Acres to square feet
AC_TO_SQFT = 43460

# Acres to hectares
AC_TO_HA = 0.404686

# Square meters to square feet
M2_TO_SQFT = 10.7639


# Coordinate Reference Systems

# Default coordinate reference system for latitudes and longitudes
CRS_LAT_LONG = 'epsg:4326'  # WGS84 Geographic

# Default coordinate reference system for area computations
CRS_AREA = 'epsg:6933'  # Equal Area Cylindrical (EPSG:6933)

# Default coordinate reference system for the computation of Poles of
# Inaccessibility (PoI) (good locations for labels)
CRS_POI = 'epsg:3395'  # World Mercator


# Raster processing

# Default raster configuration used for global raster snapping.
# (Currently inherited from Hansen)
RASTER_CONFIG = {
    'xmin': -180.00,
    'ymin': -60.00,
    'xmax': 180.00,
    'ymax': 80.00,
    'res': 0.00025,
    'crs': CRS('epsg:4326'),
}

# Default resolution for rasterized zonal statistics in meters
RASTER_ZONAL_STATISTICS_RESOLUTION_M = 15


# Vector processing

# Minimum area of polygons to be imported into `openplaces`
# Tiny parcels are rare and they're usually data processing errors.
# They are removed by default to avoid overloading the Pole of
# Inaccessibility (PoI) computations, which are approximate and fail
# when polygons get tiny.

GEO_MIN_AREA_M2 = 1


# Code length for geometry identifiers based on UBID (unique bldg. ID)
# Current default (11) is based on the observation that in several
# U.S. counties, the former default (10) created duplicate IDs for
# distinct parcels and buildings, whereas a length of 11 did not.
GEO_ID_UBID_CODELENGTH = 11


# Geohashing

# The following parameters configure the algorithm that creates globally
# unique IDs for non-overlapping parcels based on their coordinates
# (latitude and longitude) and parcel area.
#
# Changing any of the below parameters will break the link between
# `geo_id` values across versions and thus defeats the purpose of
# generating these IDs.

# Salt
GEO_ID_SALT = 'g1I9qtkKzxA3P98m80DLhuc0'

# Projection of lat/long coordinates
GEO_ID_LAT_LONG_CRS = 'epsg:4326'

# Precision of lat/long coordinates
GEO_ID_LAT_LONG_PRECISION = 1e-05

# Projection in which hectares is computed
GEO_ID_HECTARES_CRS = 'epsg:6933'

# Z transformation
GEO_ID_HECTARES_TRANSFORMATION = np.arcsinh

# Z upper boundary
GEO_ID_HECTARES_TRANSFORMATION_MAX = 20

# Z precision
GEO_ID_HECTARES_PRECISION = 0.002

# Projection in which Poles of Inaccessibility (PoI) will be computed
GEO_ID_POI_CRS = 'epsg:4326'

# Precision ratio for PoI used in geohashing
GEO_ID_POI_PRECISION_RATIO = 0.05


# Vector file extensions read with `geopandas` into a `GeoDataFrame`
GEOPANDAS_EXTENSIONS = {
    '.shp',
    '.geojson',
    '.gpkg',
    '.gdb',
    '.kml',
    '.kmz',
    '.gml',
    '.fgb',
    '.parquet',
    '.feather',
    '.gpx',
    '.tab',
    '.mif',
    '.dxf',
    '.sqlite',
    '.db',
}
