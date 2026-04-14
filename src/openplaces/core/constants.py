import numpy as np
from pyproj import CRS

# DIRECTORY STRUCTURE

# Escape directory (to signal end of administrative folder depth)

ESCAPE_DIR = '_all'

# Standard data directories for openplaces projects.
# Each entry maps a short name (used in config and recipes) to its metadata.
# - default: path relative to data_root (or None for the root itself)
# - description: human-readable label shown during interactive setup
# - shared: if True, the directory is shared across users in multi-user mode

STANDARD_DIRS = {
    'data_root': {
        'default': None,
        'description': 'Root directory for data, models, reports. None = package root',
        'shared': True,
    },
    'core': {
        'default': 'data/core',
        'description': 'Processed, standardized, analysis-ready data',
        'shared': False,
    },
    'external': {
        'default': 'data/external',
        'description': 'Downloaded data from third party sources',
        'shared': True,
    },
    'raw': {
        'default': 'data/raw',
        'description': 'Raw data from own data collection efforts',
        'shared': True,
    },
    'cache': {
        'default': 'data/cache',
        'description': 'Interim data, can be safely deleted or regenerated',
        'shared': False,
    },
    'heap': {
        'default': 'data/cache/_heap',
        'description': 'Freshly unzipped data, to be deleted after use',
        'shared': False,
    },
    'logs': {
        'default': 'data/cache/_logs',
        'description': 'Logs from script runs for performance profiling',
        'shared': False,
    },
    'out': {
        'default': 'data/out',
        'description': 'Output and results data',
        'shared': False,
    },
    'share': {
        'default': 'data/share',
        'description': 'Shared data between users',
        'shared': True,
    },
    'models': {
        'default': 'models',
        'description': 'Trained and serialized models',
        'shared': False,
    },
    'reports': {
        'default': 'reports',
        'description': 'Reports, publications, figures',
        'shared': False,
    },
}


# FILE HANDLING

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

# Tabular data file extensions read with `pandas` into a `DataFrame`
PANDAS_EXTENSIONS = {
    '.csv',
    '.xlsx',
    '.xls',
    '.dat',
}

# Extensions of companion files for shapefiles
SHAPEFILE_EXTENSIONS = [
    '.cpg',
    '.dbf',
    '.GISJOIN.atx',  # Found in US_admin-nhgis-2020
    '.prj',
    '.qpj',
    '.shp',
    '.shp.xml',
    '.shx',
    '.sbn',
    '.sbx',
]

# Extensions for compressed files
ZIP_EXTENSIONS = {'.zip', '.jar', '.kmz', '.bz2', '.tbz2'}


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

# Schema-related strings

STRING_SEPARATOR_WITHIN_IDS = '-'

STRING_SEPARATOR_BETWEEN_IDS = '_'

# Regex patterns
# Admin1: ISO and HASC codes
REGEX_ADMIN2_IDS_AA_AA = '^[A-Z]{2}\\-[A-Z]{2}$'
REGEX_ADMIN2_IDS_AA_AA_EXTRACT = '^([A-Z]{2})\\-([A-Z]{2})$'
REGEX_ADMIN2_IDS_HASC = '^[A-Z]{2}\\.[A-Z]{2}$'
REGEX_ADMIN3_IDS_HASC = '^[A-Z]{2}\\.[A-Z]{2}\\.[A-Z]{2}$'

# Extract filename from URL
REGEX_FILENAME_IN_URL = r'/([^/?]+\.[a-zA-Z0-9]+)(?:\?|$)'

# Check whether a filepath has wildcards to search the filesystem
REGEX_HAS_GLOB_WILDCARDS = r'[*?\[\]]'

# Admin IDs of countries (Admin-1) using HASC1 code for Admin-2 level
ADMIN1_IDS_USING_HASC1_FOR_ADMIN2 = ['AZ', 'BE', 'FR', 'GB', 'GN', 'IT', 'LV']


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


# RECIPE

# Keys in an ingestion recipe that are specific to a single table (layer).
# Used by build_table_recipe() to determine which keys are taken
# from an additional_layers entry and which are inherited from the primary.
RECIPE_PER_TABLE_KEYS = (
    'layer',
    'columns',
    'keep_unnamed_columns',
    'set_index',
    'create_index',
    'index_function',
    'drop',
    'query',
    'null_value_strings',
    'transformations',
    'columns_to_categorical',
    'encoding',
    'save_to',
    'overlay_admin_ids',
)


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
