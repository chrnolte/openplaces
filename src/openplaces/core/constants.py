# DIRECTORY STRUCTURE

# Escape directory (to signal end of administrative folder depth)

ESCAPE_DIR = '_all'

# Standard data directories for openplaces projects.
# Each entry maps a short name (used in config and recipes) to its metadata.
# - default: path relative to data_root (or None for the root itself)
# - description: human-readable label shown during interactive setup
# - shared: if True, the directory is shared across users in multi-user mode
# - retention: default lifecycle class for files in the directory (one of
#   RETENTION_CLASSES); overridable per bucket in openplaces.yaml and per
#   recipe via save_to.retention, except for NEVER_DELETE buckets

STANDARD_DIRS = {
    'data_root': {
        'default': None,
        'description': 'Root directory for data, models, reports. None = package root',
        'shared': True,
        'retention': 'keep',
    },
    'core': {
        'default': 'data/core',
        'description': 'Processed, standardized, analysis-ready data',
        'shared': False,
        'retention': 'keep',
    },
    'external': {
        'default': 'data/external',
        'description': 'Downloaded data from third party sources',
        'shared': True,
        'retention': 'keep',
    },
    'raw': {
        'default': 'data/raw',
        'description': 'Raw data from own data collection efforts',
        'shared': True,
        'retention': 'keep',
    },
    'cache': {
        'default': 'data/cache',
        'description': 'Interim data, can be safely deleted or regenerated',
        'shared': False,
        'retention': 'until_consumed',
    },
    'heap': {
        'default': 'data/cache/_heap',
        'description': 'Freshly unzipped data, to be deleted after use',
        'shared': False,
        'retention': 'transient',
    },
    'logs': {
        'default': 'data/cache/_logs',
        'description': 'Logs from script runs for performance profiling',
        'shared': False,
        'retention': 'keep',
    },
    'out': {
        'default': 'data/out',
        'description': 'Output and results data',
        'shared': False,
        'retention': 'keep',
    },
    'share': {
        'default': 'data/share',
        'description': 'Shared data between users',
        'shared': True,
        'retention': 'keep',
    },
    'models': {
        'default': 'models',
        'description': 'Trained and serialized models',
        'shared': False,
        'retention': 'keep',
    },
    'reports': {
        'default': 'reports',
        'description': 'Reports, publications, figures',
        'shared': False,
        'retention': 'keep',
    },
}


# DATA LIFECYCLE

# Lifecycle classes assignable to buckets (STANDARD_DIRS) and recipe outputs:
# - keep: never auto-deleted; reported by compact(), removed only manually or
#   by reprocess
# - until_consumed: deletable once ALL consuming recipes' outputs exist and
#   are complete; deletion leaves a tombstone receipt
# - transient: deleted at the end of the producing/consuming stage in the
#   same process (e.g. heap)
RETENTION_CLASSES = ('keep', 'until_consumed', 'transient')

# Hard floor enforced in code below the config layer: no configuration can
# mark these buckets deletable.
NEVER_DELETE = frozenset({'share', 'raw'})


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
    '.txt',
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
ZIP_EXTENSIONS = {
    '.zip',
    '.jar',
    '.kmz',
    '.bz2',
    '.tbz2',
    '.tgz',
    '.tar.gz',
    '.tar.bz2',
}


# STRING

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


# Vocabulary for cleaning administrative-unit names before generating admin IDs.
# These are language-extensible: add a language's terms here (or override per
# recipe) rather than editing `openplaces.io.admin.clean_geographic_name`.

# Tokens that mean "no name" (case-insensitive, after stripping).
ADMIN_NA_TOKENS = frozenset({'', 'none', 'nan', 'null', 'na', 'n.a.', 'n/a'})

# Leading articles/honorifics stripped from a name so initials come from the
# distinctive word (e.g. "San Jose" -> "Jose", "El Paso" -> "Paso").
# English + Spanish + Arabic. Order does not matter (joined into a regex).
ADMIN_NAME_PREFIXES = [
    'The',  # English
    'Al',  # Arabic
    'San',
    'Santa',
    'Santo',  # Spanish honorifics
    'El',
    'La',
    'Los',
    'Las',  # Spanish articles
]

# Generic administrative words detected (with an accompanying number) so the
# number is prefixed with the word's initial(s) instead of the place name
# (e.g. "Ward 3" -> "W3", "Comuna 5" -> "C5"). English/Filipino + Spanish.
ADMIN_GENERIC_WORDS = [
    'ward',
    'zone',
    'barangay',
    'bgy',
    'district',
    'division',
    'subd',
    'subdivision',  # English / Filipino
    'municipio',
    'comuna',
    'corregimiento',
    'vereda',
    'barrio',
    'localidad',
    'departamento',
    'provincia',  # Spanish
]

# Trailing administrative-type words extracted from a unit's long name into a
# `type` column (and used to disambiguate duplicate names within a parent).
# English (US Census) + Spanish.
REGEX_ADMIN_TYPE_EXTRACT = (
    '(Census Area|Borough|Parish|City|Town|Village|County|Municipality|'
    'Municipio|Comuna|Corregimiento|Department|Departamento|Province|Provincia)$'
)


# Essential unit conversions

# Acres to square feet
AC_TO_SQFT = 43460

# Acres to hectares
AC_TO_HA = 0.404686

# Square meters to square feet
M2_TO_SQFT = 10.7639


# Default coordinate reference system
CRS = 'epsg:4326'  # WGS84 Geographic


# RECIPE

# Keys in an ingestion recipe that are specific to a single table (layer).
# Used by build_table_recipe() to determine which keys are taken
# from an additional_layers entry and which are inherited from the primary.
RECIPE_PER_TABLE_KEYS = (
    'layer',
    'layer_key',
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
