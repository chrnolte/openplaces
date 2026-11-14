"""Constants shared by the link modules."""

from __future__ import annotations

from openplaces.geo.ids import PARCEL_ID_ALNUM_KEYS

# The key link_by_id joins on unless a recipe names another. Auto-discovery
# resolves its own key per match; anything else here is a caller override.
DEFAULT_LINK_KEY = 'parcel_id_local'

# Spine columns whose values are parcel_id_local keys, whatever the
# column is called. `parcel_link_key` (io/harmonizer/parcel_link_keys.py)
# holds the same key for every row that does not name a stacked unit, so
# a guard that recognizes only the literal default name stops guarding a
# spine that has moved onto it.
PARCEL_ID_LOCAL_KEYS = (DEFAULT_LINK_KEY, 'parcel_link_key')

# Matching keys an auto-discovered link never copies onto the spine by
# default (see link_by_id's auto_discover branch).
_LINK_KEY_COLUMNS = frozenset({DEFAULT_LINK_KEY, *PARCEL_ID_ALNUM_KEYS})

# Columns carried in spine-reference crosswalk tables (index levels excluded).
_CROSSWALK_COLS = [
    'area_intersection_m2',
    'iou',
    'area_intersection_m2_inner',
    'fraction_of_largest',
]

# Parquet footer key holding a link sidecar's validity fingerprint.
_LINK_METADATA_KEY = 'openplaces:link'
_LINK_INDEX_KEY = 'openplaces:link_index'
