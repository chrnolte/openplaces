"""
Data ingestion functions specific to parcel data
"""

import warnings

import numpy as np
import pandas as pd
from shapely import get_num_coordinates

# Parcel ID endings signaling that this is not a normal property
# (currently derived from Wisconsin state-level parcel data,
# not tested elsewhere)
PARCEL_ID_BLACKLISTED_ENDINGS = [
    # Filler identifiers
    'row',
    'gap',
    'overlap',
    # Transportation networks
    'rail',
    'railroad',
    'road',
    'rd',
    'street',
    'alley',
    'trail',
    # Water
    'hydro',
    'water',
    'river',
    'creek',
    'lake',
    'lake bed',
    'lakebed',
    'pond',
    'bog',
    'springs',
    'spring',
]

PARCEL_ID_WHITELISTED_ENDINGS = [
    'tribal',
    'new parcel',
    'condo',
    'common element',
    'common area',
    'right of way',
]

# Maximum of empty attributes before the parcel fails test #2/3
EMPTY_ATTRIBUTES_FRACTION_MAX = 0.8

# Columns that can be automatically filled based on location or update
# and therefore don't add much value to an attribute table.
EMPTY_ATTRIBUTES_IGNORE_COLUMNS = [
    'city',
    'zip_code',
    'school_district_id',
    'school_district_name',
    'source_date',
    'tax_year',
]
THINNESS_TEST_BUFFER_DEGREES = 0.00015
THINNESS_TEST_AREA_RATIO_MIN = 0.01

PARCEL_N_VERTICES_THRESHOLD = 100
THINNESS_THRESHOLD = 500


def drop_problematic_parcels(
    parcels,
    attribute_columns=None,
    parcel_id_col='parcel_id_admin3',
    parcel_n_vertices_threshold=PARCEL_N_VERTICES_THRESHOLD,
    thinness_threshold=THINNESS_THRESHOLD,
    parcel_id_blacklisted_endings=PARCEL_ID_BLACKLISTED_ENDINGS,
    parcel_id_whitelisted_endings=PARCEL_ID_WHITELISTED_ENDINGS,
    empty_attributes_ignore_columns=EMPTY_ATTRIBUTES_IGNORE_COLUMNS,
    empty_attributes_fraction_max=EMPTY_ATTRIBUTES_FRACTION_MAX,
    thinness_test_buffer_degrees=THINNESS_TEST_BUFFER_DEGREES,
    thinness_test_area_ratio_min=THINNESS_TEST_AREA_RATIO_MIN,
):
    """Drop parcels that complicate processing without adding much value

    This is an experimental function mostly designed to remove fill-in
    parcels (e.g. roads, lakes) that don't offer valuable attribute data

    First, the algorithm identifies parcels that might be unwanted:
    - Parcels with IDs that point to gap fillers, roads, water
      (detection derived from Wisconsin)
    - Parcels with a large number of vertices (water features, fill-ins)
    - Parcels with a high perimeter^2-to-area ratio (slivers)

    Then, the algorithm drops parcels that meet at least 2/3 criteria:
    - Parcels whose ID seems questionable
      (empty, blacklisted, text-only duplicates)
    - Their attributes are mostly empty
    - They are only slivers (internal buffering creates empty polygon)
    """

    # Identify parcels with shady parcel identifier
    # parcel_id_water_regex = '(^|.* )(' + '|'.join(parcel_id_water_endings) + ')$'
    parcel_id_blacklisted_regex = (
        '(^|.* )(' + '|'.join(parcel_id_blacklisted_endings) + ')$'
    )
    has_blacklisted_parcel_id = (
        parcels[parcel_id_col]
        .fillna('')
        .str.lower()
        .str.match(parcel_id_blacklisted_regex)
    )
    # Identify parcels that are candidate for inspection: many vertices, extremely thin
    # (slivers, road fillers), or identified as water
    warnings.filterwarnings('ignore', 'Geometry is in a geographic CRS')
    warnings.filterwarnings('ignore', 'invalid value encountered in divide')
    mask = (
        (get_num_coordinates(parcels.geometry.values) > parcel_n_vertices_threshold)
        | (
            (np.square(parcels.length.values) / parcels.area.values)
            > thinness_threshold
        )
        | has_blacklisted_parcel_id.values
    )
    warnings.filterwarnings('default', 'invalid value encountered in divide')
    warnings.filterwarnings('default', 'Geometry is in a geographic CRS')

    parcel_id_whitelisted_regex = (
        '(^|.* )(' + '|'.join(parcel_id_whitelisted_endings) + ')$'
    )
    parcel_ids = parcels.loc[mask, parcel_id_col]
    parcel_ids_filled = parcel_ids.fillna('').str.lower()

    has_questionable_parcel_ids = pd.Series(
        parcel_ids.isnull()
        | has_blacklisted_parcel_id.loc[mask]
        | parcel_ids_filled.str.match('(no|needs) ?(id|pin)')
        | (
            parcel_ids_filled.str.match('^[^0-9]+$')
            & parcels[parcel_id_col].duplicated(keep=False).loc[mask]
            & ~parcel_ids_filled.str.match(parcel_id_whitelisted_regex)
        ),
        index=parcel_ids.index,
        name='has_questionable_parcel_ids',
    )

    # Identify parcels lacking most attribute data
    if attribute_columns is None:
        attribute_columns = parcels.columns.drop('geometry')
    columns_with_attribute_data = [
        v
        for v in sorted(set(attribute_columns) - set([parcel_id_col]))
        if not v.startswith('admin')
        and not v.startswith('source')
        and v not in empty_attributes_ignore_columns
    ]
    lacks_most_attribute_data = pd.Series(
        parcels.loc[mask, columns_with_attribute_data]
        .isnull()
        .mean(1)
        .gt(empty_attributes_fraction_max),
        index=parcel_ids.index,
        name='lacks_most_attribute_data',
    )

    # For the most expensive operation (buffering), only pick polygons for
    # which only one of the prior tests was positive (2 of 3 are needed)
    mask_thinness_test = has_questionable_parcel_ids ^ lacks_most_attribute_data

    warnings.filterwarnings('ignore', 'Geometry is in a geographic CRS')

    parcel_test_geometry = parcels.loc[mask, 'geometry'][mask_thinness_test]
    thinness_test_area_ratio = parcel_test_geometry.buffer(
        -thinness_test_buffer_degrees
    ).area.div(parcel_test_geometry.area)

    is_a_sliver = thinness_test_area_ratio.lt(thinness_test_area_ratio_min).rename(
        'is_a_sliver'
    )
    warnings.filterwarnings('default', 'Geometry is in a geographic CRS')

    pd.set_option('future.no_silent_downcasting', True)
    three_tests = pd.concat(
        [has_questionable_parcel_ids, lacks_most_attribute_data, is_a_sliver], axis=1
    ).fillna(False)

    # Keep only parcels for which at least two of the tests fail.
    drop_parcel = three_tests.sum(1).ge(2)
    return parcels.drop(drop_parcel[drop_parcel].index)
