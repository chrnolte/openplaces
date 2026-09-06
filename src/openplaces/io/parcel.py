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
    'postal_code',
    'school_district_id',
    'school_district_name',
    'source_date',
    'tax_year',
]
# Inward buffer used by the sliver test, in meters. Converted to the
# layer's own units before it is applied, because the same distance
# is 15 meters in a projected CRS and 15 degrees in a geographic one.
# The value dates from a geographic-CRS default of 0.00015 degrees,
# which is this distance at the equator.
THINNESS_TEST_BUFFER_METERS = 16.7
THINNESS_TEST_AREA_RATIO_MIN = 0.01

# Meters per degree of latitude, for converting the buffer when the
# layer is in a geographic CRS. A degree of longitude is shorter away
# from the equator, so the buffer this gives is a conservative
# (slightly small) one outside the tropics.
_METERS_PER_DEGREE = 111_320.0

PARCEL_N_VERTICES_THRESHOLD = 100
THINNESS_THRESHOLD = 500


def _buffer_in_layer_units(parcels, buffer_meters):
    """Convert a buffer in meters to the units of a layer's own CRS.

    Shapely buffers in whatever units the coordinates are in, so one
    constant applied to a projected and a geographic copy of the same
    layer buffers by two distances that differ by five orders of
    magnitude. Identical parcels then deduplicated differently depending
    on how the source happened to be projected.

    Parameters
    ----------
    parcels : geopandas.GeoDataFrame
        The layer the buffer will be applied to. Its `crs` decides the
        conversion.
    buffer_meters : float
        The buffer distance in meters.

    Returns
    -------
    float
        The same distance expressed in the layer's coordinate units.

    Raises
    ------
    ValueError
        When the layer declares no CRS, since the units are then unknown
        and any conversion would be a guess.
    """
    crs = getattr(parcels, 'crs', None)
    if crs is None:
        raise ValueError(
            'parcels must declare a CRS: the sliver buffer is given in '
            'meters and cannot be converted to unknown coordinate units.'
        )
    if crs.is_geographic:
        return buffer_meters / _METERS_PER_DEGREE
    # A projected CRS records how many meters its own unit is worth, so
    # feet-based state plane layers convert as readily as metric ones.
    unit_meters = crs.axis_info[0].unit_conversion_factor
    return buffer_meters / unit_meters


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
    thinness_test_buffer_meters=THINNESS_TEST_BUFFER_METERS,
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

    Parameters
    ----------
    parcels : geopandas.GeoDataFrame
        The parcels to test. Its active geometry column may carry any
        name; `parcels.crs` decides the units of the sliver buffer.
    attribute_columns : sequence of str, optional
        Columns counted by the empty-attribute test. Defaults to every
        column but the geometry.
    parcel_id_col : str, default 'parcel_id_admin3'
        Column holding the parcel identifier the ID tests read.
    parcel_n_vertices_threshold : int, optional
        Vertex count above which a parcel becomes a candidate.
    thinness_threshold : float, optional
        Perimeter squared over area above which a parcel becomes a
        candidate. A zero-area geometry is always one.
    parcel_id_blacklisted_endings, parcel_id_whitelisted_endings : list of str
        Identifier endings that respectively mark a parcel as a filler
        and protect it from that reading.
    empty_attributes_ignore_columns : list of str, optional
        Columns the empty-attribute test does not count.
    empty_attributes_fraction_max : float, optional
        Share of empty attributes above which the second test fails.
    thinness_test_buffer_meters : float, optional
        Inward buffer of the sliver test, **in meters**. Converted to
        the layer's own units from `parcels.crs`, so a projected and a
        geographic copy of one layer deduplicate alike.
    thinness_test_area_ratio_min : float, optional
        Share of the original area the buffered polygon must keep to
        escape being read as a sliver.

    Returns
    -------
    geopandas.GeoDataFrame
        The input without the parcels that failed at least two tests.
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
    # Identify parcels that are candidate for inspection: many vertices,
    # extremely thin (slivers, road fillers), or identified as water.
    # The filters are scoped: a caller who silenced these warnings for
    # its own run keeps that choice once this returns.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Geometry is in a geographic CRS')
        areas = parcels.area.values
        # A collapsed polygon has no area, so perimeter^2/area is 0/0.
        # Left as NaN it compared False and the parcel survived the test
        # it most obviously fails; infinity makes it a candidate without
        # raising a divide warning to suppress.
        thinness = np.full(len(areas), np.inf, dtype=float)
        np.divide(
            np.square(parcels.length.values),
            areas,
            out=thinness,
            where=areas > 0,
        )
        mask = (
            (get_num_coordinates(parcels.geometry.values) > parcel_n_vertices_threshold)
            | (thinness > thinness_threshold)
            | has_blacklisted_parcel_id.values
        )

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

    # Identify parcels lacking most attribute data. The active geometry
    # column is not always called 'geometry' (county layers ship 'geom'
    # and 'SHAPE'), so its name is read off the frame.
    geometry_column = parcels.geometry.name
    if attribute_columns is None:
        attribute_columns = parcels.columns.drop(geometry_column)
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

    buffer_distance = _buffer_in_layer_units(parcels, thinness_test_buffer_meters)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Geometry is in a geographic CRS')

        parcel_test_geometry = parcels.loc[mask, geometry_column][mask_thinness_test]
        thinness_test_area_ratio = parcel_test_geometry.buffer(
            -buffer_distance
        ).area.div(parcel_test_geometry.area)

    is_a_sliver = thinness_test_area_ratio.lt(thinness_test_area_ratio_min).rename(
        'is_a_sliver'
    )

    # The three tests are combined as booleans. This used to set
    # 'future.no_silent_downcasting' to keep fillna from downcasting the
    # result, but that option is process-wide and outlived the call; it
    # is now deprecated, its behavior being the default, so the fill is
    # made explicit here instead.
    three_tests = (
        pd.concat(
            [has_questionable_parcel_ids, lacks_most_attribute_data, is_a_sliver],
            axis=1,
        )
        .fillna(False)
        .astype(bool)
    )

    # Keep only parcels for which at least two of the tests fail.
    drop_parcel = three_tests.sum(1).ge(2)
    return parcels.drop(drop_parcel[drop_parcel].index)
