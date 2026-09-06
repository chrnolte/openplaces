"""`drop_problematic_parcels` must read the same way whatever it is handed.

The function drops a parcel only when two of its three tests fail, so a
test that quietly never fires does not announce itself: it just leaves
fillers in the data. These pin the three ways that happened, plus the two
pieces of process-wide state the function used to change on its way
through.
"""

import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from openplaces.io.parcel import drop_problematic_parcels

# UTM 17N, meters. Every fixture is built here and reprojected from here,
# so the geographic and the projected frame hold the same parcels.
PROJECTED = 'EPSG:32617'

_X, _Y = 500_000.0, 4_000_000.0


def _rect(x0, y0, width, height):
    return Polygon(
        [
            (_X + x0, _Y + y0),
            (_X + x0 + width, _Y + y0),
            (_X + x0 + width, _Y + y0 + height),
            (_X + x0, _Y + y0 + height),
        ]
    )


def _frame(crs=PROJECTED):
    """A road-filler sliver beside an ordinary parcel."""
    frame = gpd.GeoDataFrame(
        {
            'parcel_id_admin3': [None, '123-45-678'],
            'land_value': [10_000, 20_000],
            'building_value': [0, 150_000],
            'year_built': [None, 1974],
            'geometry': [_rect(0, 0, 200, 0.5), _rect(0, 100, 100, 100)],
        },
        crs=PROJECTED,
    )
    return frame.to_crs(crs)


def test_the_sliver_test_reads_the_same_in_projected_and_geographic_crs():
    """The buffer is a distance, not a number of coordinate units.

    One constant applied in both CRSs buffered by 0.00015 meters in UTM
    and by 0.00015 degrees (about 17 meters) in EPSG:4326, so the third
    test was inert on any projected assessor layer, which is most of
    them.
    """
    projected = drop_problematic_parcels(_frame(PROJECTED))
    geographic = drop_problematic_parcels(_frame('EPSG:4326'))

    assert len(projected) == 1
    assert len(geographic) == 1
    assert projected['parcel_id_admin3'].tolist() == ['123-45-678']
    assert geographic['parcel_id_admin3'].tolist() == ['123-45-678']


def test_a_layer_without_a_crs_says_so_rather_than_guessing():
    frame = _frame().set_crs(None, allow_override=True)

    with pytest.raises(ValueError, match='must declare a CRS'):
        drop_problematic_parcels(frame)


def test_a_collapsed_polygon_is_a_candidate_rather_than_a_survivor():
    """Zero area made perimeter squared over area a 0/0 NaN.

    NaN compared False, so the geometry that most obviously fails the
    thinness test was never even inspected, and a filler with no id and
    no attributes came through untouched.
    """
    collapsed = Polygon([(_X, _Y), (_X + 10, _Y), (_X + 10, _Y), (_X, _Y)])
    frame = gpd.GeoDataFrame(
        {
            'parcel_id_admin3': [None, '123-45-678'],
            'land_value': [None, 20_000],
            'building_value': [None, 150_000],
            'year_built': [None, 1974],
            'geometry': [collapsed, _rect(0, 100, 100, 100)],
        },
        crs=PROJECTED,
    )

    with warnings.catch_warnings():
        # A divide warning here would mean the zero area is still being
        # divided by rather than handled.
        warnings.simplefilter('error', RuntimeWarning)
        kept = drop_problematic_parcels(frame)

    assert kept['parcel_id_admin3'].tolist() == ['123-45-678']


def test_the_geometry_column_may_carry_any_name():
    """County layers ship 'geom' and 'SHAPE', not only 'geometry'."""
    frame = _frame().rename_geometry('geom')

    kept = drop_problematic_parcels(frame)

    assert kept['parcel_id_admin3'].tolist() == ['123-45-678']
    assert kept.geometry.name == 'geom'


def test_the_callers_pandas_options_survive_the_call():
    """The downcasting option was set outright and never put back."""
    previous = pd.get_option('future.no_silent_downcasting')
    pd.set_option('future.no_silent_downcasting', False)
    try:
        drop_problematic_parcels(_frame())

        assert pd.get_option('future.no_silent_downcasting') is False
    finally:
        pd.set_option('future.no_silent_downcasting', previous)


def test_the_callers_warning_filters_survive_the_call():
    """Resetting a filter to 'default' overrode a caller's own ignore."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Geometry is in a geographic CRS')
        before = list(warnings.filters)

        drop_problematic_parcels(_frame())

        assert list(warnings.filters) == before


def test_a_state_plane_layer_in_feet_buffers_by_the_same_distance():
    """A projected CRS need not be metric; the conversion reads its unit."""
    feet = _frame().to_crs('EPSG:2264')  # NC state plane, US survey feet

    kept = drop_problematic_parcels(feet)

    assert kept['parcel_id_admin3'].tolist() == ['123-45-678']
    assert np.isfinite(kept.area.values).all()
