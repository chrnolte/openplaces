"""Building-point siblings in classify_footprint_priority.

When a complex's address points all land on one building, its other
buildings with a multi-dwelling building point stay primary.
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from openplaces.core.schema import SourceGeometryType
from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer.attributes import classify_footprint_priority

PARCELS = 'US_parcel-test-2026'
DWELLINGS = 'dwelling-test-2026'
BUILDINGS = 'US_building-test-2026'


def _state(areas_m2, dwelling_ids, building_dwellings):
    ids = list(areas_m2)
    spine = gpd.GeoDataFrame(
        {
            'area_ha': [areas_m2[i] / 10_000 for i in ids],
            'geometry_source': ['obm'] * len(ids),
        },
        geometry=[box(0, 0, 1, 1)] * len(ids),
        index=pd.Index(ids, name='footprint_id'),
    )
    parcel_cw = pd.DataFrame({'footprint_id': ids, 'parcel_id': ['p'] * len(ids)})
    state = HarmonizeState(recipe={}, admin_id=None, verbose=False, timer=None)
    state.spine = spine
    state.crosswalks = {
        PARCELS: parcel_cw.set_index(['footprint_id', 'parcel_id']),
        DWELLINGS: pd.DataFrame({'footprint_id': list(dwelling_ids)}),
        BUILDINGS: pd.DataFrame(
            {
                'footprint_id': list(building_dwellings),
                'n_dwellings': list(building_dwellings.values()),
            }
        ),
    }
    state.reference_types = {PARCELS: 'parcel'}
    state.source_geometry_types = {
        DWELLINGS: SourceGeometryType.single_dwelling_point,
        BUILDINGS: SourceGeometryType.single_building_point,
    }
    return state


def _roles(state, **thresholds):
    out = classify_footprint_priority(
        state, entity_type='parcel', thresholds=thresholds or None
    )
    return out.spine['priority_on_parcel'].astype(str).to_dict()


def test_equal_apartment_buildings_stay_primary():
    state = _state(
        {'a': 1000, 'b': 1000, 'c': 1000, 'shed': 60},
        dwelling_ids={'a'},
        building_dwellings={'a': 80, 'b': 80, 'c': 70, 'shed': 1},
    )
    assert _roles(state, building_point_sibling_ratio=0.5) == {
        'a': 'primary',
        'b': 'primary',
        'c': 'primary',
        'shed': 'secondary',
    }


def test_off_by_default():
    state = _state(
        {'a': 1000, 'b': 1000},
        dwelling_ids={'a'},
        building_dwellings={'b': 80},
    )
    assert _roles(state) == {'a': 'primary', 'b': 'secondary'}


def test_single_dwelling_point_or_small_outline_is_not_promoted():
    state = _state(
        {'a': 1000, 'garage': 600, 'annex': 300},
        dwelling_ids={'a'},
        building_dwellings={'garage': 1, 'annex': 20},
    )
    roles = _roles(state, building_point_sibling_ratio=0.5)
    assert roles == {'a': 'primary', 'garage': 'secondary', 'annex': 'secondary'}


def test_below_ratio_is_not_promoted():
    state = _state(
        {'a': 2000, 'b': 800},
        dwelling_ids={'a'},
        building_dwellings={'b': 30},
    )
    roles = _roles(state, building_point_sibling_ratio=0.5)
    assert roles == {'a': 'primary', 'b': 'secondary'}
