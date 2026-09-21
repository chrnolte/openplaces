"""The size override in classify_footprint_priority.

An address point on a small structure must not demote a much larger
footprint that holds a building point; everything else keeps the
dwelling-evidence rule.
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


def _state(areas_m2, parcel_of, dwelling_ids, building_ids):
    ids = list(areas_m2)
    spine = gpd.GeoDataFrame(
        {
            'area_ha': [areas_m2[i] / 10_000 for i in ids],
            'geometry_source': ['obm'] * len(ids),
        },
        geometry=[box(0, 0, 1, 1)] * len(ids),
        index=pd.Index(ids, name='footprint_id'),
    )
    parcel_cw = pd.DataFrame(
        {'footprint_id': ids, 'parcel_id': [parcel_of[i] for i in ids]}
    ).set_index(['footprint_id', 'parcel_id'])
    state = HarmonizeState(recipe={}, admin_id=None, verbose=False, timer=None)
    state.spine = spine
    state.crosswalks = {
        PARCELS: parcel_cw,
        DWELLINGS: pd.DataFrame({'footprint_id': list(dwelling_ids)}),
        BUILDINGS: pd.DataFrame({'footprint_id': list(building_ids)}),
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


def test_default_keeps_dwelling_rule():
    state = _state(
        {'shed': 40, 'house': 200},
        {'shed': 'p', 'house': 'p'},
        dwelling_ids={'shed'},
        building_ids={'house'},
    )
    assert _roles(state) == {'shed': 'primary', 'house': 'secondary'}


def test_large_building_point_footprint_stays_primary():
    state = _state(
        {'shed': 40, 'house': 200},
        {'shed': 'p', 'house': 'p'},
        dwelling_ids={'shed'},
        building_ids={'house'},
    )
    roles = _roles(state, dwelling_override_ratio=3.0)
    assert roles == {'shed': 'primary', 'house': 'primary'}


def test_below_ratio_is_not_promoted():
    state = _state(
        {'garage': 90, 'house': 200},
        {'garage': 'p', 'house': 'p'},
        dwelling_ids={'garage'},
        building_ids={'house'},
    )
    roles = _roles(state, dwelling_override_ratio=3.0)
    assert roles == {'garage': 'primary', 'house': 'secondary'}


def test_without_building_point_is_not_promoted():
    state = _state(
        {'house': 40, 'barn': 400},
        {'house': 'p', 'barn': 'p'},
        dwelling_ids={'house'},
        building_ids=set(),
    )
    roles = _roles(state, dwelling_override_ratio=3.0)
    assert roles == {'house': 'primary', 'barn': 'secondary'}


def test_building_point_requirement_can_be_dropped():
    state = _state(
        {'house': 40, 'barn': 400},
        {'house': 'p', 'barn': 'p'},
        dwelling_ids={'house'},
        building_ids=set(),
    )
    roles = _roles(
        state,
        dwelling_override_ratio=3.0,
        dwelling_override_requires_building_point=False,
    )
    assert roles == {'house': 'primary', 'barn': 'primary'}


def test_only_the_largest_is_promoted():
    areas = {'shed': 40, 'house': 300, 'shop': 150}
    state = _state(
        areas,
        dict.fromkeys(areas, 'p'),
        dwelling_ids={'shed'},
        building_ids={'house', 'shop'},
    )
    roles = _roles(state, dwelling_override_ratio=3.0)
    assert roles == {'shed': 'primary', 'house': 'primary', 'shop': 'secondary'}


def test_other_parcels_unaffected():
    areas = {'a_shed': 40, 'a_house': 200, 'b_house': 150, 'b_shed': 20}
    parcel_of = {'a_shed': 'a', 'a_house': 'a', 'b_house': 'b', 'b_shed': 'b'}
    state = _state(
        areas, parcel_of, dwelling_ids={'a_shed', 'b_house'}, building_ids={'a_house'}
    )
    roles = _roles(state, dwelling_override_ratio=3.0)
    assert roles == {
        'a_shed': 'primary',
        'a_house': 'primary',
        'b_house': 'primary',
        'b_shed': 'secondary',
    }
