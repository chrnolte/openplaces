"""classify_footprint_priority on buildings cut at parcel lines.

plans/buildings-from-parcel-pieces.md, step 3. The roles are decided
per parcel over the cut crosswalk, so a piece is primary or secondary
on its own parcel. The last tests are the transition review's finding
3.10: a house reaching 40% onto a neighbor's lot, whose dwelling point
sits on the neighbor's house. Every geometry, id and value is
fabricated (docs/5_contribute/no-personal-data.rst).
"""

from __future__ import annotations

import pandas as pd
import pytest
from shapely.geometry import Point, box

from openplaces.core.schema import SourceGeometryType
from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer.apportion import (
    apportion_reference_values,
    secondary_pairs_from,
)
from openplaces.io.harmonizer.attributes import classify_footprint_priority
from tests.io.harmonizer.test_parcel_pieces import _lots, _spine, _split, _values

PARCELS = 'US-XX_parcel-test-2026'
DWELLINGS = 'dwelling-test-2026'
BUILDINGS = 'US_building-test-2026'


def _cut(outlines, lots, points, values):
    spine, flat, stats = _split(_spine(**outlines), _lots(**lots), points, values)
    return spine, flat, stats


def _roles(spine, flat, dwelling_points, building_points):
    """Roles over the cut crosswalk, the point joins given by containment.

    *dwelling_points* and *building_points* are points; each is linked
    to the building whose polygon contains it, as the point joins'
    first pass does.
    """

    def linked(points):
        return [
            building
            for point in points
            for building, geom in spine.geometry.items()
            if geom.contains(point)
        ]

    crosswalk = flat[flat['link'].notna()].set_index(['building_id', 'parcel_id'])
    state = HarmonizeState(recipe={}, admin_id=None, verbose=False, timer=None)
    state.spine = spine.assign(area_ha=spine.geometry.area / 10_000)
    state.crosswalks = {
        PARCELS: crosswalk[['area_intersection_m2']],
        DWELLINGS: pd.DataFrame({'building_id': linked(dwelling_points)}),
        BUILDINGS: pd.DataFrame({'building_id': linked(building_points)}),
    }
    state.reference_types = {PARCELS: 'parcel'}
    state.source_geometry_types = {
        DWELLINGS: SourceGeometryType.single_dwelling_point,
        BUILDINGS: SourceGeometryType.single_building_point,
    }
    out = classify_footprint_priority(state, entity_type='parcel')
    roles = out.spine['priority_on_parcel'].astype(str).to_dict()
    return roles, crosswalk, out.spine['parcel_id_secondary']


def test_each_half_of_a_cut_duplex_is_primary_on_its_own_lot():
    """A duplex cut into two halves, a garage on the second lot: each
    half is its lot's primary building, the garage its lot's secondary
    one, and the first half is not demoted by the second lot."""
    outlines = {'d1': box(0, 0, 20, 10), 'g1': box(14, 14, 18, 18)}
    lots = {'L1': box(0, -5, 10, 20), 'L2': box(10, -6, 20, 20)}
    points = [Point(5, 5), Point(15, 5)]
    values = _values(L1=(90000.0, None), L2=(90000.0, None))
    spine, flat, stats = _cut(outlines, lots, points, values)
    assert stats['outlines_cut'] == 1
    roles, _, _ = _roles(spine, flat, points, [])
    halves = spine.index[spine['footprint_id'] == 'd1']
    assert {roles[b] for b in halves} == {'primary'}
    assert roles['g1'] == 'secondary'


def _house_reaching_onto_its_neighbor():
    """House h (10 m by 10 m) stands 60 m2 on its own lot A and 40 m2
    on lot B, where the neighbor's house n stands with the lot's
    dwelling point. h holds a building point of its own, on lot A."""
    outlines = {'h': box(0, 0, 10, 10), 'n': box(15, 0, 25, 10)}
    lots = {'A': box(-5, -5, 6, 15), 'B': box(6, -6, 30, 15)}
    dwelling = [Point(20, 5)]
    building = [Point(3, 5), Point(20, 5)]
    values = _values(A=(120000.0, None), B=(150000.0, None))
    spine, flat, stats = _cut(outlines, lots, dwelling + building, values)
    return spine, flat, stats, dwelling, building


def test_the_piece_on_the_neighbors_lot_is_refused():
    """Decision D2: the 40% piece holds no point of its own, so the
    outline stays whole and keeps both of its parcel pairs."""
    spine, flat, stats, _, _ = _house_reaching_onto_its_neighbor()
    assert stats['outlines_cut'] == 0
    assert stats['pieces_refused'] == 1
    assert sorted(spine.index) == ['h', 'n']
    kept = flat[flat['link'].notna()]
    assert sorted(kept.loc[kept['building_id'] == 'h', 'parcel_id']) == ['A', 'B']


def test_a_house_is_not_demoted_by_its_neighbors_dwelling_point():
    """Transition review 3.10: the house is primary on its own lot A,
    its dominant parcel, and secondary only on lot B."""
    spine, flat, _, dwelling, building = _house_reaching_onto_its_neighbor()
    roles, _, secondary = _roles(spine, flat, dwelling, building)
    assert roles == {'h': 'primary', 'n': 'primary'}
    assert secondary['h'] == 'B'
    assert pd.isna(secondary['n'])


def test_the_house_keeps_its_own_lots_value_and_the_neighbor_keeps_its_own():
    spine, flat, _, dwelling, building = _house_reaching_onto_its_neighbor()
    roles, crosswalk, secondary = _roles(spine, flat, dwelling, building)
    pairs = crosswalk.reset_index()[
        ['building_id', 'parcel_id', 'area_intersection_m2']
    ]
    lot_values = pd.DataFrame(
        {'improvement_value': [120000.0, 150000.0], 'n_dwellings': [1.0, 1.0]},
        index=pd.Index(['A', 'B'], name='parcel_id'),
    )
    result = apportion_reference_values(
        pairs,
        lot_values,
        spine_id_col='building_id',
        priority=pd.Series(roles),
        secondary_pairs=secondary_pairs_from(secondary),
        dwelling_linked_ids={'n'},
    )
    assert result.loc['h', 'improvement_value'] == pytest.approx(120000.0)
    assert result.loc['n', 'improvement_value'] == pytest.approx(150000.0)
    assert result.loc['h', 'n_dwellings'] == pytest.approx(1.0)
    assert result.loc['n', 'n_dwellings'] == pytest.approx(1.0)


def test_a_building_secondary_on_its_dominant_parcel_stays_secondary():
    """A shed on lot A beside the house that holds A's dwelling point,
    its corner reaching onto an empty lot C: secondary on A, its
    dominant parcel, so secondary as a building."""
    outlines = {'house': box(0, 0, 10, 10), 'shed': box(12, 0, 18, 4)}
    lots = {'A': box(-5, -5, 16, 15), 'C': box(16, -5, 30, 15)}
    points = [Point(5, 5)]
    values = _values(A=(100000.0, None), C=(0.0, None))
    spine, flat, _ = _cut(outlines, lots, points, values)
    roles, _, secondary = _roles(spine, flat, points, [])
    assert roles['house'] == 'primary'
    assert roles['shed'] == 'secondary'
    assert secondary['shed'] == 'A'
