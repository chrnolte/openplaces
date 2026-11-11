"""A building's dominant parcel where two overlaps tie.

plans/buildings-from-parcel-pieces.md, section 6.3 (tie research and
the maintainer's decision of 2026-11-10). Every geometry, id and value
below is fabricated: boxes in an equal-area projection near the origin,
made-up parcel ids and values; no record of any source is used
(docs/5_contribute/no-personal-data.rst).
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

import openplaces.io.harmonizer.attributes as attrs
import openplaces.io.harmonizer.links as links
from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer.apportion import apportion_reference_values
from openplaces.io.harmonizer.dominant_parcel import (
    OVERLAP_TIE_M2,
    dominant_overlaps,
    rank_overlaps,
)
from openplaces.io.harmonizer.links import parcel_pieces as pp
from tests.links_patching import patch_links

CRS = 'epsg:6933'


def _pairs(rows):
    return pd.DataFrame(
        rows, columns=['building_id', 'parcel_id', 'area_intersection_m2']
    )


def _pick(pairs, value=None, area=None):
    out = dominant_overlaps(
        pairs,
        spine_id_col='building_id',
        improvement_value=value,
        ref_area=area,
    )
    return out.set_index('building_id')['parcel_id'].to_dict()


# A building of 120 m2 lying whole on an outer tract (P-OUT) and an
# inner lot drawn inside it (P-IN); the two overlaps differ by float
# noise only.
NESTED = _pairs([('B1', 'P-OUT', 120.0), ('B1', 'P-IN', 120.0 + 1e-6)])
AREA = pd.Series({'P-OUT': 5000.0, 'P-IN': 400.0})


def test_improved_lot_beats_unimproved_outer_tract():
    value = pd.Series({'P-OUT': np.nan, 'P-IN': 90_000.0})
    assert _pick(NESTED, value, AREA) == {'B1': 'P-IN'}


def test_improved_outer_parcel_beats_unimproved_inner_lot():
    # The exempt half-lot drawn inside the residential parcel that
    # holds the house: geometry alone would pick the inner lot.
    value = pd.Series({'P-OUT': 224_000.0, 'P-IN': 0.0})
    assert _pick(NESTED, value, AREA) == {'B1': 'P-OUT'}


def test_nested_tie_with_both_improved_goes_to_the_inner_lot():
    value = pd.Series({'P-OUT': 10_000.0, 'P-IN': 90_000.0})
    assert _pick(NESTED, value, AREA) == {'B1': 'P-IN'}
    assert _pick(NESTED, None, AREA) == {'B1': 'P-IN'}


def test_remaining_tie_goes_to_the_lower_parcel_id():
    pairs = _pairs([('B1', 'P-9', 60.0), ('B1', 'P-2', 60.05)])
    value = pd.Series({'P-9': 1.0, 'P-2': 1.0})
    area = pd.Series({'P-9': 300.0, 'P-2': 300.0})
    assert _pick(pairs, value, area) == {'B1': 'P-2'}
    assert _pick(pairs) == {'B1': 'P-2'}


def test_pick_does_not_depend_on_row_order():
    rows = [
        ('B1', 'P-OUT', 120.0),
        ('B1', 'P-IN', 120.0 + 1e-6),
        ('B2', 'P-A', 50.0),
        ('B2', 'P-B', 50.0),
        ('B2', 'P-C', 10.0),
        ('B3', 'P-D', 80.0),
        ('B3', 'P-E', 20.0),
    ]
    value = pd.Series({'P-OUT': 0.0, 'P-IN': 5.0, 'P-A': 1.0, 'P-B': 1.0})
    area = pd.Series({'P-OUT': 9e3, 'P-IN': 4e2, 'P-A': 7e2, 'P-B': 6e2})
    expected = None
    rng = np.random.default_rng(7)
    for _ in range(20):
        shuffled = _pairs(rows).sample(frac=1, random_state=rng)
        ranked = rank_overlaps(
            shuffled,
            spine_id_col='building_id',
            improvement_value=value,
            ref_area=area,
        )
        order = list(zip(ranked['building_id'], ranked['parcel_id'], strict=True))
        if expected is None:
            expected = order
        assert order == expected
    assert expected[:2] == [('B1', 'P-IN'), ('B1', 'P-OUT')]
    assert expected[2:5] == [('B2', 'P-B'), ('B2', 'P-A'), ('B2', 'P-C')]


def test_largest_overlap_wins_outside_a_tie():
    # The smaller, unimproved, lower-id overlap loses once the gap
    # exceeds the tolerance.
    pairs = _pairs([('B1', 'P-1', 100.0), ('B1', 'P-2', 100.0 + 2 * OVERLAP_TIE_M2)])
    value = pd.Series({'P-1': 50_000.0, 'P-2': 0.0})
    area = pd.Series({'P-1': 200.0, 'P-2': 9000.0})
    assert _pick(pairs, value, area) == {'B1': 'P-2'}


def test_apportioned_address_and_land_value_follow_the_tie_rule():
    pairs = NESTED.iloc[::-1].reset_index(drop=True)
    ref = pd.DataFrame(
        {
            'address': ['1 TRACT RD', '7 LOT LN'],
            'land_value': [1000.0, 30_000.0],
            'improvement_value': [0.0, 90_000.0],
        },
        index=pd.Index(['P-OUT', 'P-IN'], name='parcel_id'),
    )
    out = apportion_reference_values(
        pairs, ref, spine_id_col='building_id', tie_ref_area=AREA
    )
    assert out.loc['B1', 'address'] == '7 LOT LN'
    assert out.loc['B1', 'land_value'] == 30_000.0


class _Entity:
    def __init__(self, entity_type):
        self.entity_type = entity_type


def _attribute(monkeypatch, ref):
    """parcel_id and land use of one building on *ref*, through the
    overlay and attribution steps the building spine runs."""
    spine = gpd.GeoDataFrame(
        {'geometry_source': ['obm']},
        geometry=[box(40, 40, 52, 50)],
        crs=CRS,
    )
    spine.index.name = 'building_id'
    state = HarmonizeState(
        recipe={'entity': _Entity('building')},
        admin_id='US-XX-YY',
        verbose=False,
        timer=None,
        spine=spine,
    )
    patch_links(monkeypatch, 'get_entities', lambda *a, **k: ref)
    state = links._link_spatial_overlay(
        state, 'parcel_ref', 'parcel', {'area_intersection_m2_min': 0}
    )
    ref_polys = state.references['parcel_ref']
    state = attrs._attribute_polygon_reference(
        state,
        'parcel_ref',
        'parcel',
        ref_polys,
        columns=['use_group_combined', 'address'],
    )
    row = state.spine.iloc[0]
    local = ref_polys.loc[row['parcel_id'], 'parcel_id_local']
    return local, row['use_group_combined_parcel'], row['address_parcel']


def _nested_ref(outer_value, inner_value):
    return gpd.GeoDataFrame(
        {
            'parcel_id_local': ['OUTER', 'INNER'],
            'use_group': ['exempt', 'residential'],
            'improvement_value': [outer_value, inner_value],
            'address': ['1 TRACT RD', '7 LOT LN'],
        },
        geometry=[box(0, 0, 200, 200), box(30, 30, 60, 60)],
        crs=CRS,
    )


@pytest.mark.parametrize(
    ('outer_value', 'inner_value', 'expected'),
    [
        (np.nan, 90_000.0, 'INNER'),
        (224_000.0, 0.0, 'OUTER'),
        (10_000.0, 90_000.0, 'INNER'),
    ],
)
@pytest.mark.parametrize('reverse', [False, True])
def test_building_spine_picks_by_the_tie_rule(
    monkeypatch, outer_value, inner_value, expected, reverse
):
    ref = _nested_ref(outer_value, inner_value)
    if reverse:
        ref = ref.iloc[::-1]
    local, use, address = _attribute(monkeypatch, ref)
    assert local == expected
    # The land use and the address come from the same side of the tie.
    assert use.startswith('residential' if expected == 'INNER' else 'exempt')
    assert address == ('7 LOT LN' if expected == 'INNER' else '1 TRACT RD')


def test_cut_candidates_list_the_tie_rule_winner_first():
    # Two lots meeting under the middle of an outline, each holding
    # half: an exact tie the dominant piece has to break.
    lots = gpd.GeoDataFrame(
        geometry=[box(0, 0, 20, 30), box(20, 0, 30, 30)],
        index=pd.Index(['P-1', 'P-2'], name='parcel_id'),
        crs=CRS,
    )
    flat = pd.DataFrame(
        {
            'footprint_id': ['F1', 'F1'],
            'parcel_id': ['P-1', 'P-2'],
            'area_intersection_m2': [50.0, 50.0],
            'link': [pp.MULTI_PARCEL_LINK, pp.MULTI_PARCEL_LINK],
        }
    )
    spine = pd.DataFrame({'footprint_id': ['F1'], 'geometry_source': ['obm']})
    plain = pp.candidate_pairs(flat, spine, 'footprint_id')
    assert plain['F1'] == ['P-1', 'P-2']
    improved = pp.candidate_pairs(
        flat.iloc[::-1],
        spine,
        'footprint_id',
        tie_improvement_value=pd.Series({'P-1': 0.0, 'P-2': 80_000.0}),
        tie_ref_area=lots.area,
    )
    assert improved['F1'] == ['P-2', 'P-1']
