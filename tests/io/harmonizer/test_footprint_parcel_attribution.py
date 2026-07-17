"""Tests for `_attribute_polygon_reference`/`_attribute_point_reference`
dtype safety, `_join_distinct` `_all`-suppression, and the synthetic-footprint
inferred-backfill path.

`n_footprints_per_parcel` (a relational count) must stay `int64`, including
0 for a footprint with no parcel link at all -- `.reindex()` alone silently
upcasts to `float64` when it introduces NaN. Every `{col}_all` sidecar
should only be populated when it would add information beyond the single
dominant/reconciled value already stored separately. A parcel-derived
synthetic fallback footprint (geometry_source contains '.') never enters the
normal overlay, so its evidence comes from a separate inferred-backfill path
that must also propagate parcel_id_local -- without it, link_curated_entity
can never find that footprint's curated parcel downstream. Id columns
(`{entity}_id*`) keep their bare name rather than the usual `_parcel`-style
suffix, unless that bare name would collide with a column the spine already
had natively.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

import openplaces.io.harmonizer.attributes as attrs
import openplaces.io.harmonizer.links as links
from openplaces.io.harmonizer import HarmonizeState


class _Entity:
    def __init__(self, entity_type):
        self.entity_type = entity_type


def _state(spine, spine_entity_type='footprint'):
    return HarmonizeState(
        recipe={'entity': _Entity(spine_entity_type)},
        admin_id='US-MA-MI',
        verbose=False,
        timer=None,
        spine=spine,
    )


def test_n_footprints_per_parcel_is_int64_with_zero_for_unlinked(monkeypatch):
    # F1 overlaps parcel P1; F2 sits far away, overlapping nothing.
    spine = gpd.GeoDataFrame(
        {'geometry_source': ['obm', 'obm']},
        geometry=[box(0, 0, 1, 1), box(100, 100, 101, 101)],
        crs='EPSG:4326',
    )
    spine.index.name = 'footprint_id'
    state = _state(spine)

    ref = gpd.GeoDataFrame(
        {'use_group_combined': ['residential | single family']},
        geometry=[box(0, 0, 1, 1)],
        crs='EPSG:4326',
    )
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: ref)
    state = links._link_spatial_overlay(
        state, 'parcel_ref', 'parcel', {'area_intersection_m2_min': 0}
    )

    state = attrs._attribute_polygon_reference(
        state,
        'parcel_ref',
        'parcel',
        state.references['parcel_ref'],
        columns=['use_group_combined'],
    )

    col = state.spine['n_footprints_per_parcel']
    assert col.dtype == 'int64'
    assert col.loc[state.spine.index[0]] == 1  # F1: linked to P1
    assert col.loc[state.spine.index[1]] == 0  # F2: no parcel link at all


def test_parcel_id_sorts_before_parcel_id_local(monkeypatch):
    # order_columns's tiebreak (both parcel_id and parcel_id_local have a
    # blank registry sort rank) falls back to creation order, so parcel_id
    # must be written to the spine before parcel_id_local/parcel_id_local_all
    # for it to sort first in the curated output.
    spine = gpd.GeoDataFrame(
        {'geometry_source': ['obm']},
        geometry=[box(0, 0, 1, 1)],
        crs='EPSG:4326',
    )
    spine.index.name = 'footprint_id'
    state = _state(spine)

    ref = gpd.GeoDataFrame(
        {'parcel_id_local': ['P1LOCAL']},
        geometry=[box(0, 0, 1, 1)],
        crs='EPSG:4326',
    )
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: ref)
    state = links._link_spatial_overlay(
        state, 'parcel_ref', 'parcel', {'area_intersection_m2_min': 0}
    )

    state = attrs._attribute_polygon_reference(
        state,
        'parcel_ref',
        'parcel',
        state.references['parcel_ref'],
        columns=['parcel_id_local'],
    )

    cols = list(state.spine.columns)
    assert cols.index('parcel_id') < cols.index('parcel_id_local')


def test_area_intersection_m2_parcel_equals_dominant_link_overlap(monkeypatch):
    # F1 (1x1 box, area 1) overlaps parcel P1 fully; a second, smaller parcel P2
    # only grazes F1 with a sliver. The dominant link (P1, the larger overlap)
    # is what parcel_id/area_intersection_m2_parcel should reflect.
    spine = gpd.GeoDataFrame(
        {'geometry_source': ['obm']},
        geometry=[box(0, 0, 1, 1)],
        crs='EPSG:4326',
    )
    spine.index.name = 'footprint_id'
    state = _state(spine)

    ref = gpd.GeoDataFrame(
        {'parcel_id_local': ['P1LOCAL', 'P2LOCAL']},
        # P1 fully contains F1 (overlap area 1.0); P2 only clips a 0.1x0.1
        # sliver off F1's corner (overlap area 0.01).
        geometry=[box(-1, -1, 2, 2), box(0.9, 0.9, 1.1, 1.1)],
        crs='EPSG:4326',
    )
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: ref)
    state = links._link_spatial_overlay(
        state, 'parcel_ref', 'parcel', {'area_intersection_m2_min': 0}
    )

    state = attrs._attribute_polygon_reference(
        state,
        'parcel_ref',
        'parcel',
        state.references['parcel_ref'],
        columns=['parcel_id_local'],
    )

    dominant_parcel_id = state.spine.loc[state.spine.index[0], 'parcel_id']
    ref_polys = state.references['parcel_ref']
    expected_parcel_id = ref_polys.index[ref_polys['parcel_id_local'] == 'P1LOCAL'][0]
    assert dominant_parcel_id == expected_parcel_id
    # The overlap area persisted alongside it should equal F1's full area (P1
    # is the dominant, larger-overlap link), not the sliver.
    assert state.spine.loc[state.spine.index[0], 'area_intersection_m2_parcel'] > 1000


def test_inferred_backfill_propagates_parcel_id_local(monkeypatch):
    # F1 overlaps parcel P1 normally. F2 is a parcel-derived synthetic
    # fallback (geometry_source contains '.') with no overlay evidence at
    # all -- infer_spine_additions would have recorded which parcel (P2, not
    # overlapping anything) it came from via state.metadata, not the overlay.
    spine = gpd.GeoDataFrame(
        {'geometry_source': ['obm', 'parcel.nconemap']},
        geometry=[box(0, 0, 1, 1), box(500, 500, 501, 501)],
        crs='EPSG:4326',
    )
    spine.index.name = 'footprint_id'
    state = _state(spine)

    ref = gpd.GeoDataFrame(
        {
            'use_group_combined': ['residential | rp1', 'residential | rp2'],
            'parcel_id_local': ['P1LOCAL', 'P2LOCAL'],
        },
        # P1 overlaps F1; P2 overlaps nothing (it's the synthetic F2's source).
        geometry=[box(0, 0, 1, 1), box(100, 100, 101, 101)],
        crs='EPSG:4326',
    )
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: ref)
    state = links._link_spatial_overlay(
        state, 'parcel_ref', 'parcel', {'area_intersection_m2_min': 0}
    )

    ref_polys = state.references['parcel_ref']
    p2_parcel_id = ref_polys.index[ref_polys['parcel_id_local'] == 'P2LOCAL'][0]
    f2_id = state.spine.index[1]
    state.metadata['inferred_from_parcel_ref'] = pd.DataFrame(
        {'parcel_id': [p2_parcel_id]}, index=pd.Index([f2_id], name='footprint_id')
    )

    state = attrs._attribute_polygon_reference(
        state,
        'parcel_ref',
        'parcel',
        ref_polys,
        columns=['use_group_combined', 'parcel_id_local'],
    )

    assert state.spine.loc[f2_id, 'parcel_id_local'] == 'P2LOCAL'


def test_join_distinct_suppresses_all_for_single_value():
    areas = pd.DataFrame(
        {
            'footprint_id': ['F1', 'F1', 'F2'],
            'use_group_combined': [
                'residential | sf',
                'residential | sf',
                'commercial',
            ],
        }
    )
    joined = attrs._join_distinct(areas, 'footprint_id', 'use_group_combined')
    # F1's two rows agree -> nothing extra to add beyond the dominant value.
    assert pd.isna(joined.loc['F1'])
    # F2 has only one row -> also a single distinct value -> missing too.
    assert pd.isna(joined.loc['F2'])


def test_join_distinct_joins_when_genuinely_multi_valued():
    areas = pd.DataFrame(
        {
            'footprint_id': ['F1', 'F1'],
            'use_group_combined': ['residential | sf', 'commercial'],
        }
    )
    joined = attrs._join_distinct(areas, 'footprint_id', 'use_group_combined')
    assert joined.loc['F1'] in (
        'residential | sf + commercial',
        'commercial + residential | sf',
    )


def test_point_reference_all_suppressed_when_points_agree():
    spine = pd.DataFrame(index=pd.Index(['F1', 'F2'], name='footprint_id'))
    state = _state(spine)

    crosswalk = pd.DataFrame(
        {
            'footprint_id': ['F1', 'F1', 'F2', 'F2'],
            'occupancy_type': [
                'Single Family',
                'Single Family',
                'Single Family',
                'Multi Family',
            ],
            'structure_value': [100.0, 50.0, 100.0, 50.0],
        }
    )

    state = attrs._attribute_point_reference(
        state,
        'US_building-nsi-2022',
        'building',
        crosswalk,
        columns=['occupancy_type', 'structure_value'],
    )

    all_col = next(
        c
        for c in state.spine.columns
        if c.startswith('occupancy_type') and c.endswith('_all')
    )
    # F1's two NSI points agree -> nothing extra beyond the dominant value.
    assert pd.isna(state.spine.loc['F1', all_col])
    # F2's points disagree -> both values are informative.
    assert state.spine.loc['F2', all_col] in (
        'Single Family + Multi Family',
        'Multi Family + Single Family',
    )


def test_point_reference_carries_generic_numeric_with_registry_agg():
    # n_stories/area_sqft are requested by the real footprint spine recipe but
    # used to be silently dropped: only structure_value/year_built/n_dwellings
    # were aggregated. Any other requested numeric column should now reach the
    # spine via the attribute registry's default aggregation (mean for both).
    spine = pd.DataFrame(index=pd.Index(['F1'], name='footprint_id'))
    state = _state(spine)

    crosswalk = pd.DataFrame(
        {
            'footprint_id': ['F1', 'F1'],
            'n_stories': [1.0, 3.0],
        }
    )

    state = attrs._attribute_point_reference(
        state,
        'US_building-nsi-2022',
        'building',
        crosswalk,
        columns=['n_stories'],
    )

    assert state.spine.loc['F1', 'n_stories_building_nsi'] == 2.0


def test_point_reference_count_column_always_written_incl_all_single_match():
    # F1 has exactly one linked NSI point (group_sizes.max() == 1, the case the
    # old `if group_sizes.max() > 1` guard skipped entirely); F2 has none.
    # n_buildings_nsi must be present for both, so its presence in the curated
    # output doesn't vary by admin unit.
    spine = pd.DataFrame(index=pd.Index(['F1', 'F2'], name='footprint_id'))
    state = _state(spine)

    crosswalk = pd.DataFrame({'footprint_id': ['F1'], 'n_stories': [1.0]})

    state = attrs._attribute_point_reference(
        state,
        'US_building-nsi-2022',
        'building',
        crosswalk,
        columns=['n_stories'],
    )

    col = state.spine['n_buildings_nsi']
    assert col.dtype == 'int64'
    assert col.loc['F1'] == 1
    assert col.loc['F2'] == 0


def test_point_reference_overture_collision_skips_count_column():
    # dwelling-overture's count_col (n_dwellings_overture) is identical to the
    # renamed n_dwellings-sum column. Before Fix 4's collision guard, writing
    # both under the same name (via a bare assignment, then later a join) would
    # raise a duplicate-column error whenever a footprint had >1 linked
    # dwelling point. The sum (1.0) must win, not the point count (2).
    spine = pd.DataFrame(index=pd.Index(['F1'], name='footprint_id'))
    state = _state(spine)

    crosswalk = pd.DataFrame({'footprint_id': ['F1', 'F1'], 'n_dwellings': [0.5, 0.5]})

    state = attrs._attribute_point_reference(
        state,
        'dwelling-overture-2025',
        'dwelling',
        crosswalk,
        columns=['n_dwellings'],
    )

    assert state.spine.loc['F1', 'n_dwellings_overture'] == 1.0


def test_polygon_reference_carries_generic_numeric_with_registry_agg(monkeypatch):
    # FEMA USA Structures has no story count, only a LiDAR height. A parcel
    # with two FEMA-linked footprints should carry the taller one (registry
    # aggregation for `height` is `max`: the tallest structure best proxies
    # the parcel's primary building).
    spine = gpd.GeoDataFrame(
        {'geometry_source': ['parcel']},
        geometry=[box(0, 0, 10, 10)],
        crs='EPSG:4326',
    )
    # Not named 'parcel_id': the overlay helper always names the reference
    # (footprint) side's identity column 'parcel_id' internally, regardless of
    # its real entity_type -- matching the real parcel spine, whose working id
    # column carries a different name until the harmonizer restores 'parcel_id'
    # at save time (see _attribute_polygon_reference's reserved_cols docs).
    spine.index.name = 'parcel_working_id'
    state = _state(spine, spine_entity_type='parcel')

    ref = gpd.GeoDataFrame(
        {'height': [5.0, 12.0]},
        geometry=[box(0, 0, 1, 1), box(2, 2, 3, 3)],
        crs='EPSG:4326',
    )
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: ref)
    state = links._link_spatial_overlay(
        state, 'US_footprint-fema-2023', 'footprint', {'area_intersection_m2_min': 0}
    )

    state = attrs._attribute_polygon_reference(
        state,
        'US_footprint-fema-2023',
        'footprint',
        state.references['US_footprint-fema-2023'],
        columns=['height'],
    )

    assert state.spine.loc[state.spine.index[0], 'height_footprint_fema'] == 12.0


def test_footprint_fema_relational_count_pluralizes_entity_not_whole_suffix(
    monkeypatch,
):
    # A parcel spine attributing FEMA footprints (entity_type='footprint',
    # source 'fema', ref_label='footprint_fema') must produce
    # 'n_footprints_fema_per_parcel', not the ungrammatical
    # 'n_footprint_femas_per_parcel' (naive 's' appended to the whole compound
    # suffix). The reverse-direction sibling ('n_parcels_per_footprint_fema')
    # has no compound entity/source to pluralize on its own side and must stay
    # unaffected.
    spine = gpd.GeoDataFrame(
        {'geometry_source': ['parcel']},
        geometry=[box(0, 0, 10, 10)],
        crs='EPSG:4326',
    )
    spine.index.name = 'parcel_working_id'
    state = _state(spine, spine_entity_type='parcel')

    ref = gpd.GeoDataFrame(
        {'occupancy_type': ['Single Family']},
        geometry=[box(0, 0, 1, 1)],
        crs='EPSG:4326',
    )
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: ref)
    state = links._link_spatial_overlay(
        state, 'US_footprint-fema-2023', 'footprint', {'area_intersection_m2_min': 0}
    )

    state = attrs._attribute_polygon_reference(
        state,
        'US_footprint-fema-2023',
        'footprint',
        state.references['US_footprint-fema-2023'],
        columns=['occupancy_type'],
    )

    assert 'n_footprints_fema_per_parcel' in state.spine.columns
    assert 'n_footprint_femas_per_parcel' not in state.spine.columns
    assert 'n_parcels_per_footprint_fema' in state.spine.columns
    assert state.spine.loc[state.spine.index[0], 'n_footprints_fema_per_parcel'] == 1


def test_sliver_link_excluded_from_relational_count_after_trim(monkeypatch):
    # F1 is a large, genuine FEMA footprint fully inside the parcel; F2 only
    # grazes the parcel with a sub-threshold sliver (far below the default
    # min_fraction_of_largest=1/6, though still above area_intersection_m2_min
    # so only the fraction floor is at play). Before the overlay-trim fix,
    # n_footprints_fema_per_parcel was built from the raw, untrimmed overlay
    # and would have counted both (2); the trimmed crosswalk used now must
    # exclude the sliver, leaving 1.
    spine = gpd.GeoDataFrame(
        {'geometry_source': ['parcel']},
        geometry=[box(0, 0, 20, 20)],
        crs='EPSG:4326',
    )
    spine.index.name = 'parcel_working_id'
    state = _state(spine, spine_entity_type='parcel')

    ref = gpd.GeoDataFrame(
        {'occupancy_type': ['Single Family', 'Single Family']},
        # F1: a full 10x10 degree footprint. F2: a tiny 0.1x0.1 degree sliver
        # elsewhere in the parcel -- its fraction_of_largest is far below 1/6.
        geometry=[box(0, 0, 10, 10), box(15, 15, 15.1, 15.1)],
        crs='EPSG:4326',
    )
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: ref)
    state = links._link_spatial_overlay(
        state, 'US_footprint-fema-2023', 'footprint', {'area_intersection_m2_min': 0}
    )

    state = attrs._attribute_polygon_reference(
        state,
        'US_footprint-fema-2023',
        'footprint',
        state.references['US_footprint-fema-2023'],
        columns=['occupancy_type'],
    )

    assert state.spine.loc[state.spine.index[0], 'n_footprints_fema_per_parcel'] == 1


def test_point_reference_excludes_flagged_points_from_n_dwellings_sum():
    # F1 has two linked NSI points reporting n_dwellings=1 each; one is flagged
    # exclude_from_upward_correction (an ESRI point colocated with the other,
    # e.g. a home-office duplicate). Only the non-excluded point's n_dwellings
    # should be summed, not both -- otherwise a genuine single-family footprint
    # would wrongly read n_dwellings=2 (Multi-Family).
    spine = pd.DataFrame(index=pd.Index(['F1'], name='footprint_id'))
    state = _state(spine)

    crosswalk = pd.DataFrame(
        {
            'footprint_id': ['F1', 'F1'],
            'n_dwellings': [1.0, 1.0],
            'source': ['Parcel', 'ESRI'],
            'exclude_from_upward_correction': [False, True],
        }
    )

    state = attrs._attribute_point_reference(
        state,
        'US_building-nsi-2022',
        'building',
        crosswalk,
        columns=['n_dwellings'],
    )

    assert state.spine.loc['F1', 'n_dwellings_nsi'] == 1.0
