"""Tests for the shared value apportionment (`apportion_reference_values`).

One implementation distributes reference-polygon values across an n:m link
for both stages: harmonize feeds it the in-memory identity overlay, curate
feeds it the persisted link sidecar joined to the curated reference. The
semantics under test are the harmonize stage's original ones (Lochhead et
al. 2026, Table 4): overlap-area shares, dwelling-linked suppression,
land_value whole on principal entities only, secondary handling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from openplaces.io.harmonizer.apportion import apportion_reference_values


def _pairs(rows):
    return pd.DataFrame(rows, columns=['fid', 'parcel_id', 'area_intersection_m2'])


def test_improvement_value_splits_by_overlap_share():
    pairs = _pairs([('F1', 'P1', 100.0), ('F2', 'P1', 300.0)])
    ref = pd.DataFrame({'improvement_value': [400000.0]}, index=pd.Index(['P1']))
    out = apportion_reference_values(pairs, ref, spine_id_col='fid')
    assert out.loc['F1', 'improvement_value'] == 100000.0
    assert out.loc['F2', 'improvement_value'] == 300000.0


def test_straddling_entity_sums_shares_from_both_references():
    # F1 is P1's only footprint and holds a quarter of P2's overlap area, so
    # it receives all of P1's value plus a quarter of P2's — the n:m
    # semantics the dominant-parcel-only id-join could not reproduce.
    pairs = _pairs([('F1', 'P1', 50.0), ('F1', 'P2', 100.0), ('F2', 'P2', 300.0)])
    ref = pd.DataFrame(
        {'improvement_value': [80000.0, 400000.0]}, index=pd.Index(['P1', 'P2'])
    )
    out = apportion_reference_values(pairs, ref, spine_id_col='fid')
    assert out.loc['F1', 'improvement_value'] == 80000.0 + 100000.0
    assert out.loc['F2', 'improvement_value'] == 300000.0


def test_dwelling_suppression_renormalizes_and_suppresses_land_value():
    # F2 has dwelling evidence, F1 does not and is secondary, as the
    # priority step makes it: F1's share is zeroed and F2 takes the
    # whole parcel value; F1's land_value is suppressed too.
    pairs = _pairs([('F1', 'P1', 300.0), ('F2', 'P1', 100.0)])
    ref = pd.DataFrame(
        {'improvement_value': [400000.0], 'land_value': [50000.0]},
        index=pd.Index(['P1']),
    )
    priority = pd.Series(['secondary', 'primary'], index=['F1', 'F2'])
    out = apportion_reference_values(
        pairs,
        ref,
        spine_id_col='fid',
        priority=priority,
        dwelling_linked_ids={'F2'},
    )
    assert pd.isna(out.loc['F1', 'improvement_value'])
    assert out.loc['F2', 'improvement_value'] == 400000.0
    assert pd.isna(out.loc['F1', 'land_value'])
    assert out.loc['F2', 'land_value'] == 50000.0


def test_primary_entity_beside_a_dwelling_point_keeps_its_area_share():
    # The priority step keeps a large building primary beside a smaller
    # dwelling-point footprint (dwelling_override_ratio) so that the
    # value splits by area; suppression must not undo that and hand
    # the whole value to the small one.
    pairs = _pairs([('F1', 'P1', 300.0), ('F2', 'P1', 100.0), ('F3', 'P1', 50.0)])
    ref = pd.DataFrame({'improvement_value': [400000.0]}, index=pd.Index(['P1']))
    priority = pd.Series(['primary', 'primary', 'secondary'], index=['F1', 'F2', 'F3'])
    out = apportion_reference_values(
        pairs,
        ref,
        spine_id_col='fid',
        priority=priority,
        dwelling_linked_ids={'F2'},
    )
    assert out.loc['F1', 'improvement_value'] == 300000.0
    assert out.loc['F2', 'improvement_value'] == 100000.0
    assert pd.isna(out.loc['F3', 'improvement_value'])


def test_stated_built_area_scales_shares_where_little_is_mapped():
    # A 14 m2 structure is all the spine maps on a parcel whose record
    # states 1,400 m2 of building ground area: it receives 1% of the
    # value, and the rest stays unallocated rather than landing on it.
    pairs = _pairs([('F1', 'P1', 14.0)])
    ref = pd.DataFrame({'improvement_value': [100e6]}, index=pd.Index(['P1']))
    out = apportion_reference_values(
        pairs, ref, spine_id_col='fid', ref_built_area=pd.Series({'P1': 1400.0})
    )
    assert out.loc['F1', 'improvement_value'] == 1e6


def test_floors_split_a_lot_between_tower_and_podium():
    # 400 m2 tower of 30 floors and 400 m2 podium of 1: by floor area
    # the tower takes 30/31 of the value, not half.
    pairs = _pairs([('TOWER', 'P1', 400.0), ('PODIUM', 'P1', 400.0)])
    ref = pd.DataFrame({'improvement_value': [31e6]}, index=pd.Index(['P1']))
    out = apportion_reference_values(
        pairs,
        ref,
        spine_id_col='fid',
        volume_weight=pd.Series({'TOWER': 30.0, 'PODIUM': 1.0}),
    )
    assert out.loc['TOWER', 'improvement_value'] == 30e6
    assert out.loc['PODIUM', 'improvement_value'] == 1e6


def test_a_lot_with_an_unknown_height_keeps_the_area_split():
    pairs = _pairs([('TOWER', 'P1', 400.0), ('PODIUM', 'P1', 400.0)])
    ref = pd.DataFrame({'improvement_value': [30e6]}, index=pd.Index(['P1']))
    out = apportion_reference_values(
        pairs,
        ref,
        spine_id_col='fid',
        volume_weight=pd.Series({'PODIUM': 1.0}),
    )
    assert out.loc['TOWER', 'improvement_value'] == 15e6


def test_stated_built_area_leaves_moderate_gaps_alone():
    # Mapped 300 m2 of a stated 1,000 m2 (coverage 0.3): within the
    # noise of floor area over story count, so the shares stand.
    pairs = _pairs([('F1', 'P1', 200.0), ('F2', 'P1', 100.0), ('F3', 'P2', 50.0)])
    ref = pd.DataFrame(
        {'improvement_value': [300000.0, 80000.0]}, index=pd.Index(['P1', 'P2'])
    )
    out = apportion_reference_values(
        pairs,
        ref,
        spine_id_col='fid',
        ref_built_area=pd.Series({'P1': 1000.0}),
    )
    assert out.loc['F1', 'improvement_value'] == 200000.0
    assert out.loc['F2', 'improvement_value'] == 100000.0
    # No stated area: unchanged.
    assert out.loc['F3', 'improvement_value'] == 80000.0


def test_secondary_entity_gets_missing_improvement_and_zero_dwellings():
    pairs = _pairs([('F1', 'P1', 100.0), ('F2', 'P1', 100.0)])
    ref = pd.DataFrame(
        {'improvement_value': [200000.0], 'n_dwellings': [2.0]},
        index=pd.Index(['P1']),
    )
    priority = pd.Series(['primary', 'secondary'], index=['F1', 'F2'])
    out = apportion_reference_values(pairs, ref, spine_id_col='fid', priority=priority)
    assert pd.isna(out.loc['F2', 'improvement_value'])
    assert out.loc['F2', 'n_dwellings'] == 0.0
    # The secondary's share passes to the primary, so the reference's
    # value and dwellings are conserved (maintainer, 2026-10-17).
    assert out.loc['F1', 'improvement_value'] == 200000.0
    assert out.loc['F1', 'n_dwellings'] == 2.0


def test_secondary_share_goes_to_primaries_by_their_weight():
    pairs = _pairs([('F1', 'P1', 150.0), ('F2', 'P1', 50.0), ('F3', 'P1', 100.0)])
    ref = pd.DataFrame({'improvement_value': [3000.0]}, index=pd.Index(['P1']))
    priority = pd.Series(['primary', 'primary', 'secondary'], index=['F1', 'F2', 'F3'])
    out = apportion_reference_values(pairs, ref, spine_id_col='fid', priority=priority)
    assert out.loc['F1', 'improvement_value'] == 2250.0
    assert out.loc['F2', 'improvement_value'] == 750.0
    assert pd.isna(out.loc['F3', 'improvement_value'])


def test_reference_with_only_secondary_entities_stays_unallocated():
    pairs = _pairs([('F1', 'P1', 40.0), ('F2', 'P1', 60.0)])
    ref = pd.DataFrame(
        {'improvement_value': [5000.0], 'n_dwellings': [1.0]},
        index=pd.Index(['P1']),
    )
    priority = pd.Series(['secondary', 'secondary'], index=['F1', 'F2'])
    out = apportion_reference_values(pairs, ref, spine_id_col='fid', priority=priority)
    assert out['improvement_value'].isna().all()
    assert (out['n_dwellings'] == 0.0).all()


def test_secondary_on_one_parcel_passes_share_on_that_parcel_only():
    # F2 is a secondary garage straddling P1 and P2; each parcel's share
    # of it goes to that parcel's own primary.
    pairs = _pairs(
        [
            ('F1', 'P1', 100.0),
            ('F2', 'P1', 50.0),
            ('F2', 'P2', 50.0),
            ('F3', 'P2', 150.0),
        ]
    )
    ref = pd.DataFrame(
        {'improvement_value': [1000.0, 2000.0]}, index=pd.Index(['P1', 'P2'])
    )
    priority = pd.Series(['primary', 'secondary', 'primary'], index=['F1', 'F2', 'F3'])
    out = apportion_reference_values(pairs, ref, spine_id_col='fid', priority=priority)
    assert out.loc['F1', 'improvement_value'] == 1000.0
    assert out.loc['F3', 'improvement_value'] == 2000.0
    assert pd.isna(out.loc['F2', 'improvement_value'])


def test_built_area_coverage_still_counts_the_secondary_as_mapped():
    # Stated ground area 1,000 m2; mapped 100 (primary) + 50 (garage) =
    # 150, coverage 0.15 < 0.2, so the shares are scaled by 0.15 after
    # the garage's share has passed to the primary.
    pairs = _pairs([('F1', 'P1', 100.0), ('F2', 'P1', 50.0)])
    ref = pd.DataFrame({'improvement_value': [1000.0]}, index=pd.Index(['P1']))
    priority = pd.Series(['primary', 'secondary'], index=['F1', 'F2'])
    out = apportion_reference_values(
        pairs,
        ref,
        spine_id_col='fid',
        priority=priority,
        ref_built_area=pd.Series([1000.0], index=['P1']),
    )
    assert out.loc['F1', 'improvement_value'] == 150.0


def test_land_value_whole_on_principal_only():
    pairs = _pairs([('F1', 'P1', 100.0), ('F2', 'P1', 300.0)])
    ref = pd.DataFrame({'land_value': [50000.0]}, index=pd.Index(['P1']))
    priority = pd.Series(['secondary', 'primary'], index=['F1', 'F2'])
    out = apportion_reference_values(pairs, ref, spine_id_col='fid', priority=priority)
    assert pd.isna(out.loc['F1', 'land_value'])
    assert out.loc['F2', 'land_value'] == 50000.0


def test_land_value_without_priority_requires_sole_entity_on_reference():
    pairs = _pairs([('F1', 'P1', 100.0), ('F2', 'P2', 100.0), ('F3', 'P2', 50.0)])
    ref = pd.DataFrame({'land_value': [50000.0, 70000.0]}, index=pd.Index(['P1', 'P2']))
    out = apportion_reference_values(pairs, ref, spine_id_col='fid')
    assert out.loc['F1', 'land_value'] == 50000.0  # sole footprint on P1
    assert pd.isna(out.loc['F2', 'land_value'])  # shares P2
    assert pd.isna(out.loc['F3', 'land_value'])


def test_year_built_zero_is_missing_and_mean_over_references():
    pairs = _pairs([('F1', 'P1', 100.0), ('F1', 'P2', 100.0), ('F2', 'P3', 1.0)])
    ref = pd.DataFrame(
        {'year_built': [1950.0, 2000.0, 0.0]}, index=pd.Index(['P1', 'P2', 'P3'])
    )
    out = apportion_reference_values(pairs, ref, spine_id_col='fid')
    assert out.loc['F1', 'year_built'] == 1975.0
    assert pd.isna(out.loc['F2', 'year_built'])


def test_address_from_dominant_reference():
    pairs = _pairs([('F1', 'P1', 10.0), ('F1', 'P2', 90.0)])
    ref = pd.DataFrame(
        {'address': ['1 Side St', '2 Main St']}, index=pd.Index(['P1', 'P2'])
    )
    out = apportion_reference_values(pairs, ref, spine_id_col='fid')
    assert out.loc['F1', 'address'] == '2 Main St'


def test_volume_weight_shifts_shares_toward_taller_entities():
    pairs = _pairs([('F1', 'P1', 100.0), ('F2', 'P1', 100.0)])
    ref = pd.DataFrame({'improvement_value': [400000.0]}, index=pd.Index(['P1']))
    stories = pd.Series([3.0, 1.0], index=['F1', 'F2'])
    out = apportion_reference_values(
        pairs, ref, spine_id_col='fid', volume_weight=stories
    )
    assert out.loc['F1', 'improvement_value'] == 300000.0
    assert out.loc['F2', 'improvement_value'] == 100000.0


def test_pairs_with_missing_reference_id_are_ignored():
    pairs = _pairs([('F1', 'P1', 100.0), ('F2', np.nan, 100.0)])
    ref = pd.DataFrame({'improvement_value': [100000.0]}, index=pd.Index(['P1']))
    out = apportion_reference_values(pairs, ref, spine_id_col='fid')
    assert out.loc['F1', 'improvement_value'] == 100000.0
    assert 'F2' not in out.index


def test_no_known_value_columns_returns_empty():
    pairs = _pairs([('F1', 'P1', 100.0)])
    ref = pd.DataFrame({'use_group': ['residential']}, index=pd.Index(['P1']))
    out = apportion_reference_values(pairs, ref, spine_id_col='fid')
    assert out.empty and len(out.columns) == 0


# --- equal_area_ref_ids ------------------------------------------------------


def test_equal_area_ref_ids_exempts_secondary_from_masking():
    # Same setup as test_secondary_entity_gets_missing_improvement_and_zero_
    # dwellings, but P1 is an equal-area reference (e.g. non-residential):
    # F2 (secondary) must keep its overlap-area share of improvement_value
    # instead of being masked to missing.
    pairs = _pairs([('F1', 'P1', 100.0), ('F2', 'P1', 100.0)])
    ref = pd.DataFrame(
        {'improvement_value': [200000.0], 'n_dwellings': [2.0]},
        index=pd.Index(['P1']),
    )
    priority = pd.Series(['primary', 'secondary'], index=['F1', 'F2'])
    out = apportion_reference_values(
        pairs,
        ref,
        spine_id_col='fid',
        priority=priority,
        equal_area_ref_ids={'P1'},
    )
    assert out.loc['F1', 'improvement_value'] == 100000.0
    assert out.loc['F2', 'improvement_value'] == 100000.0
    # n_dwellings is untouched by equal_area_ref_ids -- still zeroed on the
    # secondary entity.
    assert out.loc['F2', 'n_dwellings'] == 0.0


def test_equal_area_ref_ids_exempts_dwelling_linked_suppression():
    # Same setup as test_dwelling_suppression_renormalizes_and_suppresses_
    # land_value, but P1 is an equal-area reference: F1 (no dwelling
    # evidence) must keep its own overlap-area share of improvement_value
    # instead of being zeroed and renormalized away. land_value is unaffected
    # by equal_area_ref_ids and keeps its existing suppression.
    pairs = _pairs([('F1', 'P1', 300.0), ('F2', 'P1', 100.0)])
    ref = pd.DataFrame(
        {'improvement_value': [400000.0], 'land_value': [50000.0]},
        index=pd.Index(['P1']),
    )
    priority = pd.Series(['secondary', 'primary'], index=['F1', 'F2'])
    out = apportion_reference_values(
        pairs,
        ref,
        spine_id_col='fid',
        priority=priority,
        dwelling_linked_ids={'F2'},
        equal_area_ref_ids={'P1'},
    )
    assert out.loc['F1', 'improvement_value'] == 300000.0
    assert out.loc['F2', 'improvement_value'] == 100000.0
    assert pd.isna(out.loc['F1', 'land_value'])
    assert out.loc['F2', 'land_value'] == 50000.0


def test_equal_area_ref_ids_only_applies_to_its_own_reference():
    # A second, non-equal-area reference (P2) keeps the ordinary
    # dwelling-linked suppression even though P1 is exempted.
    pairs = _pairs(
        [
            ('F1', 'P1', 300.0),
            ('F2', 'P1', 100.0),
            ('F3', 'P2', 300.0),
            ('F4', 'P2', 100.0),
        ]
    )
    ref = pd.DataFrame(
        {'improvement_value': [400000.0, 400000.0]}, index=pd.Index(['P1', 'P2'])
    )
    out = apportion_reference_values(
        pairs,
        ref,
        spine_id_col='fid',
        dwelling_linked_ids={'F2', 'F4'},
        equal_area_ref_ids={'P1'},
    )
    assert out.loc['F1', 'improvement_value'] == 300000.0  # exempted
    assert out.loc['F2', 'improvement_value'] == 100000.0  # exempted
    assert out.loc['F3', 'improvement_value'] == 0.0  # ordinary suppression
    assert out.loc['F4', 'improvement_value'] == 400000.0  # ordinary suppression


# --- improvement_value_imputed (PROPORTIONAL_SPLIT_COLUMNS) -----------------


def test_improvement_value_imputed_splits_by_overlap_share_like_improvement_value():
    pairs = _pairs([('F1', 'P1', 100.0), ('F2', 'P1', 300.0)])
    ref = pd.DataFrame(
        {'improvement_value_imputed': [360000.0]}, index=pd.Index(['P1'])
    )
    out = apportion_reference_values(pairs, ref, spine_id_col='fid')
    assert out.loc['F1', 'improvement_value_imputed'] == 90000.0
    assert out.loc['F2', 'improvement_value_imputed'] == 270000.0


def test_improvement_value_imputed_inherits_dwelling_linked_suppression():
    pairs = _pairs([('F1', 'P1', 300.0), ('F2', 'P1', 100.0)])
    ref = pd.DataFrame(
        {'improvement_value_imputed': [400000.0]}, index=pd.Index(['P1'])
    )
    out = apportion_reference_values(
        pairs, ref, spine_id_col='fid', dwelling_linked_ids={'F2'}
    )
    assert out.loc['F1', 'improvement_value_imputed'] == 0.0
    assert out.loc['F2', 'improvement_value_imputed'] == 400000.0


def test_improvement_value_imputed_inherits_equal_area_widening():
    pairs = _pairs([('F1', 'P1', 300.0), ('F2', 'P1', 100.0)])
    ref = pd.DataFrame(
        {'improvement_value_imputed': [400000.0]}, index=pd.Index(['P1'])
    )
    out = apportion_reference_values(
        pairs,
        ref,
        spine_id_col='fid',
        dwelling_linked_ids={'F2'},
        equal_area_ref_ids={'P1'},
    )
    assert out.loc['F1', 'improvement_value_imputed'] == 300000.0
    assert out.loc['F2', 'improvement_value_imputed'] == 100000.0


# --- land_value_imputed (WHOLE_VALUE_COLUMNS, same treatment as land_value) -


def test_land_value_imputed_whole_on_principal_only():
    # land_value_imputed is conceptually "land_value, gap-filled" -- it must
    # get the exact same whole-on-principal-footprint-only treatment as
    # land_value, not improvement_value's area-proportional split.
    pairs = _pairs([('F1', 'P1', 100.0), ('F2', 'P1', 300.0)])
    ref = pd.DataFrame({'land_value_imputed': [50000.0]}, index=pd.Index(['P1']))
    priority = pd.Series(['secondary', 'primary'], index=['F1', 'F2'])
    out = apportion_reference_values(pairs, ref, spine_id_col='fid', priority=priority)
    assert pd.isna(out.loc['F1', 'land_value_imputed'])
    assert out.loc['F2', 'land_value_imputed'] == 50000.0


def test_land_value_imputed_without_priority_requires_sole_entity_on_reference():
    pairs = _pairs([('F1', 'P1', 100.0), ('F2', 'P2', 100.0), ('F3', 'P2', 50.0)])
    ref = pd.DataFrame(
        {'land_value_imputed': [50000.0, 70000.0]}, index=pd.Index(['P1', 'P2'])
    )
    out = apportion_reference_values(pairs, ref, spine_id_col='fid')
    assert out.loc['F1', 'land_value_imputed'] == 50000.0  # sole footprint on P1
    assert pd.isna(out.loc['F2', 'land_value_imputed'])  # shares P2
    assert pd.isna(out.loc['F3', 'land_value_imputed'])


def test_land_value_imputed_suppressed_by_dwelling_linked_rule():
    pairs = _pairs([('F1', 'P1', 300.0), ('F2', 'P1', 100.0)])
    ref = pd.DataFrame({'land_value_imputed': [50000.0]}, index=pd.Index(['P1']))
    priority = pd.Series(['secondary', 'primary'], index=['F1', 'F2'])
    out = apportion_reference_values(
        pairs,
        ref,
        spine_id_col='fid',
        priority=priority,
        dwelling_linked_ids={'F2'},
    )
    assert pd.isna(out.loc['F1', 'land_value_imputed'])
    assert out.loc['F2', 'land_value_imputed'] == 50000.0


def test_land_value_imputed_unaffected_by_equal_area_ref_ids():
    pairs = _pairs([('F1', 'P1', 100.0), ('F2', 'P1', 300.0)])
    ref = pd.DataFrame({'land_value_imputed': [50000.0]}, index=pd.Index(['P1']))
    priority = pd.Series(['secondary', 'primary'], index=['F1', 'F2'])
    out = apportion_reference_values(
        pairs,
        ref,
        spine_id_col='fid',
        priority=priority,
        equal_area_ref_ids={'P1'},
    )
    assert pd.isna(out.loc['F1', 'land_value_imputed'])
    assert out.loc['F2', 'land_value_imputed'] == 50000.0


def test_total_value_is_carried_whole_not_split():
    # The reason total_value rides land_value's rule rather than
    # improvement_value's: it is the parcel's land+structure sum, so
    # splitting it by building area would divide the land half by building
    # geometry, and repeating it would make a regional sum a multiple of
    # the region's assessed value.
    pairs = _pairs([('F1', 'P1', 300.0)])
    ref = pd.DataFrame({'total_value': [250000.0]}, index=pd.Index(['P1']))
    out = apportion_reference_values(pairs, ref, spine_id_col='fid')
    assert out.loc['F1', 'total_value'] == 250000.0
    assert out['total_value'].sum() == ref['total_value'].sum()


def test_total_value_is_withheld_rather_than_duplicated():
    # Two footprints on one parcel and nothing saying which is principal:
    # the value is dropped, not repeated. Losing a parcel is recoverable
    # arithmetic; double-counting one is not.
    pairs = _pairs([('F1', 'P1', 300.0), ('F2', 'P1', 100.0)])
    ref = pd.DataFrame({'total_value': [250000.0]}, index=pd.Index(['P1']))
    out = apportion_reference_values(pairs, ref, spine_id_col='fid')
    assert out['total_value'].notna().sum() == 0
    assert out['total_value'].sum() <= ref['total_value'].sum()


def test_total_value_goes_to_the_principal_entity_when_priority_is_given():
    pairs = _pairs([('F1', 'P1', 100.0), ('F2', 'P1', 300.0)])
    ref = pd.DataFrame({'total_value': [180000.0]}, index=pd.Index(['P1']))
    priority = pd.Series({'F1': 'primary', 'F2': 'secondary'})
    out = apportion_reference_values(pairs, ref, spine_id_col='fid', priority=priority)
    # F2 overlaps more, but a shed is not where a parcel's value belongs.
    assert out.loc['F1', 'total_value'] == 180000.0
    assert pd.isna(out.loc['F2', 'total_value'])
    assert out['total_value'].sum() == ref['total_value'].sum()


def test_total_value_never_exceeds_the_parcels_it_came_from():
    # The invariant to assert on real data: apportionment may drop a parcel
    # (no footprint, or no principal one), but must never manufacture value.
    pairs = _pairs([('F1', 'P1', 100.0), ('F2', 'P1', 50.0), ('F3', 'P2', 200.0)])
    ref = pd.DataFrame(
        {'total_value': [100000.0, 300000.0]}, index=pd.Index(['P1', 'P2'])
    )
    out = apportion_reference_values(pairs, ref, spine_id_col='fid')
    assert out['total_value'].sum() <= ref['total_value'].sum()
