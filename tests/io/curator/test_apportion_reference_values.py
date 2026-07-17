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
    # F2 has dwelling evidence, F1 does not: F1's share is zeroed and F2
    # takes the whole parcel value; F1's land_value is suppressed too.
    pairs = _pairs([('F1', 'P1', 300.0), ('F2', 'P1', 100.0)])
    ref = pd.DataFrame(
        {'improvement_value': [400000.0], 'land_value': [50000.0]},
        index=pd.Index(['P1']),
    )
    priority = pd.Series(['primary', 'primary'], index=['F1', 'F2'])
    out = apportion_reference_values(
        pairs,
        ref,
        spine_id_col='fid',
        priority=priority,
        dwelling_linked_ids={'F2'},
    )
    assert out.loc['F1', 'improvement_value'] == 0.0
    assert out.loc['F2', 'improvement_value'] == 400000.0
    assert pd.isna(out.loc['F1', 'land_value'])
    assert out.loc['F2', 'land_value'] == 50000.0


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
    # The primary still gets only its own share: fractions are overlap-based,
    # not renormalized by the secondary's exclusion from n_dwellings.
    assert out.loc['F1', 'improvement_value'] == 100000.0
    assert out.loc['F1', 'n_dwellings'] == 1.0


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
