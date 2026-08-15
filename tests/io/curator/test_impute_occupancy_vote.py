"""Tests for the weighted consensus vote in impute_occupancy_type.

With occupancy.evidence_mode: vote, each present evidence entry casts its
weight (default 1.0) for its residential-bucketed class; the heaviest bucket
wins, ties fall to the earliest listed entry, and occupancy_type_source
records the winning voters' labels joined with '/'. Without the flag the
first-non-null cascade is unchanged.
"""

from __future__ import annotations

import pandas as pd

import openplaces.io.curator.occupancy as occ_mod
from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.inferers import impute_occupancy_type

CLASS_MAP = [
    {
        'pattern': 'Single Family',
        'match_type': 'contains',
        'occupancy_type': 'Single-Family',
    },
    {
        'pattern': 'Single-Family',
        'match_type': 'contains',
        'occupancy_type': 'Single-Family',
    },
    {'pattern': 'Multi', 'match_type': 'contains', 'occupancy_type': 'Multi-Family'},
    {
        'pattern': 'Manufactured',
        'match_type': 'contains',
        'occupancy_type': 'Manufactured Home',
    },
]

EVIDENCE = [
    {'column': 'occupancy_type_building_nsi', 'label': 'nsi'},
    {'column': 'group_footprint_fema', 'label': 'fema'},
    {'column': 'group_parcel', 'label': 'parcel'},
    {'column': 'occupancy_type_dwelling_overture', 'label': 'overture', 'weight': 0.5},
]


def _state(df: pd.DataFrame, **config_overrides) -> CurateState:
    config = {
        'class_map': 'occupancy-class-map.csv',
        'residential_classes': ['Single-Family', 'Multi-Family', 'Manufactured Home'],
        'secondary_class': 'Secondary',
        'evidence_mode': 'vote',
        'evidence': EVIDENCE,
        **config_overrides,
    }
    return CurateState(
        recipe={
            'entity': 'footprint-cheer-2026',
            'admin_id': 'US',
            'occupancy': config,
        },
        entity_recipe={},
        admin_id=AdminId('US'),
        verbose=False,
        timer=None,
        curated=df,
    )


def _frame(nsi=None, fema=None, parcel=None, overture=None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            'occupancy_type_building_nsi': [nsi],
            'group_footprint_fema': [fema],
            'group_parcel': [parcel],
            'occupancy_type_dwelling_overture': [overture],
        }
    )


def test_cascade_default_unchanged(monkeypatch):
    # Without evidence_mode, the first present evidence still wins outright,
    # even against two agreeing lower-priority sources.
    monkeypatch.setattr(occ_mod, 'load_ruleset', lambda s, r: CLASS_MAP)
    curated = _frame(
        fema='Multi Family', parcel='Single Family', overture='Single-Family'
    )
    out = impute_occupancy_type(_state(curated, evidence_mode='cascade')).curated
    assert out['occupancy_type'].astype(object).iloc[0] == 'Multi-Family'
    assert out['occupancy_type_source'].iloc[0] == 'fema'


def test_consensus_outvotes_lone_higher_ranked_source(monkeypatch):
    # The reported US-NC-CE case: fema Multi-Family (1.0) vs parcel + overture
    # Single-Family (1.5) -> the consensus wins and the source names both voters.
    monkeypatch.setattr(occ_mod, 'load_ruleset', lambda s, r: CLASS_MAP)
    curated = _frame(
        fema='Multi Family', parcel='Single Family', overture='Single-Family'
    )
    out = impute_occupancy_type(_state(curated)).curated
    assert out['occupancy_type'].astype(object).iloc[0] == 'Single-Family'
    assert out['occupancy_type_source'].iloc[0] == 'parcel/overture'


def test_two_real_sources_beat_parcel_overture_consensus(monkeypatch):
    # nsi + fema Manufactured Home (2.0) vs parcel + overture Single-Family
    # (1.5) -> two agreeing full-weight sources still win.
    monkeypatch.setattr(occ_mod, 'load_ruleset', lambda s, r: CLASS_MAP)
    curated = _frame(
        nsi='Manufactured Home',
        fema='Manufactured Home',
        parcel='Single Family',
        overture='Single-Family',
    )
    out = impute_occupancy_type(_state(curated)).curated
    assert out['occupancy_type'].astype(object).iloc[0] == 'Manufactured Home'
    assert out['occupancy_type_source'].iloc[0] == 'nsi/fema'


def test_tie_falls_to_earliest_listed_evidence(monkeypatch):
    # fema Multi-Family vs parcel Single-Family, equal weight, no overture ->
    # tie -> the earlier listed entry (fema) keeps precedence.
    monkeypatch.setattr(occ_mod, 'load_ruleset', lambda s, r: CLASS_MAP)
    curated = _frame(fema='Multi Family', parcel='Single Family')
    out = impute_occupancy_type(_state(curated)).curated
    assert out['occupancy_type'].astype(object).iloc[0] == 'Multi-Family'
    assert out['occupancy_type_source'].iloc[0] == 'fema'


def test_half_weight_overture_loses_one_on_one(monkeypatch):
    # fema (1.0) vs overture (0.5) -> real evidence stays.
    monkeypatch.setattr(occ_mod, 'load_ruleset', lambda s, r: CLASS_MAP)
    curated = _frame(fema='Multi Family', overture='Single-Family')
    out = impute_occupancy_type(_state(curated)).curated
    assert out['occupancy_type'].astype(object).iloc[0] == 'Multi-Family'
    assert out['occupancy_type_source'].iloc[0] == 'fema'


def test_overture_alone_still_fills_the_gap(monkeypatch):
    monkeypatch.setattr(occ_mod, 'load_ruleset', lambda s, r: CLASS_MAP)
    curated = _frame(overture='Single-Family')
    out = impute_occupancy_type(_state(curated)).curated
    assert out['occupancy_type'].astype(object).iloc[0] == 'Single-Family'
    assert out['occupancy_type_source'].iloc[0] == 'overture'


def test_non_residential_sources_pool_their_votes(monkeypatch):
    # 'Retail' and 'Warehouse' match no class-map rule -> both bucket to
    # Non-Residential (2.0) and outvote the parcel's Single-Family (1.0); the
    # concrete class comes from the first-listed winning voter (nsi's Retail).
    monkeypatch.setattr(occ_mod, 'load_ruleset', lambda s, r: CLASS_MAP)
    curated = _frame(nsi='Retail', fema='Warehouse', parcel='Single Family')
    out = impute_occupancy_type(_state(curated)).curated
    assert out['occupancy_type'].astype(object).iloc[0] == 'Retail'
    assert out['occupancy_type_source'].iloc[0] == 'nsi/fema'


def test_no_evidence_leaves_the_row_unclassified(monkeypatch):
    # This step now does one thing: the evidence vote. With all evidence null
    # it assigns nothing and says so, rather than reaching for a structural
    # fallback. Secondary is decided later, as a contested vote decision --
    # see test_footprint_occupancy_vote_regressions.
    monkeypatch.setattr(occ_mod, 'load_ruleset', lambda s, r: CLASS_MAP)
    curated = _frame()
    curated['priority_on_parcel'] = ['secondary']
    out = impute_occupancy_type(_state(curated)).curated
    assert out['occupancy_type'].isna().all()
