"""Tests for NSI-first occupancy inference and the parcel-correction step.

occupancy_type is NSI-first (parcel fills gaps); reviewed parcel keywords override
the base; occupancy_type_conflict is a categorical summary with sources grouped
by unique value ('label1/label2: class | label3: class', see
_summarize_conflicts); occupancy_type_review flags a small nonzero
improvement-value share. Value- and dwelling-based class assignment is covered
by test_resolve_by_vote. All terminology comes from the recipe config.
"""

from __future__ import annotations

import pandas as pd
import pytest

import openplaces.io.curator.occupancy as occ_mod
import openplaces.io.curator.reconcilers as rec
import openplaces.path as op_path
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

OCC_CONFIG = {
    'class_map': 'occupancy-class-map.csv',
    'residential_classes': ['Single-Family', 'Multi-Family', 'Manufactured Home'],
    'secondary_class': 'Secondary',
    'evidence': [
        {'column': 'occupancy_type_building_nsi', 'label': 'nsi'},
        {'column': 'group_parcel', 'label': 'parcel'},
    ],
    'columns': {
        'improvement_value': 'improvement_value_parcel',
        'land_value': 'land_value_parcel',
        'n_dwellings': 'n_dwellings',
    },
    'rules': {
        'manufactured_home_value': {
            'zero_classifies': True,
            'review_max_ratio': 0.025,
            'class': 'Manufactured Home',
        },
        'multi_family_dwellings': {'min_dwellings': 2, 'class': 'Multi-Family'},
        'single_family_dwellings': {'max_dwellings': 1, 'class': 'Single-Family'},
    },
}


def _state(df: pd.DataFrame) -> CurateState:
    return CurateState(
        recipe={
            'entity': 'footprint-cheer-2026',
            'admin_id': 'US',
            'occupancy': OCC_CONFIG,
        },
        entity_recipe={},
        admin_id=AdminId('US'),
        verbose=False,
        timer=None,
        curated=df,
    )


def _patch(monkeypatch, tmp_path, keyword_rules):
    def fake_load(state, ruleset):
        # class-map vs keyword ruleset distinguished by filename
        return CLASS_MAP if 'class-map' in ruleset else keyword_rules

    monkeypatch.setattr(occ_mod, 'load_ruleset', fake_load)
    monkeypatch.setattr(
        op_path, 'reports_path', lambda *a, **k: tmp_path / 'occupancy-conflicts.csv'
    )


def _base_frame(**overrides) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            'use_group_combined_parcel': ['RES'],
            'group_parcel': ['Single Family'],
            'occupancy_type': ['Single-Family'],
            'occupancy_type_building_nsi': ['Single Family, 1 story'],
            'improvement_value_parcel': [100000.0],
            'land_value_parcel': [50000.0],
            'n_dwellings': [1.0],
        }
    )
    for k, v in overrides.items():
        df[k] = v
    return df


def test_impute_occupancy_type_is_nsi_first(monkeypatch):
    monkeypatch.setattr(occ_mod, 'load_ruleset', lambda s, r: CLASS_MAP)
    curated = pd.DataFrame(
        {
            'occupancy_type_building_nsi': ['Manufactured Home', None],
            'group_parcel': ['Single Family', 'Single Family'],
        }
    )
    out = impute_occupancy_type(_state(curated))
    # row 0: NSI present -> Manufactured Home (not the parcel Single-Family);
    # row 1: NSI absent -> parcel fills -> Single-Family.
    assert list(out.curated['occupancy_type'].astype(object)) == [
        'Manufactured Home',
        'Single-Family',
    ]


def test_conflict_is_categorical_summary_and_base_kept(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path, keyword_rules=[])
    curated = pd.DataFrame(
        {
            'use_group_combined_parcel': ['RES', 'RES2'],
            'group_parcel': ['Single Family', 'Single Family'],
            'occupancy_type': ['Manufactured Home', 'Single-Family'],
            'occupancy_type_building_nsi': [
                'Manufactured Home',
                'Single Family, 1 story',
            ],
            'improvement_value_parcel': [100000.0, 100000.0],
            'land_value_parcel': [50000.0, 50000.0],
            'n_dwellings': [1.0, 1.0],
        }
    )
    out = rec.resolve_occupancy(_state(curated), ruleset='kw.csv').curated
    # No corrections fire -> NSI base unchanged.
    assert list(out['occupancy_type'].astype(object)) == [
        'Manufactured Home',
        'Single-Family',
    ]
    conf = out['occupancy_type_conflict'].astype(object).tolist()
    assert conf[0] == 'nsi: Manufactured Home | parcel: Single-Family'
    assert pd.isna(conf[1])


def test_conflict_buckets_non_residential(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path, keyword_rules=[])
    curated = pd.DataFrame(
        {
            'use_group_combined_parcel': ['RES', 'COM'],
            'group_parcel': ['Single Family', 'Office'],
            'occupancy_type': ['Single-Family', 'Retail'],
            'occupancy_type_building_nsi': ['Retail', 'Warehouse'],
            'improvement_value_parcel': [100000.0, 100000.0],
            'land_value_parcel': [50000.0, 50000.0],
            'n_dwellings': [1.0, 1.0],
        }
    )
    out = rec.resolve_occupancy(_state(curated), ruleset='kw.csv').curated
    conf = out['occupancy_type_conflict'].astype(object).tolist()
    # Non-residential nsi vs residential parcel -> bucketed conflict.
    assert conf[0] == 'nsi: Non-Residential | parcel: Single-Family'
    # Two different non-residential categories collapse to one bucket, so they no
    # longer count as a conflict.
    assert pd.isna(conf[1])


def test_conflict_lists_all_present_sources(monkeypatch, tmp_path):
    # Three evidence sources: NSI + FEMA agree Single-Family, parcel says
    # Multi-Family -> the summary lists all three present sources. FEMA is selected
    # generically; occupancy_type_parcel must reflect the parcel (group_parcel),
    # not the FEMA column sitting at evidence index 1.
    _patch(monkeypatch, tmp_path, keyword_rules=[])
    config = {
        **OCC_CONFIG,
        'evidence': [
            {'column': 'occupancy_type_building_nsi', 'label': 'nsi'},
            {'column': 'occupancy_type_footprint_fema', 'label': 'fema'},
            {'column': 'group_parcel', 'label': 'parcel'},
        ],
    }
    curated = pd.DataFrame(
        {
            'use_group_combined_parcel': ['RES', 'RES'],
            'group_parcel': ['Multi Family', 'Single Family'],
            'occupancy_type_footprint_fema': ['Single Family', 'Single Family'],
            'occupancy_type': ['Single-Family', 'Single-Family'],
            'occupancy_type_building_nsi': ['Single Family', 'Single Family'],
            'improvement_value_parcel': [100000.0, 100000.0],
            'land_value_parcel': [50000.0, 50000.0],
            'n_dwellings': [1.0, 1.0],
        }
    )
    state = CurateState(
        recipe={
            'entity': 'footprint-cheer-2026',
            'admin_id': 'US',
            'occupancy': config,
        },
        entity_recipe={},
        admin_id=AdminId('US'),
        verbose=False,
        timer=None,
        curated=curated,
    )
    out = rec.resolve_occupancy(state, ruleset='kw.csv').curated
    conf = out['occupancy_type_conflict'].astype(object).tolist()
    # Row 0: parcel disagrees -> sources grouped by value (nsi + fema agree),
    # groups ordered by first-appearing label in recipe order.
    assert conf[0] == 'nsi/fema: Single-Family | parcel: Multi-Family'
    # Row 1: all agree -> null.
    assert pd.isna(conf[1])
    # occupancy_type_parcel reflects the parcel evidence (group_parcel), not FEMA.
    assert out['occupancy_type_parcel'].astype(object).tolist() == [
        'Multi-Family',
        'Single-Family',
    ]


def test_reviewed_keyword_overrides_nsi(monkeypatch, tmp_path):
    _patch(
        monkeypatch,
        tmp_path,
        keyword_rules=[
            {
                'pattern': 'MANUFACTURED',
                'match_type': 'contains',
                'occupancy_type': 'Manufactured Home',
                'reviewed': True,
            }
        ],
    )
    curated = _base_frame(use_group_combined_parcel=['MANUFACTURED HOUSING'])
    out = rec.resolve_occupancy(_state(curated), ruleset='kw.csv').curated
    assert out['occupancy_type'].iloc[0] == 'Manufactured Home'


def test_zero_improvement_keeps_base_and_no_review_flag(monkeypatch, tmp_path):
    # resolve_occupancy no longer classes zero-value as Manufactured Home (that
    # is now a resolve_by_vote indicator); the base class is kept and, since
    # the review flag is for nonzero shares only, it stays False.
    _patch(monkeypatch, tmp_path, keyword_rules=[])
    curated = _base_frame(improvement_value_parcel=[0.0])
    out = rec.resolve_occupancy(_state(curated), ruleset='kw.csv').curated
    assert out['occupancy_type'].iloc[0] == 'Single-Family'
    assert bool(out['occupancy_type_review'].iloc[0]) is False


def test_low_improvement_ratio_sets_review_flag(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path, keyword_rules=[])
    # 1000 / (1000 + 50000) = 0.0196 < 0.025 -> review, no class change.
    curated = _base_frame(improvement_value_parcel=[1000.0])
    out = rec.resolve_occupancy(_state(curated), ruleset='kw.csv').curated
    assert bool(out['occupancy_type_review'].iloc[0]) is True
    assert out['occupancy_type'].iloc[0] == 'Single-Family'


def test_dwellings_do_not_override_in_resolve_occupancy(monkeypatch, tmp_path):
    # n_dwellings>=2 multi-family assignment moved to resolve_by_vote; here the
    # base class must be left untouched.
    _patch(monkeypatch, tmp_path, keyword_rules=[])
    curated = _base_frame(n_dwellings=[3.0])
    out = rec.resolve_occupancy(_state(curated), ruleset='kw.csv').curated
    assert out['occupancy_type'].iloc[0] == 'Single-Family'


def test_missing_occupancy_config_columns_noop(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path, keyword_rules=[])
    curated = pd.DataFrame({'use_group_combined_parcel': ['RES']})  # no occupancy_type
    out = rec.resolve_occupancy(_state(curated), ruleset='kw.csv').curated
    assert 'occupancy_type_conflict' not in out.columns


def test_unknown_class_passes_through(monkeypatch):
    # Non-residential NSI label has no class-map rule -> kept verbatim.
    monkeypatch.setattr(occ_mod, 'load_ruleset', lambda s, r: CLASS_MAP)
    curated = pd.DataFrame(
        {'occupancy_type_building_nsi': ['Retail'], 'group_parcel': [None]}
    )
    out = impute_occupancy_type(_state(curated))
    assert out.curated['occupancy_type'].iloc[0] == 'Retail'


def test_secondary_priority_without_evidence_is_secondary(monkeypatch):
    # A non-primary footprint with no occupancy evidence is an accessory
    # structure -> Secondary, even though no class-map rule matched.
    monkeypatch.setattr(occ_mod, 'load_ruleset', lambda s, r: CLASS_MAP)
    curated = pd.DataFrame(
        {
            'occupancy_type_building_nsi': [None],
            'group_parcel': [None],
            'priority_on_parcel': ['secondary'],
        }
    )
    out = impute_occupancy_type(_state(curated)).curated
    assert out['occupancy_type'].astype(object).iloc[0] == 'Secondary'
    assert out['occupancy_type_source'].iloc[0] == 'secondary'


@pytest.mark.parametrize('improvement', [0.0, 1000.0])
def test_review_flag_never_true_for_zero_improvement(
    monkeypatch, tmp_path, improvement
):
    _patch(monkeypatch, tmp_path, keyword_rules=[])
    out = rec.resolve_occupancy(
        _state(_base_frame(improvement_value_parcel=[improvement])), ruleset='kw.csv'
    ).curated
    if improvement == 0.0:
        assert bool(out['occupancy_type_review'].iloc[0]) is False
