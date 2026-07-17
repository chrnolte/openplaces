"""Tests for `reconcile_land_use` (default land-use class by evidence vote).

The three group-vocabulary columns each cast one vote; ties break to
group_parcel when it voted for a tied value, else to the earliest listed
column. The winner is coarsened through the class map and fills only rows the
rule-based classifier left null. Disagreements are summarized in
land_use_class_conflict (grouped format, see _summarize_conflicts) and their
most frequent combinations are saved to the reports directory.
"""

from __future__ import annotations

import pandas as pd

import openplaces.io.transform as transform_mod
import openplaces.path as op_path
from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.reconcilers import reconcile_land_use

CLASS_MAP = pd.Series(
    {
        'Single Family': 'Single-Family',
        'Multi Family': 'Multi-Family',
        'Manufactured Home': 'Manufactured Home',
        'Retail': 'Retail',
    }
)

COLUMNS = [
    {'column': 'group_parcel', 'label': 'parcel'},
    {'column': 'group_building_nsi', 'label': 'nsi'},
    {'column': 'group_footprint_fema', 'label': 'fema'},
]


def _state(df: pd.DataFrame) -> CurateState:
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US'),
        verbose=False,
        timer=None,
        curated=df,
    )


def _patch(monkeypatch, tmp_path):
    monkeypatch.setattr(
        transform_mod, 'get_crosswalk', lambda crosswalk_dict: CLASS_MAP
    )
    monkeypatch.setattr(
        op_path, 'reports_path', lambda *a, **k: tmp_path / 'land-use-conflicts.csv'
    )


def _reconcile(df, monkeypatch, tmp_path, **kwargs):
    _patch(monkeypatch, tmp_path)
    return reconcile_land_use(
        _state(df),
        columns=COLUMNS,
        class_map_id='class-map',
        report='land-use-conflicts.csv',
        **kwargs,
    ).curated


def test_majority_vote_wins_over_tiebreaker(monkeypatch, tmp_path):
    # nsi + fema agree against group_parcel: majority wins, parcel does NOT
    # (it is only the tie-breaker, not a strict priority).
    df = pd.DataFrame(
        {
            'group_parcel': ['Manufactured Home'],
            'group_building_nsi': ['Single Family'],
            'group_footprint_fema': ['Single Family'],
        }
    )
    out = _reconcile(df, monkeypatch, tmp_path)
    assert out['land_use_class'].iloc[0] == 'Single-Family'
    assert out['land_use_class_source'].iloc[0] == 'nsi/fema'


def test_two_way_tie_breaks_to_group_parcel(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {
            'group_parcel': ['Manufactured Home'],
            'group_building_nsi': ['Single Family'],
            'group_footprint_fema': [None],
        }
    )
    out = _reconcile(df, monkeypatch, tmp_path)
    assert out['land_use_class'].iloc[0] == 'Manufactured Home'
    assert out['land_use_class_source'].iloc[0] == 'parcel'


def test_all_distinct_breaks_to_group_parcel(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {
            'group_parcel': ['Retail'],
            'group_building_nsi': ['Single Family'],
            'group_footprint_fema': ['Manufactured Home'],
        }
    )
    out = _reconcile(df, monkeypatch, tmp_path)
    assert out['land_use_class'].iloc[0] == 'Retail'


def test_tie_without_tiebreaker_falls_to_earliest_listed_column(monkeypatch, tmp_path):
    # group_parcel is missing; nsi vs fema disagree 1-1 -> nsi (listed first).
    df = pd.DataFrame(
        {
            'group_parcel': [None],
            'group_building_nsi': ['Multi Family'],
            'group_footprint_fema': ['Single Family'],
        }
    )
    out = _reconcile(df, monkeypatch, tmp_path)
    assert out['land_use_class'].iloc[0] == 'Multi-Family'
    assert out['land_use_class_source'].iloc[0] == 'nsi'


def test_missing_values_cast_no_vote(monkeypatch, tmp_path):
    # Only fema present (e.g. NSI/parcel evidence absent; FEMA 'Unclassified'
    # arrives here as missing already, via the ingest-time group remap).
    df = pd.DataFrame(
        {
            'group_parcel': [None, None],
            'group_building_nsi': [None, None],
            'group_footprint_fema': ['Single Family', None],
        }
    )
    out = _reconcile(df, monkeypatch, tmp_path)
    assert out['land_use_class'].iloc[0] == 'Single-Family'
    assert pd.isna(out['land_use_class'].iloc[1])


def test_rule_assigned_rows_are_not_overwritten(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {
            'land_use_class': pd.Categorical(['Manufactured Home Park', None]),
            'group_parcel': ['Single Family', 'Single Family'],
            'group_building_nsi': ['Single Family', 'Single Family'],
            'group_footprint_fema': [None, None],
        }
    )
    out = _reconcile(df, monkeypatch, tmp_path)
    assert out['land_use_class'].iloc[0] == 'Manufactured Home Park'
    assert out['land_use_class'].iloc[1] == 'Single-Family'


def test_winning_group_missing_from_class_map_leaves_row_unfilled(
    monkeypatch, tmp_path
):
    df = pd.DataFrame(
        {
            'group_parcel': ['Garage'],  # not in the (test) class map
            'group_building_nsi': [None],
            'group_footprint_fema': [None],
        }
    )
    out = _reconcile(df, monkeypatch, tmp_path)
    assert pd.isna(out['land_use_class'].iloc[0])
    assert 'land_use_class_source' not in out.columns


def test_conflict_column_groups_sources_by_value(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {
            'group_parcel': ['Single Family', 'Single Family'],
            'group_building_nsi': ['Single Family', 'Single Family'],
            'group_footprint_fema': ['Manufactured Home', 'Single Family'],
        }
    )
    out = _reconcile(df, monkeypatch, tmp_path)
    conflict = out['land_use_class_conflict'].astype(object)
    assert conflict.iloc[0] == 'parcel/nsi: Single Family | fema: Manufactured Home'
    assert pd.isna(conflict.iloc[1])  # agreement is not a conflict


def test_conflict_report_counts_sorted_desc(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {
            'group_parcel': ['Single Family'] * 3,
            'group_building_nsi': [
                'Manufactured Home',
                'Manufactured Home',
                'Multi Family',
            ],
            'group_footprint_fema': [None] * 3,
        }
    )
    _reconcile(df, monkeypatch, tmp_path)
    report = pd.read_csv(tmp_path / 'land-use-conflicts.csv')
    assert list(report.columns) == ['land_use_class_conflict', 'count']
    assert report['count'].tolist() == [2, 1]
    assert (
        report['land_use_class_conflict'].iloc[0]
        == 'parcel: Single Family | nsi: Manufactured Home'
    )


def test_no_evidence_columns_is_a_noop(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    df = pd.DataFrame({'other': [1]})
    out = reconcile_land_use(_state(df), columns=COLUMNS).curated
    assert 'land_use_class' not in out.columns
