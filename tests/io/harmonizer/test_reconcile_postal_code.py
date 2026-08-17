"""Tests for the `reconcile_postal_code` harmonize step."""

from __future__ import annotations

import pandas as pd

from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer.addresses import reconcile_postal_code


def _state(df: pd.DataFrame, save_statistics: bool = False) -> HarmonizeState:
    return HarmonizeState(
        recipe={'entity': {'entity_type': 'footprint'}},
        admin_id='US-NC-CE',
        verbose=False,
        timer=None,
        spine=df,
        save_statistics=save_statistics,
    )


def test_higher_priority_source_wins_when_both_present():
    df = pd.DataFrame({'address_zip': ['27510'], 'zcta5_id': ['27511']})
    state = reconcile_postal_code(_state(df), sources=['address_zip', 'zcta5_id'])
    res = state.spine
    assert res.loc[0, 'postal_code'] == '27510'
    assert res.loc[0, 'postal_code_source'] == 'address_zip'


def test_lower_priority_source_fills_when_higher_missing():
    df = pd.DataFrame({'address_zip': [pd.NA], 'zcta5_id': ['27511']})
    state = reconcile_postal_code(_state(df), sources=['address_zip', 'zcta5_id'])
    res = state.spine
    assert res.loc[0, 'postal_code'] == '27511'
    assert res.loc[0, 'postal_code_source'] == 'zcta5_id'


def test_agreement_leaves_conflict_column_null():
    df = pd.DataFrame({'address_zip': ['27510'], 'zcta5_id': ['27510']})
    state = reconcile_postal_code(_state(df), sources=['address_zip', 'zcta5_id'])
    res = state.spine
    assert res.loc[0, 'postal_code'] == '27510'
    assert pd.isna(res.loc[0, 'postal_code_conflict'])


def test_disagreement_still_picks_higher_priority_but_flags_conflict():
    df = pd.DataFrame({'address_zip': ['27510'], 'zcta5_id': ['99999']})
    state = reconcile_postal_code(_state(df), sources=['address_zip', 'zcta5_id'])
    res = state.spine
    assert res.loc[0, 'postal_code'] == '27510'
    conflict = res.loc[0, 'postal_code_conflict']
    assert 'address_zip: 27510' in conflict
    assert 'zcta5_id: 99999' in conflict


def test_extracts_five_digit_zip_from_zip_plus_four():
    df = pd.DataFrame({'address_zip': ['27510-1234']})
    state = reconcile_postal_code(_state(df), sources=['address_zip'])
    assert state.spine.loc[0, 'postal_code'] == '27510'


def test_no_source_columns_present_is_noop():
    df = pd.DataFrame({'other_column': [1]})
    state = reconcile_postal_code(_state(df), sources=['address_zip', 'zcta5_id'])
    assert 'postal_code' not in state.spine.columns


def test_spine_none_is_noop():
    state = reconcile_postal_code(
        HarmonizeState(
            recipe={}, admin_id='US-NC-CE', verbose=False, timer=None, spine=None
        ),
        sources=['address_zip'],
    )
    assert state.spine is None


def test_save_statistics_writes_diagnostics_csvs(tmp_path, monkeypatch):
    import openplaces.path as path_module

    monkeypatch.setattr(
        path_module, 'cache_path', lambda *a, **k: tmp_path / k['filename']
    )

    df = pd.DataFrame({'address_zip': ['27510'], 'zcta5_id': ['99999']})
    reconcile_postal_code(
        _state(df, save_statistics=True), sources=['address_zip', 'zcta5_id']
    )

    assert (tmp_path / 'postal-code-agreement.csv').exists()
    assert (tmp_path / 'postal-code-conflicts.csv').exists()
    assert (tmp_path / 'postal-code-conflict-cases.csv').exists()


def test_save_statistics_false_writes_nothing(tmp_path, monkeypatch):
    import openplaces.path as path_module

    monkeypatch.setattr(
        path_module, 'cache_path', lambda *a, **k: tmp_path / k['filename']
    )

    df = pd.DataFrame({'address_zip': ['27510'], 'zcta5_id': ['99999']})
    reconcile_postal_code(_state(df), sources=['address_zip', 'zcta5_id'])

    assert list(tmp_path.iterdir()) == []
