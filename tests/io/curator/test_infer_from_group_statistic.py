"""Tests for the generic grouped-statistic inference step and the parcel-group
dwelling-candidate diagnostic (curate stage)."""

from __future__ import annotations

import pandas as pd
import pytest

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.diagnostics import _group_dwelling_candidates
from openplaces.io.curator.imputers import impute_from_group_statistic


def _state(df: pd.DataFrame) -> CurateState:
    return CurateState(
        recipe={'entity': 'footprint-openplaces-2026', 'admin_id': 'US'},
        entity_recipe={},
        admin_id=AdminId('US'),
        verbose=False,
        timer=None,
        curated=df,
    )


def test_mode_learns_majority_mapping():
    df = pd.DataFrame(
        {
            'g': ['res', 'res', 'res', 'com'],
            'v': ['SF', 'SF', 'MF', 'Retail'],
        }
    )
    state = impute_from_group_statistic(_state(df), 'g', 'v', 'out')
    assert state.curated['out'].tolist() == ['SF', 'SF', 'SF', 'Retail']


def test_mean_statistic():
    df = pd.DataFrame({'g': ['a', 'a', 'b'], 'v': [1.0, 3.0, 10.0]})
    state = impute_from_group_statistic(_state(df), 'g', 'v', 'out', statistic='mean')
    assert state.curated['out'].tolist() == [2.0, 2.0, 10.0]


def test_missing_column_declares_null_output():
    # A missing cohort input (e.g. no NSI coverage in this region) still
    # declares the output column, all-null, so downstream steps and
    # curated-reference readers keep their missing-column strictness.
    df = pd.DataFrame({'g': ['a']})
    state = impute_from_group_statistic(_state(df), 'g', 'absent', 'out')
    assert 'out' in state.curated.columns
    assert state.curated['out'].isna().all()


def test_unknown_statistic_raises():
    df = pd.DataFrame({'g': ['a'], 'v': ['x']})
    with pytest.raises(ValueError):
        impute_from_group_statistic(_state(df), 'g', 'v', 'out', statistic='bogus')


def test_overrides_win_and_base_fills_gaps(monkeypatch):
    # 'res' majority is SF; override corrects it to MF. 'mfg' has no value
    # evidence (base NaN) but the override supplies it. 'com' has no override and
    # keeps its learned value.
    df = pd.DataFrame(
        {
            'g': ['res', 'res', 'com', 'mfg'],
            'v': ['SF', 'SF', 'Retail', None],
        }
    )
    import openplaces.io.transform as transform

    monkeypatch.setattr(
        transform,
        'get_crosswalk',
        lambda spec: pd.Series({'res': 'MF', 'mfg': 'Mobile'}, name='out'),
    )
    state = impute_from_group_statistic(_state(df), 'g', 'v', 'out', overrides='x')
    assert state.curated['out'].tolist() == ['MF', 'MF', 'Retail', 'Mobile']


def test_blank_override_suppresses_statistic_instead_of_falling_back(monkeypatch):
    # 'woodland' majority statistic is Single Family, but its override is an
    # explicit blank (null) -- that must suppress the statistic, not be
    # ignored in favor of it. 'marsh' has no override at all and keeps its
    # learned statistic.
    df = pd.DataFrame(
        {
            'g': ['woodland', 'woodland', 'marsh', 'marsh'],
            'v': ['Single Family', 'Single Family', 'Commercial', 'Commercial'],
        }
    )
    import openplaces.io.transform as transform

    monkeypatch.setattr(
        transform,
        'get_crosswalk',
        lambda spec: pd.Series({'woodland': None}, name='out'),
    )
    state = impute_from_group_statistic(_state(df), 'g', 'v', 'out', overrides='x')
    out = state.curated['out']
    assert out.iloc[0:2].isna().all()
    assert out.iloc[2:4].tolist() == ['Commercial', 'Commercial']


def test_diagnostic_flags_low_dwelling_multifamily():
    curated = pd.DataFrame(
        {
            'use_group_combined_parcel': ['mf_bad', 'mf_bad', 'mf_ok', 'mf_ok', 'sf'],
            'group_parcel': [
                'Multi Family',
                'Multi Family',
                'Multi Family',
                'Multi Family',
                'Single Family',
            ],
            'n_dwellings_overture': [1.0, 1.0, 3.0, 3.0, 1.0],
        }
    )
    rules = [
        {
            'pattern': 'Multi',
            'match_type': 'contains',
            'occupancy_type': 'Multi-Family',
        },
        {
            'pattern': 'Single',
            'match_type': 'contains',
            'occupancy_type': 'Single-Family',
        },
    ]
    out = _group_dwelling_candidates(curated, rules, 'Multi-Family')
    flags = out.set_index('use_group_combined_parcel')['flag_multifamily_low_dwellings']
    assert bool(flags['mf_bad']) is True  # multi-family but mean 1.0 < 1.5
    assert bool(flags['mf_ok']) is False  # multi-family, mean 3.0
    assert bool(flags['sf']) is False  # not multi-family
