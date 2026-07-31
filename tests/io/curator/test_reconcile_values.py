"""Tests for `reconcile_values`."""

from __future__ import annotations

import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.reconcilers import reconcile_values


def _state(df: pd.DataFrame) -> CurateState:
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US'),
        verbose=False,
        timer=None,
        curated=df,
    )


def test_first_non_null_source_wins_in_priority_order():
    df = pd.DataFrame(
        {
            'value_parcel': [None, 5.0],
            'value_nsi': [10.0, 20.0],
        }
    )
    out = reconcile_values(
        _state(df), priority={'value': ['value_parcel', 'value_nsi']}
    ).curated
    assert out['value'].tolist() == [10.0, 5.0]


def test_missing_source_column_is_skipped_not_raised():
    df = pd.DataFrame({'value_parcel': [1.0]})
    out = reconcile_values(
        _state(df), priority={'value': ['value_parcel', 'value_missing']}
    ).curated
    assert out['value'].tolist() == [1.0]


def test_provenance_token_is_registered_suffix_for_normal_source():
    df = pd.DataFrame({'value_parcel': [1.0]})
    out = reconcile_values(_state(df), priority={'value': ['value_parcel']}).curated
    assert out['value_source'].iloc[0] == 'parcel'


def test_provenance_token_is_imputed_not_suffix_for_imputed_source():
    # land_value_imputed_parcel ends in the registered '_parcel' suffix, but
    # its winning-token must read 'imputed' (the real/imputed distinction),
    # not 'parcel' (which would be indistinguishable from a genuine
    # land_value_parcel win).
    df = pd.DataFrame(
        {
            'land_value_parcel': [None],
            'land_value_imputed_parcel': [42_000.0],
        }
    )
    out = reconcile_values(
        _state(df),
        priority={'land_value': ['land_value_parcel', 'land_value_imputed_parcel']},
    ).curated
    assert out['land_value'].iloc[0] == 42_000.0
    assert out['land_value_source'].iloc[0] == 'imputed'


def test_real_source_still_wins_over_imputed_when_present():
    df = pd.DataFrame(
        {
            'land_value_parcel': [100_000.0],
            'land_value_imputed_parcel': [42_000.0],
        }
    )
    out = reconcile_values(
        _state(df),
        priority={'land_value': ['land_value_parcel', 'land_value_imputed_parcel']},
    ).curated
    assert out['land_value'].iloc[0] == 100_000.0
    assert out['land_value_source'].iloc[0] == 'parcel'


def test_imputed_source_wins_when_listed_first():
    # Generic mechanism test: a source listed first in priority wins via
    # bfill regardless of which of the two columns is "real" vs. "imputed" --
    # order alone decides. (US_footprint-cheer-2026 no longer reconciles
    # improvement_value this way -- land_value.py now computes the
    # real-or-imputed coalesce directly, and apportion_curated_values writes
    # the footprint's bare land_value/improvement_value straight from that;
    # this test just pins down reconcile_values' own priority-order
    # semantics in isolation.)
    df = pd.DataFrame(
        {
            'improvement_value_parcel': [500_000.0],
            'improvement_value_imputed_parcel': [380_000.0],
        }
    )
    out = reconcile_values(
        _state(df),
        priority={
            'improvement_value': [
                'improvement_value_imputed_parcel',
                'improvement_value_parcel',
            ]
        },
    ).curated
    assert out['improvement_value'].iloc[0] == 380_000.0
    assert out['improvement_value_source'].iloc[0] == 'imputed'
