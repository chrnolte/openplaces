"""Tests for `apportion_curated_values` (curate-stage parcel value attribution).

The step re-uses the n:m footprint-parcel link sidecar persisted by the
harmonize overlay (save_link: true) and the shared apportionment
implementation to distribute the *curated* parcel's values across footprints
with the same semantics the harmonize stage applies to raw parcel values —
instead of re-attaching undivided parcel totals by a dominant-parcel id-join.
"""

from __future__ import annotations

import pandas as pd
import pytest

import openplaces.io.curator.evidence as evidence_mod
import openplaces.io.harmonizer.links as links_mod
from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.evidence import apportion_curated_values


def _state(df: pd.DataFrame) -> CurateState:
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US'),
        verbose=False,
        timer=None,
        curated=df,
    )


def _curated_footprints():
    # F1 alone on P1; F2 + F3 share P2 (overlaps 100/300); F4 is a synthetic
    # parcel-fallback for P3 (postdates the sidecar); F5 is unlinked.
    return pd.DataFrame(
        {
            'parcel_id': ['P1', 'P2', 'P2', 'P3', None],
            'priority_on_parcel': [
                'primary',
                'primary',
                'primary',
                'primary',
                'unknown',
            ],
            'n_dwellings_overture': [0.0, 0.0, 0.0, 0.0, 0.0],
            'geometry_source': ['obm', 'obm', 'microsoft', 'parcel.spine', 'obm'],
        },
        index=pd.Index(['F1', 'F2', 'F3', 'F4', 'F5'], name='footprint_id'),
    )


def _sidecar():
    # The persisted overlay: link-labeled pairs plus one sub-threshold sliver
    # (link missing) that must not participate. F4/P3 is absent (synthetic).
    return pd.DataFrame(
        {
            'footprint_id': ['F1', 'F2', 'F3', 'F1'],
            'parcel_id': ['P1', 'P2', 'P2', 'P9'],
            'area_intersection_m2': [120.0, 100.0, 300.0, 0.5],
            'link': ['ok', 'ok', 'ok', None],
        }
    )


def _curated_parcels():
    return pd.DataFrame(
        {
            'parcel_id': ['P1', 'P2', 'P3', 'P9'],
            'improvement_value': [100000.0, 400000.0, 70000.0, 999999.0],
            'land_value': [30000.0, 50000.0, 10000.0, 888888.0],
            'land_value_imputed': [12000.0, 20000.0, 4000.0, 355555.0],
            'improvement_value_imputed': [88000.0, 380000.0, 66000.0, 644444.0],
            'land_use_class': [
                'Single-Family',
                'Commercial',
                'Single-Family',
                'Commercial',
            ],
        }
    )


def _patch(monkeypatch, tmp_path, sidecar: pd.DataFrame, ref: pd.DataFrame):
    sidecar_path = tmp_path / 'link.parquet'
    sidecar.to_parquet(sidecar_path)
    monkeypatch.setattr(
        evidence_mod, 'get_recipe_by_id', lambda recipe_id: {'stage': 'curate'}
    )
    monkeypatch.setattr(evidence_mod, 'get_output_path', lambda *a, **k: 'fake/path')
    monkeypatch.setattr(evidence_mod, 'read_parquet', lambda path: ref)
    # The readers resolve the sidecar's owner through the entity_recipe
    # chain (the geospine under the split); stub the resolution outright.
    monkeypatch.setattr(
        evidence_mod, 'get_link_owner_recipe_id', lambda recipe: 'entity-recipe'
    )
    monkeypatch.setattr(
        evidence_mod, 'get_entity_link_path', lambda *a, **k: sidecar_path
    )
    monkeypatch.setattr(
        links_mod, '_resolve_reference_recipe', lambda *a, **k: ('parcel-ref', None)
    )


def _apportion(
    df, monkeypatch, tmp_path, sidecar=None, ref=None, columns=None, **kwargs
):
    _patch(
        monkeypatch,
        tmp_path,
        _sidecar() if sidecar is None else sidecar,
        _curated_parcels() if ref is None else ref,
    )
    return apportion_curated_values(
        _state(df),
        recipe_id='US_parcel-openplaces-2026',
        columns=columns
        or {
            'improvement_value': 'improvement_value_parcel',
            'land_value': 'land_value_parcel',
        },
        **kwargs,
    ).curated


def test_sole_footprint_gets_full_parcel_value(monkeypatch, tmp_path):
    out = _apportion(_curated_footprints(), monkeypatch, tmp_path)
    assert out.loc['F1', 'improvement_value_parcel'] == 100000.0
    assert out.loc['F1', 'land_value_parcel'] == 30000.0


def test_shared_parcel_splits_by_overlap_share(monkeypatch, tmp_path):
    out = _apportion(_curated_footprints(), monkeypatch, tmp_path)
    assert out.loc['F2', 'improvement_value_parcel'] == 100000.0  # 100/400
    assert out.loc['F3', 'improvement_value_parcel'] == 300000.0  # 300/400


def test_synthetic_fallback_row_gets_its_parcels_value(monkeypatch, tmp_path):
    # F4 postdates the persisted overlay (infer_spine_additions), so it is
    # linked through its own parcel_id column instead, at full weight.
    out = _apportion(_curated_footprints(), monkeypatch, tmp_path)
    assert out.loc['F4', 'improvement_value_parcel'] == 70000.0
    assert out.loc['F4', 'land_value_parcel'] == 10000.0


def test_sub_threshold_sliver_pairs_do_not_participate(monkeypatch, tmp_path):
    # F1's sliver overlap with P9 has no link label: P9's value must not
    # leak into F1 (which would add 999999 to its total).
    out = _apportion(_curated_footprints(), monkeypatch, tmp_path)
    assert out.loc['F1', 'improvement_value_parcel'] == 100000.0


def test_unlinked_footprint_gets_missing_values(monkeypatch, tmp_path):
    out = _apportion(_curated_footprints(), monkeypatch, tmp_path)
    assert pd.isna(out.loc['F5', 'improvement_value_parcel'])
    assert pd.isna(out.loc['F5', 'land_value_parcel'])


def test_secondary_footprint_gets_missing_improvement_value(monkeypatch, tmp_path):
    df = _curated_footprints()
    df.loc['F3', 'priority_on_parcel'] = 'secondary'
    out = _apportion(df, monkeypatch, tmp_path)
    assert pd.isna(out.loc['F3', 'improvement_value_parcel'])
    assert pd.isna(out.loc['F3', 'land_value_parcel'])
    # F2 keeps only its own overlap share; exclusion of the secondary from
    # receiving does not re-inflate the primary's share.
    assert out.loc['F2', 'improvement_value_parcel'] == 100000.0


def test_dwelling_linked_footprint_takes_whole_parcel_value(monkeypatch, tmp_path):
    df = _curated_footprints()
    df.loc['F2', 'n_dwellings_overture'] = 1.0
    out = _apportion(df, monkeypatch, tmp_path)
    assert out.loc['F2', 'improvement_value_parcel'] == 400000.0
    assert out.loc['F3', 'improvement_value_parcel'] == 0.0
    assert pd.isna(out.loc['F3', 'land_value_parcel'])


def test_unknown_column_semantics_raise():
    # Raised before any IO: dominant-reference attributes belong to
    # link_curated_entity, not this step.
    with pytest.raises(ValueError, match='no apportionment semantics'):
        apportion_curated_values(
            _state(_curated_footprints()),
            recipe_id='ref',
            columns={'use_group_combined': 'use_group_combined_parcel'},
        )


def test_equal_area_non_residential_secondary_footprint_keeps_its_share(
    monkeypatch, tmp_path
):
    # P2 is 'Commercial' (non-residential): F3, marked secondary, must now
    # keep its overlap-area share of improvement_value instead of the
    # residential-style secondary masking test_secondary_footprint_gets_
    # missing_improvement_value exercises.
    df = _curated_footprints()
    df.loc['F3', 'priority_on_parcel'] = 'secondary'
    out = _apportion(
        df,
        monkeypatch,
        tmp_path,
        land_use_column='land_use_class',
        non_residential_classes=['Commercial'],
    )
    assert out.loc['F2', 'improvement_value_parcel'] == 100000.0  # 100/400
    assert out.loc['F3', 'improvement_value_parcel'] == 300000.0  # 300/400
    # land_value is unaffected by the non-residential equal-area rule.
    assert pd.isna(out.loc['F3', 'land_value_parcel'])


def test_equal_area_config_does_not_affect_residential_parcels(monkeypatch, tmp_path):
    # P1 ('Single-Family') is not in non_residential_classes, so its lone
    # footprint's behavior is unchanged.
    out = _apportion(
        _curated_footprints(),
        monkeypatch,
        tmp_path,
        land_use_column='land_use_class',
        non_residential_classes=['Commercial'],
    )
    assert out.loc['F1', 'improvement_value_parcel'] == 100000.0


def test_missing_land_use_column_skips_equal_area_rule_without_raising(
    monkeypatch, tmp_path
):
    df = _curated_footprints()
    df.loc['F3', 'priority_on_parcel'] = 'secondary'
    out = _apportion(
        df,
        monkeypatch,
        tmp_path,
        land_use_column='not_a_real_column',
        non_residential_classes=['Commercial'],
    )
    # Falls back to ordinary (non-equal-area) secondary masking.
    assert pd.isna(out.loc['F3', 'improvement_value_parcel'])


_IMPUTED_COLUMNS = {
    'land_value_imputed': 'land_value_imputed_parcel',
    'improvement_value_imputed': 'improvement_value_imputed_parcel',
}


def test_improvement_value_imputed_splits_like_improvement_value(monkeypatch, tmp_path):
    # improvement_value_imputed keeps improvement_value's own area-proportional
    # split treatment (PROPORTIONAL_SPLIT_COLUMNS).
    out = _apportion(
        _curated_footprints(), monkeypatch, tmp_path, columns=_IMPUTED_COLUMNS
    )
    assert out.loc['F1', 'improvement_value_imputed_parcel'] == 88000.0
    assert out.loc['F2', 'improvement_value_imputed_parcel'] == 95000.0
    assert out.loc['F3', 'improvement_value_imputed_parcel'] == 285000.0


def test_land_value_imputed_whole_on_principal_footprint(monkeypatch, tmp_path):
    # land_value_imputed is conceptually "land_value, gap-filled" -- it gets
    # land_value's own whole-on-principal-footprint-only treatment
    # (WHOLE_VALUE_COLUMNS), not improvement_value's proportional split.
    df = _curated_footprints()
    df.loc['F3', 'priority_on_parcel'] = 'secondary'
    out = _apportion(df, monkeypatch, tmp_path, columns=_IMPUTED_COLUMNS)
    assert out.loc['F1', 'land_value_imputed_parcel'] == 12000.0  # sole on P1
    assert out.loc['F2', 'land_value_imputed_parcel'] == 20000.0  # whole value
    assert pd.isna(out.loc['F3', 'land_value_imputed_parcel'])  # secondary


def test_imputed_value_columns_respect_secondary_masking_by_default(
    monkeypatch, tmp_path
):
    df = _curated_footprints()
    df.loc['F3', 'priority_on_parcel'] = 'secondary'
    out = _apportion(df, monkeypatch, tmp_path, columns=_IMPUTED_COLUMNS)
    assert pd.isna(out.loc['F3', 'land_value_imputed_parcel'])
    assert pd.isna(out.loc['F3', 'improvement_value_imputed_parcel'])


def test_imputed_value_columns_map_directly_to_bare_footprint_columns(
    monkeypatch, tmp_path
):
    # Mirrors the actual recipe: land_value_imputed/improvement_value_imputed
    # apportion straight into the footprint's own bare land_value/
    # improvement_value -- these become the canonical, final columns, with no
    # further reconciliation step needed.
    out = _apportion(
        _curated_footprints(),
        monkeypatch,
        tmp_path,
        columns={
            'land_value_imputed': 'land_value',
            'improvement_value_imputed': 'improvement_value',
        },
    )
    assert out.loc['F1', 'land_value'] == 12000.0
    assert out.loc['F1', 'improvement_value'] == 88000.0


def test_default_params_leave_existing_improvement_value_land_value_behavior_unchanged(
    monkeypatch, tmp_path
):
    # No land_use_column/non_residential_classes passed: identical to the
    # pre-existing behavior exercised by the tests above this section.
    out = _apportion(_curated_footprints(), monkeypatch, tmp_path)
    assert out.loc['F1', 'improvement_value_parcel'] == 100000.0
    assert out.loc['F1', 'land_value_parcel'] == 30000.0


def test_missing_sidecar_raises(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path, _sidecar(), _curated_parcels())
    monkeypatch.setattr(
        evidence_mod,
        'get_entity_link_path',
        lambda *a, **k: tmp_path / 'absent.parquet',
    )
    with pytest.raises(FileNotFoundError, match='save_link'):
        apportion_curated_values(
            _state(_curated_footprints()),
            recipe_id='ref',
            columns={'improvement_value': 'improvement_value_parcel'},
        )
