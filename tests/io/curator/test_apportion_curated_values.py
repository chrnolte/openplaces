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
    monkeypatch.setattr(evidence_mod, 'get_recipe_id', lambda recipe: 'entity-recipe')
    monkeypatch.setattr(
        evidence_mod, 'get_entity_link_path', lambda *a, **k: sidecar_path
    )
    monkeypatch.setattr(
        links_mod, '_resolve_reference_recipe', lambda *a, **k: ('parcel-ref', None)
    )


def _apportion(df, monkeypatch, tmp_path, sidecar=None, ref=None, **kwargs):
    _patch(
        monkeypatch,
        tmp_path,
        _sidecar() if sidecar is None else sidecar,
        _curated_parcels() if ref is None else ref,
    )
    return apportion_curated_values(
        _state(df),
        recipe_id='US_parcel-openplaces-2026',
        columns={
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
