"""Tests for `link_curated_entity` (formerly `link_curated_parcels`).

Both default keys are the bare `parcel_id`: an id column (`{entity}_id*`)
never carries the usual `_{source}`/`_{entity}_{source}` provenance suffix
other cross-attributed columns get, since it already names the row it
identifies (see `_attributed_name` in the harmonizer) -- so the current
entity's cross-attributed parcel reference (`parcel_id`, written by the
harmonizer) matches the referenced entity's own key (also `parcel_id`)
without any suffix on either side. The globally-unique `parcel_id` (geo_id)
is used rather than `parcel_id_local`: the latter is only locally
cross-comparable and can collide within a single admin unit, which would
silently misattribute the joined columns to an unrelated parcel.
"""

from __future__ import annotations

import pandas as pd
import pytest

import openplaces.io.curator.evidence as evidence_mod
from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.evidence import link_curated_entity


def _state(df: pd.DataFrame) -> CurateState:
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US'),
        verbose=False,
        timer=None,
        curated=df,
    )


def _patch(monkeypatch, ref_df: pd.DataFrame, stage: str = 'curate'):
    monkeypatch.setattr(
        evidence_mod, 'get_recipe_by_id', lambda recipe_id: {'stage': stage}
    )
    monkeypatch.setattr(evidence_mod, 'get_output_path', lambda *a, **k: 'fake/path')
    monkeypatch.setattr(evidence_mod, 'read_parquet', lambda path: ref_df)


def test_joins_on_default_bare_id_key(monkeypatch):
    entity = pd.DataFrame({'parcel_id': ['p1', 'p2', None]})
    ref = pd.DataFrame({'parcel_id': ['p1', 'p2'], 'group_parcel': ['A', 'B']})
    _patch(monkeypatch, ref)

    out = link_curated_entity(
        _state(entity), recipe_id='ref', columns={'group_parcel': 'group_parcel'}
    ).curated

    assert out['group_parcel'].iloc[0] == 'A'
    assert out['group_parcel'].iloc[1] == 'B'
    assert pd.isna(out['group_parcel'].iloc[2])


def test_missing_ref_key_raises(monkeypatch):
    entity = pd.DataFrame({'parcel_id': ['p1']})
    ref = pd.DataFrame({'other_id': ['p1'], 'group_parcel': ['WRONG']})
    _patch(monkeypatch, ref)

    with pytest.raises(ValueError, match="no 'parcel_id' column"):
        link_curated_entity(
            _state(entity), recipe_id='ref', columns={'group_parcel': 'group_parcel'}
        )


def test_ref_key_as_index_is_reset(monkeypatch):
    # The curated reference's own key is typically its DataFrame index
    # (e.g. a parcel entity indexed by parcel_id), not a plain column.
    entity = pd.DataFrame({'parcel_id': ['p1', 'p2']})
    ref = pd.DataFrame(
        {'parcel_id': ['p1', 'p2'], 'group_parcel': ['A', 'B']}
    ).set_index('parcel_id')
    _patch(monkeypatch, ref)

    out = link_curated_entity(
        _state(entity), recipe_id='ref', columns={'group_parcel': 'group_parcel'}
    ).curated

    assert out['group_parcel'].tolist() == ['A', 'B']


def test_missing_entity_key_skips(monkeypatch):
    entity = pd.DataFrame({'other': [1]})
    ref = pd.DataFrame({'parcel_id': ['p1'], 'group_parcel': ['A']})
    _patch(monkeypatch, ref)

    out = link_curated_entity(
        _state(entity), recipe_id='ref', columns={'group_parcel': 'group_parcel'}
    ).curated
    assert 'group_parcel' not in out.columns


def test_custom_keys(monkeypatch):
    entity = pd.DataFrame({'building_id': ['b1', 'b2']})
    ref = pd.DataFrame({'nsi_id': ['b1', 'b2'], 'occupancy_type': ['SF', 'MF']})
    _patch(monkeypatch, ref)

    out = link_curated_entity(
        _state(entity),
        recipe_id='ref',
        columns={'occupancy_type': 'occupancy_type_nsi'},
        entity_key='building_id',
        ref_key='nsi_id',
    ).curated
    assert out['occupancy_type_nsi'].tolist() == ['SF', 'MF']


def test_requires_curate_stage(monkeypatch):
    entity = pd.DataFrame({'parcel_id': ['p1']})
    ref = pd.DataFrame({'parcel_id': ['p1']})
    _patch(monkeypatch, ref, stage='harmonize')

    with pytest.raises(ValueError, match="must have stage 'curate'"):
        link_curated_entity(_state(entity), recipe_id='ref', columns={})
