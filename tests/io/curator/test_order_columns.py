"""Tests for the order_columns curate step (_all adjacency)."""

import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.formatters import order_columns


def _state(columns):
    curated = pd.DataFrame(
        {col: ['x'] for col in columns},
        index=pd.Index(['f1'], name='footprint_id'),
    )
    return CurateState(
        recipe={'recipe_id': 'US_footprint-cheer-2026'},
        entity_recipe={},
        admin_id=AdminId('US-NC-BR'),
        verbose=False,
        timer=None,
        curated=curated,
    )


def test_all_column_follows_its_base():
    # collect_link_ids appends parcel_id_all last; ordering must still place
    # it directly behind parcel_id, like every other _all column.
    state = order_columns(
        _state(
            [
                'parcel_id',
                'parcel_id_local',
                'parcel_id_local_all',
                'geometry',
                'parcel_id_all',
            ]
        )
    )
    assert list(state.curated.columns) == [
        'parcel_id',
        'parcel_id_all',
        'parcel_id_local',
        'parcel_id_local_all',
        'geometry',
    ]


def test_all_without_base_untouched():
    # An _all column with no base present falls back to the existing rules
    # and must not crash. It stays in the canonical block (0), which sorts
    # ahead of parcel_id (now leading the parcel group in the source block).
    state = order_columns(_state(['parcel_id', 'other_all']))
    assert list(state.curated.columns) == ['other_all', 'parcel_id']


def test_conflict_column_follows_its_source_sidecar():
    # address_conflict must sort directly behind the address_source sidecar
    # (band 0.5), not stay in the canonical block beside address.
    state = order_columns(
        _state(
            [
                'address_conflict',
                'year_built',
                'address',
                'geometry',
                'year_built_source',
                'address_source',
            ]
        )
    )
    assert list(state.curated.columns) == [
        'year_built',
        'address',
        'year_built_source',
        'address_source',
        'address_conflict',
        'geometry',
    ]


def test_flag_conflict_columns_keep_their_flag_slot():
    # occupancy_type_conflict has an explicit _FLAG_COLUMNS slot (block 2) and
    # must not be pulled behind its sidecar by the generic _conflict rule.
    state = order_columns(
        _state(
            [
                'occupancy_type_conflict',
                'occupancy_type',
                'geometry',
                'occupancy_type_source',
                'improvement_value_parcel',
            ]
        )
    )
    assert list(state.curated.columns) == [
        'occupancy_type',
        'occupancy_type_source',
        'improvement_value_parcel',
        'occupancy_type_conflict',
        'geometry',
    ]


def test_bare_parcel_ids_lead_the_parcel_evidence_group():
    # Bare parcel-id columns carry no provenance suffix and no registry sort
    # rank, so they used to strand at the tail of the canonical block (between
    # roof_shape and m2). They identify which parcel the _parcel evidence
    # columns describe, so they must lead that group in the source block
    # instead: canonical (roof_shape, m2) < relational counts < parcel ids <
    # _parcel evidence, geometry last.
    state = order_columns(
        _state(
            [
                'parcel_id',
                'parcel_id_local',
                'use_group_combined_parcel',
                'roof_shape',
                'geometry',
                'improvement_value_parcel',
                'area_m2',
                'n_parcels_per_footprint',
                'parcel_id_all',
            ]
        )
    )
    assert list(state.curated.columns) == [
        'roof_shape',
        'area_m2',
        'n_parcels_per_footprint',
        'parcel_id',
        'parcel_id_all',
        'parcel_id_local',
        'use_group_combined_parcel',
        'improvement_value_parcel',
        'geometry',
    ]
