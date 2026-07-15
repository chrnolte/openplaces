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
        admin_id=AdminId('US-NC-BS'),
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
    # and must not crash or be reordered relative to its peers.
    state = order_columns(_state(['parcel_id', 'other_all']))
    assert list(state.curated.columns) == ['parcel_id', 'other_all']
