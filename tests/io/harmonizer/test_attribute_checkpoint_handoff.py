"""The attribute checkpoint carries the pipeline metadata the skipped
steps would have set, so a resumed parcel spine gets its index name back.

Found 2026-09-22: a parcel spine rerun through a valid checkpoint shipped
under the working index name 'spine_id' with no 'parcel_id' column,
because resolve_spine (skipped on restore) is what records the original
name in state.metadata['spine_index_name'] for the save step to restore.

Every id below is fabricated.
"""

import pandas as pd

from openplaces.io.harmonizer import (
    _apply_handoff_metadata,
    _handoff_metadata,
    _read_checkpoint,
    _write_checkpoint,
)


class _State:
    def __init__(self, **metadata):
        self.metadata = dict(metadata)


def _spine():
    frame = pd.DataFrame(
        {'land_value': [1000.0, 2500.0], 'parcel_id_local': ['a-1', 'a-2']},
        index=pd.Index(['p1', 'p2'], name='spine_id'),
    )
    return frame


CHAIN = {'format': 1, 'steps': ['h0', 'h1'], 'source': None}


def test_handoff_round_trips_through_the_checkpoint_footer(tmp_path):
    path = tmp_path / 'checkpoint.parquet'
    state = _State(
        spine_index_name='parcel_id',
        spine_source_recipe_ids={'US-XX_parcel-a-2026', 'US-XX_parcel-b-2026'},
        spine_keep_columns={'zoning'},
    )
    _write_checkpoint(path, _spine(), CHAIN, _handoff_metadata(state))

    loaded = _read_checkpoint(path, CHAIN)
    assert loaded is not None
    spine, handoff = loaded
    assert spine.index.name == 'spine_id'
    assert list(spine.index) == ['p1', 'p2']
    assert handoff['spine_index_name'] == 'parcel_id'
    # Sets are written sorted so the footer is deterministic.
    assert handoff['spine_source_recipe_ids'] == [
        'US-XX_parcel-a-2026',
        'US-XX_parcel-b-2026',
    ]


def test_applying_the_handoff_gives_the_save_step_its_index_name_back(tmp_path):
    path = tmp_path / 'checkpoint.parquet'
    _write_checkpoint(
        path, _spine(), CHAIN, _handoff_metadata(_State(spine_index_name='parcel_id'))
    )
    _, handoff = _read_checkpoint(path, CHAIN)

    resumed = _State()
    _apply_handoff_metadata(resumed, handoff)
    # This is the value the harmonizer's save step renames the index to;
    # before the fix it was None after a restore and the rename was a no-op.
    assert resumed.metadata['spine_index_name'] == 'parcel_id'


def test_sets_come_back_as_sets():
    state = _State()
    _apply_handoff_metadata(
        state, {'spine_keep_columns': ['b', 'a'], 'spine_source_recipe_ids': ['r']}
    )
    assert state.metadata['spine_keep_columns'] == {'a', 'b'}
    assert state.metadata['spine_source_recipe_ids'] == {'r'}


def test_a_checkpoint_without_a_handoff_reads_back_empty(tmp_path):
    path = tmp_path / 'checkpoint.parquet'
    _write_checkpoint(path, _spine(), CHAIN, None)
    _, handoff = _read_checkpoint(path, CHAIN)
    assert handoff == {}
    state = _State()
    _apply_handoff_metadata(state, handoff)
    assert state.metadata == {}


def test_a_stale_chain_is_refused(tmp_path):
    path = tmp_path / 'checkpoint.parquet'
    _write_checkpoint(path, _spine(), CHAIN, {'spine_index_name': 'parcel_id'})
    assert _read_checkpoint(path, {**CHAIN, 'steps': ['h0']}) is None


def test_no_metadata_means_no_handoff():
    assert _handoff_metadata(_State()) == {}
