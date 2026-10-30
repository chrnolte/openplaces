"""A restored attribute checkpoint skips the steps that wrote side files
(link sidecars, link tables), so it is refused when one of them is gone
or changed, and when the package code changed.

Found 2026-10-29: Snakemake deleted a forced building spine job's
declared footprint-to-building link table, the job restored a valid
checkpoint, skipped link_entities_by_id, and failed with the table
missing. Every id below is fabricated.
"""

import pandas as pd

import openplaces.io.harmonizer as harmonizer
from openplaces.io.harmonizer import _read_checkpoint, _write_checkpoint
from openplaces.io.harmonizer.written import journal_writes, note_written

CHAIN = {'format': 3, 'steps': ['h0', 'h1'], 'sources': [], 'code': 'c0'}


def _spine():
    return pd.DataFrame(
        {'n_dwellings': [1, 2]}, index=pd.Index(['b1', 'b2'], name='building_id')
    )


def _side_file(tmp_path, content=b'links'):
    path = tmp_path / 'US-XX-YY_building-geospine-2026_link.parquet'
    path.write_bytes(content)
    return path


def test_a_checkpoint_restores_while_its_side_files_stand(tmp_path):
    side = _side_file(tmp_path)
    path = tmp_path / 'checkpoint.parquet'
    _write_checkpoint(path, _spine(), CHAIN, {}, [side])
    assert _read_checkpoint(path, CHAIN) is not None


def test_a_deleted_side_file_refuses_the_restore(tmp_path):
    side = _side_file(tmp_path)
    path = tmp_path / 'checkpoint.parquet'
    _write_checkpoint(path, _spine(), CHAIN, {}, [side])
    side.unlink()
    assert _read_checkpoint(path, CHAIN) is None


def test_a_rewritten_side_file_refuses_the_restore(tmp_path):
    side = _side_file(tmp_path)
    path = tmp_path / 'checkpoint.parquet'
    _write_checkpoint(path, _spine(), CHAIN, {}, [side])
    side.write_bytes(b'other links')
    assert _read_checkpoint(path, CHAIN) is None


def test_a_checkpoint_from_before_the_stamps_is_refused(tmp_path):
    import pyarrow.parquet as pq

    path = tmp_path / 'checkpoint.parquet'
    _write_checkpoint(path, _spine(), CHAIN, {}, [])
    table = pq.read_table(path)
    meta = dict(table.schema.metadata)
    del meta[harmonizer.CHECKPOINT_WRITES_KEY.encode()]
    pq.write_table(table.replace_schema_metadata(meta), path)
    assert _read_checkpoint(path, CHAIN) is None


def test_the_journal_records_only_inside_its_block(tmp_path):
    note_written(tmp_path / 'before.parquet')
    with journal_writes() as written:
        note_written(tmp_path / 'a.parquet')
        note_written(tmp_path / 'b.parquet')
    note_written(tmp_path / 'after.parquet')
    assert [p.name for p in written] == ['a.parquet', 'b.parquet']


def test_the_chain_carries_the_code_stamp(monkeypatch):
    monkeypatch.setattr(harmonizer, '_code_stamp', lambda: 'c1')
    chain = harmonizer._checkpoint_chain({}, [{'step': 'x'}], 0, None)
    assert chain['code'] == 'c1'


def test_the_code_stamp_is_a_stable_hash():
    assert harmonizer._code_stamp() == harmonizer._code_stamp()
    assert len(harmonizer._code_stamp()) == 16
