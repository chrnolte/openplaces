"""An input the lifecycle consumed and deleted, with a receipt, is not a
dependency the graph has to satisfy again.

The census block tile grid is `until_consumed`: once the geospines have
read it, the cleanup deletes it and leaves a tombstone receipt beside
it. To the orchestrator the file was simply missing, so a county rebuild
scheduled the tile grid and, behind it, every consumer. A job whose own
output exists keeps only the inputs that exist or carry no receipt; a
job whose output is missing keeps every input, so a real rebuild still
regenerates what it needs.
"""

from pathlib import Path

from openplaces.flow.dag import RecipeDAG
from openplaces.io.cleanup import write_receipt

TARGET = 'US_footprint-openplaces-2026'
COUNTY = 'US-NC-BRU'


def _dag():
    return RecipeDAG(TARGET, admin_ids=[COUNTY], deliver=False)


def test_a_receipted_absent_input_is_dropped_when_the_output_exists(
    tmp_path, monkeypatch
):
    dag = _dag()
    consumed = tmp_path / 'consumed.parquet'
    present = tmp_path / 'present.parquet'
    present.write_bytes(b'')
    unreceipted = tmp_path / 'gone.parquet'
    own = tmp_path / 'own.parquet'
    own.write_bytes(b'')
    write_receipt(consumed, {'consumers_verified': [{'path': str(own)}]})
    monkeypatch.setattr(dag, 'output_path', lambda *a, **k: own)

    kept = dag._without_consumed_intermediates(
        [consumed, present, unreceipted], 'harmonize', 'r', COUNTY, None
    )

    assert kept == [present, unreceipted]


def test_every_input_is_kept_when_the_output_is_missing(tmp_path, monkeypatch):
    dag = _dag()
    consumed = tmp_path / 'consumed.parquet'
    write_receipt(consumed, {'consumers_verified': []})
    monkeypatch.setattr(dag, 'output_path', lambda *a, **k: tmp_path / 'absent.parquet')

    kept = dag._without_consumed_intermediates(
        [consumed], 'harmonize', 'r', COUNTY, None
    )

    assert kept == [consumed]
    assert isinstance(kept[0], Path)
