"""Tests for the detector prediction checkpoint sidecar."""

import os

import pandas as pd

from openplaces.io.enricher.detectors import checkpoint as checkpoint_module
from openplaces.io.enricher.detectors.checkpoint import (
    PredictionCheckpoint,
    local_checkpoint_path,
)


def test_load_without_file_is_empty(tmp_path):
    checkpoint = PredictionCheckpoint(tmp_path / 'ckpt.parquet')
    assert checkpoint.load() == {}
    assert not checkpoint.path.exists()


def test_roundtrip_preserves_none_values(tmp_path):
    path = tmp_path / 'ckpt.parquet'
    checkpoint = PredictionCheckpoint(path, save_every=100)
    checkpoint.add('a', 'Gable')
    checkpoint.add('b', None)
    checkpoint.add('c', 3)
    checkpoint.flush()

    resumed = PredictionCheckpoint(path).load()
    assert resumed == {'a': 'Gable', 'b': None, 'c': 3}


def test_auto_flush_every_save_every(tmp_path):
    path = tmp_path / 'ckpt.parquet'
    checkpoint = PredictionCheckpoint(path, save_every=2)
    checkpoint.add('a', 1)
    assert not path.exists()
    checkpoint.add('b', 2)
    assert path.exists()
    assert len(pd.read_parquet(path)) == 2
    # Third addition stays buffered until the next flush trigger.
    checkpoint.add('c', 3)
    assert len(pd.read_parquet(path)) == 2
    checkpoint.flush()
    assert len(pd.read_parquet(path)) == 3


def test_flush_without_new_predictions_is_a_noop(tmp_path):
    path = tmp_path / 'ckpt.parquet'
    checkpoint = PredictionCheckpoint(path)
    checkpoint.flush()
    assert not path.exists()


def test_resume_accumulates_across_instances(tmp_path):
    path = tmp_path / 'ckpt.parquet'
    first = PredictionCheckpoint(path, save_every=1)
    first.add('a', 'Flat')

    second = PredictionCheckpoint(path, save_every=1)
    assert second.load() == {'a': 'Flat'}
    second.add('b', 'Hip')

    assert PredictionCheckpoint(path).load() == {'a': 'Flat', 'b': 'Hip'}


def test_delete_removes_file_and_state(tmp_path):
    path = tmp_path / 'ckpt.parquet'
    checkpoint = PredictionCheckpoint(path, save_every=1)
    checkpoint.add('a', 1)
    assert path.exists()
    checkpoint.delete()
    assert not path.exists()
    assert checkpoint.load() == {}


def test_local_checkpoint_path_is_stable_and_outside_output(tmp_path):
    output_side = tmp_path / 'synced' / 'evidence_checkpoint.parquet'

    first = local_checkpoint_path(output_side)
    second = local_checkpoint_path(output_side)

    assert first == second
    assert first.name == output_side.name
    assert tmp_path not in first.parents


def test_load_migrates_largest_legacy_checkpoint(tmp_path):
    current = tmp_path / 'local' / 'checkpoint.parquet'
    legacy = tmp_path / 'synced' / 'checkpoint.parquet'
    legacy_tmp = legacy.with_suffix('.tmp')
    legacy.parent.mkdir()
    pd.DataFrame({'value': ['1']}, index=['a']).to_parquet(legacy)
    pd.DataFrame(
        {'value': ['1', '2']},
        index=['a', 'b'],
    ).to_parquet(legacy_tmp)

    checkpoint = PredictionCheckpoint(
        current,
        legacy_paths=[legacy, legacy_tmp],
    )

    assert checkpoint.load() == {'a': 1, 'b': 2}
    assert current.exists()
    assert not legacy.exists()
    assert not legacy_tmp.exists()


def test_replace_retries_transient_permission_error(tmp_path, monkeypatch):
    path = tmp_path / 'checkpoint.parquet'
    checkpoint = PredictionCheckpoint(path, save_every=1)
    real_replace = os.replace
    attempts = []

    def flaky_replace(source, target):
        attempts.append(target)
        if len(attempts) < 3:
            raise PermissionError('locked')
        real_replace(source, target)

    monkeypatch.setattr(checkpoint_module.os, 'replace', flaky_replace)
    monkeypatch.setattr(checkpoint_module.time, 'sleep', lambda seconds: None)

    checkpoint.add('a', 1)

    assert len(attempts) == 3
    assert PredictionCheckpoint(path).load() == {'a': 1}
