"""Tests for the Enricher's partial-run coverage and evidence merging."""

import json

import pandas as pd
import pytest

from openplaces.io import read_parquet, save_parquet
from openplaces.io.enricher import Enricher


@pytest.fixture
def enricher():
    """Enricher instance without recipe resolution (methods under test
    only touch output paths and footers)."""
    return Enricher.__new__(Enricher)


@pytest.fixture
def evidence_path(tmp_path):
    """Evidence parquet covering one town, with a coverage footer."""
    evidence = pd.DataFrame(
        {'roof_shape_brails': ['Gable', None, 'Hip']},
        index=['a', 'b', 'c'],
    )
    path = tmp_path / 'evidence.parquet'
    save_parquet(
        evidence,
        path,
        file_metadata={'openplaces:partitions': json.dumps(['US-NC-BR-SH'])},
    )
    return path


def test_missing_file_is_not_covered(enricher, tmp_path):
    assert not enricher._is_covered(tmp_path / 'missing.parquet', ['US-NC-BR-SH'])


def test_town_coverage_footer(enricher, evidence_path):
    assert enricher._is_covered(evidence_path, ['US-NC-BR-SH'])
    assert not enricher._is_covered(evidence_path, ['US-NC-BR-SM'])
    assert not enricher._is_covered(evidence_path, ['US-NC-BR-SH', 'US-NC-BR-SM'])
    # A full process-level request must re-run over a partial file.
    assert not enricher._is_covered(evidence_path, None)


def test_legacy_file_without_footer_counts_as_complete(enricher, tmp_path):
    path = tmp_path / 'legacy.parquet'
    save_parquet(pd.DataFrame({'x': [1]}, index=['a']), path)
    assert enricher._is_covered(path, ['US-NC-BR-SM'])
    assert enricher._is_covered(path, None)


def test_merged_coverage_unions_towns(enricher, evidence_path):
    coverage = enricher._merged_coverage(evidence_path, ['US-NC-BR-SM'])
    assert coverage == ['US-NC-BR-SH', 'US-NC-BR-SM']


def test_merge_evidence_updates_only_attempted_rows(evidence_path):
    existing = read_parquet(evidence_path)
    new = pd.DataFrame(
        {'roof_shape_brails': [None, 'Flat', None, 'Hip']},
        index=['a', 'b', 'c', 'd'],
    )
    merged = Enricher._merge_evidence(existing, new, attempted_keys={'b', 'd'})
    assert merged.loc['a', 'roof_shape_brails'] == 'Gable'  # kept
    assert merged.loc['b', 'roof_shape_brails'] == 'Flat'  # updated
    assert merged.loc['c', 'roof_shape_brails'] == 'Hip'  # kept
    assert merged.loc['d', 'roof_shape_brails'] == 'Hip'  # new row


@pytest.mark.parametrize('attempted_keys', [{'d', 'e'}, None])
def test_merge_evidence_keeps_rows_of_towns_not_in_this_run(
    evidence_path, attempted_keys
):
    """A second town's run must not discard the first town's rows.

    `new` is deliberately disjoint from `existing`: a sub-admin run hands
    in only its own town's rows, and the earlier test above cannot see
    the defect because its `new` frame is a superset of `existing`.
    """
    existing = read_parquet(evidence_path)
    new = pd.DataFrame({'roof_shape_brails': ['Flat', None]}, index=['d', 'e'])
    merged = Enricher._merge_evidence(existing, new, attempted_keys)
    assert set(merged.index) == {'a', 'b', 'c', 'd', 'e'}
    assert merged.loc['a', 'roof_shape_brails'] == 'Gable'
    assert pd.isna(merged.loc['b', 'roof_shape_brails'])
    assert merged.loc['c', 'roof_shape_brails'] == 'Hip'
    assert merged.loc['d', 'roof_shape_brails'] == 'Flat'
    assert pd.isna(merged.loc['e', 'roof_shape_brails'])


def test_sequential_town_runs_accumulate_evidence(monkeypatch, tmp_path):
    """Enriching town A, then town B, leaves both in the saved file.

    Drives `_enrich_one` end to end with a stub spine and a stub step so
    the file on disk, not just the merge helper, is what is checked: the
    footer records both towns, so a row lost here would never be redone.
    """
    import openplaces.io.enricher as enricher_mod

    spine = pd.DataFrame(
        {'admin4_id': ['US-NC-BR-SH', 'US-NC-BR-SH', 'US-NC-BR-SM']},
        index=pd.Index(['a', 'b', 'c'], name='footprint_id'),
    )
    out_path = tmp_path / 'evidence.parquet'
    monkeypatch.setattr(enricher_mod, 'get_entities', lambda *a, **k: spine)
    monkeypatch.setattr(enricher_mod, 'get_output_path', lambda *a, **k: out_path)

    def stub_step(state, **params):
        state.evidence['town'] = state.spine['admin4_id']
        state.metadata['attempted_keys'] = set(state.spine.index)
        return state

    monkeypatch.setitem(enricher_mod._STEP_REGISTRY, 'stub_step', stub_step)

    enricher = Enricher.__new__(Enricher)
    enricher.recipe = {'pipeline': [{'step': 'stub_step'}]}
    enricher.entity_recipe = {}
    enricher.verbose = False
    enricher._timer = None

    enricher._enrich_one('US-NC-BR', ['US-NC-BR-SH'])
    enricher._enrich_one('US-NC-BR', ['US-NC-BR-SM'])

    saved = read_parquet(out_path)
    assert saved['town'].to_dict() == {
        'a': 'US-NC-BR-SH',
        'b': 'US-NC-BR-SH',
        'c': 'US-NC-BR-SM',
    }
    assert enricher._is_covered(out_path, ['US-NC-BR-SH', 'US-NC-BR-SM'])
