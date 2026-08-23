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
