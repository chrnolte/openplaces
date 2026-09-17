"""Confusion matrices, producer's/consumer's accuracy and the report writer.

Every value here is fabricated: class labels, counts, county codes and
years are made up to exercise one behavior each.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from openplaces.io.curator.validation import (
    ABSTAIN_LABEL,
    NON_RESIDENTIAL_LABEL,
    OTHER_LABEL,
    ValidationContext,
    accuracy_from_matrix,
    bin_year_agreement,
    cohens_kappa,
    confusion_matrix,
    paired_disagreement,
    read_confusion_matrix,
    score_classification,
    write_confusion_report,
    write_year_agreement_report,
    year_error_summary,
)

CLASSES = ['A', 'B', 'C']


def _expand(matrix_rows, classes=CLASSES):
    """Truth/predicted series realizing a square count matrix."""
    truth, predicted = [], []
    for reference, counts in zip(classes, matrix_rows, strict=True):
        for column, n in zip(classes, counts, strict=True):
            truth += [reference] * n
            predicted += [column] * n
    return pd.Series(truth), pd.Series(predicted)


def test_matrix_orientation_and_sentinel_columns():
    truth = pd.Series(['A', 'A', 'B', 'B', 'C'])
    predicted = pd.Series(['A', None, 'Z', 'B', 'A'])

    matrix = confusion_matrix(truth, predicted, CLASSES)

    assert list(matrix.index) == CLASSES
    assert list(matrix.columns) == [*CLASSES, OTHER_LABEL, ABSTAIN_LABEL]
    # Rows are the reference: the C predicted as A sits in row C.
    assert matrix.loc['C', 'A'] == 1
    assert matrix.loc['A', ABSTAIN_LABEL] == 1
    assert matrix.loc['B', OTHER_LABEL] == 1
    # No row is dropped.
    assert matrix.to_numpy().sum() == len(truth)


def test_a_class_never_predicted_and_one_without_support_stay_visible():
    truth = pd.Series(['A', 'A', 'B'])
    predicted = pd.Series(['A', 'B', 'B'])

    matrix = confusion_matrix(truth, predicted, CLASSES)
    table = accuracy_from_matrix(matrix).set_index('class')

    assert matrix.loc['C'].sum() == 0
    assert matrix['C'].sum() == 0
    # No reference support and nothing predicted: undefined, not zero.
    assert np.isnan(table.loc['C', 'producers_accuracy_recall'])
    assert np.isnan(table.loc['C', 'consumers_accuracy_precision'])
    assert np.isnan(table.loc['C', 'f1'])
    assert table.loc['A', 'producers_accuracy_recall'] == 0.5
    assert table.loc['B', 'consumers_accuracy_precision'] == 0.5


def test_a_reference_outside_the_classes_counts_against_consumers_accuracy():
    truth = pd.Series(['A', 'Office', 'A'])
    predicted = pd.Series(['A', 'A', 'A'])

    matrix = confusion_matrix(truth, predicted, CLASSES)
    table = accuracy_from_matrix(matrix, CLASSES).set_index('class')

    assert OTHER_LABEL in matrix.index
    assert table.loc['A', 'consumers_accuracy_precision'] == pytest.approx(2 / 3)
    # The overall figures cover reference rows in the classes only.
    assert table.loc['ALL', 'n_reference'] == 2
    assert table.loc['ALL', 'overall_accuracy_answered'] == 1.0


def test_margins_agree_with_score_classification():
    truth = pd.Series(['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'A', 'B'])
    predicted = pd.Series(['A', 'B', None, 'B', 'B', 'C', 'A', None, 'A', 'Z'])

    scores = score_classification(truth, predicted, CLASSES).set_index('class')
    table = accuracy_from_matrix(confusion_matrix(truth, predicted, CLASSES))
    table = table.set_index('class')

    for cls in CLASSES:
        assert table.loc[cls, 'n_reference'] == scores.loc[cls, 'n_truth']
        assert table.loc[cls, 'n_answered'] == scores.loc[cls, 'n_scored']
        assert table.loc[cls, 'n_correct'] == scores.loc[cls, 'n_correct']
        assert table.loc[cls, 'n_predicted'] == scores.loc[cls, 'n_predicted']
        assert (
            round(table.loc[cls, 'producers_accuracy_recall'], 4)
            == (scores.loc[cls, 'recall'])
        )
        assert (
            round(table.loc[cls, 'consumers_accuracy_precision'], 4)
            == (scores.loc[cls, 'precision'])
        )
    assert table.loc['ALL', 'abstention_rate'] == pytest.approx(2 / 10)
    assert table.loc['ALL', 'overall_accuracy_all_rows'] == pytest.approx(5 / 10)
    assert table.loc['ALL', 'overall_accuracy_answered'] == pytest.approx(5 / 8)


def test_score_classification_output_is_unchanged():
    """Pinned against the pre-matrix implementation's exact output."""
    truth = pd.Series(['A', 'A', 'B', None, 'Q', 'Q', 'C'])
    predicted = pd.Series(['A', 'B', 'B', 'A', 'Q', None, 'Z'])

    table = score_classification(truth, predicted, CLASSES)

    expected = pd.DataFrame(
        [
            ['A', 2, 2, 1, 2, 0.5, 0.5, 0.5],
            ['B', 1, 1, 1, 2, 1.0, 0.5, 0.6667],
            ['C', 1, 1, 0, 0, 0.0, None, 0.0],
            # Exact agreement over every answered row, Q included.
            ['ALL', 7, 6, 3, 6, 0.5, 0.5, 0.5],
        ],
        columns=[
            'class',
            'n_truth',
            'n_scored',
            'n_correct',
            'n_predicted',
            'recall',
            'precision',
            'f1',
        ],
    )
    pd.testing.assert_frame_equal(table, expected)


def test_kappa_matches_a_hand_computed_three_by_three():
    # n = 100, observed 0.75; row margins 25/30/45, column margins
    # 30/25/45, so expected = (750 + 750 + 2025) / 10000 = 0.3525 and
    # kappa = (0.75 - 0.3525) / (1 - 0.3525).
    truth, predicted = _expand([[20, 5, 0], [10, 15, 5], [0, 5, 40]])
    matrix = confusion_matrix(truth, predicted, CLASSES)

    assert cohens_kappa(matrix) == pytest.approx(0.3975 / 0.6475)
    table = accuracy_from_matrix(matrix).set_index('class')
    assert table.loc['ALL', 'kappa'] == pytest.approx(0.3975 / 0.6475)
    assert table.loc['ALL', 'macro_f1'] == pytest.approx(
        np.mean([40 / 55, 30 / 55, 80 / 90])
    )


def test_kappa_ignores_abstentions():
    truth, predicted = _expand([[20, 5, 0], [10, 15, 5], [0, 5, 40]])
    truth = pd.concat([truth, pd.Series(['A'] * 7)], ignore_index=True)
    predicted = pd.concat([predicted, pd.Series([None] * 7)], ignore_index=True)

    matrix = confusion_matrix(truth, predicted, CLASSES)

    assert cohens_kappa(matrix) == pytest.approx(0.3975 / 0.6475)


def test_paired_disagreement_counts_discordant_rows():
    truth = pd.Series(['A'] * 10)
    a = pd.Series(['A'] * 8 + ['B', None])
    b = pd.Series(['A'] * 5 + ['B'] * 5)

    result = paired_disagreement(truth, a, b)

    assert result['n_both_right'] == 5
    assert result['n_a_only'] == 3
    assert result['n_b_only'] == 0
    assert result['n_both_wrong'] == 2
    # Two-sided binomial on 3 discordant rows, all one way: 2 / 8.
    assert result['exact_p'] == pytest.approx(0.25)
    assert result['chi_square'] == pytest.approx(4 / 3)


def _survey_frame():
    truth, predicted = _expand([[20, 5, 0], [10, 15, 5], [0, 5, 40]])
    counties = pd.Series(['X01'] * 60 + ['X02'] * 36 + ['X03'] * 4)
    return truth, predicted, counties


def test_writer_refuses_thin_strata_and_round_trips(tmp_path):
    truth, predicted, counties = _survey_frame()
    other = predicted.where(predicted.ne('C'))

    paths = write_confusion_report(
        truth,
        {'vote': predicted, 'single': other},
        CLASSES,
        tmp_path,
        'demo',
        strata={'county': counties},
        reference='fabricated reference',
        tier='strong',
    )

    confusion = pd.read_csv(paths['confusion'], keep_default_na=False)
    assert set(confusion['stratum']) == {'all', 'X01', 'X02'}
    metadata = json.loads(paths['metadata'].read_text(encoding='utf-8'))
    assert metadata['refused_strata'] == [
        {'stratum_by': 'county', 'stratum': 'X03', 'n': 4}
    ]
    assert metadata['n_rows'] == 100
    assert metadata['tier'] == 'strong'
    assert metadata['reference'] == 'fabricated reference'

    for source, values in (('vote', predicted), ('single', other)):
        wide = read_confusion_matrix(paths['confusion'], source)
        pd.testing.assert_frame_equal(
            wide, confusion_matrix(truth, values, CLASSES), check_names=True
        )
        in_county = counties.eq('X02').to_numpy()
        wide = read_confusion_matrix(
            confusion, source, stratum_by='county', stratum='X02'
        )
        pd.testing.assert_frame_equal(
            wide,
            confusion_matrix(truth[in_county], values[in_county], CLASSES),
        )

    accuracy = pd.read_csv(paths['accuracy'])
    pooled = accuracy[
        accuracy['source'].eq('vote')
        & accuracy['stratum'].eq('all')
        & accuracy['class'].eq('ALL')
    ].iloc[0]
    assert pooled['kappa'] == pytest.approx(0.3975 / 0.6475, abs=1e-6)


def test_writer_refuses_a_reference_below_the_minimum(tmp_path):
    with pytest.raises(ValueError, match='minimum'):
        write_confusion_report(
            pd.Series(['A'] * 9),
            {'vote': pd.Series(['A'] * 9)},
            CLASSES,
            tmp_path,
            'thin',
        )
    assert not list(tmp_path.iterdir())


def test_writer_rejects_misaligned_predictions(tmp_path):
    truth = pd.Series(['A'] * 12)
    with pytest.raises(ValueError, match='aligned'):
        write_confusion_report(
            truth, {'vote': pd.Series(['A'] * 11)}, CLASSES, tmp_path, 'bad'
        )


def test_year_bins():
    reference = pd.Series([2000, 2000, 2000, 2000, 2000, None, 2000])
    predicted = pd.Series([2000, 2001, 1997, 2006, None, 1990, 1999.5])

    bins = bin_year_agreement(reference, predicted)

    assert bins.tolist()[:4] == [
        'exact',
        'within 1 year',
        'within 5 years',
        'more than 5 years',
    ]
    assert pd.isna(bins.iloc[4]) and pd.isna(bins.iloc[5])
    assert bins.iloc[6] == 'within 1 year'

    summary = year_error_summary(reference, predicted)
    assert summary['n_reference'] == 6
    assert summary['n_answered'] == 5
    assert summary['share_within_1_year'] == pytest.approx(3 / 5)
    assert summary['mean_absolute_error'] == pytest.approx((0 + 1 + 3 + 6 + 0.5) / 5)
    assert summary['bias'] == pytest.approx((0 + 1 - 3 + 6 - 0.5) / 5)


def test_year_report_writes_bins_through_the_same_files(tmp_path):
    reference = pd.Series([1990] * 12)
    inventory = pd.Series([1990] * 6 + [1991] * 3 + [2000] * 2 + [None])

    paths = write_year_agreement_report(
        reference, {'inventory': inventory}, tmp_path, 'years'
    )

    wide = read_confusion_matrix(paths['confusion'], 'inventory')
    assert list(wide.index) == ['reference year']
    assert wide.loc['reference year'].tolist() == [6, 3, 0, 2, 1]
    metadata = json.loads(paths['metadata'].read_text(encoding='utf-8'))
    assert metadata['kind'] == 'year_agreement'
    errors = pd.read_csv(paths['accuracy']).iloc[0]
    assert errors['n_exact'] == 6
    # Shares are written rounded to six places.
    assert errors['abstention_rate'] == pytest.approx(1 / 12, abs=1e-6)


class _Context(ValidationContext):
    """A context carrying only the fields scoring reads."""

    def __init__(self, config=None):
        self.recipe_id = 'fabricated_recipe'
        self.config = config or {}
        self.classes = ('A', 'B')
        self.collapse = {'A2': 'A'}
        self.source_columns = {'final_vote': 'predicted', 'nsi': 'nsi_class_inv'}
        self.derived_source_columns = {}
        self.dwelling_count_column = None
        self.inventory_suffix = '_inv'
        self.class_map = None


def test_score_sources_writes_matrices_by_default_when_given_a_folder(tmp_path):
    linked = pd.DataFrame(
        {
            'occupancy_type_canonical': ['A'] * 8 + ['B'] * 8,
            'predicted': ['A'] * 7 + ['B'] + ['B'] * 8,
            'nsi_class_inv': ['A'] * 16,
            'admin_id': ['X01'] * 16,
        }
    )
    context = _Context()

    without = context.score_sources(linked)
    with_dir = context.score_sources(linked, tmp_path)

    pd.testing.assert_frame_equal(without, with_dir)
    assert not (tmp_path / 'other').exists()
    confusion = pd.read_csv(
        tmp_path / 'fabricated_recipe_occupancy-survey_confusion.csv',
        keep_default_na=False,
    )
    assert set(confusion['source']) == {'final_vote', 'nsi'}
    assert set(confusion['stratum_by']) == {'all', 'county'}


def test_entity_source_values_reads_unsuffixed_columns():
    entities = pd.DataFrame({'occupancy_type': ['A2', 'B'], 'nsi_class': ['A', None]})
    values = _Context().entity_source_values(entities)
    assert values['final_vote'].tolist() == ['A', 'B']
    assert values['nsi'].tolist()[0] == 'A'


def test_notebooks_resolve_to_the_region_their_reference_scores():
    context = _Context(
        {
            'ground_truth': {'region': 'region-one'},
            'references': {'S1': {'region': 'region-two'}},
            'notebooks': [
                {'notebook': 'survey.ipynb', 'reference': 'ground_truth'},
                {'notebook': 'permits.ipynb', 'reference': 'S1'},
                {'notebook': 'absent.ipynb', 'reference': 'S2'},
            ],
        }
    )
    assert context.notebooks_for_region('region-one') == ['survey.ipynb']
    assert context.notebooks_for_region('region-two') == ['permits.ipynb']
    assert context.notebooks_for_region('region-three') == []


def test_secondary_and_non_residential_get_their_own_columns():
    truth = pd.Series(['A', 'A', 'A', 'B', 'B'])
    predicted = pd.Series(['A', 'Shed', 'Warehouse', None, 'B'])

    matrix = confusion_matrix(
        truth, predicted, CLASSES, secondary='Shed', other=NON_RESIDENTIAL_LABEL
    )

    assert list(matrix.columns) == [
        *CLASSES,
        'Shed',
        NON_RESIDENTIAL_LABEL,
        ABSTAIN_LABEL,
    ]
    assert ABSTAIN_LABEL == 'No class'
    assert matrix.loc['A', 'Shed'] == 1
    assert matrix.loc['A', NON_RESIDENTIAL_LABEL] == 1
    assert matrix.loc['B', ABSTAIN_LABEL] == 1
    # Asserting Secondary or a non-residential class is an answer, and
    # a wrong one; only No class is left out of the answered rows.
    table = accuracy_from_matrix(
        matrix, CLASSES, secondary='Shed', other=NON_RESIDENTIAL_LABEL
    ).set_index('class')
    assert table.loc['A', 'n_answered'] == 3
    assert table.loc['A', 'producers_accuracy_recall'] == pytest.approx(1 / 3)
    assert table.loc['B', 'n_answered'] == 1


class _RulesetContext(_Context):
    """A context whose class map knows residential text only."""

    def __init__(self):
        super().__init__()
        self.residential_classes = ('A', 'B')
        self.secondary_class = 'Shed'
        self.source_columns = {'final_vote': 'predicted'}
        self.derived_source_columns = {
            'nsi': {'column': 'nsi_raw', 'keep_unmapped': True},
            'keyword': {'column': 'text_raw'},
        }

    def class_from_ruleset(self, terms, ruleset=None, **kwargs):
        mapping = {'type a home': 'A', 'type b home': 'B'}
        return terms.astype(object).map(mapping)


def test_an_asserted_non_residential_class_is_not_no_class(tmp_path):
    linked = pd.DataFrame(
        {
            'occupancy_type_canonical': ['A'] * 12,
            'predicted': ['A'] * 9 + ['Shed', 'Office', None],
            # Fabricated source vocabulary: two mapped, one unmapped
            # assertion, one blank and one missing record.
            'nsi_raw_inv': ['type a home'] * 8 + ['type b home', 'Office', '', None],
            'text_raw_inv': ['free text'] * 12,
        }
    )
    context = _RulesetContext()

    values = context.source_values(linked)

    assert values['nsi'].tolist()[8:10] == ['B', 'Office']
    assert values['nsi'].iloc[10:].isna().all()
    # Without keep_unmapped, text no rule matched asserts nothing.
    assert values['keyword'].isna().all()

    labels = context.matrix_labels()
    assert labels == {
        'secondary': 'Shed',
        'other': NON_RESIDENTIAL_LABEL,
        'kept_classes': [],
    }
    nsi = confusion_matrix(
        linked['occupancy_type_canonical'], values['nsi'], ['A', 'B'], **labels
    )
    assert nsi.loc['A', NON_RESIDENTIAL_LABEL] == 1
    assert nsi.loc['A', ABSTAIN_LABEL] == 2
    vote = confusion_matrix(
        linked['occupancy_type_canonical'], values['final_vote'], ['A', 'B'], **labels
    )
    assert vote.loc['A', 'Shed'] == 1
    assert vote.loc['A', NON_RESIDENTIAL_LABEL] == 1
    assert vote.loc['A', ABSTAIN_LABEL] == 1

    context.score_sources(linked, tmp_path)
    confusion = pd.read_csv(
        tmp_path / 'fabricated_recipe_occupancy-survey_confusion.csv',
        keep_default_na=False,
    )
    assert {'Shed', NON_RESIDENTIAL_LABEL, ABSTAIN_LABEL} <= set(confusion['predicted'])


class _UnscoredResidentialContext(_RulesetContext):
    """A recipe with a residential class the reference never labels."""

    def __init__(self):
        super().__init__()
        self.residential_classes = ('A', 'B', 'C')


def test_an_unscored_residential_class_is_not_non_residential(tmp_path):
    linked = pd.DataFrame(
        {
            'occupancy_type_canonical': ['A'] * 11 + ['C'],
            'predicted': ['A'] * 8 + ['C', 'C', 'Office', 'C'],
            'nsi_raw_inv': ['type a home'] * 12,
            'text_raw_inv': ['free text'] * 12,
        }
    )
    context = _UnscoredResidentialContext()
    labels = context.matrix_labels()
    assert labels['kept_classes'] == ['C']

    matrix = confusion_matrix(
        linked['occupancy_type_canonical'], linked['predicted'], ['A', 'B'], **labels
    )
    # Its own column, never folded into Non-residential, and its own
    # row where the reference uses it.
    assert matrix.loc['A', 'C'] == 2
    assert matrix.loc['A', NON_RESIDENTIAL_LABEL] == 1
    assert matrix.loc['C', 'C'] == 1
    table = accuracy_from_matrix(
        matrix, ['A', 'B'], secondary='Shed', other=NON_RESIDENTIAL_LABEL
    ).set_index('class')
    # A C prediction on an A is an answered miss.
    assert table.loc['A', 'n_answered'] == 11
    assert table.loc['A', 'n_correct'] == 8

    context.score_sources(linked, tmp_path)
    confusion = pd.read_csv(
        tmp_path / 'fabricated_recipe_occupancy-survey_confusion.csv',
        keep_default_na=False,
    )
    assert 'C' in set(confusion['predicted'])
