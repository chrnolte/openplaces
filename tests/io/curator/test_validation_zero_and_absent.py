"""The validation scorer must tell a zero score from an absent one.

Two defects that both hid a real result behind a missing value: a class
scored 0.0 was reported as NaN and then dropped by the regression gate,
and a source with no evidence for an entity was scored as though it had
predicted Single-Family there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from openplaces.io.curator.validation import (
    ValidationContext,
    compare_classifications_paired,
    score_classification,
)


def test_a_fully_misclassified_class_scores_zero_not_missing():
    """Every real Multi-Family called Single-Family is F1 0.0, not NaN."""
    truth = pd.Series(['Multi-Family', 'Multi-Family', 'Single-Family'])
    predicted = pd.Series(['Single-Family'] * 3)

    scored = score_classification(
        truth, predicted, ['Single-Family', 'Multi-Family']
    ).set_index('class')

    assert scored.loc['Multi-Family', 'recall'] == 0.0
    assert scored.loc['Multi-Family', 'f1'] == 0.0


def test_a_class_with_no_support_stays_undefined():
    """Undefined must stay undefined: nothing to be right or wrong about."""
    truth = pd.Series(['Single-Family', 'Single-Family'])
    predicted = pd.Series(['Single-Family', 'Single-Family'])

    scored = score_classification(
        truth, predicted, ['Single-Family', 'Manufactured Home']
    ).set_index('class')

    assert pd.isna(scored.loc['Manufactured Home', 'f1'])
    assert pd.isna(scored.loc['Manufactured Home', 'recall'])
    assert scored.loc['Manufactured Home', 'n_scored'] == 0


def test_the_paired_gate_sees_a_collapse_to_zero():
    """A class the proposal collapses must reach the gate as a loss.

    With the collapse reported as NaN, every bootstrap draw failed
    np.isfinite, n_draws fell to 0 and p_worse came back NaN, so the
    gate could not fire on the regression it exists to catch.
    """
    truth = pd.Series(['Multi-Family'] * 8 + ['Single-Family'] * 8)
    baseline = truth.copy()
    proposed = pd.Series(['Single-Family'] * 16)

    report = compare_classifications_paired(
        truth,
        baseline,
        proposed,
        ['Single-Family', 'Multi-Family'],
        n_draws=50,
    ).set_index('class')

    row = report.loc['Multi-Family']
    assert row['f1_base'] == 1.0
    assert row['f1_new'] == 0.0
    assert row['d_f1'] == -1.0
    assert row['n_draws'] > 0
    assert row['p_worse'] == 1.0


class _Context(ValidationContext):
    """A context carrying only the fields source_values reads."""

    def __init__(self):
        self.collapse = {}
        self.source_columns = {}
        self.derived_source_columns = {}
        self.dwelling_count_column = 'n_dwellings_overture_inv'
        self.inventory_suffix = '_inv'
        self.class_map = None


def test_an_absent_dwelling_count_is_not_an_asserted_single_family():
    """No Overture record must score as no prediction, not Single-Family."""
    linked = pd.DataFrame(
        {
            'n_dwellings_overture_inv': [np.nan, 1.0, 3.0],
            'occupancy_type_canonical': [
                'Manufactured Home',
                'Single-Family',
                'Multi-Family',
            ],
        }
    )
    values = _Context().source_values(linked)['overture']

    assert values.isna().tolist() == [True, False, False]
    assert values.tolist()[1:] == ['Single-Family', 'Multi-Family']

    scored = score_classification(
        linked['occupancy_type_canonical'],
        values,
        ['Single-Family', 'Multi-Family', 'Manufactured Home'],
    ).set_index('class')

    # The manufactured home Overture never saw is neither scored against
    # it nor counted among the rows it predicted.
    assert scored.loc['Manufactured Home', 'n_scored'] == 0
    assert scored.loc['ALL', 'n_scored'] == 2
    assert scored.loc['Single-Family', 'precision'] == 1.0
