"""Confusion matrices and the accuracies derived from them:
producer's and consumer's accuracy, kappa, paired comparisons
and year agreement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compare_classifications_paired(
    truth: pd.Series,
    baseline: pd.Series,
    proposed: pd.Series,
    classes: list[str],
    n_draws: int = 400,
    seed: int = 0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Paired bootstrap of the per-class F1 change from *baseline* to *proposed*.

    Comparing two independent point estimates cannot resolve a change
    smaller than the sample's own spread -- on a survey of a thousand-odd
    labelled points that spread is several F1 points, far wider than the
    change a careful edit produces. Resampling the *same* point indices for
    both predictions cancels the variation the two runs share, because most
    points are classified identically either way, so the interval reflects
    only the rows that actually moved.

    Parameters
    ----------
    truth : pandas.Series
        Hand-assigned class per point.
    baseline : pandas.Series
        The accepted run's prediction for the same points, already aligned
        to *truth* (same index, same order).
    proposed : pandas.Series
        The candidate run's prediction for those points.
    classes : list of str
        Classes to score, in report order. An ``ALL`` row is appended.
    n_draws : int, optional
        Bootstrap draws (default 400).
    seed : int, optional
        Seed for the resampler, so a gate decision is reproducible.
    alpha : float, optional
        Two-sided interval width (default 0.05, i.e. a 95% interval).

    Returns
    -------
    pandas.DataFrame
        One row per class plus ``ALL``, with ``f1_base``, ``f1_new``,
        ``d_f1`` (the point estimate of new minus base), ``d_low``/``d_high``
        (the paired interval), and ``p_worse`` (share of draws in which the
        class lost F1). A class the proposed run never predicts and the
        baseline never predicted scores NaN, as in
        :func:`score_classification`.
    """
    truth = truth.reset_index(drop=True)
    baseline = baseline.reset_index(drop=True)
    proposed = proposed.reset_index(drop=True)
    if not (len(truth) == len(baseline) == len(proposed)):
        raise ValueError(
            f'paired comparison needs aligned inputs; got {len(truth)} truth, '
            f'{len(baseline)} baseline and {len(proposed)} proposed rows.'
        )

    def _f1(idx, predicted):
        table = score_classification(
            truth.iloc[idx].reset_index(drop=True),
            predicted.iloc[idx].reset_index(drop=True),
            classes,
        )
        return table.set_index('class')['f1']

    rng = np.random.default_rng(seed)
    deltas: dict[str, list[float]] = {}
    for _ in range(n_draws):
        idx = rng.integers(0, len(truth), len(truth))
        base_f1 = _f1(idx, baseline)
        new_f1 = _f1(idx, proposed)
        for label in base_f1.index:
            deltas.setdefault(label, []).append(new_f1[label] - base_f1[label])

    whole = np.arange(len(truth))
    base_point = _f1(whole, baseline)
    new_point = _f1(whole, proposed)

    rows = []
    for label in base_point.index:
        draws = np.array(deltas.get(label, []), dtype=float)
        finite = draws[np.isfinite(draws)]
        low, high = (
            np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            if len(finite)
            else (np.nan, np.nan)
        )
        rows.append(
            {
                'class': label,
                'f1_base': base_point[label],
                'f1_new': new_point[label],
                'd_f1': new_point[label] - base_point[label],
                'd_low': low,
                'd_high': high,
                'p_worse': float((finite < 0).mean()) if len(finite) else np.nan,
                'n_draws': int(len(finite)),
            }
        )
    return pd.DataFrame(rows)


# Sentinel labels of a confusion matrix. 'No class' means the source
# asserted nothing; an asserted class outside the scored vocabulary is
# 'Non-residential' where the recipe declares its residential classes,
# and the generic '(other class)' where it does not.
ABSTAIN_LABEL = 'No class'
OTHER_LABEL = '(other class)'
NON_RESIDENTIAL_LABEL = 'Non-residential'
NO_REFERENCE_LABEL = '(no reference)'

# A year-agreement report files every scored entity under one reference
# row: the reference carries a year, not a class, so the columns say how
# far the prediction lands from it.
YEAR_REFERENCE_LABEL = 'reference year'
YEAR_AGREEMENT_BINS = (
    'exact',
    'within 1 year',
    'within 5 years',
    'more than 5 years',
)

# Rows a stratum must hold before any of its cells is written. A matrix
# cell over a handful of rows can point at one surveyed address.
MIN_STRATUM_ROWS = 10


def confusion_matrix(
    truth: pd.Series,
    predicted: pd.Series,
    classes: list[str],
    *,
    abstain: str = ABSTAIN_LABEL,
    other: str = OTHER_LABEL,
    secondary: str | None = None,
    collapse_other: bool = True,
    kept_classes: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Count matrix of reference classes (rows) by predicted classes (columns).

    Every input row lands in exactly one cell, so the matrix total is the
    number of rows passed in. Columns separate, in order, the scored
    *classes*, any *kept_classes*, the *secondary* class (an outbuilding,
    where one is named), every other asserted class (*other*, e.g.
    `Non-residential`) and no assertion at all (*abstain*). Dropping or
    merging those would hide whether a source declined to answer or
    answered outside the vocabulary, which call for different fixes.

    Parameters
    ----------
    truth : pandas.Series
        Reference class per row.
    predicted : pandas.Series
        Predicted class per row, positionally aligned to *truth*.
    classes : list of str
        Classes in report order. Always present as rows and columns, so a
        class never predicted, or with no reference support, shows up as
        zeros rather than disappearing.
    abstain : str, optional
        Column label for rows with no prediction.
    other : str, optional
        Row and column label for asserted labels outside *classes* and
        *secondary*.
    secondary : str, optional
        A class kept in a row and column of its own rather than folded
        into *other* (the recipe's `occupancy: secondary_class`).
    collapse_other : bool, optional
        True (default) folds every label outside *classes* and
        *secondary* into *other*. False keeps each such label as its own
        row and column, after *classes* in order of first appearance,
        which is what an exact label-equality agreement needs.
    kept_classes : list of str, optional
        Classes the reference does not score but that must not be
        folded into *other*: a residential class of the recipe that
        this reference never labels (`RV Dwelling` against a survey or
        permits that have no such class). Each is a column of its own,
        and a row only where the reference uses it. Being outside
        *classes*, a prediction of one counts as an answered miss in
        :func:`accuracy_from_matrix`, never as a non-residential one.

    Returns
    -------
    pandas.DataFrame
        Integer counts. Rows are *classes*, then *kept_classes*,
        *secondary* and *other* (or the extra labels) where a reference
        falls outside *classes*, then `(no reference)` where a reference
        is missing. Columns are *classes*, *kept_classes*, *secondary*
        when given, *other* (or the extra labels), then *abstain*.
    """
    truth = pd.Series(truth).astype(object).reset_index(drop=True)
    predicted = pd.Series(predicted).astype(object).reset_index(drop=True)
    if len(truth) != len(predicted):
        raise ValueError(
            f'confusion_matrix needs aligned inputs; got {len(truth)} '
            f'reference and {len(predicted)} predicted rows.'
        )
    classes = list(classes)
    known = set(classes)
    extra_classes = [
        label
        for label in dict.fromkeys(kept_classes or ())
        if label not in known and label != secondary
    ]

    reference = truth.where(truth.notna(), NO_REFERENCE_LABEL)
    answer = predicted.where(predicted.notna(), abstain)
    if collapse_other:
        kept = known | set(extra_classes) | ({secondary} if secondary else set())
        reference = reference.where(
            reference.isin(kept) | reference.eq(NO_REFERENCE_LABEL), other
        )
        answer = answer.where(answer.isin(kept) | answer.eq(abstain), other)
        extra_columns = [*extra_classes, *([secondary] if secondary else []), other]
        in_reference = set(reference)
        extra_rows = [label for label in extra_columns if label in in_reference]
    else:
        sentinels = {NO_REFERENCE_LABEL, abstain}
        extra_columns = [
            label
            for label in dict.fromkeys([*reference, *answer])
            if label not in known and label not in sentinels
        ]
        in_reference = set(reference)
        extra_rows = [label for label in extra_columns if label in in_reference]
    rows = classes + extra_rows
    if reference.eq(NO_REFERENCE_LABEL).any():
        rows.append(NO_REFERENCE_LABEL)
    columns = classes + extra_columns + [abstain]

    counts = (
        pd.DataFrame({'reference': reference, 'predicted': answer})
        .groupby(['reference', 'predicted'], sort=False)
        .size()
    )
    matrix = pd.DataFrame(0, index=rows, columns=columns, dtype=int)
    for (row, column), n in counts.items():
        matrix.loc[row, column] = int(n)
    matrix.index.name = 'reference'
    matrix.columns.name = 'predicted'
    return matrix


def _ratio(numerator, denominator):
    """A share, or None where the denominator is zero."""
    return numerator / denominator if denominator else None


def _f1(precision, recall):
    """F1 from a precision and recall that may each be undefined (None)."""
    # A real zero is a score, not a missing measurement. Testing the
    # floats for truthiness collapsed the two, so a class we got
    # entirely wrong reported None (NaN), and
    # compare_classifications_paired's np.isfinite filter then dropped
    # every bootstrap draw of it, leaving the regression gate blind to
    # exactly the collapse it exists to catch. F1 is undefined only
    # where precision and recall are both undefined: no truth support
    # and nothing predicted.
    if precision is None and recall is None:
        return None
    # sklearn's zero_division convention: an undefined half counts as 0
    # once the other half is measurable, because a class we found none
    # of, or predicted only wrongly, scored zero rather than went
    # unmeasured.
    hit = (precision or 0.0, recall or 0.0)
    return 2 * hit[0] * hit[1] / sum(hit) if sum(hit) else 0.0


def _class_counts(matrix: pd.DataFrame, cls, abstain: str) -> dict:
    """Row, column and diagonal counts of one class in a matrix."""
    in_rows = cls in matrix.index
    in_columns = cls in matrix.columns
    n_reference = int(matrix.loc[cls].sum()) if in_rows else 0
    n_abstained = (
        int(matrix.loc[cls, abstain]) if in_rows and abstain in matrix.columns else 0
    )
    return {
        'n_reference': n_reference,
        'n_answered': n_reference - n_abstained,
        'n_correct': int(matrix.loc[cls, cls]) if in_rows and in_columns else 0,
        'n_predicted': int(matrix[cls].sum()) if in_columns else 0,
    }


def _matrix_classes(matrix, classes, *, abstain, other, secondary=None):
    """The reference classes of a matrix: given, or every non-sentinel row."""
    if classes is not None:
        return list(classes)
    sentinels = {abstain, other, secondary, NO_REFERENCE_LABEL}
    return [label for label in matrix.index if label not in sentinels]


def cohens_kappa(
    matrix: pd.DataFrame,
    classes: list[str] | None = None,
    *,
    abstain: str = ABSTAIN_LABEL,
    other: str = OTHER_LABEL,
    secondary: str | None = None,
) -> float:
    """Cohen's kappa over the answered rows of a confusion matrix.

    Agreement beyond what the two margins alone would produce by chance.
    It is computed over the reference rows in *classes* and every
    answered column. A row is answered when the source asserted any
    class, a secondary or non-residential one included; only *abstain*
    is excluded, and the *secondary* and *other* columns count as
    categories whose reference margin is zero. A classifier that
    abstains is therefore neither rewarded nor punished here, which is
    why the abstention rate is reported beside it.

    Parameters
    ----------
    matrix : pandas.DataFrame
        Output of :func:`confusion_matrix`.
    classes : list of str, optional
        Reference classes to include. Default: every non-sentinel row.
    abstain, other, secondary : str, optional
        Sentinel labels, as passed to :func:`confusion_matrix`.

    Returns
    -------
    float
        Kappa, or NaN when no row was answered or chance agreement is
        total.
    """
    classes = _matrix_classes(
        matrix, classes, abstain=abstain, other=other, secondary=secondary
    )
    answered = [column for column in matrix.columns if column != abstain]
    block = matrix.reindex(index=classes, columns=answered, fill_value=0)
    n = int(block.to_numpy().sum())
    if not n:
        return float('nan')
    labels = list(dict.fromkeys([*classes, *answered]))
    row_margin = block.sum(axis=1).reindex(labels, fill_value=0)
    column_margin = block.sum(axis=0).reindex(labels, fill_value=0)
    observed = sum(int(block.loc[c, c]) for c in classes if c in answered) / n
    expected = float((row_margin * column_margin).sum()) / n**2
    if expected >= 1:
        return float('nan')
    return (observed - expected) / (1 - expected)


# Column order of accuracy_from_matrix, shared with the report writer.
ACCURACY_COLUMNS = (
    'class',
    'n_reference',
    'n_answered',
    'n_correct',
    'n_predicted',
    'producers_accuracy_recall',
    'producers_accuracy_all_rows',
    'consumers_accuracy_precision',
    'f1',
    'overall_accuracy_answered',
    'overall_accuracy_all_rows',
    'macro_f1',
    'kappa',
    'abstention_rate',
)


def accuracy_from_matrix(
    matrix: pd.DataFrame,
    classes: list[str] | None = None,
    *,
    abstain: str = ABSTAIN_LABEL,
    other: str = OTHER_LABEL,
    secondary: str | None = None,
) -> pd.DataFrame:
    """Producer's and consumer's accuracy per class, plus overall figures.

    Producer's accuracy is row-based: of the reference rows of a class
    that the classifier answered, the share it got right (recall). A
    row is answered when the source asserted any class, including a
    secondary or non-residential one, which therefore counts as a miss;
    only the *abstain* column is left out.
    Consumer's accuracy is column-based: of the rows it called a class,
    the share that really are (precision). A map producer reads the
    first, a map user the second, and a classifier can be excellent at
    one while failing the other.

    Parameters
    ----------
    matrix : pandas.DataFrame
        Output of :func:`confusion_matrix`.
    classes : list of str, optional
        Reference classes to score, in report order. Default: every
        non-sentinel row.
    abstain, other, secondary : str, optional
        Sentinel labels, as passed to :func:`confusion_matrix`.

    Returns
    -------
    pandas.DataFrame
        One row per class and a final `ALL` row, with the columns of
        `ACCURACY_COLUMNS`. Class rows carry the counts,
        `producers_accuracy_recall` (over answered rows),
        `producers_accuracy_all_rows` (abstentions counted as misses),
        `consumers_accuracy_precision` and `f1`. The `ALL` row carries
        the counts pooled over the class rows, and
        `overall_accuracy_answered`, `overall_accuracy_all_rows`,
        `macro_f1` (mean over classes whose F1 is defined), `kappa`
        (:func:`cohens_kappa`) and `abstention_rate`. Undefined shares
        are NaN.
    """
    classes = _matrix_classes(
        matrix, classes, abstain=abstain, other=other, secondary=secondary
    )
    records = []
    for cls in classes:
        counts = _class_counts(matrix, cls, abstain)
        recall = _ratio(counts['n_correct'], counts['n_answered'])
        precision = _ratio(counts['n_correct'], counts['n_predicted'])
        records.append(
            {
                'class': cls,
                **counts,
                'producers_accuracy_recall': recall,
                'producers_accuracy_all_rows': _ratio(
                    counts['n_correct'], counts['n_reference']
                ),
                'consumers_accuracy_precision': precision,
                'f1': _f1(precision, recall),
            }
        )

    n_reference = sum(record['n_reference'] for record in records)
    n_answered = sum(record['n_answered'] for record in records)
    n_correct = sum(record['n_correct'] for record in records)
    defined_f1 = [record['f1'] for record in records if record['f1'] is not None]
    records.append(
        {
            'class': 'ALL',
            'n_reference': n_reference,
            'n_answered': n_answered,
            'n_correct': n_correct,
            'n_predicted': n_answered,
            'overall_accuracy_answered': _ratio(n_correct, n_answered),
            'overall_accuracy_all_rows': _ratio(n_correct, n_reference),
            'macro_f1': float(np.mean(defined_f1)) if defined_f1 else None,
            'kappa': cohens_kappa(
                matrix, classes, abstain=abstain, other=other, secondary=secondary
            ),
            'abstention_rate': (1 - n_answered / n_reference if n_reference else None),
        }
    )
    table = pd.DataFrame.from_records(records, columns=list(ACCURACY_COLUMNS))
    for column in ACCURACY_COLUMNS[5:]:
        table[column] = pd.to_numeric(table[column], errors='coerce').astype(float)
    return table


def score_classification(
    truth: pd.Series, predicted: pd.Series, classes: list[str]
) -> pd.DataFrame:
    """Per-class precision, recall and F1, plus an overall agreement row.

    Recall answers "of the real Xs, how many did we find"; precision answers
    "of those we called X, how many really are". Reporting only one invites
    the failure this whole module exists to prevent.

    Computed from :func:`confusion_matrix` with every label kept apart
    (`collapse_other=False`), so each number here has the same definition
    as the matching one in :func:`accuracy_from_matrix`. The `ALL` row is
    exact label agreement over every answered row, whatever its class.
    """
    matrix = confusion_matrix(truth, predicted, classes, collapse_other=False)

    def _rounded(value):
        return round(value, 4) if value is not None else None

    records = []
    for cls in classes:
        counts = _class_counts(matrix, cls, ABSTAIN_LABEL)
        recall = _ratio(counts['n_correct'], counts['n_answered'])
        precision = _ratio(counts['n_correct'], counts['n_predicted'])
        records.append(
            {
                'class': cls,
                'n_truth': counts['n_reference'],
                'n_scored': counts['n_answered'],
                'n_correct': counts['n_correct'],
                'n_predicted': counts['n_predicted'],
                'recall': _rounded(recall),
                'precision': _rounded(precision),
                'f1': _rounded(_f1(precision, recall)),
            }
        )

    answered = [column for column in matrix.columns if column != ABSTAIN_LABEL]
    n_scored = int(matrix[answered].to_numpy().sum())
    n_correct = sum(
        int(matrix.loc[label, label]) for label in matrix.index if label in answered
    )
    agreement = _rounded(_ratio(n_correct, n_scored))
    records.append(
        {
            'class': 'ALL',
            'n_truth': int(matrix.to_numpy().sum()),
            'n_scored': n_scored,
            'n_correct': n_correct,
            'n_predicted': n_scored,
            'recall': agreement,
            'precision': agreement,
            'f1': agreement,
        }
    )
    return pd.DataFrame(records)


def paired_disagreement(
    truth: pd.Series, a: pd.Series, b: pd.Series
) -> dict[str, float]:
    """McNemar's test of two classifiers scored on the same rows.

    Only the rows where exactly one of the two is right carry information
    about which is better; rows both get right or both get wrong cancel.
    A missing prediction counts as wrong, and rows without a reference
    are dropped.

    Parameters
    ----------
    truth : pandas.Series
        Reference class per row.
    a, b : pandas.Series
        The two predictions, positionally aligned to *truth*.

    Returns
    -------
    dict
        `n` (rows with a reference), `n_both_right`, `n_a_only`
        (a right, b wrong), `n_b_only`, `n_both_wrong`, `exact_p` (the
        two-sided binomial test on the discordant rows), and
        `chi_square` with `chi_square_p` (continuity-corrected, one
        degree of freedom). The p-values are NaN when no row is
        discordant.
    """
    import math

    truth = pd.Series(truth).astype(object).reset_index(drop=True)
    a = pd.Series(a).astype(object).reset_index(drop=True)
    b = pd.Series(b).astype(object).reset_index(drop=True)
    if not (len(truth) == len(a) == len(b)):
        raise ValueError(
            f'paired_disagreement needs aligned inputs; got {len(truth)}, '
            f'{len(a)} and {len(b)} rows.'
        )
    keep = truth.notna()
    truth, a, b = truth[keep], a[keep], b[keep]
    a_right = (a.notna() & a.eq(truth)).to_numpy(dtype=bool)
    b_right = (b.notna() & b.eq(truth)).to_numpy(dtype=bool)
    a_only = int((a_right & ~b_right).sum())
    b_only = int((b_right & ~a_right).sum())
    discordant = a_only + b_only
    if discordant:
        tail = sum(math.comb(discordant, k) for k in range(min(a_only, b_only) + 1))
        exact_p = min(1.0, 2 * tail / 2**discordant)
        chi_square = (abs(a_only - b_only) - 1) ** 2 / discordant
        chi_square_p = math.erfc(math.sqrt(chi_square / 2))
    else:
        exact_p = chi_square = chi_square_p = float('nan')
    return {
        'n': int(keep.sum()),
        'n_both_right': int((a_right & b_right).sum()),
        'n_a_only': a_only,
        'n_b_only': b_only,
        'n_both_wrong': int((~a_right & ~b_right).sum()),
        'exact_p': exact_p,
        'chi_square': chi_square,
        'chi_square_p': chi_square_p,
    }


def bin_year_agreement(
    reference_year: pd.Series, predicted_year: pd.Series
) -> pd.Series:
    """Label how far each predicted year lands from its reference year.

    Parameters
    ----------
    reference_year, predicted_year : pandas.Series
        Years, aligned on the same index. Non-numeric values read as
        missing.

    Returns
    -------
    pandas.Series
        One of `YEAR_AGREEMENT_BINS` (`exact`, `within 1 year` for a
        difference above 0 and at most 1, `within 5 years` above 1 and at
        most 5, `more than 5 years`), or missing where either year is.
    """
    reference = pd.to_numeric(reference_year, errors='coerce').astype(float)
    predicted = pd.to_numeric(predicted_year, errors='coerce').astype(float)
    distance = (predicted - reference).abs()
    labels = pd.Series(None, index=reference.index, dtype=object)
    exact, within_1, within_5, beyond = YEAR_AGREEMENT_BINS
    labels[distance.eq(0)] = exact
    labels[distance.gt(0) & distance.le(1)] = within_1
    labels[distance.gt(1) & distance.le(5)] = within_5
    labels[distance.gt(5)] = beyond
    return labels


def year_error_summary(reference_year: pd.Series, predicted_year: pd.Series) -> dict:
    """Agreement-bin counts and error statistics for one year prediction.

    Parameters
    ----------
    reference_year, predicted_year : pandas.Series
        Years, aligned on the same index. Rows without a reference year
        are not counted at all.

    Returns
    -------
    dict
        `n_reference`, `n_answered`, one `n_` count per agreement bin,
        the cumulative shares `share_exact`, `share_within_1_year` and
        `share_within_5_years` over answered rows, `mean_absolute_error`,
        `median_absolute_error`, `bias` (mean of predicted minus
        reference, so positive means the prediction is later) and
        `abstention_rate`.
    """
    reference = pd.to_numeric(reference_year, errors='coerce').astype(float)
    predicted = pd.to_numeric(predicted_year, errors='coerce').astype(float)
    has_reference = reference.notna()
    reference, predicted = reference[has_reference], predicted[has_reference]
    answered = predicted.notna()
    error = (predicted - reference)[answered]
    distance = error.abs()
    n_answered = int(answered.sum())
    bins = bin_year_agreement(reference, predicted).value_counts()
    summary = {
        'n_reference': int(len(reference)),
        'n_answered': n_answered,
        **{
            'n_' + label.replace(' ', '_'): int(bins.get(label, 0))
            for label in YEAR_AGREEMENT_BINS
        },
        'share_exact': _ratio(int(distance.eq(0).sum()), n_answered),
        'share_within_1_year': _ratio(int(distance.le(1).sum()), n_answered),
        'share_within_5_years': _ratio(int(distance.le(5).sum()), n_answered),
        'mean_absolute_error': float(distance.mean()) if n_answered else None,
        'median_absolute_error': float(distance.median()) if n_answered else None,
        'bias': float(error.mean()) if n_answered else None,
        'abstention_rate': (
            1 - n_answered / len(reference) if len(reference) else None
        ),
    }
    return summary
