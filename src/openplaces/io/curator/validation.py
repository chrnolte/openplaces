"""Score a curated classification against hand-labeled ground-truth points.

Vocabulary-neutral and geography-neutral by construction: no class names, no
admin units, and no source paths appear here. Callers supply the labelled
points, the curated entities, and the class list, so the same code validates
building occupancy in North Carolina, land use elsewhere, or any future
labelled set (a state manufactured-housing registry, building permits) without
being edited.

Two things this module insists on that a naive accuracy check gets wrong:

- **Identity beats proximity when linking.** The nearest footprint to a survey
  pin is very often a shed or the neighbor's house, so an address match is
  tried first and distance is only the fallback. Which route matched is
  recorded, because a distance-linked row is weaker evidence than an
  address-linked one and the difference should stay visible downstream.
  Identity alone does not finish the job, though: a house and its garage
  share one address, so several entities routinely match the same point.
  Callers rank those ties with `prefer_column`; without it the pick falls to
  row order, which is arbitrary.
- **Precision and recall are reported separately, per class.** A rule that
  labels almost everything one class scores excellent recall for it, and an
  aggregate agreement figure hides the whole problem: it is entirely possible
  for overall agreement to rise while two of three classes get worse.
- **The full confusion matrix is the default output.** Every scoring
  path builds one (:func:`confusion_matrix`, reference rows, predicted
  columns, abstentions and off-vocabulary predictions kept) and derives
  producer's and consumer's accuracy from it
  (:func:`accuracy_from_matrix`); :func:`write_confusion_report` writes
  both, aggregates only, into a delivery's accuracies folder.
"""

from __future__ import annotations

from datetime import UTC
from pathlib import Path

import numpy as np
import pandas as pd

# Per-row validation outcomes. A misclassified row is, strictly, both an
# omission for its true class and a commission for the predicted one; as a
# single per-row label, 'commission' means "we asserted a class and it was
# wrong" and 'omission' means "we asserted nothing where a label exists".
RESULT_CORRECT = 'correct'
RESULT_OMISSION = 'omission'
RESULT_COMMISSION = 'commission'


def normalize_house_number(value) -> str | None:
    """Normalize a house number so it compares across dtypes.

    A number that round-tripped through a float column arrives as ``5114.0``
    while the entity side stores ``'5114'``. Comparing the raw strings matches
    nothing at all, and does so silently -- every row falls through to the
    distance fallback and the linkage still looks like it worked.
    """
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.endswith('.0'):
        text = text[:-2]
    return text or None


def link_points_to_entities(
    points,
    entities,
    *,
    number_column: str = 'address_number',
    street_column: str = 'address_street',
    lon_column: str = 'lon',
    lat_column: str = 'lat',
    max_distance_m: float = 15.0,
    street_threshold: float = 80.0,
    admin1_id: str | None = None,
    suffix: str = '_inv',
    prefer_column: str | None = None,
    prefer_values: tuple = (),
) -> pd.DataFrame:
    """Link labelled *points* to *entities*, by address first then distance.

    Parameters
    ----------
    points : pandas.DataFrame
        Labelled points, carrying *lon_column*/*lat_column* and optionally the
        address columns.
    entities : geopandas.GeoDataFrame
        Curated entities to link against; every column is carried through.
    number_column, street_column : str, optional
        Address component columns, expected on both sides. Address matching is
        skipped when either is absent from *entities*.
    max_distance_m : float, optional
        Fallback radius for rows with no address match (default 15).
    street_threshold : float, optional
        Fuzzy street-similarity cutoff, 0-100 (default 80).
    admin1_id : str, optional
        Region hint for street canonicalization (e.g. ``'US-NC'``).
    suffix : str, optional
        Appended to every entity column (default ``'_inv'``). Both sides
        routinely share column names -- the class being validated most of all
        -- and overwriting the ground-truth side would make every comparison
        trivially agree with itself.
    prefer_column : str, optional
        Entity column used to rank several entities matching one point's
        address. Entities whose value is in *prefer_values* win; remaining
        ties keep row order. Without this, an address shared by a house and
        its outbuildings resolves to whichever happens to come first.
    prefer_values : tuple, optional
        Values of *prefer_column* that mark an entity as preferred. Named by
        the caller, since this module knows no vocabulary of its own.

    Returns
    -------
    pandas.DataFrame
        One row per linked point: the point's own columns, every entity column
        suffixed, and ``matched_by`` recording ``'address'`` or ``'distance'``.
        Points that matched nothing are absent.
    """
    import geopandas as gpd

    from openplaces.geo.address import match_streets

    if points.empty or entities is None or len(entities) == 0:
        return pd.DataFrame()

    entities = entities.reset_index(drop=True)
    has_address = {number_column, street_column} <= set(entities.columns)

    matched: dict[int, int] = {}
    if has_address and {number_column, street_column} <= set(points.columns):
        # An exact house number narrows the candidates to a handful, so the
        # fuzzy street comparison stays cheap.
        by_number: dict[str, list[int]] = {}
        for position, number in enumerate(entities[number_column]):
            key = normalize_house_number(number)
            if key:
                by_number.setdefault(key, []).append(position)

        preferred = None
        if prefer_column and prefer_column in entities.columns:
            wanted = set(prefer_values)
            preferred = entities[prefer_column].isin(wanted).to_numpy()

        for index, row in points.iterrows():
            key = normalize_house_number(row.get(number_column))
            street = row.get(street_column)
            if key is None or pd.isna(street):
                continue
            candidates: list[int] = []
            for position in by_number.get(key, ()):
                if match_streets(
                    street,
                    entities[street_column].iloc[position],
                    threshold=street_threshold,
                    admin1_id=admin1_id,
                ):
                    candidates.append(position)
                    if preferred is None:
                        # Nothing to rank by, so the rest of the
                        # bucket cannot change the answer.
                        break
            if not candidates:
                continue
            if len(candidates) > 1:
                # Stable sort: preferred entities move ahead, and
                # anything still tied keeps row order.
                candidates.sort(key=lambda position: not preferred[position])
            matched[index] = candidates[0]

    nearest: dict[int, int] = {}
    remaining = points.drop(index=list(matched))
    if not remaining.empty:
        metric_crs = entities.estimate_utm_crs()
        located = gpd.GeoDataFrame(
            remaining,
            geometry=gpd.points_from_xy(remaining[lon_column], remaining[lat_column]),
            crs='EPSG:4326',
        ).to_crs(metric_crs)
        joined = gpd.sjoin_nearest(
            located[['geometry']],
            entities[['geometry']].to_crs(metric_crs),
            how='left',
            distance_col='dist_m',
            lsuffix='pt',
            rsuffix='ent',
        )
        joined = joined.sort_values('dist_m').groupby(level=0).first()
        joined = joined[joined['dist_m'] <= max_distance_m]
        nearest = joined['index_ent'].dropna().astype(int).to_dict()

    pairs = {**matched, **nearest}
    if not pairs:
        return pd.DataFrame()

    point_index = list(pairs)
    linked = points.loc[point_index].reset_index(drop=True)
    picked = entities.iloc[[pairs[i] for i in point_index]].reset_index(drop=True)

    linked['matched_by'] = [
        'address' if i in matched else 'distance' for i in point_index
    ]
    for column in picked.columns:
        linked[f'{column}{suffix}'] = picked[column].to_numpy()
    return linked


def classify_validation_result(truth: pd.Series, predicted: pd.Series) -> pd.Series:
    """Label each row ``correct``, ``omission`` or ``commission``.

    ``omission`` is reserved for rows where a label exists but nothing was
    predicted -- the classifier declined to answer. ``commission`` is an
    answer that was wrong. Keeping them apart matters because they have
    different fixes: an omission means evidence was missing or a threshold was
    too strict, a commission means the evidence present was misread.
    """
    truth_values = truth.astype(object)
    predicted_values = predicted.astype(object)
    result = pd.Series(RESULT_COMMISSION, index=truth.index, dtype=object)
    result[predicted_values.isna()] = RESULT_OMISSION
    result[predicted_values.notna() & (predicted_values == truth_values)] = (
        RESULT_CORRECT
    )
    return result


def summarize_sources(
    sources: dict[str, pd.Series], separator: str = ' | '
) -> pd.Series:
    """Render each row's per-source values as one readable string.

    Produces e.g. ``'truth: Multi-Family | nsi: Single-Family | fema: -'``, so
    a reviewer can see the whole evidence picture for a disputed row without
    scanning a dozen columns. Missing values are rendered rather than dropped:
    a source that said nothing is itself informative.
    """
    if not sources:
        return pd.Series(dtype=object)
    frame = pd.DataFrame(
        {label: series.astype(object) for label, series in sources.items()}
    )
    return frame.apply(
        lambda row: separator.join(
            f'{label}: {"-" if pd.isna(row[label]) else row[label]}'
            for label in frame.columns
        ),
        axis=1,
    )


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
) -> pd.DataFrame:
    """Count matrix of reference classes (rows) by predicted classes (columns).

    Every input row lands in exactly one cell, so the matrix total is the
    number of rows passed in. Columns separate, in order, the scored
    *classes*, the *secondary* class (an outbuilding, where one is
    named), every other asserted class (*other*, e.g. `Non-residential`)
    and no assertion at all (*abstain*). Dropping or merging those would
    hide whether a source declined to answer or answered outside the
    vocabulary, which call for different fixes.

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

    Returns
    -------
    pandas.DataFrame
        Integer counts. Rows are *classes*, then *secondary* and *other*
        (or the extra labels) where a reference falls outside *classes*,
        then `(no reference)` where a reference is missing. Columns are
        *classes*, *secondary* when given, *other* (or the extra labels),
        then *abstain*.
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

    reference = truth.where(truth.notna(), NO_REFERENCE_LABEL)
    answer = predicted.where(predicted.notna(), abstain)
    if collapse_other:
        kept = known | ({secondary} if secondary else set())
        reference = reference.where(
            reference.isin(kept) | reference.eq(NO_REFERENCE_LABEL), other
        )
        answer = answer.where(answer.isin(kept) | answer.eq(abstain), other)
        extra_columns = [*([secondary] if secondary else []), other]
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


def _as_strata(strata) -> dict[str, pd.Series]:
    """Normalize a strata argument to {stratum kind: labels}."""
    if strata is None:
        return {}
    if isinstance(strata, pd.Series):
        return {str(strata.name or 'stratum'): strata}
    return {str(kind): labels for kind, labels in strata.items()}


def _positional(values, n_rows: int, what: str) -> pd.Series:
    """A Series of *values* reset to positions, checked against *n_rows*."""
    series = pd.Series(values).reset_index(drop=True)
    if len(series) != n_rows:
        raise ValueError(
            f'{what} has {len(series)} rows; the reference has {n_rows}. '
            'Pass them positionally aligned.'
        )
    return series


def _stratum_masks(strata: dict[str, pd.Series], keep: np.ndarray, min_rows: int):
    """Pooled and per-stratum row masks, and the strata held back.

    Returns
    -------
    tuple of (list, list)
        `(stratum_by, stratum, mask)` for every stratum written, pooled
        first, and a `{stratum_by, stratum, n}` record for every stratum
        refused for holding fewer than *min_rows* rows.
    """
    n_pooled = int(keep.sum())
    if n_pooled < min_rows:
        raise ValueError(
            f'Only {n_pooled} rows carry a reference, fewer than the '
            f'minimum of {min_rows} a written stratum needs.'
        )
    groups = [('all', 'all', keep)]
    refused = []
    for kind, labels in strata.items():
        labels = labels.astype(object).where(labels.notna(), '(missing)')
        for value in sorted(labels[keep].unique(), key=str):
            mask = keep & labels.eq(value).to_numpy(dtype=bool)
            n = int(mask.sum())
            if n < min_rows:
                refused.append({'stratum_by': kind, 'stratum': str(value), 'n': n})
                continue
            groups.append((kind, str(value), mask))
    return groups, refused


def _long_form(matrix: pd.DataFrame, **labels) -> list[dict]:
    """Matrix cells as long-form records, in row-then-column order."""
    return [
        {**labels, 'reference': row, 'predicted': column, 'n': int(n)}
        for row, values in matrix.iterrows()
        for column, n in values.items()
    ]


def _write_report(
    out_dir,
    name: str,
    confusion: list[dict],
    accuracy: pd.DataFrame,
    metadata: dict,
) -> dict[str, Path]:
    """Write the long-form matrix, the accuracy table and the JSON sidecar."""
    import json
    from datetime import datetime

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        'confusion': out_dir / f'{name}_confusion.csv',
        'accuracy': out_dir / f'{name}_accuracy.csv',
        'metadata': out_dir / f'{name}_confusion.json',
    }
    pd.DataFrame.from_records(
        confusion,
        columns=['source', 'stratum_by', 'stratum', 'reference', 'predicted', 'n'],
    ).to_csv(paths['confusion'], index=False)
    accuracy.round(6).to_csv(paths['accuracy'], index=False)

    try:
        from openplaces.path import recipe_roots_footer

        recipe_roots = json.loads(recipe_roots_footer())
    except Exception:  # noqa: BLE001 - provenance is best effort
        recipe_roots = None
    record = {
        'name': name,
        **metadata,
        'orientation': 'rows are the reference, columns the prediction',
        'generated_at': datetime.now(UTC).isoformat(timespec='seconds'),
        'recipe_roots': recipe_roots,
        'files': {role: path.name for role, path in paths.items()},
    }
    paths['metadata'].write_text(
        json.dumps(record, indent=2, default=str) + '\n', encoding='utf-8'
    )
    return paths


def write_confusion_report(
    truth: pd.Series,
    predictions: dict[str, pd.Series],
    classes: list[str],
    out_dir,
    name: str,
    *,
    strata=None,
    reference: str | None = None,
    notes: str | None = None,
    tier: str | None = None,
    min_rows: int = MIN_STRATUM_ROWS,
    abstain: str = ABSTAIN_LABEL,
    other: str = OTHER_LABEL,
    secondary: str | None = None,
) -> dict[str, Path]:
    """Write confusion matrices and accuracies for every prediction source.

    The default output of every validation step. For each source, and
    for the pooled rows plus every stratum, it writes the full matrix
    (reference rows, predicted columns, abstentions and off-vocabulary
    predictions included) and the table of
    :func:`accuracy_from_matrix`. Aggregates only: nothing row-level is
    written, and a stratum with fewer than *min_rows* reference rows is
    refused and listed in the sidecar instead, so no cell can point at
    one surveyed building.

    Parameters
    ----------
    truth : pandas.Series
        Reference class per row. Rows without one are left out.
    predictions : dict of str to pandas.Series
        Source label to predicted classes, each positionally aligned to
        *truth*.
    classes : list of str
        Reference classes, in report order.
    out_dir : str or pathlib.Path
        Directory to write into, created if missing (normally
        `io.delivery.delivery_accuracy_dir`).
    name : str
        File stem shared by the three outputs.
    strata : pandas.Series or dict of str to pandas.Series, optional
        Stratum labels per row (a county id, a geometry source),
        positionally aligned to *truth*. A Series is named by its own
        name; a dict names each kind by its key.
    reference : str, optional
        What the reference is, recorded in the sidecar.
    notes : str, optional
        Free text recorded in the sidecar.
    tier : str, optional
        The match tier the rows were selected at, recorded in the
        sidecar.
    min_rows : int, optional
        Fewest reference rows a written stratum may hold (default 10).
    abstain, other, secondary : str, optional
        Sentinel labels, as in :func:`confusion_matrix`. Pass
        :meth:`ValidationContext.matrix_labels` to follow the recipe.

    Returns
    -------
    dict of str to pathlib.Path
        `confusion` (`{name}_confusion.csv`, long form: `source`,
        `stratum_by`, `stratum`, `reference`, `predicted`, `n`),
        `accuracy` (`{name}_accuracy.csv`, one row per source, stratum
        and class) and `metadata` (`{name}_confusion.json`).

    Raises
    ------
    ValueError
        If fewer than *min_rows* rows carry a reference, or a prediction
        or stratum is not aligned to *truth*.
    """
    truth = pd.Series(truth).astype(object).reset_index(drop=True)
    n_rows = len(truth)
    keep = truth.notna().to_numpy(dtype=bool)
    predictions = {
        str(label): _positional(values, n_rows, f'prediction {label!r}')
        for label, values in predictions.items()
    }
    strata = {
        kind: _positional(labels, n_rows, f'stratum {kind!r}')
        for kind, labels in _as_strata(strata).items()
    }
    groups, refused = _stratum_masks(strata, keep, min_rows)

    confusion: list[dict] = []
    tables = []
    for source, values in predictions.items():
        for stratum_by, stratum, mask in groups:
            matrix = confusion_matrix(
                truth[mask],
                values[mask],
                classes,
                abstain=abstain,
                other=other,
                secondary=secondary,
            )
            confusion += _long_form(
                matrix, source=source, stratum_by=stratum_by, stratum=stratum
            )
            table = accuracy_from_matrix(
                matrix, classes, abstain=abstain, other=other, secondary=secondary
            )
            table.insert(0, 'stratum', stratum)
            table.insert(0, 'stratum_by', stratum_by)
            table.insert(0, 'source', source)
            tables.append(table)

    return _write_report(
        out_dir,
        name,
        confusion,
        pd.concat(tables, ignore_index=True),
        {
            'kind': 'classification',
            'reference': reference,
            'notes': notes,
            'tier': tier,
            'classes': list(classes),
            'sources': list(predictions),
            'abstain_label': abstain,
            'other_label': other,
            'secondary_label': secondary,
            'n_rows': int(keep.sum()),
            'n_rows_without_reference': int(n_rows - keep.sum()),
            'min_rows': min_rows,
            'strata': sorted({kind for kind, _, _ in groups[1:]}),
            'refused_strata': refused,
        },
    )


def write_year_agreement_report(
    reference_year: pd.Series,
    predictions: dict[str, pd.Series],
    out_dir,
    name: str,
    *,
    strata=None,
    reference: str | None = None,
    notes: str | None = None,
    tier: str | None = None,
    min_rows: int = MIN_STRATUM_ROWS,
) -> dict[str, Path]:
    """Write year-built agreement bins and error statistics per source.

    A year is not a class, so the matrix has a single reference row
    (`reference year`) whose columns are the agreement bins of
    :func:`bin_year_agreement` plus the abstention column; the accuracy
    file holds :func:`year_error_summary` (bin counts, cumulative shares,
    mean and median absolute error, bias) instead of producer's and
    consumer's accuracy. Files, sidecar and the minimum-stratum guard are
    those of :func:`write_confusion_report`.

    Parameters
    ----------
    reference_year : pandas.Series
        Reference year per row. Rows without one are left out.
    predictions : dict of str to pandas.Series
        Source label to predicted years, positionally aligned.
    out_dir : str or pathlib.Path
        Directory to write into.
    name : str
        File stem shared by the three outputs.
    strata, reference, notes, tier, min_rows
        As in :func:`write_confusion_report`.

    Returns
    -------
    dict of str to pathlib.Path
        `confusion`, `accuracy` and `metadata`, as in
        :func:`write_confusion_report`.
    """
    reference_year = pd.to_numeric(
        pd.Series(reference_year).reset_index(drop=True), errors='coerce'
    )
    n_rows = len(reference_year)
    keep = reference_year.notna().to_numpy(dtype=bool)
    predictions = {
        str(label): pd.to_numeric(
            _positional(values, n_rows, f'prediction {label!r}'), errors='coerce'
        )
        for label, values in predictions.items()
    }
    strata = {
        kind: _positional(labels, n_rows, f'stratum {kind!r}')
        for kind, labels in _as_strata(strata).items()
    }
    groups, refused = _stratum_masks(strata, keep, min_rows)

    columns = [*YEAR_AGREEMENT_BINS, ABSTAIN_LABEL]
    confusion: list[dict] = []
    records = []
    for source, values in predictions.items():
        for stratum_by, stratum, mask in groups:
            bins = bin_year_agreement(reference_year[mask], values[mask])
            counts = bins.fillna(ABSTAIN_LABEL).value_counts()
            matrix = pd.DataFrame(
                [[int(counts.get(column, 0)) for column in columns]],
                index=pd.Index([YEAR_REFERENCE_LABEL], name='reference'),
                columns=pd.Index(columns, name='predicted'),
            )
            confusion += _long_form(
                matrix, source=source, stratum_by=stratum_by, stratum=stratum
            )
            records.append(
                {
                    'source': source,
                    'stratum_by': stratum_by,
                    'stratum': stratum,
                    **year_error_summary(reference_year[mask], values[mask]),
                }
            )

    return _write_report(
        out_dir,
        name,
        confusion,
        pd.DataFrame.from_records(records),
        {
            'kind': 'year_agreement',
            'reference': reference,
            'notes': notes,
            'tier': tier,
            'classes': list(YEAR_AGREEMENT_BINS),
            'sources': list(predictions),
            'abstain_label': ABSTAIN_LABEL,
            'other_label': None,
            'secondary_label': None,
            'n_rows': int(keep.sum()),
            'n_rows_without_reference': int(n_rows - keep.sum()),
            'min_rows': min_rows,
            'strata': sorted({kind for kind, _, _ in groups[1:]}),
            'refused_strata': refused,
        },
    )


def read_confusion_matrix(
    confusion, source: str, *, stratum_by: str = 'all', stratum: str = 'all'
) -> pd.DataFrame:
    """Rebuild one wide matrix from a long-form `{name}_confusion.csv`.

    Parameters
    ----------
    confusion : pandas.DataFrame or str or pathlib.Path
        The long-form table, or the path of the CSV holding it.
    source : str
        Prediction source to select.
    stratum_by, stratum : str, optional
        Stratum to select; the pooled matrix by default.

    Returns
    -------
    pandas.DataFrame
        Counts with reference rows and predicted columns, in the order
        they were written.

    Raises
    ------
    KeyError
        If the file holds no such source and stratum.
    """
    if not isinstance(confusion, pd.DataFrame):
        confusion = pd.read_csv(confusion, keep_default_na=False)
    block = confusion[
        confusion['source'].astype(str).eq(str(source))
        & confusion['stratum_by'].astype(str).eq(str(stratum_by))
        & confusion['stratum'].astype(str).eq(str(stratum))
    ]
    if block.empty:
        raise KeyError(
            f'No matrix for source {source!r}, stratum {stratum_by}={stratum}.'
        )
    matrix = block.pivot(index='reference', columns='predicted', values='n')
    matrix = matrix.reindex(
        index=list(dict.fromkeys(block['reference'])),
        columns=list(dict.fromkeys(block['predicted'])),
    ).astype(int)
    matrix.index.name = 'reference'
    matrix.columns.name = 'predicted'
    return matrix


def class_from_ruleset(
    recipe,
    terms: pd.Series,
    ruleset: str,
    *,
    reviewed_only: bool = False,
) -> pd.Series | None:
    """Reconstruct a class column from raw evidence through a recipe ruleset.

    Applies the same ordered ruleset a curate vote used, so a
    reconstructed column scores identically to one the recipe's
    formatting stage dropped from published output.

    Parameters
    ----------
    recipe : str or dict
        Curate recipe (id or dict) whose sidecar rulesets to read.
    terms : pandas.Series
        Raw label text the class column is derived from.
    ruleset : str
        Filename of the ruleset CSV beside the curate recipe.
    reviewed_only : bool, optional
        Keep only matches whose winning rule is marked reviewed,
        mirroring the recipe's own flag. Nulling unreviewed matches
        after the fact, rather than dropping those rules up front, is
        deliberate: pre-filtering would let a term fall through to a
        later reviewed rule and assert a class the vote never saw.

    Returns
    -------
    pandas.Series or None
        The reconstructed class column, or None when the ruleset cannot
        be located, leaving the caller to omit that source.
    """
    from types import SimpleNamespace

    from openplaces.io.curator.occupancy import load_ruleset, match_ruleset
    from openplaces.recipe import get_recipe_by_id

    if isinstance(recipe, str):
        recipe = get_recipe_by_id(recipe)
    try:
        state = SimpleNamespace(recipe=recipe)
        rules = load_ruleset(state, ruleset)
    except (FileNotFoundError, KeyError):
        return None
    proposal, reviewed = match_ruleset(terms.astype(object), rules)
    if reviewed_only:
        proposal = proposal.where(reviewed)
    return proposal


def reference_confidence_tier(
    frame: pd.DataFrame,
    *,
    count_column: str = 'n_permits_with_occupancy_type',
    mode_pct_column: str = 'occupancy_type_mode_pct',
    mode_column: str = 'occupancy_type_mode',
    matched_via_column: str = 'matched_via',
    id_match_value: str = 'parcel_id_local',
) -> pd.Series:
    """Confidence tier for a reference-label table, high to low.

    Two things separate a strong claim from a weak one: how the
    reference reached the entity (an id join beats an address or point
    match) and whether its records agree (a unanimous mode over at
    least two label-bearing records beats a single uncorroborated one).
    A point match scores in the `addr` tiers, with an address match:
    both locate the parcel rather than naming it, which is why
    `id_match_value` names only the id route.

    Parameters
    ----------
    frame : pandas.DataFrame
        Entity-keyed reference labels carrying the four columns named by
        the keyword arguments (the permit pair-table schema by default).

    Returns
    -------
    pandas.Series
        One of `1_id_strong`, `2_id_weak`, `3_addr_strong`,
        `4_addr_weak`, or `none` where no record names a label.
    """
    n_labels = pd.to_numeric(frame[count_column], errors='coerce')
    unanimous = frame[mode_pct_column].ge(0.999) & n_labels.ge(2)
    by_id = frame[matched_via_column].eq(id_match_value)
    spoke = frame[mode_column].notna()
    tier = pd.Series('none', index=frame.index)
    tier[spoke & ~by_id & ~unanimous] = '4_addr_weak'
    tier[spoke & ~by_id & unanimous] = '3_addr_strong'
    tier[spoke & by_id & ~unanimous] = '2_id_weak'
    tier[spoke & by_id & unanimous] = '1_id_strong'
    return tier


class ValidationContext:
    """Everything a validation notebook needs, built from recipe data.

    A curate recipe declares its validation configuration in a
    `validation:` block, the same way delivery columns live in `share:`:
    the hand-labelled reference it is scored against, the class
    vocabulary and how the inventory's finer bands collapse onto it,
    linkage thresholds, and the evidence columns each vote input is
    scored from. Reference tables whose source is licence-restricted
    are declared in an untracked sidecar
    (`{recipe_id}_validation-references.yaml` beside the recipe) that
    is merged over the committed block when present, so the committed
    surface never names such a source.

    Notebooks build one context and read the same names the block
    declares; nothing geography- or source-specific lives in this
    class.
    """

    def __init__(self, recipe, references_state=None):
        """Build a context for one curate recipe.

        Parameters
        ----------
        recipe : str or dict
            Curate recipe id or dict carrying a `validation:` block.
        references_state : str, optional
            Which entry of the sidecar's `references:` mapping to
            activate (e.g. a state code). Without it the
            reference-table helpers raise when used.
        """
        from openplaces.recipe import get_recipe_by_id, get_recipe_id

        if isinstance(recipe, str):
            recipe = get_recipe_by_id(recipe)
        self.recipe = recipe
        self.recipe_id = get_recipe_id(recipe)
        config = dict(recipe.get('validation') or {})
        sidecar = self._load_references_sidecar()
        if sidecar:
            config = {**config, **sidecar}
        if not config:
            raise ValueError(
                f'{self.recipe_id} declares no validation: block and has '
                'no validation-references sidecar.'
            )
        self.config = config
        self.classes = tuple(config.get('classes') or ())
        self.collapse = dict(config.get('collapse') or {})
        self.single_dwelling_classes = tuple(
            config.get('single_dwelling_classes') or ()
        )
        link = dict(config.get('link') or {})
        self.max_distance_m = link.get('max_distance_m', 15)
        self.street_threshold = link.get('street_threshold', 80.0)
        self.prefer_column = link.get('prefer_column')
        self.prefer_values = tuple(link.get('prefer_values') or ())
        self.prediction_key = list(config.get('prediction_key') or [])
        self.class_map = config.get('class_map')
        self.keyword_ruleset = config.get('keyword_ruleset')
        self.source_columns = dict(config.get('source_columns') or {})
        self.derived_source_columns = dict(config.get('derived_source_columns') or {})
        self.dwelling_count_column = config.get('dwelling_count_column')
        self.inventory_suffix = config.get('inventory_suffix', '_inv')
        occupancy = dict(recipe.get('occupancy') or {})
        self.secondary_class = occupancy.get('secondary_class')
        self.residential_classes = tuple(occupancy.get('residential_classes') or ())
        self.references_state = references_state
        self.reference = None
        if references_state is not None:
            table = dict(config.get('references') or {})
            if references_state not in table:
                raise KeyError(
                    f'No validation reference declared for '
                    f'{references_state!r}; the untracked sidecar '
                    f'{self.recipe_id}_validation-references.yaml '
                    f'declares: {sorted(table)}'
                )
            self.reference = dict(table[references_state])

    # Configuration resolution

    def _load_references_sidecar(self):
        import yaml

        from openplaces.path import recipe_path

        base = recipe_path(
            self.recipe.get('admin_id'),
            self.recipe.get('entity') or self.recipe.get('dataset'),
            # recipe_path prefixes the recipe id itself
            filename='validation-references',
        )
        path = Path(str(base))
        if path.suffix != '.yaml':
            path = path.with_suffix('.yaml')
        if not path.exists():
            return {}
        with open(path, encoding='utf-8') as f:
            return yaml.safe_load(f) or {}

    @property
    def ground_truth_path(self):
        from openplaces.core.schema import Entity
        from openplaces.path import external_dir

        spec = dict(self.config.get('ground_truth') or {})
        entity = Entity(spec['entity_type'], spec['source'], str(spec['version']))
        return external_dir(spec['admin_id'], entity=entity) / spec['filename']

    def _cache_path(self, filename=None, **kwargs):
        from openplaces.path import cache_path

        return cache_path(
            str(self.recipe.get('admin_id') or 'US'),
            entity=self.recipe.get('entity'),
            filename=filename,
            **kwargs,
        )

    @property
    def validation_dir(self):
        return self._cache_path(as_dir=True)

    @property
    def linked_path(self):
        return self._cache_path('validation-footprints')

    @property
    def baseline_path(self):
        return self._cache_path('occupancy-baseline', default_extension='csv')

    @property
    def baseline_predictions_path(self):
        return self._cache_path(
            'occupancy-baseline-predictions', default_extension='csv'
        )

    # Reference tables (entity-keyed label pairs), from the sidecar

    def _reference_dir(self):
        from openplaces.core.schema import Entity
        from openplaces.path import external_dir

        if not self.reference:
            raise ValueError(
                'This context was built without references_state; pass '
                "one, e.g. ValidationContext(recipe, 'NC')."
            )
        entity = Entity(
            self.reference['entity_type'],
            self.reference['source'],
            str(self.reference['version']),
        )
        return external_dir(
            self.reference['admin_id'], entity=entity
        ) / self.reference.get('subdir', 'validation')

    @property
    def reference_region(self):
        return self.reference['region'] if self.reference else None

    @property
    def reference_dir(self):
        return self._reference_dir()

    @property
    def reference_strong_tiers(self):
        return tuple(
            (self.reference or {}).get(
                'strong_tiers', ('1_id_strong', '2_id_weak', '3_addr_strong')
            )
        )

    def reference_admin_ids(self):
        """Admin units with a complete footprint+parcel pair on disk.

        An incomplete pair means a mid-write unit, not a unit without
        records, so it is skipped rather than read. The admin id
        pattern is anchored to the sidecar's declared admin scope and
        code width so files written under superseded id mints cannot
        double-count their units.
        """
        import re

        scope = self.reference['admin_id']
        width = int(self.reference.get('admin_code_width', 3))
        kinds = {}
        for path in sorted(
            self._reference_dir().glob(f'{scope}-*_occupancy_validation.parquet')
        ):
            match = re.match(
                rf'({re.escape(scope)}-\w{{{width}}})_'
                r'(footprint|parcel)_occupancy_validation',
                path.stem,
            )
            if match:
                kinds.setdefault(match.group(1), set()).add(match.group(2))
        return sorted(c for c, k in kinds.items() if k == {'footprint', 'parcel'})

    def load_reference(self, admin_id, kind='footprint'):
        """Load one unit's entity-level reference labels, or None."""
        path = self._reference_dir() / f'{admin_id}_{kind}_occupancy_validation.parquet'
        if not path.exists():
            return None
        frame = pd.read_parquet(path)
        frame.index.name = f'{kind}_id'
        return frame

    def reference_tier(self, frame):
        """Confidence tier of loaded reference labels (module helper)."""
        return reference_confidence_tier(frame)

    # Stands in for "no class" when two class columns are compared, so a
    # missing value on either side compares equal to itself and unequal
    # to every real class.
    _NO_CLASS = '<none>'

    def link_ground_truth(self, counties=None, *, verbose=False, save=True):
        """Link the hand-labelled points to curated entities, per unit.

        Address identity is tried before proximity (see
        :func:`link_points_to_entities`): the nearest footprint to a
        survey pin is very often a shed or the neighbour's house.

        Parameters
        ----------
        counties : tuple of str, optional
            Admin units to link. Defaults to :meth:`survey_admin_ids`.
        verbose : bool, optional
            Report per-unit linkage counts.
        save : bool, optional
            Write the linked frame to :attr:`linked_path` (parquet, plus
            a CSV sidecar for review). Default True.

        Returns
        -------
        geopandas.GeoDataFrame
            One row per linked point: the reference columns, every
            curated column suffixed with :attr:`inventory_suffix`,
            `matched_by`, and the derived comparison columns
            `predicted`, `is_single_dwelling`, `validation_result`,
            `occupancy_type_conflict_sources` and `sources_disagree`.
            The geometry is the matched entity, not the reference pin.
        """
        import geopandas as gpd

        import openplaces as op

        counties = tuple(counties) if counties else self.survey_admin_ids()
        admin1_id = str((self.config.get('ground_truth') or {}).get('admin_id') or '')
        frames = []
        crs = None
        for admin_id in counties:
            points = self.load_ground_truth((admin_id,))
            if points.empty:
                continue
            entities = op.get_entities(
                self.recipe_id, admin_id, geom=True, missing='ignore'
            )
            if entities is None or entities.empty:
                if verbose:
                    print(f'{admin_id}: no curated output on disk, skipped')
                continue
            crs = entities.crs
            # reset_index carries the entity id through as a column, so
            # a reference-table notebook can join on an id that is
            # stable across branches.
            linked = link_points_to_entities(
                points,
                entities.reset_index(),
                max_distance_m=self.max_distance_m,
                street_threshold=self.street_threshold,
                admin1_id=admin1_id or None,
                prefer_column=self.prefer_column,
                prefer_values=self.prefer_values,
            )
            if linked.empty:
                if verbose:
                    print(f'{admin_id}: {len(points)} points, none linked')
                continue
            if verbose:
                by_route = linked['matched_by'].value_counts()
                print(
                    f'{admin_id}: {len(linked)}/{len(points)} points linked '
                    f'(address {by_route.get("address", 0)}, '
                    f'distance {by_route.get("distance", 0)})'
                )
            frames.append(linked)

        if not frames:
            return gpd.GeoDataFrame()

        suffix = self.inventory_suffix
        linked = pd.concat(frames, ignore_index=True)
        linked['predicted'] = self.collapse_bands(linked[f'occupancy_type{suffix}'])
        linked['is_single_dwelling'] = linked['occupancy_type_canonical'].isin(
            self.single_dwelling_classes
        )
        linked['validation_result'] = classify_validation_result(
            linked['occupancy_type_canonical'], linked['predicted']
        )

        sources = self.source_values(linked)
        linked['occupancy_type_conflict_sources'] = summarize_sources(
            {'ground_truth': linked['occupancy_type_canonical'], **sources}
        )
        # Flag the rows worth reading by hand: any input that spoke and
        # was overruled. Both sides compare through a sentinel: a row
        # the vote declined to classify makes `ne` return pd.NA on a
        # nullable column, and reading a missing vote as a disagreement
        # is the intended answer, not a convenience.
        predicted = linked['predicted'].astype(object).fillna(self._NO_CLASS)
        disagree = pd.Series(False, index=linked.index)
        for label, values in sources.items():
            if label == 'final_vote':
                continue
            values = values.astype(object)
            disagree |= values.notna() & values.fillna(self._NO_CLASS).ne(predicted)
        linked['sources_disagree'] = disagree

        # link_points_to_entities returns a plain DataFrame, so the
        # matched geometry arrives as a CRS-less object column; restore
        # it from the entities it came from.
        linked = gpd.GeoDataFrame(
            linked.drop(columns=f'geometry{suffix}'),
            geometry=gpd.GeoSeries(linked[f'geometry{suffix}'], crs=crs),
        )
        if save:
            self.validation_dir.mkdir(parents=True, exist_ok=True)
            linked.to_parquet(self.linked_path)
            linked.drop(columns='geometry').to_csv(
                Path(str(self.linked_path)).with_suffix('.csv'), index=False
            )
            if verbose:
                print(f'wrote {len(linked)} linked points to {self.linked_path}')
        return linked

    # Vocabulary

    def collapse_bands(self, values):
        """Map the inventory's finer class bands onto the reference's."""
        return values.astype(object).replace(self.collapse)

    def class_from_ruleset(self, terms, ruleset=None, **kwargs):
        """Recipe-bound wrapper for the module-level class_from_ruleset."""
        return class_from_ruleset(
            self.recipe, terms, ruleset or self.class_map, **kwargs
        )

    # Survey ground truth

    def survey_admin_ids(self):
        """Admin units present in the ground-truth table, sorted.

        Discovered rather than hardcoded: the table's own admin column
        already reflects points that landed outside their declared
        source sheet.
        """
        path = self.ground_truth_path
        if not path.exists():
            spec = dict(self.config.get('ground_truth') or {})
            raise FileNotFoundError(
                f'Ground truth not found at {path}. ' + spec.get('regenerate_hint', '')
            )
        admin_ids = pd.read_csv(path, usecols=['admin_id'])['admin_id']
        return tuple(sorted(admin_ids.dropna().unique()))

    def load_ground_truth(self, counties=None):
        """Load the hand-labelled points, optionally restricted by unit."""
        points = pd.read_csv(self.ground_truth_path)
        points = points[points['admin_id'].notna()]
        if counties:
            points = points[points['admin_id'].isin(tuple(counties))]
        return points.reset_index(drop=True)

    # Sources and scoring

    def source_values(self, linked):
        """The vote and each input it arbitrates, on the reference vocabulary.

        Parameters
        ----------
        linked : pandas.DataFrame
            Linked frame carrying curated columns under
            `inventory_suffix`.

        Returns
        -------
        dict of str to pandas.Series
            Source label to comparable class values, bands collapsed. A
            derived source declaring `keep_unmapped: true` keeps its raw
            evidence value wherever the ruleset maps it to no class, so
            a source that asserts a class outside the ruleset (NSI's
            `Agricultural`, say) scores as that assertion rather than as
            no prediction; a row without raw evidence stays missing.
        """
        values = {
            label: self.collapse_bands(linked[column])
            for label, column in self.source_columns.items()
            if column in linked.columns
        }
        for label, spec in self.derived_source_columns.items():
            column = spec['column'] + self.inventory_suffix
            if column not in linked.columns:
                continue
            keep_unmapped = spec.get('keep_unmapped', False)
            if label in values and not keep_unmapped:
                continue
            ruleset = spec.get('ruleset', self.class_map)
            derived = self.class_from_ruleset(
                linked[column],
                ruleset,
                reviewed_only=spec.get('reviewed_only', False),
            )
            if label in values:
                # A class column, where present, already holds this
                # ruleset's answer; only its unmapped rows are filled.
                derived = values[label]
            elif derived is None:
                continue
            derived = self.collapse_bands(derived).astype(object)
            if keep_unmapped:
                derived = self._with_unmapped_evidence(derived, linked[column], ruleset)
            values[label] = derived
        count_col = self.dwelling_count_column
        if count_col and count_col in linked.columns:
            dwellings = pd.to_numeric(linked[count_col], errors='coerce')
            # No dwelling record is not a count of zero. Filling it made
            # "Overture never saw this entity" indistinguishable from
            # "Overture saw one dwelling", so the source was credited and
            # penalized for a Single-Family it never asserted. Absent
            # evidence stays missing, and score_classification's
            # predicted.notna() gate excludes it from this source's score.
            implied = pd.Series(None, index=linked.index, dtype=object)
            observed = dwellings.notna()
            implied[observed & (dwellings >= 2)] = 'Multi-Family'
            implied[observed & (dwellings < 2)] = 'Single-Family'
            values['overture'] = implied
        return values

    def _with_unmapped_evidence(self, classes, raw, ruleset):
        """Fill rows no rule matched with the raw evidence value itself.

        Before this, a class map covering residential classes only
        turned every non-residential assertion (NSI's `Professional
        Technical Services` on a surveyed manufactured home) into a
        missing prediction, so the matrices could not tell "the source
        said nothing" from "the source said something non-residential".
        Rows a rule matched keep the ruleset's answer, even one nulled
        for being unreviewed, and blank raw values stay missing.
        """
        raw = raw.astype(object)
        present = raw.notna() & raw.astype(str).str.strip().ne('')
        matched = self.class_from_ruleset(raw, ruleset)
        unmatched = (
            matched.isna() if matched is not None else pd.Series(True, raw.index)
        )
        fill = classes.isna() & unmatched & present
        return classes.mask(fill, raw)

    def matrix_labels(self):
        """Confusion-matrix sentinel labels that follow this recipe.

        Returns
        -------
        dict
            `secondary` (the recipe's `occupancy: secondary_class`) and
            `other`, which reads `Non-residential` when the recipe
            declares its residential classes. Pass as keyword arguments
            to :func:`write_confusion_report` or
            :func:`confusion_matrix`.
        """
        residential = getattr(self, 'residential_classes', ())
        return {
            'secondary': getattr(self, 'secondary_class', None),
            'other': NON_RESIDENTIAL_LABEL if residential else OTHER_LABEL,
        }

    def entity_source_values(self, entities):
        """:meth:`source_values` for curated entities read straight from disk.

        Scoring against an entity-keyed reference (permits) starts from
        the curated table itself rather than a survey linkage, so its
        columns carry no inventory suffix. Suffixing them here lets both
        references score exactly the same sources, reconstructed the same
        way.

        Parameters
        ----------
        entities : pandas.DataFrame
            Curated entity columns, unsuffixed. `occupancy_type` becomes
            the vote.

        Returns
        -------
        dict of str to pandas.Series
            As :meth:`source_values`, indexed like *entities*.
        """
        frame = pd.DataFrame(entities).drop(columns='geometry', errors='ignore')
        frame = frame.add_suffix(self.inventory_suffix)
        if 'occupancy_type' in entities.columns:
            frame['predicted'] = self.collapse_bands(entities['occupancy_type'])
        return self.source_values(frame)

    def survey_strata(self, linked):
        """Strata the survey matrices are split by, where present.

        The county the point was surveyed in, and the source of the
        footprint geometry it matched (a building outline traced from
        imagery and a parcel-derived placeholder fail differently).
        """
        strata = {}
        if 'admin_id' in linked.columns:
            strata['county'] = linked['admin_id']
        geometry_source = f'geometry_source{self.inventory_suffix}'
        if geometry_source in linked.columns:
            strata['geometry_source'] = linked[geometry_source]
        return strata

    def score_sources(
        self,
        linked,
        out_dir=None,
        *,
        name=None,
        strata=None,
        min_rows=MIN_STRATUM_ROWS,
        notes=None,
    ):
        """Score the vote and each of its inputs against the hand labels.

        Parameters
        ----------
        linked : pandas.DataFrame
            Output of :meth:`link_ground_truth`.
        out_dir : str or pathlib.Path, optional
            When given, confusion matrices and producer's/consumer's
            accuracies for every scored source are written there through
            :func:`write_confusion_report`, pooled and per stratum. This
            is the default output of a validation run; the return value
            is unchanged either way.
        name : str, optional
            File stem, default `{recipe_id}_occupancy-survey`.
        strata : dict of str to pandas.Series, optional
            Default :meth:`survey_strata`.
        min_rows : int, optional
            Fewest points a written stratum may hold (default 10).
        notes : str, optional
            Recorded in the report's JSON sidecar.

        Returns
        -------
        pandas.DataFrame
            :func:`score_classification` per source, with a `source`
            column in front.
        """
        truth = linked['occupancy_type_canonical']
        sources = self.source_values(linked)
        tables = []
        for label, values in sources.items():
            table = score_classification(truth, values, list(self.classes))
            table.insert(0, 'source', label)
            tables.append(table)
        if out_dir is not None:
            spec = dict(self.config.get('ground_truth') or {})
            write_confusion_report(
                truth,
                sources,
                list(self.classes),
                out_dir,
                name or f'{self.recipe_id}_occupancy-survey',
                strata=self.survey_strata(linked) if strata is None else strata,
                reference=' '.join(
                    str(part)
                    for part in ('survey', spec.get('source'), spec.get('version'))
                    if part
                ),
                notes=notes,
                min_rows=min_rows,
                **self.matrix_labels(),
            )
        return pd.concat(tables, ignore_index=True)

    # Which notebooks score which delivered region

    def reference_regions(self):
        """Delivery region each declared reference scores, by reference key.

        `ground_truth` maps to the survey's `region`; every entry of the
        sidecar's `references:` to its own `region`. A reference without a
        region, or a sidecar that is absent, contributes nothing.
        """
        regions = {}
        survey_region = (self.config.get('ground_truth') or {}).get('region')
        if survey_region:
            regions['ground_truth'] = str(survey_region)
        for key, spec in (self.config.get('references') or {}).items():
            if (spec or {}).get('region'):
                regions[str(key)] = str(spec['region'])
        return regions

    def notebooks_for_region(self, region):
        """Validation notebooks that score one delivered region.

        Read from the `notebooks:` list of the `validation:` block, each
        entry naming a notebook (relative to the repository root) and
        the reference it scores against. A notebook whose reference is
        not available (the untracked sidecar is absent) is not listed.

        Parameters
        ----------
        region : str
            A delivery region id.

        Returns
        -------
        list of str
            Notebook paths, in declared order.
        """
        regions = self.reference_regions()
        return [
            str(entry['notebook'])
            for entry in self.config.get('notebooks') or []
            if regions.get(str(entry.get('reference'))) == str(region)
        ]

    # Baseline bookkeeping for the paired gate

    def save_baseline_predictions(self, linked, path=None):
        """Write the accepted run's per-point predictions."""
        path = Path(path or self.baseline_predictions_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = [*self.prediction_key, 'occupancy_type_canonical', 'predicted']
        linked[columns].to_csv(path, index=False)
        return path

    def load_baseline_predictions(self, path=None):
        """Read the baseline predictions, failing with a how-to hint."""
        path = Path(path or self.baseline_predictions_path)
        if not path.exists():
            raise FileNotFoundError(
                f'No baseline predictions at {path}. Run the validation '
                'once with --write_baseline to record the accepted run '
                'before gating against it.'
            )
        return pd.read_csv(path)

    def align_to_baseline(self, linked, baseline):
        """Line the current run's predictions up with the baseline's.

        Returns the points both runs share; points only one run has are
        counted in the report rather than silently dropped, so the gate
        cannot quietly score a different set of buildings than the
        baseline did.
        """
        key = self.prediction_key
        current = linked[[*key, 'occupancy_type_canonical', 'predicted']].copy()
        merged = current.merge(
            baseline, on=key, how='inner', suffixes=('', '_base'), validate='1:1'
        )
        report = {
            'n_shared': len(merged),
            'n_baseline_only': len(baseline) - len(merged),
            'n_current_only': len(current) - len(merged),
        }
        report['n_truth_changed'] = int(
            merged['occupancy_type_canonical']
            .astype(object)
            .ne(merged['occupancy_type_canonical_base'].astype(object))
            .sum()
        )
        return (
            merged['occupancy_type_canonical'],
            merged['predicted_base'],
            merged['predicted'],
            report,
        )

    @staticmethod
    def check_baseline_coverage(table, baseline):
        """Fail loudly when a baseline row finds no counterpart in table.

        The gate merges on (source, class); a source missing from the
        scored table would silently shrink the comparison while the
        gate still reports a pass.
        """
        expected = set(map(tuple, baseline[['source', 'class']].to_numpy()))
        actual = set(map(tuple, table[['source', 'class']].to_numpy()))
        missing = sorted(expected - actual)
        if missing:
            raise SystemExit(
                f'FAIL: {len(missing)} baseline row(s) had no counterpart '
                f'to compare against, so the gate would have scored only '
                f'{len(actual)} of {len(expected)} rows: {missing}'
            )


def validation_context(recipe, references_state=None):
    """Build a :class:`ValidationContext`; see the class docstring."""
    return ValidationContext(recipe, references_state)


def validation_notebooks(recipe, region) -> list[str]:
    """Validation notebooks declared for one delivery region of a recipe.

    Parameters
    ----------
    recipe : str or dict
        Curate recipe id or dict.
    region : str
        Delivery region id.

    Returns
    -------
    list of str
        Notebook paths relative to the repository root; empty when the
        recipe declares no `validation:` block or nothing scores
        *region*.
    """
    try:
        context = ValidationContext(recipe)
    except ValueError:
        return []
    return context.notebooks_for_region(region)
