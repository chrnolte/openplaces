"""Write and read the aggregate confusion and year-agreement
reports, stratified, never below the minimum stratum size.
"""

from __future__ import annotations

from datetime import UTC
from pathlib import Path

import numpy as np
import pandas as pd

from openplaces.io.curator.validation.accuracy import (
    ABSTAIN_LABEL,
    MIN_STRATUM_ROWS,
    OTHER_LABEL,
    YEAR_AGREEMENT_BINS,
    YEAR_REFERENCE_LABEL,
    accuracy_from_matrix,
    bin_year_agreement,
    confusion_matrix,
    year_error_summary,
)


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
    kept_classes: list[str] | tuple[str, ...] | None = None,
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
    kept_classes : list of str, optional
        Unscored classes kept apart from *other*, as in
        :func:`confusion_matrix`.

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
                kept_classes=kept_classes,
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
            'kept_classes': list(kept_classes or ()),
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
