"""Ordered label rulesets: read one, apply it to a column of labels.

A ruleset is a CSV of rows ``pattern``, ``match_type`` (contains |
regex; default contains), an output class column and an optional
``reviewed`` flag. Rows apply top to bottom and the first match wins,
case-insensitively. The curate stage classes labels with these files
(``io.curator.occupancy``); this module sits in ``core`` so that a
harmonize step can apply the same file rather than keep a second list
of terms (``consolidate_condo_cluster_footprints`` reads the parcel
curate's land-use keywords).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd


def as_matchable_text(series: pd.Series) -> pd.Series:
    """Return *series* as nullable-string text safe for the ``.str`` accessor.

    A code column can arrive numeric (a land-use or occupancy code read
    without a string cast), and pandas' ``.str`` accessor raises
    AttributeError on an integer dtype rather than simply matching
    nothing, which aborts a whole county's curate. This is the same dtype
    slip :func:`~openplaces.io.curator._coerce_registry_numerics` absorbs
    on the numeric side. Whole-numbered values render without a decimal
    point, so a float-typed code column matches the pattern a person
    wrote for it (92, not 92.0). Missing values stay missing.

    Parameters
    ----------
    series : pandas.Series
        Column of labels or codes to match patterns against.

    Returns
    -------
    pandas.Series
        The same values as pandas' nullable string dtype.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.astype('string')
    if pd.api.types.is_float_dtype(series):
        finite = series.dropna()
        if len(finite) and np.isfinite(finite).all() and (finite % 1 == 0).all():
            return series.astype('Int64').astype('string')
    return series.astype('string')


def read_ruleset(csv_path: pathlib.Path, class_column: str | None = None) -> list[dict]:
    """Read an ordered label ruleset CSV.

    Parameters
    ----------
    csv_path : pathlib.Path
        The ruleset file.
    class_column : str, optional
        CSV column holding the output class. Defaults to whichever of
        ``occupancy_type`` or ``land_use_class`` the file has.

    Returns
    -------
    list of dict
        One rule per row, in file order, with the keys ``pattern``,
        ``match_type``, ``occupancy_type`` (the output class, whatever
        the CSV calls it) and ``reviewed``.

    Raises
    ------
    FileNotFoundError
        If *csv_path* does not exist.
    KeyError
        If the file has no *class_column*.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f'Classification ruleset not found: {csv_path}')
    table = pd.read_csv(csv_path)
    if class_column is None:
        class_column = next(
            (c for c in ('occupancy_type', 'land_use_class') if c in table.columns),
            'occupancy_type',
        )
    if class_column not in table.columns:
        raise KeyError(
            f'Ruleset {csv_path.name} has no {class_column!r} column; '
            f'found {list(table.columns)}.'
        )
    rules = []
    for _, row in table.iterrows():
        rules.append(
            {
                'pattern': str(row['pattern']),
                'match_type': str(row.get('match_type', 'contains')).strip().lower(),
                'occupancy_type': str(row[class_column]),
                'reviewed': str(row.get('reviewed', '')).strip().lower()
                in ('true', '1', 'yes'),
            }
        )
    return rules


def match_ruleset(terms: pd.Series, rules: list[dict]) -> tuple[pd.Series, pd.Series]:
    """Apply an ordered label ruleset, returning (matched class, reviewed).

    First match wins, mirroring
    :func:`~openplaces.io.curator.occupancy.coerce_to_class`, except
    that a term matching no rule yields a missing class rather than
    passing through unchanged, because this answers "which class did
    the label assert?", not "normalize this label".

    Parameters
    ----------
    terms : pandas.Series
        Labels or codes.
    rules : list of dict
        From :func:`read_ruleset`.

    Returns
    -------
    tuple of pandas.Series
        The matched class (missing where no rule matched) and whether
        the matching rule is marked reviewed.
    """
    text = as_matchable_text(terms)
    proposal = pd.Series(pd.NA, index=terms.index, dtype=object)
    reviewed = pd.Series(False, index=terms.index)
    unmatched = pd.Series(True, index=terms.index)
    for rule in rules:
        mask = unmatched & text.str.contains(
            rule['pattern'],
            case=False,
            na=False,
            regex=rule['match_type'] == 'regex',
        ).astype(bool)
        if mask.any():
            proposal.loc[mask] = rule['occupancy_type']
            reviewed.loc[mask] = rule['reviewed']
            unmatched.loc[mask] = False
    return proposal, reviewed
