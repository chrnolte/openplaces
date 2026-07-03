"""Shared weighted-indicator vocabulary for the vote-based curation steps.

A single predicate language reused by every step that tallies weighted
evidence toward a class (``resolve_by_vote`` in ``reconcilers.py``,
``classify_parcel_land_use`` in ``inferers.py``): each indicator references
existing columns and contributes a boolean vote, so any recipe can mix and
match the same vocabulary regardless of which voting step it configures.
"""

from __future__ import annotations

import pandas as pd


def evaluate_indicator(curated: pd.DataFrame, indicator: dict) -> pd.Series:
    """Return a boolean Series marking rows that satisfy one voting indicator.

    Indicators reference columns by name; any referenced column that is absent
    yields an all-False result, so the indicator simply contributes no votes.

    Supported ``type`` values:
    - ``value_share_below``: ``value / sum(total) < max_ratio``. With
      ``include_zero`` true, a zero ``value`` also matches (covers a zero total).
    - ``keyword``: case-insensitive ``str.contains(pattern)`` on ``column``
      (``regex`` defaults to true; set false for a literal substring match).
    - ``equals``: ``column`` equals ``value``.
    - ``in_set``: ``column`` value is in ``values``.
    - ``numeric_at_least`` (alias ``count_at_least``): ``column >= min``.
    - ``numeric_at_most``: ``column <= max``.
    """
    false = pd.Series(False, index=curated.index)
    kind = indicator['type']

    if kind == 'value_share_below':
        value_col = indicator['value']
        total_cols = indicator['total']
        if value_col not in curated.columns or any(
            c not in curated.columns for c in total_cols
        ):
            return false
        value = pd.to_numeric(curated[value_col], errors='coerce')
        total = sum(pd.to_numeric(curated[c], errors='coerce') for c in total_cols)
        max_ratio = float(indicator['max_ratio'])
        ratio = value.where(total > 0) / total.where(total > 0)
        matched = (ratio < max_ratio).fillna(False)
        if indicator.get('include_zero'):
            matched = matched | (value == 0).fillna(False)
        return matched

    col = indicator.get('column')
    if col is None or col not in curated.columns:
        return false

    if kind == 'keyword':
        return (
            curated[col]
            .astype(object)
            .str.contains(
                indicator['pattern'],
                case=False,
                na=False,
                regex=bool(indicator.get('regex', True)),
            )
        )

    if kind == 'equals':
        return (curated[col].astype(object) == indicator['value']).fillna(False)

    if kind == 'in_set':
        return curated[col].astype(object).isin(set(indicator['values'])).fillna(False)

    if kind in ('numeric_at_least', 'count_at_least'):
        return (
            pd.to_numeric(curated[col], errors='coerce') >= float(indicator['min'])
        ).fillna(False)

    if kind == 'numeric_at_most':
        return (
            pd.to_numeric(curated[col], errors='coerce') <= float(indicator['max'])
        ).fillna(False)

    raise ValueError(f'Unknown voting indicator type: {kind!r}.')
