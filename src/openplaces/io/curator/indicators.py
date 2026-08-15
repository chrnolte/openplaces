"""Shared voting vocabulary and scoring cores for the curation steps.

A single predicate language reused by every step that tallies weighted
evidence toward a class (:func:`evaluate_indicator`), plus the two scoring
cores that consume it. Every classification in the curate stage resolves
through one of exactly two mechanisms, both living here:

- :func:`score_decisions` -- an *enumerated* vote: the recipe lists the
  candidate classes, each with its own weighted indicators, ``min_score``,
  and optional hard-veto ``require`` list.
- :func:`vote_dynamic_values` -- an *open-vocabulary* vote: the candidate
  classes are whatever values the evidence columns happen to hold, so they
  cannot be enumerated in the recipe; weight is pooled at an optional
  coarser bucket granularity.

Both share one tie-break rule -- strict ``>``, so the earliest-listed
candidate wins a tie and recipe order is precedence.
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
    - ``value_share_at_least``: ``value / sum(total) >= min_ratio``, the mirror
      of ``value_share_below``. No ``include_zero`` option: a zero ``value``
      can never satisfy ``>= min_ratio`` for any positive ``min_ratio``.
    - ``keyword``: case-insensitive ``str.contains(pattern)`` on ``column``
      (``regex`` defaults to true; set false for a literal substring match).
    - ``equals``: ``column`` equals ``value``.
    - ``in_set``: ``column`` value is in ``values``.
    - ``numeric_at_least`` (alias ``count_at_least``): ``column >= min``.
    - ``numeric_at_most``: ``column <= max``.
    - ``any_of``: true where any of the nested ``indicators`` matches. Lets a
      backing signal corroborate an existing indicator (e.g. an independent
      source agreeing with a generic column) without contributing an extra
      weighted vote of its own -- the whole group counts once.
    - ``all_of``: true only where every nested indicator matches, again
      counting once. For a composite signal whose parts mean nothing apart --
      an elongation ratio is evidence of a manufactured home only together
      with a small footprint area, since a long warehouse satisfies the
      ratio alone.
    """
    false = pd.Series(False, index=curated.index)
    kind = indicator['type']

    if kind == 'any_of':
        matched = false
        for sub in indicator.get('indicators', []):
            matched = matched | evaluate_indicator(curated, sub)
        return matched

    if kind == 'all_of':
        nested = indicator.get('indicators', [])
        if not nested:
            return false
        matched = pd.Series(True, index=curated.index)
        for sub in nested:
            matched = matched & evaluate_indicator(curated, sub)
        return matched

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

    if kind == 'value_share_at_least':
        value_col = indicator['value']
        total_cols = indicator['total']
        if value_col not in curated.columns or any(
            c not in curated.columns for c in total_cols
        ):
            return false
        value = pd.to_numeric(curated[value_col], errors='coerce')
        total = sum(pd.to_numeric(curated[c], errors='coerce') for c in total_cols)
        min_ratio = float(indicator['min_ratio'])
        ratio = value.where(total > 0) / total.where(total > 0)
        return (ratio >= min_ratio).fillna(False)

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


def score_decisions(
    curated: pd.DataFrame,
    decisions: list[dict],
    score_columns: dict[str, str] | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Score every row against each ordered *decision* and pick a winner.

    The enumerated-vote core shared by every curate step that chooses among
    recipe-listed candidate classes. For each row a decision's score is the
    sum of the weights of its matched indicators; the decision is eligible
    where that score reaches its ``min_score`` and every one of its
    ``require`` indicators also holds. Among the eligible decisions the
    highest score wins, ties broken by recipe order.

    Parameters
    ----------
    curated : pandas.DataFrame
        Frame the indicators reference by column name.
    decisions : list of dict
        Ordered candidates, each ``{class, min_score, indicators, require,
        source}``. ``min_score`` defaults to 1 and each indicator's
        ``weight`` to 1. ``require`` is a hard AND-gate applied on top of
        ``min_score``, for evidence that should veto a decision outright
        however strongly its other indicators score.
    score_columns : dict of {class: column}, optional
        For named classes, write that decision's raw weighted score to the
        given column of *curated*. Deliberately ungated by ``min_score`` and
        by whether the decision won, so a later step can use the graded
        signal (e.g. a vacancy likelihood) even where the class lost.

    Returns
    -------
    winner : pandas.Series
        Winning class per row, missing where no decision was eligible.
    token : pandas.Series
        Each winning decision's ``source``, missing where it set none --
        callers apply their own default.
    best_score, second_score : pandas.Series
        Winning and runner-up scores among decisions that individually
        reached their own ``min_score``; the gap between them is how
        callers flag narrow, review-worthy wins.
    """
    winner = pd.Series(pd.NA, index=curated.index, dtype=object)
    token = pd.Series(pd.NA, index=curated.index, dtype=object)
    best_score = pd.Series(-1.0, index=curated.index)
    second_score = pd.Series(-1.0, index=curated.index)

    for decision in decisions:
        score = pd.Series(0.0, index=curated.index)
        for indicator in decision.get('indicators', []):
            weight = float(indicator.get('weight', 1.0))
            matched = evaluate_indicator(curated, indicator).astype(float)
            score = score + matched * weight
        if score_columns and decision['class'] in score_columns:
            curated[score_columns[decision['class']]] = score

        eligible = score >= float(decision.get('min_score', 1))
        for req in decision.get('require', []):
            eligible = eligible & evaluate_indicator(curated, req)

        # Strict > keeps the earlier decision on ties (recipe order).
        take = eligible & (score > best_score)
        # The decision this one displaces becomes the new runner-up.
        second_score.loc[take] = best_score.loc[take]
        # An eligible decision that doesn't win outright may still beat
        # the current runner-up.
        runner_up = eligible & ~take & (score > second_score)
        second_score.loc[runner_up] = score.loc[runner_up]

        winner.loc[take] = decision['class']
        source = decision.get('source')
        token.loc[take] = source if source is not None else pd.NA
        best_score.loc[take] = score.loc[take]

    return winner, token, best_score, second_score


def vote_dynamic_values(
    values: dict[str, pd.Series],
    weights: dict[str, float] | None = None,
    *,
    buckets: dict[str, pd.Series] | None = None,
    tiebreaker: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Row-wise weighted vote across labeled columns of an open vocabulary.

    The counterpart to :func:`score_decisions` for the case where the
    candidate classes cannot be listed in the recipe because they are
    whatever values the evidence columns hold. Each label casts its weight
    for its own value, or for *buckets*[label] when given -- letting two
    differing values pool their weight at a coarser granularity while the
    reported winner stays a concrete value.

    Parameters
    ----------
    values : dict of {label: pandas.Series}
        Evidence columns keyed by label, in precedence order. Must be
        non-empty; callers skip this function when no evidence is present.
    weights : dict of {label: float}, optional
        Per-label vote weight (default 1.0), tuning each source's say.
    buckets : dict of {label: pandas.Series}, optional
        Coarser value each label actually votes for. The winning bucket
        decides, but the returned value is the earliest winning label's own
        concrete value -- so a pooled win still names a specific class.
    tiebreaker : pandas.Series, optional
        Breaks a tie when its own value is among the tied ones. Without it
        (or when its value is not tied) the earliest-listed label wins.

    Returns
    -------
    winner : pandas.Series
        Winning concrete value per row, missing where nothing voted.
    token : pandas.Series
        Winning labels joined with '/' (e.g. ``'nsi/parcel'``).
    """
    labels = list(values)
    weights = weights or {}
    keyed = buckets if buckets is not None else values

    frames: dict[tuple[str, str], pd.Series] = {}
    for label in labels:
        frames[('key', label)] = keyed[label].astype(object)
        frames[('value', label)] = values[label].astype(object)
    if tiebreaker is not None:
        frames[('tiebreak', '')] = tiebreaker.astype(object)
    stacked = pd.concat(frames, axis=1)

    def _row_vote(row) -> tuple:
        totals: dict[str, float] = {}
        for label in labels:
            key = row[('key', label)]
            if pd.notna(key):
                totals[key] = totals.get(key, 0.0) + float(weights.get(label, 1.0))
        best = max(totals.values())
        # dict preserves first-vote order, so ties fall to the earliest
        # listed label unless the tiebreaker claims one of the tied values.
        tied = [key for key, total in totals.items() if total == best]
        winning_key = tied[0]
        if len(tied) > 1 and tiebreaker is not None:
            tiebreak_value = row[('tiebreak', '')]
            if pd.notna(tiebreak_value) and tiebreak_value in tied:
                winning_key = tiebreak_value
        voters = [
            label
            for label in labels
            if pd.notna(row[('key', label)]) and row[('key', label)] == winning_key
        ]
        return (row[('value', voters[0])], '/'.join(voters))

    index = stacked.index
    winner = pd.Series(pd.NA, index=index, dtype=object)
    token = pd.Series(pd.NA, index=index, dtype=object)
    has_vote = pd.concat([keyed[label].notna() for label in labels], axis=1).any(axis=1)
    if has_vote.any():
        voted = stacked.loc[has_vote].apply(_row_vote, axis=1)
        winner.loc[has_vote] = voted.str[0]
        token.loc[has_vote] = voted.str[1]
    return winner, token
