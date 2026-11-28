"""The open-vocabulary vote decides each distinct row once, as before.

plans/slow-steps-same-output-faster.md, R5: `vote_dynamic_values` calls
its row vote once per distinct combination of evidence instead of once
per row, for every caller (`classify_occupancy_by_source_vote`, the pooled-vote
indicator, `reconcile_land_use`). The previous function is kept below;
both run on the same fabricated evidence and their winners and tokens
are compared exactly, including the Python type of every value: rows
differing only by 1, 1.0 and True, by None, NaN and pd.NA, or by 0.0
and -0.0 must not share a result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openplaces.io.curator.indicators import vote_dynamic_values


def _old_vote_dynamic_values(values, weights=None, *, buckets=None, tiebreaker=None):
    labels = list(values)
    weights = weights or {}
    keyed = buckets if buckets is not None else values

    frames = {}
    for label in labels:
        frames[('key', label)] = keyed[label].astype(object)
        frames[('value', label)] = values[label].astype(object)
    if tiebreaker is not None:
        frames[('tiebreak', '')] = tiebreaker.astype(object)
    stacked = pd.concat(frames, axis=1)

    def _row_vote(row) -> tuple:
        totals = {}
        for label in labels:
            key = row[('key', label)]
            if pd.notna(key):
                totals[key] = totals.get(key, 0.0) + float(weights.get(label, 1.0))
        best = max(totals.values())
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


def _same(new, old):
    for a, b in zip(new, old, strict=True):
        pd.testing.assert_series_equal(a, b, check_exact=True, check_dtype=True)
        assert [type(v) for v in a] == [type(v) for v in b]


def _run(values, **kwargs):
    _same(
        vote_dynamic_values(values, **kwargs),
        _old_vote_dynamic_values(values, **kwargs),
    )


INDEX = pd.Index([f'b{i}' for i in range(9)], name='building_id')
NSI = [
    'Single Family',
    'Multi Family',
    None,
    'Retail',
    'Single Family',
    'Single Family',
    np.nan,
    pd.NA,
    'Hotel',
]
PARCEL = [
    'Single Family',
    'Single Family',
    'Mobile Home',
    None,
    'Retail',
    'Single Family',
    np.nan,
    None,
    'Retail',
]
FEMA = [
    'Multi Family',
    'Multi Family',
    'Mobile Home',
    None,
    None,
    'Single Family',
    None,
    pd.NA,
    'Hotel',
]


@pytest.mark.parametrize('dtype', [object, 'str', 'category'])
def test_plain_vote(dtype):
    values = {
        label: pd.Series(raw, index=INDEX, dtype=object).astype(dtype)
        for label, raw in (('nsi', NSI), ('parcel', PARCEL), ('fema', FEMA))
    }
    _run(values)
    _run(values, weights={'nsi': 2.0, 'fema': 0.5})


def test_buckets_and_tiebreaker():
    values = {
        label: pd.Series(raw, index=INDEX, dtype=object)
        for label, raw in (('nsi', NSI), ('parcel', PARCEL), ('fema', FEMA))
    }
    residential = {'Single Family', 'Multi Family', 'Mobile Home'}
    buckets = {
        label: series.where(series.isna() | series.isin(residential), 'Non-Residential')
        for label, series in values.items()
    }
    tiebreak = pd.Series(PARCEL, index=INDEX, dtype=object)
    _run(values, buckets=buckets)
    _run(values, buckets=buckets, tiebreaker=tiebreak)
    _run(values, weights={'parcel': 1.5}, tiebreaker=tiebreak)


def test_equal_values_of_other_types_stay_apart():
    index = pd.RangeIndex(8)
    values = {
        'a': pd.Series([1, 1.0, True, 0.0, -0.0, None, np.nan, pd.NA], dtype=object),
        'b': pd.Series([1.0, 1, 1, -0.0, 0.0, 'x', 'x', 'x'], dtype=object),
    }
    for series in values.values():
        series.index = index
    _run(values)


def test_no_row_votes():
    values = {'a': pd.Series([None, np.nan], dtype=object)}
    _run(values)


def test_random_evidence():
    rng = np.random.default_rng(21)
    n = 4000
    classes = np.array(
        ['Single Family', 'Multi Family', 'Mobile Home', 'Retail', None], dtype=object
    )
    index = pd.Index([f'b{i:05d}' for i in range(n)], name='building_id')
    values = {
        label: pd.Series(rng.choice(classes, n), index=index, dtype=object)
        for label in ('nsi', 'parcel', 'fema', 'overture')
    }
    _run(values, weights={'nsi': 1.0, 'parcel': 1.0, 'fema': 0.5, 'overture': 2.0})
    _run(values, tiebreaker=values['parcel'])
