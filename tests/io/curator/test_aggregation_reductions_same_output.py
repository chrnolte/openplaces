"""The `distinct` reduction and the token union give what they gave before.

plans/slow-steps-same-output-faster.md, S1: `_reduce` joins only the
groups of several values, and `_union_tokens` calls Python only for
keys of several rows. Each test runs the previous implementation, kept
below, and the current one on the same rows and compares them exactly:
values, dtypes, index, names and the kind of missing value. Every id,
class and token is fabricated (docs/5_contribute/no-personal-data.rst).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openplaces.io.curator.aggregation import (
    AGREE,
    DISTINCT,
    DISTINCT_SEPARATOR,
    UNION,
    _reduce,
    _union_lists,
    _union_tokens,
)


def _old_reduce(values: pd.DataFrame, group_key, functions: dict) -> pd.DataFrame:
    """`_reduce` as it stood before S1."""
    special = (AGREE, DISTINCT, UNION)
    plain = {c: f for c, f in functions.items() if f not in special}
    grouped = values[list(functions)].groupby(group_key)
    if plain:
        reduced = grouped.agg(plain)
        reduced = reduced.where(grouped.count()[list(plain)] > 0)
    else:
        reduced = pd.DataFrame(index=grouped.size().index)
    for column in functions:
        by_group = values[column].groupby(group_key)
        if functions[column] == UNION:
            reduced[column] = _union_lists(values, group_key, column).reindex(
                reduced.index
            )
        elif functions[column] == AGREE:
            reduced[column] = by_group.first().where(by_group.nunique() == 1)
        elif functions[column] == DISTINCT:
            single = by_group.first().where(by_group.nunique() == 1)
            several = by_group.nunique() > 1
            if several.any():
                stated = values[column].notna()
                joined = (
                    values.loc[stated, column]
                    .astype(str)
                    .groupby(group_key[stated])
                    .agg(lambda s: DISTINCT_SEPARATOR.join(dict.fromkeys(s)))
                )
                single = single.astype(object).where(~several, joined)
            reduced[column] = single
    return reduced[list(functions)]


def _old_union_tokens(values: pd.DataFrame, group_key, column: str, sidecar: str):
    """`_union_tokens` as it stood before S1."""
    stating = values[column].notna() & values[sidecar].notna()
    tokens = values.loc[stating, sidecar].astype(str)
    keys = group_key[stating]
    if not keys.duplicated().any():
        return pd.Series(tokens.to_numpy(), index=keys.to_numpy())

    def union(series):
        parts = dict.fromkeys(t for v in series for t in v.split('+') if t)
        return '+'.join(parts) or None

    return tokens.groupby(keys).agg(union)


def _assert_same_series(new: pd.Series, old: pd.Series) -> None:
    pd.testing.assert_series_equal(new, old, check_dtype=True, check_exact=True)
    pd.testing.assert_index_equal(new.index, old.index, exact=True)
    # The kind of missing value (None, NaN, pd.NA) as well.
    assert [type(v) for v in new.to_numpy(dtype=object)] == [
        type(v) for v in old.to_numpy(dtype=object)
    ]


def _frame(keys, classes, numbers, tokens, key_dtype='string'):
    values = pd.DataFrame(
        {
            'cls_obj': pd.Series(classes, dtype=object),
            'cls_str': pd.Series(classes, dtype='str'),
            'cls_cat': pd.Series(classes, dtype='category'),
            'number': pd.Series(numbers, dtype='float64'),
            'cls_obj_source': pd.Series(tokens, dtype=object),
        }
    )
    values['number_source'] = values['cls_obj_source']
    return values, pd.Series(keys, dtype=key_dtype)


# A lone row, a group of two equal classes, a group of two different
# classes, a group with one class and a missing one, an all-missing
# group, a group with repeated and new classes, and a missing key.
KEYS = ['a', 'b', 'b', 'c', 'c', 'd', 'd', 'e', 'e', 'f', 'f', 'f', None, 'g']
CLASSES = [
    'Single-Family',
    'Multi-Family',
    'Multi-Family',
    'Single-Family',
    'Multi-Family',
    'Mobile',
    None,
    None,
    None,
    'Shed',
    'Barn',
    'Shed',
    'Barn',
    'Barn',
]
NUMBERS = [1.0, 2.0, 2.0, 3.0, 4.0, 5.0, np.nan, np.nan, np.nan, 6.0, 7.0, 6.0, 8, 9]
TOKENS = [
    'nsi+nsi',
    'parcel',
    'parcel+imputed',
    '',
    'fema+',
    None,
    'nsi',
    'nsi',
    None,
    '+',
    'parcel',
    'nsi+parcel',
    'nsi',
    'a+b+a',
]


@pytest.mark.parametrize('key_dtype', ['string', object, 'str'])
def test_reduce_distinct_and_agree_match_the_previous_code(key_dtype):
    values, keys = _frame(KEYS, CLASSES, NUMBERS, TOKENS, key_dtype=key_dtype)
    functions = {
        'cls_obj': DISTINCT,
        'cls_str': DISTINCT,
        'cls_cat': DISTINCT,
        'number': AGREE,
    }
    new = _reduce(values, keys, functions)
    old = _old_reduce(values, keys, functions)
    pd.testing.assert_frame_equal(new, old, check_dtype=True, check_exact=True)
    for column in functions:
        _assert_same_series(new[column], old[column])


def test_reduce_without_any_group_of_several_values():
    values, keys = _frame(
        ['a', 'b', 'b'], ['x', 'y', 'y'], [1.0, 2.0, 2.0], ['p', 'q', 'q']
    )
    functions = {'cls_obj': DISTINCT, 'cls_cat': DISTINCT}
    pd.testing.assert_frame_equal(
        _reduce(values, keys, functions),
        _old_reduce(values, keys, functions),
        check_dtype=True,
        check_exact=True,
    )


def test_reduce_on_empty_rows():
    values, keys = _frame([], [], [], [])
    functions = {'cls_obj': DISTINCT, 'number': AGREE}
    pd.testing.assert_frame_equal(
        _reduce(values, keys, functions),
        _old_reduce(values, keys, functions),
        check_dtype=True,
        check_exact=True,
    )


@pytest.mark.parametrize('column', ['cls_obj', 'number'])
@pytest.mark.parametrize('key_dtype', ['string', object])
def test_union_tokens_match_the_previous_code(column, key_dtype):
    values, keys = _frame(KEYS, CLASSES, NUMBERS, TOKENS, key_dtype=key_dtype)
    sidecar = f'{column}_source'
    _assert_same_series(
        _union_tokens(values, keys, column, sidecar),
        _old_union_tokens(values, keys, column, sidecar),
    )


@pytest.mark.parametrize(
    'tokens',
    [
        # Every union empty: pandas holds them as missing.
        ['', '+', '++', ''],
        # Every union a token.
        ['nsi', 'nsi+parcel', 'parcel', 'fema'],
        # A lone row repeats a token, which the union removes.
        ['nsi+nsi', 'a', 'b', 'c'],
    ],
)
def test_union_tokens_by_kind_of_result(tokens):
    values, keys = _frame(
        ['a', 'b', 'b', 'c'], ['x'] * 4, [1.0] * 4, tokens, key_dtype=object
    )
    _assert_same_series(
        _union_tokens(values, keys, 'cls_obj', 'cls_obj_source'),
        _old_union_tokens(values, keys, 'cls_obj', 'cls_obj_source'),
    )


def test_union_tokens_with_integer_keys_and_a_repeated_index():
    values, keys = _frame(
        [3, 1, 1, 2, 3], ['x'] * 5, [1.0] * 5, ['a', 'b+c', 'c', '', 'a+d'], 'int64'
    )
    values.index = [0, 0, 1, 1, 2]
    keys.index = values.index
    _assert_same_series(
        _union_tokens(values, keys, 'cls_obj', 'cls_obj_source'),
        _old_union_tokens(values, keys, 'cls_obj', 'cls_obj_source'),
    )


def _random_frame(n_rows: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    # Most keys hold one row, some two to five.
    sizes = rng.choice([1, 1, 1, 1, 1, 1, 1, 2, 3, 5], size=n_rows)
    keys = np.repeat(np.arange(len(sizes)), sizes)[:n_rows]
    rng.shuffle(keys)
    pool = np.array(['Single-Family', 'Multi-Family', 'Mobile', 'Shed', None])
    classes = pool[rng.integers(0, len(pool), n_rows)]
    numbers = np.where(rng.random(n_rows) < 0.2, np.nan, rng.integers(0, 4, n_rows))
    token_pool = np.array(
        ['nsi', 'parcel', 'nsi+parcel', 'parcel+imputed', '', '+', 'nsi+nsi', None]
    )
    tokens = token_pool[rng.integers(0, len(token_pool), n_rows)]
    key_text = pd.Series(keys).astype(str).to_numpy(dtype=object)
    key_text[rng.random(n_rows) < 0.01] = None
    return _frame(list(key_text), list(classes), numbers, list(tokens))


def test_randomized_rows_match_the_previous_code():
    values, keys = _random_frame(4000)
    functions = {
        'cls_obj': DISTINCT,
        'cls_str': DISTINCT,
        'cls_cat': DISTINCT,
        'number': AGREE,
    }
    pd.testing.assert_frame_equal(
        _reduce(values, keys, functions),
        _old_reduce(values, keys, functions),
        check_dtype=True,
        check_exact=True,
    )
    for column in ('cls_obj', 'number'):
        _assert_same_series(
            _union_tokens(values, keys, column, f'{column}_source'),
            _old_union_tokens(values, keys, column, f'{column}_source'),
        )
