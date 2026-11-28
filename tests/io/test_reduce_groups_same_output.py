"""The shared group reducer returns what pandas' call per group did.

plans/slow-steps-same-output-faster.md, R1 to R3 and R7:
`table.reduce_groups` replaces ``groupby(...).agg(f)`` where f joins a
group's values, `table.map_distinct` replaces ``Series.map(f)`` where
a column holds few distinct values, and `aggregate_rows` runs its join
reducers through the first. The expressions the callers used before
are kept below as references; old and new run on the same fabricated
rows and are compared exactly: values, the Python type of each value,
dtype, index, name, column and row order.

Pinned dtype behavior: where the reducer returns text, pandas infers
the text dtype ('str') for the result, and `reduce_groups` reproduces
it by letting pandas aggregate one group of each kind of result. The
tests fail if a pandas upgrade changes that inference in only one of
the two paths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openplaces.table import (
    _aggregate_columns,
    aggregate_rows,
    join_distinct_text,
    join_nonnull_addresses,
    join_nonnull_strings,
    map_distinct,
    reduce_groups,
)


def _old_join_ids(s: pd.Series) -> str:
    """The lambda polygon.py, point.py and collect_link_ids used."""
    return '|'.join(s.dropna().astype(str).unique())


def _types(series: pd.Series) -> list:
    return [type(v) for v in series.to_numpy(dtype=object)]


def _assert_series_same(new: pd.Series, old: pd.Series) -> None:
    pd.testing.assert_series_equal(new, old, check_exact=True, check_dtype=True)
    pd.testing.assert_index_equal(new.index, old.index, exact=True)
    assert _types(new) == _types(old)


def _assert_frame_same(new: pd.DataFrame, old: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(new, old, check_exact=True, check_dtype=True)
    for column in old.columns:
        assert _types(new[column]) == _types(old[column]), column


# Fabricated keys and values. k1 one value; k2 two; k3 one value twice;
# k4 two values, one repeated, unsorted; k5 only missing values; a row
# with no key; k6 an empty string.
KEYS = ['k1', 'k2', 'k2', 'k3', 'k3', 'k4', 'k4', 'k4', 'k5', None, 'k6']
TEXTS = ['a1', 'b2', 'b3', 'c4', 'c4', 'd9', 'd5', 'd9', None, 'x7', '']
ADDRESSES = [
    '12 Fernvale Rd',
    '40 Quillon Ave Unit 2',
    '40 Quillon Ave Unit 3',
    '7 Brantmoor St',
    '7 Brantmoor St',
    '3 Oakhurst Ln',
    '5 Oakhurst Ln',
    '3 Oakhurst Ln Apt 1',
    None,
    '99 Pellow Way',
    '',
]

JOINS = [
    (join_distinct_text, lambda s: join_distinct_text(s.to_numpy(dtype=object))),
    (join_nonnull_strings, join_nonnull_strings),
    (join_nonnull_addresses, join_nonnull_addresses),
]


def _old(values, keys, series_reducer, sort=True):
    return values.groupby(keys, sort=sort).agg(series_reducer)


@pytest.mark.parametrize('value_dtype', [object, 'str', 'string', 'category'])
@pytest.mark.parametrize('key_dtype', [object, 'str', 'category'])
@pytest.mark.parametrize('sort', [True, False])
@pytest.mark.parametrize('pair', range(len(JOINS)))
def test_joins_match_pandas_per_group(value_dtype, key_dtype, sort, pair):
    reducer, series_reducer = JOINS[pair]
    for raw in (TEXTS, ADDRESSES):
        values = pd.Series(raw, dtype=value_dtype, name='value')
        keys = pd.Series(KEYS, dtype=key_dtype, name='key')
        _assert_series_same(
            reduce_groups(values, keys, reducer, sort=sort),
            _old(values, keys, series_reducer, sort=sort),
        )


@pytest.mark.parametrize('value_dtype', [object, 'str', 'string', 'category'])
def test_join_distinct_text_is_the_old_id_lambda(value_dtype):
    values = pd.Series(TEXTS, dtype=value_dtype, name='parcel_id')
    keys = pd.Series(KEYS, name='footprint_id')
    _assert_series_same(
        reduce_groups(values, keys, join_distinct_text),
        _old(values, keys, _old_join_ids),
    )


@pytest.mark.parametrize(
    'raw',
    [
        [1.0, 2.0, 2.0, np.nan, 3.0, 1.5],
        [1, 2, 2, 3, 3, 4],
        [1, '1', 1.0, 'p', None, True],
        [None, None, None, None, None, None],
        [np.nan, None, pd.NA, 'q', 'q', np.nan],
    ],
)
@pytest.mark.parametrize('dtype', [None, object, 'Int64', 'Float64'])
def test_numeric_mixed_and_nullable_ids(raw, dtype):
    try:
        values = pd.Series(raw, dtype=dtype, name='parcel_id')
    except (TypeError, ValueError):
        pytest.skip('values do not fit the dtype')
    keys = pd.Series([10, 11, 11, 12, 12, 13], name='footprint_id')
    _assert_series_same(
        reduce_groups(values, keys, join_distinct_text),
        _old(values, keys, _old_join_ids),
    )


def test_reducer_returning_none_only():
    values = pd.Series([None, None, None], dtype=object)
    keys = pd.Series(['a', 'b', 'b'])
    _assert_series_same(
        reduce_groups(values, keys, join_nonnull_strings),
        _old(values, keys, join_nonnull_strings),
    )


def test_numeric_keys_and_missing_keys():
    values = pd.Series(['a', 'b', 'c', 'd', 'e'], dtype='str')
    keys = pd.Series([3.0, np.nan, 1.0, 3.0, 2.0])
    for sort in (True, False):
        _assert_series_same(
            reduce_groups(values, keys, join_nonnull_strings, sort=sort),
            _old(values, keys, join_nonnull_strings, sort=sort),
        )


def test_keys_as_an_array_and_repeated_index():
    values = pd.Series(['a', 'b', 'c'], index=[5, 5, 5], name='v')
    keys = np.array(['y', 'x', 'y'], dtype=object)
    old = (
        values.reset_index(drop=True).groupby(pd.Series(keys)).agg(join_nonnull_strings)
    )
    _assert_series_same(reduce_groups(values, keys, join_nonnull_strings), old)


def test_empty_input():
    values = pd.Series([], dtype='str', name='v')
    keys = pd.Series([], dtype='str', name='k')
    _assert_series_same(
        reduce_groups(values, keys, join_distinct_text),
        _old(values, keys, _old_join_ids),
    )


def test_every_key_missing():
    values = pd.Series(['a', 'b'], name='v')
    keys = pd.Series([None, None], dtype=object, name='k')
    _assert_series_same(
        reduce_groups(values, keys, join_distinct_text),
        _old(values, keys, _old_join_ids),
    )


def test_geometry_union_per_group():
    shapely = pytest.importorskip('shapely')
    boxes = [shapely.box(i, 0, i + 2, 1) for i in range(6)]
    values = pd.Series(np.asarray(boxes, dtype=object))
    keys = pd.Series(['a', 'b', 'a', 'c', 'b', 'a'], name='footprint_id')
    new = reduce_groups(values, keys, shapely.union_all)
    old = (
        pd.DataFrame({'footprint_id': keys, '_og': values})
        .groupby('footprint_id')['_og']
        .agg(shapely.union_all)
    )
    pd.testing.assert_index_equal(new.index, old.index, exact=True)
    assert list(shapely.to_wkb(new.to_numpy())) == list(shapely.to_wkb(old.to_numpy()))


def _random(n_rows: int, seed: int) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    sizes = rng.choice([1, 1, 1, 1, 1, 1, 2, 2, 3, 6], size=n_rows)
    keys = np.repeat(np.arange(len(sizes)), sizes)[:n_rows]
    order = rng.permutation(n_rows)
    texts = pd.Series(keys + rng.integers(-1, 2, n_rows)).map('v{:05d}'.format)
    texts = texts.to_numpy(dtype=object)[order]
    texts[rng.random(n_rows) < 0.03] = None
    key_text = pd.Series(keys[order]).map('k{:05d}'.format)
    key_text[rng.random(n_rows) < 0.01] = None
    return pd.Series(texts, dtype='str', name='value'), key_text.rename('key')


@pytest.mark.parametrize('sort', [True, False])
@pytest.mark.parametrize('pair', range(len(JOINS)))
def test_random_groups(sort, pair):
    reducer, series_reducer = JOINS[pair]
    values, keys = _random(4000, seed=7)
    _assert_series_same(
        reduce_groups(values, keys, reducer, sort=sort),
        _old(values, keys, series_reducer, sort=sort),
    )


# map_distinct, with the token functions split_curated_values maps.
TOKENS = [
    'parcel.roll',
    'roll',
    'roll+imputed',
    'roll+open+imputed',
    None,
    np.nan,
    pd.NA,
    'roll',
    '',
    'parcel.roll+imputed',
]


def _token_functions():
    from openplaces.core.provenance import mark_imputed
    from openplaces.io.curator.evidence import _strip_open_marker, _with_open_marker

    lane = 'parcel'
    return [
        lambda t: t if pd.isna(t) or str(t).startswith(lane) else f'{lane}.{t}',
        _strip_open_marker,
        mark_imputed,
        _with_open_marker,
    ]


@pytest.mark.parametrize('dtype', [object, 'str', 'string', 'category'])
@pytest.mark.parametrize('position', range(4))
def test_map_distinct_is_series_map(dtype, position):
    function = _token_functions()[position]
    raw = TOKENS if dtype is object else [t for t in TOKENS if t is not pd.NA]
    series = pd.Series(raw, dtype=dtype, index=range(10, 10 + len(raw)), name='t')
    try:
        old = series.map(function)
    except TypeError:
        pytest.skip('the function does not take this dtype at all')
    _assert_series_same(map_distinct(series, function), old)


def test_map_distinct_keeps_equal_values_of_other_types_apart():
    series = pd.Series([1, 1.0, True, '1', None, np.nan, pd.NA], dtype=object)
    _assert_series_same(map_distinct(series, repr), series.map(repr))
    _assert_series_same(map_distinct(series, type), series.map(type))


def test_map_distinct_random_tokens():
    rng = np.random.default_rng(11)
    series = pd.Series(rng.choice(np.array(TOKENS, dtype=object), 5000), dtype=object)
    for function in _token_functions():
        _assert_series_same(map_distinct(series, function), series.map(function))


def test_map_distinct_empty():
    series = pd.Series([], dtype=object)
    _assert_series_same(map_distinct(series, str), series.map(str))


# aggregate_rows: joins through reduce_groups, the rest by pandas.
def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'geo_id': ['g2', 'g1', 'g2', 'g3', 'g1', None, 'g4'],
            'address': ADDRESSES[:7],
            'use_group': ['res', 'com', None, 'res', 'res', 'ind', None],
            'improvement_value': [1.0, 2.0, 3.0, np.nan, 5.0, 6.0, np.nan],
        }
    )


@pytest.mark.parametrize('address_dtype', [object, 'str'])
def test_aggregate_rows_join_columns(address_dtype):
    rows = _rows().astype({'address': address_dtype})
    grouped = rows.groupby('geo_id')
    agg_cols = {
        'address': join_nonnull_addresses,
        'use_group': join_nonnull_strings,
        'improvement_value': 'sum',
    }
    _assert_frame_same(
        _aggregate_columns(rows, 'geo_id', grouped, agg_cols), grouped.agg(agg_cols)
    )
    only_joins = {'address': join_nonnull_addresses}
    _assert_frame_same(
        _aggregate_columns(rows, 'geo_id', grouped, only_joins),
        grouped.agg(only_joins),
    )


def test_aggregate_rows_end_to_end_matches_plain_agg(monkeypatch):
    rows = _rows()
    new = aggregate_rows(rows, by='geo_id')
    import openplaces.table as table

    monkeypatch.setattr(
        table,
        '_aggregate_columns',
        lambda df, by, grouped, agg_cols: grouped.agg(agg_cols) if agg_cols else None,
    )
    old = aggregate_rows(rows, by='geo_id')
    _assert_frame_same(new, old)


def test_aggregate_rows_random_addresses(monkeypatch):
    rng = np.random.default_rng(5)
    n = 3000
    streets = np.array(['Fernvale Rd', 'Quillon Ave', 'Brantmoor St', 'Pellow Way'])
    numbers = rng.integers(1, 60, n)
    units = np.where(
        rng.random(n) < 0.3, ' Unit ' + rng.integers(1, 9, n).astype(str), ''
    )
    addresses = pd.Series(
        [f'{a} {b}{c}' for a, b, c in zip(numbers, rng.choice(streets, n), units)],
        dtype=object,
    )
    addresses[rng.random(n) < 0.05] = None
    rows = pd.DataFrame(
        {
            'geo_id': pd.Series(rng.integers(0, 900, n)).map('g{:04d}'.format),
            'address': addresses,
            'improvement_value': rng.random(n),
        }
    )
    new = aggregate_rows(rows, by='geo_id')
    import openplaces.table as table

    monkeypatch.setattr(
        table,
        '_aggregate_columns',
        lambda df, by, grouped, agg_cols: grouped.agg(agg_cols) if agg_cols else None,
    )
    _assert_frame_same(new, aggregate_rows(rows, by='geo_id'))
