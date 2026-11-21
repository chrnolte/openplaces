"""Collected link ids are the strings the lambda per footprint gave.

plans/slow-steps-same-output-faster.md, S4: `collect_link_ids` joins
only the groups of several ids in Python. The previous expression is
kept below; both run on the same links and are compared exactly:
strings, order, separators, dtype, index and the kind of missing
value. Every id is fabricated (docs/5_contribute/no-personal-data.rst).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openplaces.io.curator.evidence import _join_link_ids


def _old_join_link_ids(links: pd.DataFrame, id_col: str, ref_key: str) -> pd.Series:
    """The expression `collect_link_ids` used before S4."""
    return links.groupby(id_col)[ref_key].agg(
        lambda s: '|'.join(s.dropna().astype(str).unique())
    )


def _assert_same(links, id_col='footprint_id', ref_key='parcel_id'):
    new = _join_link_ids(links, id_col, ref_key)
    old = _old_join_link_ids(links, id_col, ref_key)
    pd.testing.assert_series_equal(new, old, check_dtype=True, check_exact=True)
    pd.testing.assert_index_equal(new.index, old.index, exact=True)
    assert [type(v) for v in new.to_numpy(dtype=object)] == [
        type(v) for v in old.to_numpy(dtype=object)
    ]
    # What the step writes from it.
    pd.testing.assert_series_equal(
        new.where(new != ''), old.where(old != ''), check_dtype=True, check_exact=True
    )


# f1 one parcel; f2 two parcels; f3 one parcel listed twice; f4 two
# parcels, one listed twice, in an order that is not sorted; f5 no
# parcel stated; a link with no footprint; f6 an empty id.
FOOTPRINTS = ['f1', 'f2', 'f2', 'f3', 'f3', 'f4', 'f4', 'f4', 'f5', None, 'f6']
PARCELS = ['p1', 'p2', 'p3', 'p4', 'p4', 'p9', 'p5', 'p9', None, 'p7', '']


@pytest.mark.parametrize('parcel_dtype', [object, 'str', 'string', 'category'])
@pytest.mark.parametrize('footprint_dtype', [object, 'str', 'category'])
def test_string_ids_match_the_previous_code(parcel_dtype, footprint_dtype):
    links = pd.DataFrame(
        {
            'footprint_id': pd.Series(FOOTPRINTS, dtype=footprint_dtype),
            'parcel_id': pd.Series(PARCELS, dtype=parcel_dtype),
        }
    )
    _assert_same(links)


@pytest.mark.parametrize(
    'parcels',
    [
        [1.0, 2.0, 2.0, np.nan, 3.0],
        [1, 2, 2, 3, 3],
        [1, '1', 1.0, 'p', None],
    ],
)
def test_numeric_and_mixed_ids(parcels):
    links = pd.DataFrame(
        {
            'footprint_id': [10, 11, 11, 12, 12],
            'parcel_id': pd.Series(parcels, dtype=object if None in parcels else None),
        }
    )
    _assert_same(links)


@pytest.mark.parametrize('parcel_dtype', ['Int64', 'category'])
@pytest.mark.parametrize('footprints', [[10, 11, 12], [10, 11, 11]])
def test_ids_pandas_may_cast_back(parcel_dtype, footprints):
    # Joined ids that fit the column's dtype come back in it.
    links = pd.DataFrame(
        {
            'footprint_id': footprints,
            'parcel_id': pd.Series([1, 2, 3], dtype=parcel_dtype),
        }
    )
    _assert_same(links)


def test_every_footprint_with_one_id():
    links = pd.DataFrame({'footprint_id': ['f2', 'f1'], 'parcel_id': ['p2', 'p1']})
    _assert_same(links)


def test_no_stated_id():
    links = pd.DataFrame(
        {
            'footprint_id': ['f1', 'f2'],
            'parcel_id': pd.Series([None, None], dtype=object),
        }
    )
    _assert_same(links)


def test_no_links():
    links = pd.DataFrame(
        {
            'footprint_id': pd.Series([], dtype='str'),
            'parcel_id': pd.Series([], dtype='str'),
        }
    )
    _assert_same(links)


def test_repeated_index_labels():
    links = pd.DataFrame(
        {'footprint_id': ['f1', 'f1', 'f2'], 'parcel_id': ['p2', 'p1', 'p3']},
        index=[0, 0, 0],
    )
    _assert_same(links)


def _random_links(n_rows: int, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sizes = rng.choice([1, 1, 1, 1, 1, 1, 2, 2, 3, 6], size=n_rows)
    footprints = np.repeat(np.arange(len(sizes)), sizes)[:n_rows]
    parcels = footprints + rng.integers(-1, 2, n_rows)
    order = rng.permutation(n_rows)
    parcel_ids = pd.Series(parcels[order]).map('p{:05d}'.format).to_numpy(dtype=object)
    parcel_ids[rng.random(n_rows) < 0.02] = None
    return pd.DataFrame(
        {
            'footprint_id': pd.Series(footprints[order]).map('f{:05d}'.format),
            'parcel_id': pd.Series(parcel_ids, dtype='str'),
        }
    )


def test_randomized_links_match_the_previous_code():
    _assert_same(_random_links(4000))
