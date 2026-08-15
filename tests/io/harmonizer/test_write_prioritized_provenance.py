"""Tests for `_write_prioritized`'s optional per-cell provenance tracking
(`provenance_token`), which stamps a `{name}_source` sidecar for exactly the
cells a call actually changes -- opt-in, so most callers never create the
sidecar at all.
"""

import pandas as pd

import openplaces.io.harmonizer.links as links
from openplaces.io.harmonizer import _ensure_object_source_column


def _source_list(series):
    """Normalize a source sidecar's NA cells (pd.NA or NaN) to plain None."""
    return [None if pd.isna(v) else v for v in series]


def test_new_column_with_token_stamps_every_written_cell():
    spine = pd.DataFrame(index=['a', 'b'])
    new_vals = pd.Series([1.0, 2.0], index=['a', 'b'])
    links._write_prioritized(spine, 'value', new_vals, provenance_token='county')
    assert spine['value'].tolist() == [1.0, 2.0]
    assert _source_list(spine['value_source']) == ['county', 'county']


def test_gap_fill_stamps_only_the_filled_cell():
    spine = pd.DataFrame({'value': [10.0, 20.0, None]}, index=['a', 'b', 'c'])
    # Sparse (1/3, below the default 0.5 majority threshold): only fills c's
    # gap, taking the gap-fill (not overwrite) branch.
    new_vals = pd.Series([None, None, 30.0], index=['a', 'b', 'c'])
    links._write_prioritized(spine, 'value', new_vals, provenance_token='county')
    assert spine['value'].tolist() == [10.0, 20.0, 30.0]
    assert _source_list(spine['value_source']) == [None, None, 'county']


def test_majority_overwrite_stamps_every_changed_cell_but_not_unchanged_ones():
    spine = pd.DataFrame({'value': [10.0, 20.0, 30.0]}, index=['a', 'b', 'c'])
    # Majority coverage: overwrites outright, but 'b' happens to get the
    # exact same value it already had -- that cell must not be flagged.
    new_vals = pd.Series([11.0, 20.0, 31.0], index=['a', 'b', 'c'])
    links._write_prioritized(spine, 'value', new_vals, provenance_token='state')
    assert spine['value'].tolist() == [11.0, 20.0, 31.0]
    assert _source_list(spine['value_source']) == ['state', None, 'state']


def test_no_token_creates_no_source_column():
    spine = pd.DataFrame({'value': [10.0]}, index=['a'])
    new_vals = pd.Series([11.0], index=['a'])
    links._write_prioritized(spine, 'value', new_vals)
    assert 'value_source' not in spine.columns


def test_repeated_calls_only_move_the_sidecar_for_actually_changed_cells():
    spine = pd.DataFrame({'value': [None, None]}, index=['a', 'b'])
    links._write_prioritized(
        spine,
        'value',
        pd.Series([1.0, None], index=['a', 'b']),
        provenance_token='first',
    )
    # A second source only supplies 'b' -- 'a' must keep its original value
    # and provenance untouched.
    links._write_prioritized(
        spine,
        'value',
        pd.Series([None, 2.0], index=['a', 'b']),
        provenance_token='second',
    )
    assert spine['value'].tolist() == [1.0, 2.0]
    assert _source_list(spine['value_source']) == ['first', 'second']


def test_ensure_object_source_column_casts_existing_numeric_column():
    # A prior join can leave an all-NaN float64 column behind (e.g. pandas
    # infers a numeric dtype for an all-missing column) without ever
    # populating it -- that dtype can't hold an arbitrary source-id string,
    # so it must be cast to object rather than crash on the next assignment.
    df = pd.DataFrame({'value_source': pd.Series([float('nan'), float('nan')])})
    assert pd.api.types.is_numeric_dtype(df['value_source'])
    _ensure_object_source_column(df, 'value_source')
    assert df['value_source'].dtype == object
    df.loc[0, 'value_source'] = 'county'
    assert df['value_source'].tolist()[0] == 'county'


def test_ensure_object_source_column_casts_existing_categorical_column():
    df = pd.DataFrame({'value_source': pd.Categorical(['a', 'a'])})
    _ensure_object_source_column(df, 'value_source')
    assert df['value_source'].dtype == object
    df.loc[0, 'value_source'] = 'county'
    assert df['value_source'].tolist() == ['county', 'a']
