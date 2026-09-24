"""A crosswalk key matches the column it is mapped against.

`_read_crosswalk_table` keeps a key column as text when one of its keys
is zero-padded, because that is a sure sign of an identifier. A key like
Canada's province code `24` carries no such sign, so pandas reads it as
an integer, which matches nothing in a column read under
`csv_dtype: str`, and the all-null result used to pass silently: all 293
Canadian census divisions came out parentless, and only
`assign_admin_ids` refusing a parentless unit made it visible.
"""

import pandas as pd
import pytest

from openplaces.io.transform import _map_through_crosswalk


def test_numeric_keys_match_a_text_column():
    """The case that broke Canada: no zero padding anywhere."""
    crosswalk = pd.Series(['CA-QC', 'CA-ON'], index=[24, 35])
    codes = pd.Series(['24', '35', '24'], dtype='string')
    assert list(_map_through_crosswalk(codes, crosswalk)) == [
        'CA-QC',
        'CA-ON',
        'CA-QC',
    ]


def test_zero_padded_text_keys_still_match():
    crosswalk = pd.Series(['a', 'b'], index=['037', '045'])
    codes = pd.Series(['037', '045'], dtype='string')
    assert list(_map_through_crosswalk(codes, crosswalk)) == ['a', 'b']


def test_a_numeric_column_against_numeric_keys_is_left_alone():
    """Aligning to text must not break a genuinely numeric mapping."""
    crosswalk = pd.Series(['low', 'high'], index=[1, 2])
    values = pd.Series([1, 2, 1])
    assert list(_map_through_crosswalk(values, crosswalk)) == ['low', 'high', 'low']


def test_an_unmatched_value_is_still_missing():
    crosswalk = pd.Series(['CA-QC'], index=[24])
    codes = pd.Series(['24', '99'], dtype='string')
    mapped = _map_through_crosswalk(codes, crosswalk)
    assert mapped.iloc[0] == 'CA-QC'
    assert pd.isna(mapped.iloc[1])


def test_matching_nothing_at_all_warns():
    """The silent failure this exists to end."""
    crosswalk = pd.Series(['a', 'b'], index=['x', 'y'])
    codes = pd.Series(['24', '35'], dtype='string')
    with pytest.warns(UserWarning, match='matched none'):
        mapped = _map_through_crosswalk(codes, crosswalk)
    assert mapped.isna().all()


def test_a_partial_match_does_not_warn():
    crosswalk = pd.Series(['a'], index=['24'])
    codes = pd.Series(['24', '99'], dtype='string')
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter('error')
        _map_through_crosswalk(codes, crosswalk)


def test_an_empty_column_neither_warns_nor_raises():
    crosswalk = pd.Series(['a'], index=[1])
    empty = pd.Series([], dtype='string')
    assert len(_map_through_crosswalk(empty, crosswalk)) == 0


def test_an_all_missing_column_does_not_warn():
    """Nothing to match is not the same as matching nothing."""
    crosswalk = pd.Series(['a'], index=[1])
    blank = pd.Series([None, None], dtype='string')
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter('error')
        _map_through_crosswalk(blank, crosswalk)
