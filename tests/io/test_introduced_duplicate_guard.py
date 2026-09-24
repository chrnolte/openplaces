"""A derived matching key may never merge entities the source told apart.

Every id below is fabricated.
"""

import warnings

import pandas as pd
import pytest

from openplaces.geo.ids import (
    add_parcel_id_alnum,
    introduced_duplicate_mask,
    refuse_introduced_duplicates,
)


def _mask(derived, source):
    return introduced_duplicate_mask(
        pd.Series(derived, dtype='string'), pd.Series(source, dtype='string')
    )


def test_a_key_that_merges_two_distinct_ids_is_flagged():
    # '1-23' and '12-3' both strip to '123'.
    assert _mask(['123', '123'], ['1-23', '12-3']).tolist() == [True, True]


def test_a_duplicate_the_source_already_had_is_not_flagged():
    # The source itself repeats the id; stripping caused nothing.
    assert _mask(['123', '123'], ['1-23', '1-23']).tolist() == [False, False]


def test_a_difference_the_key_is_meant_to_merge_is_not_flagged():
    # Case folding is intended, so the caller passes a case-folded source.
    assert _mask(['12A', '12A'], ['12-A', '12-A']).tolist() == [False, False]


def test_untouched_and_missing_rows_are_not_flagged():
    mask = _mask(['1', '2', None], ['1', '2', None])
    assert mask.tolist() == [False, False, False]


def test_one_merging_key_does_not_condemn_the_others():
    mask = _mask(['123', '123', '456'], ['1-23', '12-3', '456'])
    assert mask.tolist() == [True, True, False]


def test_refusing_blanks_only_the_merging_rows_and_warns():
    derived = pd.Series(['123', '123', '456'], dtype='string')
    source = pd.Series(['1-23', '12-3', '456'], dtype='string')
    with pytest.warns(UserWarning, match='told apart'):
        out = refuse_introduced_duplicates(derived, source, 'parcel_id_alnum')
    assert out.isna().tolist() == [True, True, False]
    assert out.dropna().tolist() == ['456']


def test_refusing_returns_the_key_untouched_when_nothing_merges():
    derived = pd.Series(['1', '2'], dtype='string')
    assert refuse_introduced_duplicates(derived, derived) is derived


def test_the_fallback_key_refuses_a_collapse_it_would_have_made():
    # Two parcels the source punctuates differently, which the fallback
    # key would otherwise make one.
    frame = pd.DataFrame({'parcel_id_assessor': ['1-23', '12-3', '9-9']})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = add_parcel_id_alnum(frame)['parcel_id_alnum']
    assert out.isna().tolist() == [True, True, False]
    assert out.dropna().tolist() == ['99']


def test_the_fallback_key_survives_where_it_only_removes_punctuation():
    # The case the fallback exists for: one side punctuates, the other
    # does not, and no two parcels collide.
    frame = pd.DataFrame({'parcel_id_assessor': ['4071-68-1844', '4071-68-1845']})
    out = add_parcel_id_alnum(frame)['parcel_id_alnum']
    assert out.tolist() == ['4071681844', '4071681845']


def test_a_degenerate_source_column_is_caught_by_the_stricter_key():
    # The Carteret shape: the column the fallback coalesces from is a
    # block code shared by several parcels, so the raw and the stripped
    # key agree and nothing looks merged; parcel_id_local, built from a
    # different column, is what says these are three parcels.
    frame = pd.DataFrame(
        {
            'parcel_id_assessor': ['BLOCK7', 'BLOCK7', 'BLOCK8'],
            'parcel_id_local': ['7-1', '7-2', '8-1'],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = add_parcel_id_alnum(frame)['parcel_id_alnum']
    assert out.isna().tolist() == [True, True, False]
    assert out.dropna().tolist() == ['BLOCK8']


def test_a_strict_key_that_distinguishes_nothing_does_not_veto_the_fallback():
    # The Pender shape: parcel_id_local is one value on every row, so it
    # tells no two parcels apart and cannot claim the fallback merged
    # any. The fallback is exactly what gives this source identity.
    frame = pd.DataFrame(
        {
            'parcel_id_assessor': ['4071-68-1844', '4071-68-1845'],
            'parcel_id_local': ['SAME', 'SAME'],
        }
    )
    out = add_parcel_id_alnum(frame)['parcel_id_alnum']
    assert out.tolist() == ['4071681844', '4071681845']


def test_a_source_that_repeats_its_own_id_still_gets_the_key():
    # Two rows of one parcel: the duplication is the source's, so the
    # key is not refused and the join can still aggregate them.
    frame = pd.DataFrame({'parcel_id_assessor': ['4071-68-1844', '4071-68-1844']})
    out = add_parcel_id_alnum(frame)['parcel_id_alnum']
    assert out.tolist() == ['4071681844', '4071681844']
