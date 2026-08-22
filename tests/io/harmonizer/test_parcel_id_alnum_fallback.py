"""Tests for the punctuation-free parcel-id fallback join key.

``parcel_id_local`` is the right join key when both sides standardize the
same way, and useless the moment they do not. Two sources describing one
parcel can hold the same id and never meet: they may punctuate it
differently, and ``compute_parcel_id_local``'s duplicate guard may pick a
different conversion on each side. Measured on Northampton NC, 20,132
parcels shared an id and **zero** shared a ``parcel_id_local``; across seven
eastern NC counties the statewide layer's land use reached 117 of 195,245
parcels.
"""

from __future__ import annotations

import pandas as pd

from openplaces.geo.ids import (
    PARCEL_ID_ALNUM,
    PARCEL_ID_MATCH_CANDIDATES,
    add_parcel_id_alnum,
)
from openplaces.io.harmonizer.links import DEFAULT_LINK_KEY


def test_punctuation_and_case_are_the_only_things_discarded():
    df = pd.DataFrame({'parcel_id_assessor': ['4071-68-1844', 'ab 12/34', '1.2.3']})
    out = add_parcel_id_alnum(df)
    assert list(out[PARCEL_ID_ALNUM]) == ['4071681844', 'AB1234', '123']


def test_two_punctuations_of_one_id_meet_on_the_key():
    # The whole point: a county roll writing 4071-68-1844 and a statewide
    # layer writing 4071681844 describe the same parcel.
    county = add_parcel_id_alnum(pd.DataFrame({'parcel_id_assessor': ['4071-68-1844']}))
    state = add_parcel_id_alnum(pd.DataFrame({'parcel_id_admin3': ['4071681844']}))
    assert county[PARCEL_ID_ALNUM].iloc[0] == state[PARCEL_ID_ALNUM].iloc[0]


def test_a_zero_filled_id_column_falls_through_to_the_next_candidate():
    # NC OneMap leaves altparno zero-filled and carries the real county PIN
    # in parno. A zero-filled column is a placeholder, not an id.
    df = pd.DataFrame(
        {
            'parcel_id_assessor': ['0', '0000', '', None, '3280-07-1600'],
            'parcel_id_admin3': [
                '2223-78-7833',
                '2229-42-3660',
                '2229-31-9889',
                '1-1',
                'ignored',
            ],
        }
    )
    out = add_parcel_id_alnum(df)
    assert list(out[PARCEL_ID_ALNUM]) == [
        '2223787833',
        '2229423660',
        '2229319889',
        '11',
        '3280071600',
    ]


def test_a_row_with_no_usable_id_anywhere_gets_no_key():
    df = pd.DataFrame({'parcel_id_assessor': ['0'], 'parcel_id_admin3': [None]})
    assert pd.isna(add_parcel_id_alnum(df)[PARCEL_ID_ALNUM].iloc[0])


def test_candidate_order_prefers_the_sources_own_assessor_id():
    assert PARCEL_ID_MATCH_CANDIDATES[0] == 'parcel_id_assessor'
    df = pd.DataFrame({'parcel_id_assessor': ['AAA-1'], 'parcel_id_admin3': ['BBB-2']})
    assert add_parcel_id_alnum(df)[PARCEL_ID_ALNUM].iloc[0] == 'AAA1'


def test_a_frame_with_no_id_columns_is_untouched():
    df = pd.DataFrame({'other': [1, 2]})
    out = add_parcel_id_alnum(df)
    assert PARCEL_ID_ALNUM not in out.columns


def test_deriving_twice_is_stable():
    # link_by_id recomputes rather than trusting a carried copy, so the
    # function has to be idempotent against its own output.
    df = pd.DataFrame({'parcel_id_assessor': ['4071-68-1844']})
    once = add_parcel_id_alnum(df)
    twice = add_parcel_id_alnum(once.copy())
    assert once[PARCEL_ID_ALNUM].equals(twice[PARCEL_ID_ALNUM])


def test_a_sparse_carried_key_is_overwritten_not_trusted():
    # resolve_spine can carry a nearly-empty copy from whichever source won
    # the geometry -- for the counties this fallback serves, that is the
    # source with no usable id. Pender arrived with 49 of 55,101 filled.
    df = pd.DataFrame(
        {
            'parcel_id_assessor': ['4071-68-1844', '2229-42-3660'],
            PARCEL_ID_ALNUM: [None, 'STALE'],
        }
    )
    out = add_parcel_id_alnum(df)
    assert list(out[PARCEL_ID_ALNUM]) == ['4071681844', '2229423660']


def test_the_default_link_key_is_the_standardized_one():
    # The fallback must never become the default: it cannot tell 1-23 from
    # 12-3, which is exactly the collision parcel_id_local's guard avoids.
    assert DEFAULT_LINK_KEY == 'parcel_id_local'
    assert DEFAULT_LINK_KEY != PARCEL_ID_ALNUM


def test_a_tz_aware_and_a_naive_date_column_can_still_be_combined():
    # One source stamps its sale dates UTC and another writes naive local
    # dates. combine_first raises on that mix rather than picking one, and
    # it took out a whole county's harmonize the first time a statewide
    # layer's dates reached a county whose own source is naive.
    from openplaces.io.harmonizer.links import _align_for_combine

    aware = pd.Series(pd.to_datetime(['2020-01-01']).tz_localize('UTC'))
    naive = pd.Series(pd.to_datetime(['2021-01-01']))
    left, right = _align_for_combine(aware, naive)
    combined = right.combine_first(left)
    assert str(combined.dtype).endswith('UTC]')
    assert combined.iloc[0].year == 2021


def test_alignment_leaves_non_datetime_columns_alone():
    from openplaces.io.harmonizer.links import _align_for_combine

    s = pd.Series(['a', 'b'])
    left, right = _align_for_combine(s, s)
    assert left.dtype == s.dtype and right.dtype == s.dtype


def test_two_naive_date_columns_are_not_forced_to_utc():
    from openplaces.io.harmonizer.links import _align_for_combine

    a = pd.Series(pd.to_datetime(['2020-01-01']))
    b = pd.Series(pd.to_datetime(['2021-01-01']))
    left, right = _align_for_combine(a, b)
    assert getattr(left.dtype, 'tz', None) is None
    assert getattr(right.dtype, 'tz', None) is None


def test_a_datetime_column_against_a_non_datetime_one_does_not_raise():
    # The `.dt` accessor exists only on datetimes. Gating on "either side
    # is a datetime" and then using `.dt` on both crashed every county
    # whose two sources disagree about the dtype, not just the timezone.
    from openplaces.io.harmonizer.links import _align_for_combine

    dates = pd.Series(pd.to_datetime(['2020-01-01', '2021-01-01']))
    text = pd.Series(['2019-05-05', None], dtype=object)
    left, right = _align_for_combine(dates, text)
    combined = right.combine_first(left)
    assert len(combined) == 2
    left2, right2 = _align_for_combine(text, dates)
    assert len(right2.combine_first(left2)) == 2
