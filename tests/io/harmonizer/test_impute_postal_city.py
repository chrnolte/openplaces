from __future__ import annotations

import pandas as pd

from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer.addresses import impute_postal_city


def make_state(df: pd.DataFrame, admin_id: str = 'US-MA-MI') -> HarmonizeState:
    return HarmonizeState(
        recipe={},
        admin_id=admin_id,
        verbose=False,
        timer=None,
        spine=df,
    )


def test_impute_postal_city_resolves_preferred_and_acceptable_cities():
    df = pd.DataFrame({'postal_code': ['02459']})
    state = impute_postal_city(make_state(df))
    res = state.spine

    assert res.loc[0, 'postal_zip5'] == '02459'
    assert res.loc[0, 'postal_city'] == 'Newton Center'
    assert 'Newton' in res.loc[0, 'postal_city_acceptable'].split('; ')
    assert 'Newton Centre' in res.loc[0, 'postal_city_acceptable'].split('; ')
    assert 'Newton Cntr' in res.loc[0, 'postal_city_unacceptable'].split('; ')
    assert res.loc[0, 'postal_city_source'] == 'zipcodes'


def test_impute_postal_city_truncates_zip_plus_four():
    df = pd.DataFrame({'postal_code': ['02459-1234']})
    state = impute_postal_city(make_state(df))
    res = state.spine

    assert res.loc[0, 'postal_zip5'] == '02459'
    assert res.loc[0, 'postal_city'] == 'Newton Center'


def test_impute_postal_city_invalid_zip_is_missing():
    df = pd.DataFrame({'postal_code': ['00000']})
    state = impute_postal_city(make_state(df))
    res = state.spine

    assert res.loc[0, 'postal_zip5'] == '00000'
    assert pd.isna(res.loc[0, 'postal_city'])
    assert pd.isna(res.loc[0, 'postal_city_acceptable'])
    assert pd.isna(res.loc[0, 'postal_city_unacceptable'])
    assert 'postal_city_source' not in res.columns or pd.isna(
        res.loc[0, 'postal_city_source']
    )


def test_impute_postal_city_missing_value_passes_through():
    df = pd.DataFrame({'postal_code': [pd.NA]})
    state = impute_postal_city(make_state(df))
    res = state.spine

    assert pd.isna(res.loc[0, 'postal_zip5'])
    assert pd.isna(res.loc[0, 'postal_city'])


def test_impute_postal_city_no_column_is_noop():
    df = pd.DataFrame({'other_column': [1, 2]})
    state = impute_postal_city(make_state(df))

    assert list(state.spine.columns) == ['other_column']


def test_impute_postal_city_spine_none_is_noop():
    state = HarmonizeState(
        recipe={}, admin_id='US-MA-MI', verbose=False, timer=None, spine=None
    )
    assert impute_postal_city(state).spine is None


def test_impute_postal_city_non_us_admin_id_degrades_gracefully():
    df = pd.DataFrame({'postal_code': ['02459']})
    state = impute_postal_city(make_state(df, admin_id='DE'))
    res = state.spine

    # ZIP5 extraction is country-agnostic string parsing, so it still runs;
    # the actual USPS lookup is gated to admin1_id == 'US' and degrades to
    # missing rather than returning US data for a non-US admin unit.
    assert res.loc[0, 'postal_zip5'] == '02459'
    assert pd.isna(res.loc[0, 'postal_city'])
    assert pd.isna(res.loc[0, 'postal_city_acceptable'])
    assert pd.isna(res.loc[0, 'postal_city_unacceptable'])


def test_impute_postal_city_backfills_city_and_rerenders_address():
    # Mirrors the real gap: an address was assembled at harmonize time with
    # no city (e.g. dwelling_overture's own city was empty), but a ZIP
    # resolves one. `city`/`address` should both pick it up.
    df = pd.DataFrame(
        {
            'postal_code': ['01862'],
            'city': [pd.NA],
            'address': ['1 Theresa Ave, MA 01862'],
            'address_number': ['1'],
            'address_street': ['THERESA AVE'],
            'address_unit': [pd.NA],
        }
    )
    state = impute_postal_city(make_state(df))
    res = state.spine

    assert res.loc[0, 'city'] == 'North Billerica'
    assert res.loc[0, 'city_source'] == 'zipcodes'
    assert res.loc[0, 'address'] == '1 Theresa Ave, North Billerica, MA 01862'


def test_impute_postal_city_does_not_override_existing_city():
    df = pd.DataFrame(
        {
            'postal_code': ['01862'],
            'city': ['Billerica'],
            'address': ['1 Theresa Ave, Billerica, MA 01862'],
            'address_number': ['1'],
            'address_street': ['THERESA AVE'],
            'address_unit': [pd.NA],
        }
    )
    state = impute_postal_city(make_state(df))
    res = state.spine

    assert res.loc[0, 'city'] == 'Billerica'
    assert 'city_source' not in res.columns or pd.isna(res.loc[0, 'city_source'])
    assert res.loc[0, 'address'] == '1 Theresa Ave, Billerica, MA 01862'


def test_impute_postal_city_backfill_needs_no_address_number_or_unit_columns():
    # A recipe that only opted into city_output_col (not number/unit) should
    # still get a correctly re-rendered address, just without those parts.
    df = pd.DataFrame(
        {
            'postal_code': ['01862'],
            'city': [pd.NA],
            'address': ['Theresa Ave, MA 01862'],
            'address_street': ['THERESA AVE'],
        }
    )
    state = impute_postal_city(make_state(df))
    res = state.spine

    assert res.loc[0, 'city'] == 'North Billerica'
    assert res.loc[0, 'address'] == 'Theresa Ave, North Billerica, MA 01862'
