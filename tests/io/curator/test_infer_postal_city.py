from __future__ import annotations

import pandas as pd

from openplaces.io.curator import CurateState
from openplaces.io.curator.inferers import infer_postal_city


def make_state(df: pd.DataFrame, admin_id: str = 'US-MA-MI') -> CurateState:
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=admin_id,
        verbose=False,
        timer=None,
        curated=df,
    )


def test_infer_postal_city_resolves_preferred_and_acceptable_cities():
    df = pd.DataFrame({'postal_code': ['02459']})
    state = infer_postal_city(make_state(df))
    res = state.curated

    assert res.loc[0, 'postal_zip5'] == '02459'
    assert res.loc[0, 'postal_city'] == 'Newton Center'
    assert 'Newton' in res.loc[0, 'postal_city_acceptable'].split('; ')
    assert 'Newton Centre' in res.loc[0, 'postal_city_acceptable'].split('; ')
    assert 'Newton Cntr' in res.loc[0, 'postal_city_unacceptable'].split('; ')
    assert res.loc[0, 'postal_city_source'] == 'zipcodes'


def test_infer_postal_city_truncates_zip_plus_four():
    df = pd.DataFrame({'postal_code': ['02459-1234']})
    state = infer_postal_city(make_state(df))
    res = state.curated

    assert res.loc[0, 'postal_zip5'] == '02459'
    assert res.loc[0, 'postal_city'] == 'Newton Center'


def test_infer_postal_city_invalid_zip_is_missing():
    df = pd.DataFrame({'postal_code': ['00000']})
    state = infer_postal_city(make_state(df))
    res = state.curated

    assert res.loc[0, 'postal_zip5'] == '00000'
    assert pd.isna(res.loc[0, 'postal_city'])
    assert pd.isna(res.loc[0, 'postal_city_acceptable'])
    assert pd.isna(res.loc[0, 'postal_city_unacceptable'])
    assert 'postal_city_source' not in res.columns or pd.isna(
        res.loc[0, 'postal_city_source']
    )


def test_infer_postal_city_missing_value_passes_through():
    df = pd.DataFrame({'postal_code': [pd.NA]})
    state = infer_postal_city(make_state(df))
    res = state.curated

    assert pd.isna(res.loc[0, 'postal_zip5'])
    assert pd.isna(res.loc[0, 'postal_city'])


def test_infer_postal_city_no_column_is_noop():
    df = pd.DataFrame({'other_column': [1, 2]})
    state = infer_postal_city(make_state(df))

    assert list(state.curated.columns) == ['other_column']


def test_infer_postal_city_non_us_admin_id_degrades_gracefully():
    df = pd.DataFrame({'postal_code': ['02459']})
    state = infer_postal_city(make_state(df, admin_id='DE'))
    res = state.curated

    # ZIP5 extraction is country-agnostic string parsing, so it still runs;
    # the actual USPS lookup is gated to admin1_id == 'US' and degrades to
    # missing rather than returning US data for a non-US admin unit.
    assert res.loc[0, 'postal_zip5'] == '02459'
    assert pd.isna(res.loc[0, 'postal_city'])
    assert pd.isna(res.loc[0, 'postal_city_acceptable'])
    assert pd.isna(res.loc[0, 'postal_city_unacceptable'])
