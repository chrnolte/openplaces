"""Tests for the harmonizer `derive_address_id_local` step.

Companion matching key to `parcel_id_local` for entities with no shared
parcel id to join on (e.g. MA masslandrecords transactions, which carry no
parcel identifier at all). `address_id_local` is scoped by `admin4_id` (a
town), not `city`, since a town can contain several villages that each use
their own name as a street address's city (e.g. Newton, MA contains Newton
Centre, Waban, ...) -- two real, distinct streets of the same name in two
different villages of one town would otherwise collide if `city` were
trusted as the disambiguator.
"""

import pandas as pd

from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer.addresses import derive_address_id_local


def _state(spine):
    return HarmonizeState(
        recipe={}, admin_id='US-MA-MI', verbose=False, timer=None, spine=spine
    )


def test_same_street_different_village_same_town_does_not_collide():
    # Real Newton, MA pattern: "Boylston St" runs through several villages;
    # different house numbers on the same street should not collide.
    spine = pd.DataFrame(
        {
            'address_street': ['Boylston St', 'Boylston St'],
            'address_number': ['528', '662'],
            'city': ['Newton Centre', 'Newton Centre'],
            'admin4_id': ['US-MA-MI-NE', 'US-MA-MI-NE'],
        }
    )
    state = derive_address_id_local(_state(spine))

    assert state.spine['address_id_local'].nunique() == 2
    assert state.spine['address_id_local'].notna().all()


def test_city_omitted_from_plain_key_but_included_in_city_key():
    spine = pd.DataFrame(
        {
            'address_street': ['Hawthorn St'],
            'address_number': ['10'],
            'city': ['Waban'],
            'admin4_id': ['US-MA-MI-NE'],
        }
    )
    state = derive_address_id_local(_state(spine))

    assert 'WABAN' not in state.spine['address_id_local'].iloc[0]
    assert 'WABAN' in state.spine['address_id_local_city'].iloc[0]


def test_missing_city_leaves_city_key_null():
    spine = pd.DataFrame(
        {
            'address_street': ['Main St'],
            'address_number': ['5'],
            'city': [None],
            'admin4_id': ['US-MA-MI-NE'],
        }
    )
    state = derive_address_id_local(_state(spine))

    assert pd.notna(state.spine['address_id_local'].iloc[0])
    assert pd.isna(state.spine['address_id_local_city'].iloc[0])


def test_missing_street_or_number_yields_null_key():
    spine = pd.DataFrame(
        {
            'address_street': [None, 'Main St'],
            'address_number': ['5', None],
            'city': ['Newton', 'Newton'],
            'admin4_id': ['US-MA-MI-NE', 'US-MA-MI-NE'],
        }
    )
    state = derive_address_id_local(_state(spine))

    assert state.spine['address_id_local'].isna().all()


def test_different_towns_with_same_street_and_number_do_not_collide():
    spine = pd.DataFrame(
        {
            'address_street': ['Main St', 'Main St'],
            'address_number': ['5', '5'],
            'city': ['Newton', 'Newton'],
            'admin4_id': ['US-MA-MI-NE', 'US-MA-MI-WA'],
        }
    )
    state = derive_address_id_local(_state(spine))

    assert state.spine['address_id_local'].nunique() == 2


def test_missing_columns_is_a_no_op():
    spine = pd.DataFrame({'other': [1, 2]})
    state = derive_address_id_local(_state(spine))

    assert 'address_id_local' not in state.spine.columns
