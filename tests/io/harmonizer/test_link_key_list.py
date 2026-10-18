"""link_entities_by_id reads the first key both tables carry, as the
parcel curate's key join does: a roll keyed on parcel_id_admin2 alone got
an empty link table (Somerville MA, 2026-10-17). Every id is fabricated.
"""

import pandas as pd
import pytest

from openplaces.io.harmonizer.entity_links import _first_shared_key


def _parcels():
    return pd.DataFrame(
        {'parcel_id_local': ['L1', 'L2'], 'parcel_id_admin2': ['M1', 'M2']},
        index=pd.Index(['p1', 'p2'], name='spine_id'),
    )


def test_a_single_key_is_used_as_before():
    props = pd.DataFrame({'parcel_id_local': ['L1']})
    assert _first_shared_key(
        _parcels(), props, 'parcel_id_local', 'parcel_id_local'
    ) == ('parcel_id_local', 'parcel_id_local')


def test_the_first_key_both_tables_carry_is_used():
    props = pd.DataFrame({'parcel_id_admin2': ['M1']})
    keys = ['parcel_id_local', 'parcel_id_admin2']
    assert _first_shared_key(_parcels(), props, keys, keys) == (
        'parcel_id_admin2',
        'parcel_id_admin2',
    )


def test_a_key_with_no_value_on_one_side_is_skipped():
    props = pd.DataFrame({'parcel_id_local': [None], 'parcel_id_admin2': ['M1']})
    keys = ['parcel_id_local', 'parcel_id_admin2']
    assert _first_shared_key(_parcels(), props, keys, keys)[0] == 'parcel_id_admin2'


def test_one_reference_key_pairs_with_every_spine_key():
    # Position by position: the first spine key holding values pairs
    # with the single reference key.
    props = pd.DataFrame({'parcel_id_admin2': ['M1']})
    assert _first_shared_key(
        _parcels(), props, ['parcel_id_local', 'parcel_id_admin2'], 'parcel_id_admin2'
    ) == ('parcel_id_local', 'parcel_id_admin2')


def test_no_shared_key_falls_back_to_the_first():
    props = pd.DataFrame({'other': ['x']})
    keys = ['parcel_id_local', 'parcel_id_admin2']
    assert _first_shared_key(_parcels(), props, keys, keys) == (
        'parcel_id_local',
        'parcel_id_local',
    )


def test_lists_of_different_length_are_refused():
    with pytest.raises(ValueError, match='different numbers'):
        _first_shared_key(_parcels(), _parcels(), ['a', 'b'], ['a', 'b', 'c'])
