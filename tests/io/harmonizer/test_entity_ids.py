"""Stable entity ids for a non-spatial spine (io.harmonizer.entity_ids).

Every id and value below is fabricated.
"""

import pandas as pd
import pytest

from openplaces.io.harmonizer.entity_ids import (
    merge_on_id,
    mint_ids,
    normalize_issued_id,
)

ADMIN = 'US-XX-ABC'


def _roll(numbers, **columns):
    return pd.DataFrame({'account': numbers, 'source': 'roll', **columns})


def test_two_publications_of_one_number_normalize_alike():
    values = pd.Series(['12-34 ab', '12_34__AB', ' 12.34.ab. '])
    assert normalize_issued_id(values).tolist() == ['12-34-AB'] * 3


def test_separator_positions_are_part_of_the_number():
    # Map 12, block A, lot 1, unit 2 is not map 12, block A, lot 12.
    values = pd.Series(['12_A_1_2', '12_A_12'])
    assert normalize_issued_id(values).is_unique


def test_blank_and_all_zero_numbers_are_not_numbers():
    values = pd.Series(['', '000-00', None, '0', ' - '])
    assert normalize_issued_id(values).isna().all()


def test_the_id_is_the_admin_unit_and_the_issued_number():
    ids, report = mint_ids(_roll(['a 1', 'A-2']), ADMIN, 'account', 'roll')
    assert ids.tolist() == ['US-XX-ABC_A-1', 'US-XX-ABC_A-2']
    assert report == {'n_without_number': 0, 'n_repeated': 0, 'n_exact_duplicates': 0}


def test_ids_do_not_depend_on_row_order():
    rows = _roll(['A1', 'A2', 'A3'], value=[1.0, 2.0, 3.0])
    forward, _ = mint_ids(rows, ADMIN, 'account', 'roll')
    shuffled = rows.iloc[[2, 0, 1]]
    backward, _ = mint_ids(shuffled, ADMIN, 'account', 'roll')
    assert dict(zip(rows['value'], forward)) == dict(zip(shuffled['value'], backward))


def test_a_repeated_number_gets_a_suffix_ordered_by_content_not_position():
    rows = _roll(['A1'] * 2 + [f'B{i}' for i in range(200)], value=range(202))
    ids, report = mint_ids(rows, ADMIN, 'account', 'roll')
    assert ids.is_unique
    assert sorted(ids[:2]) == ['US-XX-ABC_A1', 'US-XX-ABC_A1_2']
    assert report['n_repeated'] == 2
    flipped = rows.iloc[[1, 0, *range(2, 202)]]
    again, _ = mint_ids(flipped, ADMIN, 'account', 'roll')
    assert dict(zip(rows['value'], ids)) == dict(zip(flipped['value'], again))


def test_exact_duplicate_rows_share_one_id():
    rows = _roll(['A1', 'A1', 'A2'], value=[5.0, 5.0, 6.0])
    ids, report = mint_ids(rows, ADMIN, 'account', 'roll')
    assert ids.tolist() == ['US-XX-ABC_A1', 'US-XX-ABC_A1', 'US-XX-ABC_A2']
    assert report['n_exact_duplicates'] == 1


def test_a_column_naming_the_lot_instead_of_the_account_raises():
    # Ten units on each of ten lots: the lot number repeats everywhere.
    rows = _roll([f'LOT{i // 10}' for i in range(100)], unit=range(100))
    with pytest.raises(ValueError, match='coarser than the'):
        mint_ids(rows, ADMIN, 'account', 'roll')


def test_split_units_keep_the_numbers_that_do_name_one_unit():
    # A parcel layer that repeats the lot's number on every unit of one
    # stack, and gives the units of another their own.
    rows = _roll(['LOT'] * 30 + [f'U{i}' for i in range(30)], unit=range(60))
    ids, report = mint_ids(
        rows, ADMIN, 'account', 'layer:units', content_if_coarse=True
    )
    assert ids.is_unique
    assert ids[30:].tolist() == [f'US-XX-ABC_U{i}' for i in range(30)]
    assert ids[:30].str.startswith('US-XX-ABC_layer:units:').all()
    assert report['n_coarse_numbers'] == 30


def test_a_row_without_a_number_is_named_by_its_content():
    rows = _roll([None, 'A2'], value=[1.0, 2.0])
    ids, report = mint_ids(rows, ADMIN, 'account', 'roll')
    assert ids[0].startswith('US-XX-ABC_roll:')
    assert report['n_without_number'] == 1
    again, _ = mint_ids(rows.iloc[[1, 0]], ADMIN, 'account', 'roll')
    assert again[0] == ids[0]


def test_two_sources_with_the_same_ids_merge_into_one_row_each():
    county = pd.DataFrame(
        {'account': ['A1', 'A2'], 'rooms': [5, None], 'source': 'county'}
    )
    state = pd.DataFrame(
        {
            'account': ['a1', ' a2', 'a3.'],
            'rooms': [9, 4, 7],
            'use': ['R', 'R', 'C'],
            'source': 'state',
        }
    )
    rows = pd.concat([county, state], ignore_index=True)
    ids = pd.concat(
        [
            mint_ids(county, ADMIN, 'account', 'county')[0],
            mint_ids(state, ADMIN, 'account', 'state')[0],
        ],
        ignore_index=True,
    )
    merged = merge_on_id(rows, ids, 'property_id')

    assert merged.index.name == 'property_id'
    assert list(merged.index) == ['US-XX-ABC_A1', 'US-XX-ABC_A2', 'US-XX-ABC_A3']
    # The first source wins a cell; the second fills what it left empty.
    assert merged['rooms'].tolist() == [5, 4, 7]
    assert merged['use'].tolist() == ['R', 'R', 'C']
    assert merged['source'].tolist() == ['county+state', 'county+state', 'state']
