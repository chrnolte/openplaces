"""Several house numbers on one building or lot become one range."""

from __future__ import annotations

import pytest

from openplaces.geo.address import (
    compress_number_list,
    normalize_address_components,
    split_address_list,
)
from openplaces.io.harmonizer.addresses import _parse_address_list


@pytest.mark.parametrize(
    'text, expected',
    [
        ('241; 217; 215', '215-241'),
        ('241 217 215', '215-241'),
        ('62 60', '60-62'),
        ('708 716A 706 716B', '706-716'),
        ('12; 12', '12'),
        ('12', None),
        ('12 1/2', None),  # a half-numbered address, not a list
        ('12 MAIN', None),
        ('', None),
        (None, None),
    ],
)
def test_compress_number_list(text, expected):
    assert compress_number_list(text) == expected


def test_a_number_field_with_several_numbers_normalizes_to_a_range():
    out = normalize_address_components('HANCOCK STREET', address_number='241; 217; 215')
    assert out['address_number'] == '215-241'
    # A range already written as one stays as it is.
    assert normalize_address_components('MAIN ST', '704-706')['address_number'] == (
        '704-706'
    )


def test_split_address_list():
    assert split_address_list('53 Vale St + 102 Mill Rd') == [
        '53 Vale St',
        '102 Mill Rd',
    ]
    assert split_address_list('1 Main St; 3 Main St') == ['1 Main St', '3 Main St']
    assert split_address_list('5-7 Harris St') == ['5-7 Harris St']


def test_a_joined_list_keeps_the_first_street_and_its_numbers_as_a_range():
    out = _parse_address_list(
        '53 Vale St + 55 Vale St + 102 Mill Rd + 31 Vale St', 'US'
    )
    assert out['address_street'] == 'VALE ST'
    assert out['address_number'] == '31-55'


def test_a_single_address_parses_as_before():
    out = _parse_address_list('62 Boylston St', 'US')
    assert out['address_number'] == '62'
    assert out['address_street'] == 'BOYLSTON ST'
