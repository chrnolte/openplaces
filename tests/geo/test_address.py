"""Tests for address-number range handling in `geo/address.py`.

A raw address number like '704-706' (a standard multi-unit/multi-family deed
notation) must survive `normalize_address_components` intact -- the generic
`clean_text` punctuation stripping used for every other component would
otherwise squash it into the unmatchable digit string '704706'.
"""

from openplaces.geo.address import (
    _parse_fallback,
    canonicalize_for_match,
    normalize_address_components,
    parse_address,
    split_number_range,
)


def test_digit_range_is_preserved():
    result = normalize_address_components(
        address_street='Broadway', address_number='704-706'
    )
    assert result['address_number'] == '704-706'


def test_range_with_trailing_letter_is_preserved():
    result = normalize_address_components(
        address_street='Sycamore St', address_number='36-36A'
    )
    assert result['address_number'] == '36-36A'


def test_non_range_number_is_unaffected():
    result = normalize_address_components(
        address_street='Main St', address_number='123'
    )
    assert result['address_number'] == '123'


def test_three_part_range_falls_back_to_plain_cleaning():
    # Not a two-part range -- behaves exactly as before the fix.
    result = normalize_address_components(
        address_street='Main St', address_number='38-40-42'
    )
    assert result['address_number'] == '384042'


def test_non_range_punctuation_falls_back_to_plain_cleaning():
    result = normalize_address_components(
        address_street='Main St', address_number='38 REAR'
    )
    assert result['address_number'] == '38 REAR'


def test_split_number_range():
    assert split_number_range('704-706') == ('704', '706')
    assert split_number_range('36-36A') == ('36', '36A')


def test_split_number_range_returns_none_for_non_range():
    assert split_number_range('123') is None
    assert split_number_range('38-40-42') is None
    assert split_number_range('') is None
    assert split_number_range(None) is None


def test_parse_address_preserves_range_via_usaddress_path():
    # A well-formed full address string usaddress tags cleanly (the primary
    # path, _parse_cached) -- confirms the fix at the tagged-component
    # cleaning step, not just normalize_address_components downstream.
    result = parse_address('704-706 Broadway', admin1_id='US')
    assert result.components['address_number'] == '704-706'

    result = parse_address('57-59 Sycamore St', admin1_id='US')
    assert result.components['address_number'] == '57-59'


def test_parse_fallback_preserves_leading_range():
    # Directly exercises the regex fallback path (used when usaddress
    # raises RepeatedLabelError), which has its own independent
    # whole-string clean_text call that would otherwise squash the range
    # before the number is ever pulled out.
    parsed = _parse_fallback('704-706 Broadway Extra Words Here', 'US')
    assert parsed['address_number'] == '704-706'


def test_parse_fallback_plain_number_unaffected():
    parsed = _parse_fallback('123 Main St', 'US')
    assert parsed['address_number'] == '123'


# Real MA deed-vs-parcel spelling mismatches found while investigating the
# transaction-linkage match rate: the transaction source truncates or
# informally abbreviates these street suffixes/words, while MassGIS parcel
# data uses the fuller or differently-abbreviated form.
def test_terr_matches_terrace_abbreviation():
    assert canonicalize_for_match('Gilman Ter', 'US') == canonicalize_for_match(
        'Gilman Terr', 'US'
    )


def test_blv_matches_blvd_abbreviation():
    a = canonicalize_for_match('Powder House Blv', 'US')
    b = canonicalize_for_match('Powder House Blvd', 'US')
    assert a == b


def test_pkw_matches_pkwy_abbreviation():
    a = canonicalize_for_match('Alewife Brook Pkw', 'US')
    b = canonicalize_for_match('Alewife Brook Pkwy', 'US')
    assert a == b


def test_pk_matches_park_abbreviation():
    assert canonicalize_for_match('Wesley Pk', 'US') == canonicalize_for_match(
        'Wesley Park', 'US'
    )


def test_powderhouse_matches_powder_house_phrase():
    a = canonicalize_for_match('Powderhouse Terr', 'US')
    b = canonicalize_for_match('Powder House Terr', 'US')
    assert a == b
