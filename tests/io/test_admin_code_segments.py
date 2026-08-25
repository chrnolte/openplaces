"""Tests for reading a named part out of a hierarchical national code.

The behavior under test is that a caller names a *segment* in the
source's own vocabulary and never an openplaces admin level, because the
level a segment corresponds to varies by geography and changes when a
government reorganizes.
"""

import pandas as pd
import pytest

from openplaces.io.admin_codes.segments import (
    admin_code_segment,
    get_segment,
    load_code_segments,
    slice_segment,
)

# Fabricated codes in the shape of the real scheme. A five-character
# value is a county FIPS, a ten-character value a county subdivision.
COUNTY_CODE = '99001'
COUSUB_CODE = '9900112345'


def test_county_segment_of_a_subdivision_code_is_its_prefix():
    out = slice_segment(pd.Series([COUSUB_CODE]), segment='county')
    assert out.tolist() == [COUNTY_CODE]


def test_county_segment_of_a_county_code_is_the_code_itself():
    """The identity case is why nothing has to be stored.

    A unit whose national code already *is* the segment needs no lookup
    row and no extra column: slicing returns the same string. That is
    what keeps `county_fips` from being duplicated onto the 3,076 US
    units where it equals `admin3_id_admin1`.
    """
    out = slice_segment(pd.Series([COUNTY_CODE]), segment='county')
    assert out.tolist() == [COUNTY_CODE]


def test_state_segment_is_shared_by_both_widths():
    out = slice_segment(pd.Series([COUNTY_CODE, COUSUB_CODE]), segment='state')
    assert out.tolist() == ['99', '99']


def test_blank_and_missing_codes_yield_missing_results():
    out = slice_segment(pd.Series(['', None, COUNTY_CODE]), segment='county')
    assert out.isna().tolist() == [True, True, False]


def test_undeclared_width_raises_rather_than_truncating():
    """A code that lost a leading zero must not slice quietly.

    Four characters is not a width this scheme declares, so a `county`
    lookup on it cannot be answered. Returning the whole short string
    would look like a valid answer.
    """
    with pytest.raises(ValueError, match='does not declare'):
        slice_segment(pd.Series(['9001']), segment='county')


def test_undeclared_width_is_skipped_when_not_strict():
    out = slice_segment(
        pd.Series(['9001', COUSUB_CODE]), segment='county', strict=False
    )
    assert out.isna().tolist() == [True, False]
    assert out.dropna().tolist() == [COUNTY_CODE]


def test_unknown_segment_lists_what_is_declared():
    with pytest.raises(KeyError, match='Declared segments'):
        get_segment('parish')


def test_unknown_scheme_lists_what_is_declared():
    with pytest.raises(KeyError, match='Declared schemes'):
        get_segment('county', scheme='not-a-scheme')


def test_every_declaration_is_internally_consistent():
    for segment in load_code_segments().values():
        assert segment.start >= 0
        assert segment.length > 0
        assert segment.total_length >= segment.start + segment.length
        assert segment.admin1_id, f'{segment.segment} declares no country'


class TestAgainstTheCommittedSpine:
    """The cases this mechanism exists for, read off the real spine."""

    def test_new_england_towns_resolve_to_their_counties(self):
        counties = admin_code_segment(segment='county').dropna()
        towns = counties[counties.index.str.match(r'US-(CT|MA|ME|NH|RI|VT)-')]
        # Six states whose level 3 is towns, not counties: the county is
        # only reachable through the segment.
        assert len(towns) == 1603
        assert towns.nunique() == 68
        assert towns.str.len().eq(5).all()

    def test_counties_resolve_to_themselves(self):
        counties = admin_code_segment(segment='county').dropna()
        elsewhere = counties[~counties.index.str.match(r'US-(CT|MA|ME|NH|RI|VT)-')]
        assert len(elsewhere) == elsewhere.nunique()

    def test_a_scheme_is_applied_only_to_its_own_country(self):
        """A five-character code elsewhere is not a US county FIPS."""
        counties = admin_code_segment(segment='county')
        assert counties.index.str.startswith('US-').all()
