from __future__ import annotations

import pandas as pd
import pytest

from openplaces.geo.address import (
    DIRECTIONALS,
    SECONDARY_UNITS,
    STREET_SUFFIXES,
    ParsedAddress,
    canonicalize_for_match,
    get_address_format,
    get_admin2_codes,
    harmonize_address_case,
    match_streets,
    parse_address,
    parse_address_string,
)
from openplaces.io.curator import CurateState
from openplaces.io.curator.reconcilers import reconcile_addresses


def make_state(df: pd.DataFrame) -> CurateState:
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id='US-MA-MI',
        verbose=False,
        timer=None,
        curated=df,
    )


def test_parse_address_string():
    assert parse_address_string('123 N Main St, Apt 4B') == {
        'address_number': '123',
        'address_street': 'N MAIN ST',
        'unit_number': 'APT 4B',
        'city': None,
        'state': None,
        'postal_code': None,
    }
    # Suffix-less street must not produce a false city split
    parsed = parse_address_string('N Broadway')
    assert parsed['address_street'] == 'N BROADWAY'
    assert parsed['address_number'] is None
    assert parsed['city'] is None
    # Empty input
    assert parse_address_string('') == {
        'address_number': None,
        'address_street': None,
        'unit_number': None,
        'city': None,
        'state': None,
        'postal_code': None,
    }


def test_parse_address_string_city_state_zip():
    parsed = parse_address_string('5006 BOGUE SOUND DR EMERALD ISLE NC 28594')
    assert parsed['address_number'] == '5006'
    assert parsed['address_street'] == 'BOGUE SOUND DR'
    assert parsed['city'] == 'EMERALD ISLE'
    assert parsed['state'] == 'NC'
    assert parsed['postal_code'] == '28594'


def test_parse_address_metadata_and_strings():
    result = parse_address('5006 BOGUE SOUND DR EMERALD ISLE NC 28594')
    assert isinstance(result, ParsedAddress)
    assert result.address_raw == '5006 BOGUE SOUND DR EMERALD ISLE NC 28594'
    assert result.address_normalized == '5006 BOGUE SOUND DR, EMERALD ISLE, NC 28594'
    assert result.address_formatted == '5006 Bogue Sound Dr, Emerald Isle, NC 28594'
    assert result.metadata['parser'] == 'usaddress'
    assert result.metadata['status'] == 'success'
    assert result.metadata['version']


def test_parse_address_backend_none_preserves_raw():
    result = parse_address('Musterstrasse 12, 10115 Berlin', backend='none')
    assert result.address_raw == 'Musterstrasse 12, 10115 Berlin'
    assert all(v is None for v in result.components.values())
    assert result.metadata == {'parser': 'none', 'version': None, 'status': 'success'}
    # Raw string survives, cleaned/title-cased, in the output strings
    assert result.address_normalized == 'MUSTERSTRASSE 12 10115 BERLIN'
    assert result.address_formatted == 'Musterstrasse 12 10115 Berlin'


def test_parse_address_auto_non_us_falls_back():
    result = parse_address('Musterstrasse 12, 10115 Berlin', admin1_id='DE')
    assert result.metadata['parser'] == 'none'
    assert result.metadata['status'] == 'fallback'
    assert result.address_raw == 'Musterstrasse 12, 10115 Berlin'
    assert all(v is None for v in result.components.values())


def test_parse_address_placeholder_backend_warns():
    with pytest.warns(UserWarning, match='placeholder'):
        result = parse_address('123 Main St', backend='libpostal')
    # Falls back to auto resolution (usaddress for US) but flags the fallback
    assert result.metadata['status'] == 'fallback'
    assert result.components['address_street'] == 'MAIN ST'


def test_parse_address_unknown_backend_raises():
    with pytest.raises(ValueError, match='Unknown address parser backend'):
        parse_address('123 Main St', backend='geocoder3000')


def test_equivalence_tables_loaded_from_csv():
    assert STREET_SUFFIXES['STREET'] == 'ST'
    assert STREET_SUFFIXES['HIGHWAY'] == 'HWY'
    assert DIRECTIONALS['NORTH'] == 'N'
    assert SECONDARY_UNITS['APARTMENT'] == 'APT'


def test_match_streets():
    # Pre-type notation + dangling trailing directional (Tyrrell County case)
    assert match_streets('HWY 94 S', 'HIGHWAY 94')
    # Multi-word match phrase: UNITED STATES -> US
    assert match_streets('OLD US 64', 'OLD UNITED STATES HIGHWAY 64')
    # Noise token drop: literal NULL in dirty parcel strings
    assert match_streets('MAIN ST NULL NULL', 'Main Street')
    # Case-insensitive by construction
    assert match_streets('bogue sound drive', 'BOGUE SOUND DR')
    # Genuinely different streets stay unmatched
    assert not match_streets('MAIN ST', 'ATLANTIC AVE')
    assert not match_streets('HWY 64 E', 'CARLENO DR')
    assert not match_streets('', 'MAIN ST')


def test_harmonize_address_case():
    # Standard directional & suffix; the unit joins the street segment
    assert (
        harmonize_address_case('N MAIN ST', '123', 'APT 4B') == '123 N Main St Apt 4B'
    )
    # A bare unit identifier renders with the '#' notation
    assert harmonize_address_case('MAIN ST', '123', '4B') == '123 Main St #4B'
    # Ordinal street name
    assert harmonize_address_case('E 56TH AVE', '1234') == '1234 E 56th Ave'
    # Roman numeral
    assert harmonize_address_case('HENRY III RD', '5') == '5 Henry III Rd'
    # Directional without suffix
    assert harmonize_address_case('N BROADWAY', '100') == '100 N Broadway'
    # Words that look like Roman numerals stay title-cased
    assert harmonize_address_case('MILL CREEK RD', '7') == '7 Mill Creek Rd'


def test_harmonize_address_case_city_state_zip():
    result = harmonize_address_case(
        'BOGUE SOUND DR',
        '5006',
        city='EMERALD ISLE',
        state='NC',
        postal_code='28594',
    )
    assert result == '5006 Bogue Sound Dr, Emerald Isle, NC 28594'


def test_harmonize_address_case_german_format():
    # The DE row in address_formats.csv renders street-first with the postal
    # code preceding the city and no state segment.
    result = harmonize_address_case(
        'HAUPTSTRASSE',
        '12',
        city='MUNICH',
        postal_code='80331',
        admin1_id='DE',
    )
    assert result == 'Hauptstrasse 12, 80331 Munich'
    # Without a postal code the city segment still renders
    assert (
        harmonize_address_case('HAUPTSTRASSE', '12', city='MUNICH', admin1_id='DE')
        == 'Hauptstrasse 12, Munich'
    )


def test_get_address_format_rows():
    assert get_address_format('DE')['format'].startswith('{address_street}')
    # Unknown countries fall back to the default row
    assert get_address_format('FR') == get_address_format(None)


def test_canonicalize_for_match_is_country_scoped():
    # USPS abbreviations must not leak into other countries' matching
    assert canonicalize_for_match('MAIN STREET') == 'MAIN ST'
    assert canonicalize_for_match('MAIN STREET', admin1_id='DE') == 'MAIN STREET'


def test_reconcile_addresses_curate_step():
    df = pd.DataFrame(
        {
            'address_parcel': [
                '123 MAIN ST, APT 4B',
                '456 BROADWAY AVE',
                None,
                '789 ELM ST',
            ],
            'address_street_overture': ['Main St', None, 'Oak Rd', 'Pine St'],
            # 999 doesn't match 789 in the last row
            'address_number_overture': ['123', None, '100', '999'],
        }
    )
    state = reconcile_addresses(
        make_state(df),
        sources={
            'parcel': {'address_full': 'address_parcel'},
            'dwelling_overture': {
                'address_street': 'address_street_overture',
                'address_number': 'address_number_overture',
            },
        },
    )
    res = state.curated

    # Row 0: both agree -> reconciled, parcel base keeps its unit
    assert res.loc[0, 'address'] == '123 Main St Apt 4B'
    assert res.loc[0, 'address_source'] == 'reconciled'

    # Row 1: only parcel present
    assert res.loc[1, 'address'] == '456 Broadway Ave'
    assert res.loc[1, 'address_source'] == 'parcel'

    # Row 2: only the dwelling point present
    assert res.loc[2, 'address'] == '100 Oak Rd'
    assert res.loc[2, 'address_source'] == 'dwelling_overture'

    # Row 3: house numbers disagree -> higher-priority parcel wins alone,
    # and the disagreement is summarized in address_conflict
    assert res.loc[3, 'address'] == '789 Elm St'
    assert res.loc[3, 'address_source'] == 'parcel'
    assert (
        res.loc[3, 'address_conflict']
        == 'parcel: 789 ELM ST | dwelling_overture: 999 PINE ST'
    )
    # Agreeing and single-source rows carry no conflict
    assert res.loc[[0, 1, 2], 'address_conflict'].isna().all()


def test_reconcile_addresses_writes_street_output_col():
    df = pd.DataFrame({'address_parcel': ['9-11 PEARSON AVE UNIT 1', '7 PEARSON AVE']})
    state = reconcile_addresses(
        make_state(df),
        sources={'parcel': {'address_full': 'address_parcel'}},
    )
    res = state.curated
    # Both resolve to the same parsed street, regardless of the unit suffix
    # on the first -- the whole point of grouping by this column instead of
    # the full formatted `address` string. Case-formatted, like `address`.
    assert res.loc[0, 'address_street'] == 'Pearson Ave'
    assert res.loc[1, 'address_street'] == 'Pearson Ave'


def test_reconcile_addresses_street_output_col_disabled():
    df = pd.DataFrame({'address_parcel': ['123 MAIN ST']})
    state = reconcile_addresses(
        make_state(df),
        sources={'parcel': {'address_full': 'address_parcel'}},
        street_output_col=None,
    )
    assert 'address_street' not in state.curated.columns


def test_reconcile_addresses_writes_number_unit_city_output_cols():
    df = pd.DataFrame(
        {'address_parcel': ['9 PEARSON AVE APT 4B, EMERALD ISLE NC 28594']}
    )
    state = reconcile_addresses(
        make_state(df),
        sources={'parcel': {'address_full': 'address_parcel'}},
        number_output_col='address_number',
        unit_output_col='address_unit',
        city_output_col='city',
    )
    res = state.curated
    # The persisted components are case-formatted, matching what ended up in
    # the formatted `address` string -- not the internal matching-only
    # uppercase representation.
    assert res.loc[0, 'address'] == '9 Pearson Ave Apt 4B, Emerald Isle, NC 28594'
    assert res.loc[0, 'address_number'] == '9'
    assert res.loc[0, 'address_unit'] == 'Apt 4B'
    assert res.loc[0, 'city'] == 'Emerald Isle'


def test_reconcile_addresses_output_cols_missing_component_stays_na():
    # No unit/city on this row -> the opted-in output columns stay missing
    # rather than getting written as empty strings.
    df = pd.DataFrame({'address_parcel': ['123 MAIN ST']})
    state = reconcile_addresses(
        make_state(df),
        sources={'parcel': {'address_full': 'address_parcel'}},
        unit_output_col='address_unit',
        city_output_col='city',
    )
    res = state.curated
    assert pd.isna(res.loc[0, 'address_unit'])
    assert pd.isna(res.loc[0, 'city'])


def test_reconcile_addresses_source_column_survives_when_distinct_from_output():
    # Regression guard for the parcel spine's former bug: a source column
    # sharing a name with output_col/city_output_col gets silently
    # overwritten by the reconciled value (no surviving raw copy). As long as
    # the source's evidence columns are named distinctly from the output
    # columns -- as US_parcel-spine-2026.yaml now ensures via a
    # rename_columns step ahead of this call -- the raw values are untouched.
    df = pd.DataFrame(
        {
            'address_original': ['1 theresa ave'],
            'city_original': ['billerica'],
        }
    )
    state = reconcile_addresses(
        make_state(df),
        sources={
            'assessor': {
                'address_full': 'address_original',
                'city': 'city_original',
            },
        },
        city_output_col='city',
    )
    res = state.curated
    # Raw evidence untouched -- still exactly what came in.
    assert res.loc[0, 'address_original'] == '1 theresa ave'
    assert res.loc[0, 'city_original'] == 'billerica'
    # Reconciled output lands in the separate, canonical columns.
    assert res.loc[0, 'address'] == '1 Theresa Ave, Billerica'
    assert res.loc[0, 'city'] == 'Billerica'


def test_reconcile_addresses_usaddress_city_annotation():
    # Same one-line string, same parsed city -- but one source declares an
    # explicit city role that overrides the parsed value (per the class
    # docstring), so only the address_full-only source's row is annotated.
    df = pd.DataFrame(
        {
            'address_full_a': ['5006 BOGUE SOUND DR EMERALD ISLE NC 28594', None],
            'address_full_b': [None, '123 MAIN ST EMERALD ISLE NC 28594'],
            'city_b': [None, 'EMERALD ISLE'],
        }
    )
    state = reconcile_addresses(
        make_state(df),
        sources={
            'a': {'address_full': 'address_full_a'},
            'b': {'address_full': 'address_full_b', 'city': 'city_b'},
        },
    )
    res = state.curated
    assert res.loc[0, 'address_source'] == 'a+usaddress'
    assert res.loc[1, 'address_source'] == 'b'


def test_reconcile_addresses_full_string_with_city_state_zip():
    df = pd.DataFrame({'address_parcel': ['5006 BOGUE SOUND DR EMERALD ISLE NC 28594']})
    state = reconcile_addresses(
        make_state(df),
        sources={'parcel': {'address_full': 'address_parcel'}},
    )
    res = state.curated
    assert res.loc[0, 'address'] == '5006 Bogue Sound Dr, Emerald Isle, NC 28594'
    # 'parcel' declares only address_full (no explicit city role), so its
    # city could only come from parsing that string -- annotated accordingly.
    assert res.loc[0, 'address_source'] == 'parcel+usaddress'


def test_reconcile_addresses_three_sources_priority_and_fill():
    df = pd.DataFrame(
        {
            # Highest priority: missing on rows 1-2
            'address_parcel': ['12 OAK ST', None, None],
            # Second: agrees with parcel on row 0 and fills its missing city
            'address_full_x': ['12 OAK STREET BOSTON MA 02129', '9 ELM RD', None],
            # Third: only usable source on row 2
            'address_street_y': [None, None, 'Pine St'],
            'address_number_y': [None, None, '77'],
        }
    )
    state = reconcile_addresses(
        make_state(df),
        sources={
            'parcel': {'address_full': 'address_parcel'},
            'x': {'address_full': 'address_full_x'},
            'y': {
                'address_street': 'address_street_y',
                'address_number': 'address_number_y',
            },
        },
    )
    res = state.curated

    # Row 0: parcel base; agreeing source x fills city/state/zip. x also only
    # declares address_full, so the filled city is usaddress-derived too.
    assert res.loc[0, 'address'] == '12 Oak St, Boston, MA 02129'
    assert res.loc[0, 'address_source'] == 'reconciled+usaddress'

    # Row 1: falls through to x
    assert res.loc[1, 'address'] == '9 Elm Rd'
    assert res.loc[1, 'address_source'] == 'x'

    # Row 2: falls through to y
    assert res.loc[2, 'address'] == '77 Pine St'
    assert res.loc[2, 'address_source'] == 'y'


def test_reconcile_addresses_typo_agreement_not_flagged():
    # Streets differ exactly (typo) but pass the fuzzy check -> no conflict
    df = pd.DataFrame(
        {
            'address_parcel': ['270 KILKENY LOOP RD'],
            'address_street_overture': ['Kilkenny Loop Rd'],
            'address_number_overture': ['270'],
        }
    )
    state = reconcile_addresses(
        make_state(df),
        sources={
            'parcel': {'address_full': 'address_parcel'},
            'dwelling_overture': {
                'address_street': 'address_street_overture',
                'address_number': 'address_number_overture',
            },
        },
    )
    res = state.curated
    assert res.loc[0, 'address_source'] == 'reconciled'
    assert pd.isna(res.loc[0, 'address_conflict'])


def test_reconcile_addresses_notation_variants_reconcile():
    # HIGHWAY vs HWY pre-type notation + dangling directional -> agree
    df = pd.DataFrame(
        {
            'address_parcel': ['22846 HWY 94 S'],
            'address_street_overture': ['Highway 94'],
            'address_number_overture': ['22846'],
        }
    )
    state = reconcile_addresses(
        make_state(df),
        sources={
            'parcel': {'address_full': 'address_parcel'},
            'dwelling_overture': {
                'address_street': 'address_street_overture',
                'address_number': 'address_number_overture',
            },
        },
    )
    res = state.curated
    assert res.loc[0, 'address'] == '22846 Hwy 94 S'
    assert res.loc[0, 'address_source'] == 'reconciled'
    assert pd.isna(res.loc[0, 'address_conflict'])


def test_reconcile_addresses_street_mismatch_flagged():
    # Same house number but unrelated streets -> conflict summarized
    df = pd.DataFrame(
        {
            'address_parcel': ['5 MAIN ST'],
            'address_street_overture': ['Atlantic Ave'],
            'address_number_overture': ['5'],
        }
    )
    state = reconcile_addresses(
        make_state(df),
        sources={
            'parcel': {'address_full': 'address_parcel'},
            'dwelling_overture': {
                'address_street': 'address_street_overture',
                'address_number': 'address_number_overture',
            },
        },
    )
    res = state.curated
    assert res.loc[0, 'address'] == '5 Main St'
    assert res.loc[0, 'address_source'] == 'parcel'
    assert (
        res.loc[0, 'address_conflict']
        == 'parcel: 5 MAIN ST | dwelling_overture: 5 ATLANTIC AVE'
    )


def test_get_admin2_codes_from_iso_table():
    assert 'NC' in get_admin2_codes('US')
    assert 'MA' in get_admin2_codes('US')
    assert 'BY' in get_admin2_codes('DE')  # Bavaria
    assert 'NC' not in get_admin2_codes('DE')


def test_reconcile_addresses_component_source_city_zip_and_admin_state():
    # A component source contributing city/postal cascades into the output,
    # and the missing state is completed from the run's admin unit when the
    # recipe opts in via complete_from_admin
    # (the Emerald Isle marina case: parcel address empty, dwelling is base).
    df = pd.DataFrame(
        {
            'address_parcel': [None, None],
            'address_street_overture': ['DAVID Drive', 'Marina Dr'],
            'address_number_overture': ['502', '8618'],
            'city_overture': ['EMERALD ISLE', None],
            'postal_code_overture': ['28594', None],
        }
    )
    state = reconcile_addresses(
        make_state(df),  # admin_id US-MA-MI -> state MA
        sources={
            'parcel': {'address_full': 'address_parcel'},
            'dwelling_overture': {
                'address_street': 'address_street_overture',
                'address_number': 'address_number_overture',
                'city': 'city_overture',
                'postal_code': 'postal_code_overture',
            },
        },
        complete_from_admin={'state': 2},
    )
    res = state.curated
    assert res.loc[0, 'address'] == '502 David Dr, Emerald Isle, MA 28594'
    assert res.loc[0, 'address_source'] == 'dwelling_overture'
    # No city/zip -> no dangling state appended
    assert res.loc[1, 'address'] == '8618 Marina Dr'


def test_reconcile_addresses_complete_city_from_postal():
    # No source supplies a city (mirrors dwelling_overture's real-world gap
    # in MA), but a ZIP is present: complete_city_from_postal fills city
    # from the USPS-preferred name for that ZIP.
    df = pd.DataFrame(
        {
            'address_street_overture': ['Beacon Street'],
            'address_number_overture': ['123'],
            'postal_code_overture': ['02459'],
        }
    )
    state = reconcile_addresses(
        make_state(df),  # admin_id US-MA-MI -> state MA
        sources={
            'dwelling_overture': {
                'address_street': 'address_street_overture',
                'address_number': 'address_number_overture',
                'postal_code': 'postal_code_overture',
            },
        },
        complete_from_admin={'state': 2},
        complete_city_from_postal=True,
    )
    res = state.curated
    assert res.loc[0, 'address'] == '123 Beacon St, Newton Center, MA 02459'


def test_reconcile_addresses_complete_city_from_postal_default_off():
    # Same input as above, but the flag defaults to False: existing recipes
    # that don't opt in keep their current (city-less) output unchanged.
    df = pd.DataFrame(
        {
            'address_street_overture': ['Beacon Street'],
            'address_number_overture': ['123'],
            'postal_code_overture': ['02459'],
        }
    )
    state = reconcile_addresses(
        make_state(df),
        sources={
            'dwelling_overture': {
                'address_street': 'address_street_overture',
                'address_number': 'address_number_overture',
                'postal_code': 'postal_code_overture',
            },
        },
        complete_from_admin={'state': 2},
    )
    res = state.curated
    assert res.loc[0, 'address'] == '123 Beacon St, MA 02459'


def test_reconcile_addresses_complete_from_admin_non_us():
    # The admin completion is country-agnostic: a German admin id fills the
    # Bavarian level-2 code, validated against ISO 3166-2 for DE.
    df = pd.DataFrame(
        {
            'street_col': ['Hauptstrasse'],
            'number_col': ['12'],
            'city_col': ['MUNICH'],
        }
    )
    state = CurateState(
        recipe={},
        entity_recipe={},
        admin_id='DE-BY',
        verbose=False,
        timer=None,
        curated=df,
    )
    state = reconcile_addresses(
        state,
        sources={
            'cadastre': {
                'address_street': 'street_col',
                'address_number': 'number_col',
                'city': 'city_col',
            },
        },
        complete_from_admin={'state': 2},
    )
    res = state.curated
    # The DE template renders street-first and (per German convention) no
    # state segment; the admin-completion mechanism itself is exercised by
    # the US test above.
    assert res.loc[0, 'address'] == 'Hauptstrasse 12, Munich'
    assert res.loc[0, 'address_source'] == 'cadastre'


def test_reconcile_addresses_missing_columns_tolerated():
    df = pd.DataFrame({'address_parcel': ['1 MAIN ST']})
    state = reconcile_addresses(
        make_state(df),
        sources={
            'parcel': {'address_full': 'address_parcel'},
            'ghost': {'address_full': 'not_a_column'},
        },
    )
    res = state.curated
    assert res.loc[0, 'address'] == '1 Main St'
    assert res.loc[0, 'address_source'] == 'parcel'


def test_reconcile_addresses_no_sources_is_noop():
    df = pd.DataFrame({'other': [1]})
    state = reconcile_addresses(
        make_state(df),
        sources={'parcel': {'address_full': 'address_parcel'}},
    )
    assert 'address' not in state.curated.columns


def test_reconcile_addresses_unknown_role_raises():
    df = pd.DataFrame({'address_parcel': ['1 MAIN ST']})
    with pytest.raises(ValueError, match='unknown role'):
        reconcile_addresses(
            make_state(df),
            sources={'parcel': {'street_col': 'address_parcel'}},
        )
