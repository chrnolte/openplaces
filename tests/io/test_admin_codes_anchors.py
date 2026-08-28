"""Tests for prior-code anchoring and the derive_codes entry point.

The US state codes are the motivating case. AK, AZ and CT are conventions
rather than abbreviations, so the rules alone reach only 30 of 51; anchoring
on the country's own published codes is what closes the gap.
"""

import pandas as pd
import pytest

from openplaces.io.admin_codes import (
    derive_codes,
    get_anchor_codes,
    get_code_length_convention,
    is_valid_code,
    load_code_conventions,
    load_prior_codes,
    normalize_name,
)

SPINE_ADMIN2 = (
    'src/openplaces/recipes/_all/admin/spine/2026/admin-spine-2026_admin2.csv'
)


@pytest.fixture(scope='module')
def spine2():
    return pd.read_csv(SPINE_ADMIN2, dtype=str, keep_default_na=False, encoding='utf-8')


class TestReferenceTables:
    def test_prior_codes_table_loads_and_is_populated(self):
        table = load_prior_codes()
        assert len(table) > 5000
        assert table['country_code'].nunique() > 190
        assert set(table.columns) >= {
            'country_code',
            'code',
            'subdivision_code',
            'name',
            'name_native',
            'parent_code',
            'code_kind',
            'code_length',
        }

    def test_code_kinds_are_classified(self):
        kinds = set(load_prior_codes()['code_kind'])
        assert kinds <= {'alpha', 'numeric', 'mixed'}
        assert 'alpha' in kinds and 'numeric' in kinds

    def test_conventions_table_covers_the_same_countries(self):
        codes = load_prior_codes()
        conventions = load_code_conventions()
        assert set(conventions['country_code']) == set(codes['country_code'])


class TestNormalizeName:
    @pytest.mark.parametrize(
        ('raw', 'expected'),
        [
            ('Stockholm County', 'STOCKHOLM'),
            ('Ardèche', 'ARDECHE'),
            ('Baja California Sur', 'BAJA CALIFORNIA SUR'),
        ],
    )
    def test_strips_type_words_and_diacritics(self, raw, expected):
        assert normalize_name(raw) == expected

    def test_a_name_of_only_type_words_survives(self):
        # Dropping every word would make the name unmatchable, so the
        # folded original is kept instead of an empty string.
        assert normalize_name('County') == 'COUNTY'


class TestAnchors:
    @pytest.mark.parametrize(
        ('name', 'code'),
        [('Alaska', 'AK'), ('Arizona', 'AZ'), ('Connecticut', 'CT')],
    )
    def test_us_conventions_no_rule_would_recover(self, name, code):
        assert get_anchor_codes('US')[normalize_name(name)] == code

    def test_numeric_codes_are_excluded(self):
        assert all(v.isalpha() for v in get_anchor_codes('US').values())

    def test_unknown_country_yields_no_anchors(self):
        assert get_anchor_codes('ZZ') == {}

    def test_length_convention_is_national_not_universal(self):
        # 25 countries use one character, 96 use two and 30 use three,
        # so a single global width would be wrong for a third of the
        # world. The reviewed policy outranks the published scheme where
        # it has an entry: Colombia's ISO codes are three characters,
        # but two cover every department, so the policy says two.
        assert get_code_length_convention('US') == 2
        assert get_code_length_convention('CO') == 2
        assert get_code_length_convention('RU') == 3
        assert get_code_length_convention('ZZ') is None


class TestDeriveCodes:
    def test_anchoring_recovers_every_us_state_code(self, spine2):
        sub = spine2[spine2['admin2_id'].str.startswith('US-')]
        truth = dict(zip(sub['name'], sub['admin2_id'].str.split('-').str[-1]))
        result = derive_codes(list(sub['name']), admin1_id='US', lengths=(2,))
        assert len(result) == 51
        assert all(result[n][0] == truth[n] for n in truth)

    def test_rules_alone_do_worse_than_anchoring(self, spine2):
        # The gap is the whole reason anchoring exists. Asserted as a
        # strict inequality rather than a fixed number so the test
        # tracks the claim, not a snapshot.
        sub = spine2[spine2['admin2_id'].str.startswith('US-')]
        truth = dict(zip(sub['name'], sub['admin2_id'].str.split('-').str[-1]))
        names = list(sub['name'])
        anchored = derive_codes(names, admin1_id='US', lengths=(2,))
        unanchored = derive_codes(
            names, admin1_id='US', lengths=(2,), use_anchors=False
        )
        n_anchored = sum(anchored[n][0] == truth[n] for n in truth)
        n_unanchored = sum(unanchored[n][0] == truth[n] for n in truth)
        assert n_anchored == 51
        assert n_unanchored < n_anchored

    def test_default_length_follows_the_country_convention(self, spine2):
        # Mexico publishes three-letter state codes and has no reviewed
        # overrides, so the convention decides the width; the reviewed
        # length policy sets it to two because two characters cover
        # every state.
        sub = spine2[spine2['admin2_id'].str.startswith('MX-')]
        result = derive_codes(list(sub['name']), admin1_id='MX')
        assert all(len(code) == 2 for code, _ in result.values())

    def test_a_reviewed_override_outranks_the_convention(self):
        # Colombia publishes three-letter codes, but every department
        # carries a reviewed two-letter override, and a decision a person
        # made about these specific units beats a length inferred from a
        # published scheme. Antioquia is AN, not ANT.
        result = derive_codes(['Antioquia', 'Amazonas'], admin1_id='CO')
        assert result['Antioquia'][0] == 'AN'
        assert all(len(code) == 2 for code, _ in result.values())

    def test_the_convention_still_holds_where_overrides_are_partial(self):
        # The override width wins only when it covers the whole sibling
        # group. A group where it does not must not end up split across
        # two widths, so it falls back to the convention. Colombia's
        # municipalities carry no overrides at all, and the convention
        # for Colombia is the reviewed policy's two characters.
        result = derive_codes(['Abejorral', 'Abriaquí', 'Amalfi'], admin1_id='CO')
        assert all(len(code) == 2 for code, _ in result.values())

    def test_codes_are_unique_and_well_formed(self, spine2):
        sub = spine2[spine2['admin2_id'].str.startswith('CO-')]
        result = derive_codes(list(sub['name']), admin1_id='CO')
        codes = [code for code, _ in result.values()]
        assert len(set(codes)) == len(codes)
        assert all(is_valid_code(code) for code in codes)

    def test_anchor_rule_is_reported_as_provenance(self):
        assert derive_codes(['Alaska'], admin1_id='US', lengths=(2,))['Alaska'] == (
            'AK',
            'anchor',
        )

    def test_reserved_codes_are_respected(self):
        result = derive_codes(['Alaska'], admin1_id='US', lengths=(2,), reserved={'AK'})
        assert result['Alaska'][0] != 'AK'

    def test_duplicate_names_are_collapsed(self):
        assert len(derive_codes(['Alaska', 'Alaska'], admin1_id='US')) == 1

    def test_no_country_still_produces_valid_codes(self):
        result = derive_codes(['Northern Region', 'Southern Region'])
        assert len(result) == 2
        assert all(is_valid_code(code) for code, _ in result.values())
