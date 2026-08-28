"""Tests for intuitive two-letter coverage and the code-length policy.

The substantive claim under test is that group size does not predict
whether two letters suffice. Shared honorifics and crowded romanizations
do, which is why the policy is a measured coverage test rather than a
count cutoff.
"""

import pandas as pd
import pytest

from openplaces.io.admin_codes import (
    intuitive_codes,
    intuitive_coverage,
    recommend_code_length,
    syllable_onsets,
)

SPINE_ADMIN2 = (
    'src/openplaces/recipes/_all/admin/spine/2026/admin-spine-2026_admin2.csv'
)
POLICY = (
    'src/openplaces/recipes/_all/admin/openplaces/2026/'
    'admin-openplaces-2026_code-length-policy.csv'
)


@pytest.fixture(scope='module')
def spine2():
    spine = pd.read_csv(
        SPINE_ADMIN2, dtype=str, keep_default_na=False, encoding='utf-8'
    )
    spine['admin1_id'] = spine['admin2_id'].str.slice(0, 2)
    return spine


class TestSyllableOnsets:
    @pytest.mark.parametrize(
        ('word', 'expected'),
        [
            ('SOMERVILLE', ['S', 'M', 'V', 'L']),
            ('ANTIOQUIA', ['A', 'T', 'Q']),
            ('OSLO', ['O', 'L']),
            # Y glides to a consonant before a vowel, which is what makes
            # Boyaca give BY rather than BC.
            ('BOYACA', ['B', 'Y', 'C']),
        ],
    )
    def test_onsets(self, word, expected):
        assert syllable_onsets(word) == expected


class TestIntuitiveCodes:
    def test_four_mappings_are_offered(self):
        # first-two SO, syllables SM, ends SE.
        assert intuitive_codes('Somerville', admin1_id='US') == {'SO', 'SM', 'SE'}

    def test_acronym_from_two_words(self):
        assert 'NE' in intuitive_codes('North East', admin1_id='US')

    def test_a_saint_offers_both_readings(self):
        # Naming a place for a saint is deliberate and speakers keep the
        # particle (San Francisco is SF), so both readings are offered:
        # George alone (GE) and the full name (SA, SG). A mostly-Saint
        # sibling group can still drop the particle.
        codes = intuitive_codes('Saint George', admin1_id='BB')
        assert 'GE' in codes
        assert 'SG' in codes

    def test_articles_are_stripped(self):
        assert 'BU' in intuitive_codes('Al Butnan', admin1_id='LY')

    def test_only_two_letter_alphabetic_codes_are_returned(self):
        codes = intuitive_codes('Region 5', admin1_id='US')
        assert all(len(c) == 2 and c.isalpha() for c in codes)


class TestIntuitiveCoverage:
    def test_distinct_names_are_fully_covered(self):
        assert intuitive_coverage(['Alabama', 'Alaska'], admin1_id='US') == (2, 2)

    def test_coverage_is_order_independent(self):
        names = ['Saint Andrew', 'Saint George', 'Saint James', 'Christ Church']
        assert intuitive_coverage(names, admin1_id='BB') == intuitive_coverage(
            list(reversed(names)), admin1_id='BB'
        )

    def test_identical_names_cannot_all_be_covered(self):
        # Four identical names share one candidate set, so at most as many
        # can be placed as that set has members.
        names = ['Springfield'] * 4
        matched, total = intuitive_coverage(names, admin1_id='US')
        assert total == 4
        assert matched == len(intuitive_codes('Springfield', admin1_id='US'))
        assert matched < total

    def test_empty_input(self):
        assert intuitive_coverage([], admin1_id='US') == (0, 0)

    def test_shared_honorific_no_longer_defeats_two_letters(self, spine2):
        # Barbados is eleven parishes, nearly all 'Saint X'. Stripping the
        # honorific is what makes the group coverable.
        names = list(spine2.loc[spine2.admin1_id == 'BB', 'name'])
        matched, total = intuitive_coverage(names, admin1_id='BB')
        assert matched == total


class TestRecommendCodeLength:
    def test_published_convention_wins(self):
        assert recommend_code_length(['Antioquia'], 'CO', iso_length=3) == (
            3,
            'published convention',
        )

    def test_falls_back_to_measured_coverage(self):
        length, reason = recommend_code_length(
            ['Alabama', 'Alaska'], 'US', iso_length=None
        )
        assert length == 2
        assert 'intuitive coverage' in reason

    def test_poor_coverage_recommends_three(self):
        length, _ = recommend_code_length(['Springfield'] * 5, 'US')
        assert length == 3

    def test_iso_length_of_one_is_not_adopted(self):
        # A one-character convention cannot satisfy the id format rule, so
        # the group is measured instead of inheriting it.
        length, reason = recommend_code_length(['Alpha', 'Beta'], 'US', iso_length=1)
        assert length in (2, 3)
        assert reason != 'published convention'


class TestPolicyTable:
    def test_policy_table_covers_every_country_in_the_spine(self, spine2):
        policy = pd.read_csv(POLICY, dtype=str, keep_default_na=False)
        with_names = spine2[spine2['name'].str.strip() != '']
        assert set(policy['admin1_id']) == set(with_names['admin1_id'])

    def test_recommended_lengths_are_two_or_three(self):
        policy = pd.read_csv(POLICY)
        assert set(policy['recommended_length']) <= {2, 3}

    def test_size_does_not_predict_coverage(self):
        # The claim behind using a measured test rather than a count
        # cutoff: the two groups overlap in size, so no threshold
        # separates them. Currently the smallest country that cannot be
        # covered has 8 subdivisions while the largest that can has 81.
        # If this ever fails, a size threshold has become defensible and
        # the policy should be revisited.
        policy = pd.read_csv(POLICY)
        covered = policy[policy.intuitive_2char_share == 1.0]['n_units']
        not_covered = policy[policy.intuitive_2char_share < 1.0]['n_units']
        assert len(not_covered) > 0
        assert not_covered.min() < covered.max()
