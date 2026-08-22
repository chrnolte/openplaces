"""Tests for generating and assigning admin id code segments.

The France cases are the substantive ones. French departement names exercise
every rule in the generator, and eight of them have abbreviations that French
administrative usage already established. Those eight are held-out ground
truth: the generator is never given them, so reproducing them shows the rules
track how speakers actually abbreviate rather than fitting a table.
"""

import random

import pandas as pd
import pytest

from openplaces.io.admin_codes import (
    Candidate,
    assign_codes,
    generate_candidates,
    get_language_pack,
    is_valid_code,
    tokenize,
)

# Abbreviations in established French administrative use. Held-out
# ground truth, not inputs to the generator.
KNOWN_FRENCH_ABBREVIATIONS = {
    'Bouches-du-Rhône': 'BDR',
    'Pas-de-Calais': 'PDC',
    'Alpes-de-Haute-Provence': 'AHP',
    'Seine-Saint-Denis': 'SSD',
    'Val-de-Marne': 'VDM',
    "Val-d'Oise": 'VDO',
    'Hauts-de-Seine': 'HDS',
    'Puy-de-Dôme': 'PDD',
}

SPINE_ADMIN3 = (
    'src/openplaces/recipes/_all/admin/spine/2026/admin-spine-2026_admin3.csv'
)


@pytest.fixture(scope='module')
def french_departements():
    """Return the real departement names from the committed admin spine."""
    spine = pd.read_csv(
        SPINE_ADMIN3, dtype=str, keep_default_na=False, encoding='utf-8'
    )
    names = spine.loc[spine['admin3_id'].str.startswith('FR-'), 'name']
    return sorted(names)


@pytest.fixture(scope='module')
def french_assignment(french_departements):
    """Assign a three-character code to every French departement."""
    candidates = {
        name: generate_candidates(name, admin1_id='FR', lengths=(3,))
        for name in french_departements
    }
    return candidates, assign_codes(candidates)


class TestCodeValidity:
    """The format rule: a letter, then one or two alphanumerics."""

    @pytest.mark.parametrize('code', ['HSA', 'BDR', 'MA', 'X01', 'A1'])
    def test_accepts_well_formed_codes(self, code):
        assert is_valid_code(code)

    @pytest.mark.parametrize(
        'code',
        [
            'BU3SAM',  # present in the current admin4 spine
            'S137',  # present in the current admin4 spine
            '3DR',  # leading digit
            'A',  # too short
            'ab',  # lower case
            'US-MA',  # separator
            '',
        ],
    )
    def test_rejects_malformed_codes(self, code):
        assert not is_valid_code(code)

    def test_every_generated_french_code_is_valid(self, french_assignment):
        _, assignment = french_assignment
        assert all(is_valid_code(code) for code, _ in assignment.values())


class TestAdministrativeTypeWords:
    """A type word says what a unit is, not which one it is."""

    @pytest.mark.parametrize(
        ('name', 'expected'),
        [
            ('Haines Borough', 'HA'),
            ('Dillingham Census Area', 'DI'),
            ('Middlesex County', 'MI'),
            ('Orleans Parish', 'OR'),
        ],
    )
    def test_type_words_do_not_reach_the_code(self, name, expected):
        # Leaving them in gave HB and DC, which is how 56% of US county
        # codes came out unrecognizable before this was fixed.
        assert generate_candidates(name, admin1_id='US', lengths=(2,))[0].code == (
            expected
        )

    def test_a_distinctive_word_survives_alongside_a_type_word(self):
        codes = [
            c.code
            for c in generate_candidates(
                'Fairbanks North Star Borough', admin1_id='US', lengths=(2,)
            )
        ]
        assert codes[0] == 'FN'

    def test_a_name_reduced_to_one_character_does_not_crash(self):
        # Stripping type words can leave a single character, and the
        # first-and-last rule reads the second one unconditionally.
        assert generate_candidates('A County', admin1_id='US', lengths=(2,)) == []

    def test_a_name_that_is_only_a_type_word_still_yields_something(self):
        assert generate_candidates('County', admin1_id='US', lengths=(2,))


class TestTypeWordVocabulary:
    """Type words are data, and only reachable through a language."""

    def test_universal_type_words_apply_to_every_language(self):
        for language in ('und', 'ja', 'tr', 'ru'):
            assert get_language_pack(language).is_type_word('county')

    def test_language_specific_words_do_not_leak(self):
        # 'shi' is a Japanese suffix. Applied globally it would eat real
        # names, so it must not reach other languages.
        assert get_language_pack('ja').is_type_word('shi')
        assert not get_language_pack('fr').is_type_word('shi')

    @pytest.mark.parametrize(
        ('name', 'country', 'expected'),
        [
            ('Chiryu-shi', 'JP', 'CHI'),
            ('Kas ilcesi', 'TR', 'KAS'),
            ('Giaginskiy rayon', 'RU', 'GIA'),
            ('Haines Borough', 'US', 'HAI'),
        ],
    )
    def test_each_language_strips_its_own_type_words(self, name, country, expected):
        codes = generate_candidates(name, admin1_id=country, lengths=(3,))
        assert codes[0].code == expected

    def test_every_country_with_a_recipe_has_a_language(self):
        # A language pack is only reachable through the country-language
        # table, so a country missing there silently falls back to 'und'
        # and loses its own type words. That regression is invisible until
        # a code comes out wrong, so it is pinned here.
        for country in ('BR', 'PH', 'JP', 'TR', 'RU', 'CO', 'US'):
            pack = get_language_pack(admin1_id=country)
            assert pack.language != 'und', (
                f'{country} has a recipe but no language assigned'
            )


class TestTokenize:
    """Prepositions stay in an initialism; conjunctions do not."""

    def test_preposition_is_kept_as_a_token_but_not_significant(self):
        pack = get_language_pack('fr')
        tokens, significant = tokenize('Bouches-du-Rhône', pack)
        assert tokens == ['BOUCHES', 'DU', 'RHONE']
        assert significant == ['BOUCHES', 'RHONE']

    def test_conjunction_is_kept_as_a_token_but_not_significant(self):
        pack = get_language_pack('fr')
        tokens, significant = tokenize('Eure-et-Loir', pack)
        assert tokens == ['EURE', 'ET', 'LOIR']
        assert significant == ['EURE', 'LOIR']

    def test_diacritics_are_folded(self):
        pack = get_language_pack('fr')
        _, significant = tokenize('Ardèche', pack)
        assert significant == ['ARDECHE']

    def test_unknown_country_falls_back_to_undetermined(self):
        assert get_language_pack(admin1_id='ZZ').language == 'und'


class TestFrenchGroundTruth:
    """Held-out validation against abbreviations speakers already use."""

    @pytest.mark.parametrize(
        ('name', 'expected'), sorted(KNOWN_FRENCH_ABBREVIATIONS.items())
    )
    def test_established_abbreviation_is_the_first_choice(self, name, expected):
        candidates = generate_candidates(name, admin1_id='FR', lengths=(3,))
        assert candidates[0].code == expected

    def test_established_abbreviations_survive_assignment(self, french_assignment):
        _, assignment = french_assignment
        for name, expected in KNOWN_FRENCH_ABBREVIATIONS.items():
            assert assignment[name][0] == expected

    def test_a_preposition_compound_keeps_the_preposition(self):
        # Bouches-du-Rhone gives BDR, never BOR: French initialisms keep
        # the preposition binding one compound name.
        codes = [
            c.code for c in generate_candidates('Bouches-du-Rhône', admin1_id='FR')
        ]
        assert codes.index('BDR') < codes.index('BOR')

    def test_a_conjunction_compound_drops_the_conjunction(self):
        # Eure-et-Loir gives EUL, never EEL.
        codes = [c.code for c in generate_candidates('Eure-et-Loir', admin1_id='FR')]
        assert 'EUL' in codes
        assert 'EEL' not in codes

    def test_a_leading_qualifier_pairs_with_its_unqualified_unit(self):
        # Savoie gives SAV, so Haute-Savoie should read as its qualified
        # form.
        assert generate_candidates('Savoie', admin1_id='FR')[0].code == 'SAV'
        assert generate_candidates('Haute-Savoie', admin1_id='FR')[0].code == 'HSA'

    def test_a_buried_qualifier_is_not_treated_as_leading(self):
        # Alpes-de-Haute-Provence gives AHP, not HAL: the qualifier is
        # part of the name proper rather than a modifier on it.
        assert (
            generate_candidates('Alpes-de-Haute-Provence', admin1_id='FR')[0].code
            == 'AHP'
        )

    def test_a_qualifier_split_by_a_preposition_is_not_treated_as_leading(self):
        # Hauts-de-Seine gives HDS, not HSE.
        assert generate_candidates('Hauts-de-Seine', admin1_id='FR')[0].code == 'HDS'


class TestAssignment:
    """Assignment is unique, order-independent, and weight-sensitive."""

    def test_every_departement_gets_a_distinct_code(self, french_assignment):
        _, assignment = french_assignment
        codes = [code for code, _ in assignment.values()]
        assert len(assignment) == 96
        assert len(set(codes)) == 96

    def test_result_does_not_depend_on_input_order(self, french_departements):
        shuffled = list(french_departements)
        random.Random(0).shuffle(shuffled)

        def solve(names):
            candidates = {
                name: generate_candidates(name, admin1_id='FR', lengths=(3,))
                for name in names
            }
            return {k: v[0] for k, v in assign_codes(candidates).items()}

        assert solve(french_departements) == solve(shuffled)

    def test_weight_decides_a_contested_code(self):
        # Two units whose first choice is the same code. The heavier
        # unit should win it; this is the mechanism population weighting
        # uses.
        contested = {
            'alpha': [Candidate('AB', 'name', 2), Candidate('AC', 'name', 2)],
            'beta': [Candidate('AB', 'name', 2), Candidate('AD', 'name', 2)],
        }
        heavy_alpha = assign_codes(contested, weights={'alpha': 100.0, 'beta': 1.0})
        heavy_beta = assign_codes(contested, weights={'alpha': 1.0, 'beta': 100.0})
        assert heavy_alpha['alpha'][0] == 'AB'
        assert heavy_beta['beta'][0] == 'AB'

    def test_length_penalty_prefers_the_shorter_code(self):
        candidates = {'unit': [Candidate('ABC', 'x', 3), Candidate('AB', 'y', 2)]}
        assert assign_codes(candidates, length_penalty=10.0)['unit'][0] == 'AB'
        assert assign_codes(candidates, length_penalty=0.0)['unit'][0] == 'ABC'

    def test_reserved_codes_are_not_reused(self):
        candidates = {'unit': [Candidate('AB', 'name', 2), Candidate('AC', 'name', 2)]}
        assert assign_codes(candidates, reserved={'AB'})['unit'][0] == 'AC'

    def test_more_units_than_candidates_still_resolves_uniquely(self):
        # Every unit proposes the same single code, so all but one must
        # fall through to the sequential pool without colliding.
        candidates = {f'unit{i}': [Candidate('AB', 'name', 2)] for i in range(10)}
        assignment = assign_codes(candidates)
        codes = [code for code, _ in assignment.values()]
        assert len(set(codes)) == 10
        assert all(is_valid_code(code) for code in codes)

    def test_empty_group_returns_empty(self):
        assert assign_codes({}) == {}
