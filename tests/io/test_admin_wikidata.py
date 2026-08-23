"""Tests for matching spine units to Wikidata items.

No network: the harvest is a fixture, so these pin the matching rules
rather than Wikidata's current contents. The rule that matters is that an
ambiguous match is reported, never resolved by picking one.
"""

import pandas as pd
import pytest

from openplaces.io.admin_wikidata import (
    FUZZY_CUTOFF,
    MATCH_AMBIGUOUS,
    MATCH_FUZZY,
    MATCH_MISSING,
    MATCH_UNIQUE,
    children_query,
    index_harvest,
    match_summary,
    match_units,
)

PARENT_ISO = {'CO-AN': 'CO-AN', 'CO-BY': 'CO-BY'}


@pytest.fixture
def harvest():
    return pd.DataFrame(
        [
            # A clean municipality.
            {
                'item': 'http://www.wikidata.org/entity/Q1',
                'itemLabel': 'Abejorral',
                'native': '',
                'parentIso': 'CO-AN',
            },
            # Same name under a different parent: must not be matched across.
            {
                'item': 'http://www.wikidata.org/entity/Q2',
                'itemLabel': 'Abejorral',
                'native': '',
                'parentIso': 'CO-BY',
            },
            # Two items sharing a name under one parent: ambiguous.
            {
                'item': 'http://www.wikidata.org/entity/Q3',
                'itemLabel': 'Otanche',
                'native': '',
                'parentIso': 'CO-BY',
            },
            {
                'item': 'http://www.wikidata.org/entity/Q4',
                'itemLabel': 'Otanche',
                'native': '',
                'parentIso': 'CO-BY',
            },
            # Diacritics and an administrative type word.
            {
                'item': 'http://www.wikidata.org/entity/Q5',
                'itemLabel': 'Chocó Municipality',
                'native': '',
                'parentIso': 'CO-AN',
            },
        ]
    )


def units(rows):
    return pd.DataFrame(rows, columns=['admin_id', 'name', 'parent_admin_id'])


class TestChildrenQuery:
    def test_targets_the_containment_property(self):
        assert 'wdt:P131' in children_query('CO-')

    def test_scopes_to_the_requested_prefix(self):
        assert '"CO-"' in children_query('CO-')

    def test_does_not_filter_by_type(self):
        # Filtering by a country's municipality class scored worse (89.7%
        # against 91.7%) and costs a manual lookup per country.
        assert 'P279' not in children_query('CO-')

    def test_limit_is_settable(self):
        assert 'LIMIT 5' in children_query('CO-', limit=5)


class TestIndexHarvest:
    def test_keys_on_parent_and_normalized_name(self, harvest):
        index = index_harvest(harvest)
        assert ('CO-AN', 'ABEJORRAL') in index
        assert ('CO-BY', 'OTANCHE') in index

    def test_collects_every_item_sharing_a_key(self, harvest):
        assert len(index_harvest(harvest)[('CO-BY', 'OTANCHE')]) == 2


class TestMatchUnits:
    def test_unique_match_yields_a_qid(self, harvest):
        result = match_units(
            units([('CO-AN-ABE', 'Abejorral', 'CO-AN')]), harvest, PARENT_ISO
        )
        assert result.iloc[0]['status'] == MATCH_UNIQUE
        assert result.iloc[0]['wikidata_id'] == 'Q1'

    def test_ambiguous_match_is_reported_not_resolved(self, harvest):
        result = match_units(
            units([('CO-BY-OTA', 'Otanche', 'CO-BY')]), harvest, PARENT_ISO
        )
        row = result.iloc[0]
        assert row['status'] == MATCH_AMBIGUOUS
        assert row['n_candidates'] == 2
        # Picking one silently would sometimes be wrong, so nothing is picked.
        assert row['wikidata_id'] == ''

    def test_missing_match_is_reported(self, harvest):
        result = match_units(
            units([('CO-AN-ZZZ', 'Nowhere', 'CO-AN')]), harvest, PARENT_ISO
        )
        assert result.iloc[0]['status'] == MATCH_MISSING
        assert result.iloc[0]['n_candidates'] == 0

    def test_matching_stays_inside_the_parent(self, harvest):
        # Abejorral exists under both parents; each must match its own.
        result = match_units(
            units(
                [
                    ('CO-AN-ABE', 'Abejorral', 'CO-AN'),
                    ('CO-BY-ABE', 'Abejorral', 'CO-BY'),
                ]
            ),
            harvest,
            PARENT_ISO,
        )
        assert list(result['wikidata_id']) == ['Q1', 'Q2']

    def test_type_words_and_diacritics_are_normalized(self, harvest):
        result = match_units(
            units([('CO-AN-CHO', 'Choco', 'CO-AN')]), harvest, PARENT_ISO
        )
        assert result.iloc[0]['wikidata_id'] == 'Q5'

    def test_unknown_parent_does_not_raise(self, harvest):
        result = match_units(
            units([('XX-YY-ZZ', 'Abejorral', 'XX-YY')]), harvest, PARENT_ISO
        )
        assert result.iloc[0]['status'] == MATCH_MISSING


class TestMatchSummary:
    def test_shares_sum_to_one(self, harvest):
        result = match_units(
            units(
                [
                    ('CO-AN-ABE', 'Abejorral', 'CO-AN'),
                    ('CO-BY-OTA', 'Otanche', 'CO-BY'),
                    ('CO-AN-ZZZ', 'Nowhere', 'CO-AN'),
                ]
            ),
            harvest,
            PARENT_ISO,
        )
        assert match_summary(result).sum() == pytest.approx(1.0)

    def test_empty_input(self):
        assert match_summary(pd.DataFrame()).empty


class TestFuzzyMatching:
    """A near miss is a suggestion for review, never an exact match."""

    def test_off_by_default(self, harvest):
        result = match_units(
            units([('CO-AN-ABE', 'Abejorral ', 'CO-AN')]), harvest, PARENT_ISO
        )
        assert result.iloc[0]['status'] in (MATCH_UNIQUE, MATCH_MISSING)

    def test_recovers_a_spelling_variant(self, harvest):
        result = match_units(
            units([('CO-AN-ABE', 'Abejoral', 'CO-AN')]),
            harvest,
            PARENT_ISO,
            fuzzy_cutoff=FUZZY_CUTOFF,
        )
        row = result.iloc[0]
        assert row['status'] == MATCH_FUZZY
        assert row['wikidata_id'] == 'Q1'

    def test_never_reaches_outside_the_parent(self, harvest):
        # Otanche exists only under CO-BY. A unit under CO-AN must not
        # borrow it however close the spelling.
        result = match_units(
            units([('CO-AN-OTA', 'Otanch', 'CO-AN')]),
            harvest,
            PARENT_ISO,
            fuzzy_cutoff=FUZZY_CUTOFF,
        )
        assert result.iloc[0]['status'] == MATCH_MISSING

    def test_a_fuzzy_hit_on_an_ambiguous_name_is_not_taken(self, harvest):
        # Otanche has two items under CO-BY; a near miss must not pick one.
        result = match_units(
            units([('CO-BY-OTA', 'Otanch', 'CO-BY')]),
            harvest,
            PARENT_ISO,
            fuzzy_cutoff=FUZZY_CUTOFF,
        )
        assert result.iloc[0]['wikidata_id'] == ''

    def test_a_distant_name_is_not_matched(self, harvest):
        result = match_units(
            units([('CO-AN-XXX', 'Zzzzzzz', 'CO-AN')]),
            harvest,
            PARENT_ISO,
            fuzzy_cutoff=FUZZY_CUTOFF,
        )
        assert result.iloc[0]['status'] == MATCH_MISSING
