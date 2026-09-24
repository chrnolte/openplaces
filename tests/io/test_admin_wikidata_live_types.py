"""An item's types are its type statements that have not ended.

`wdt:P31` keeps a statement whose own qualifier says the membership
ended, so a municipality abolished in 1812 still reads as a current one.
P576 does not catch it: what was recorded is that the thing stopped
being a municipality, not that it stopped existing.

These tests read the generated SPARQL rather than calling the endpoint,
so they pin the rule without a network round trip. The measurement
behind it is in the `LIVE_TYPES` docstring.
"""

import pytest

from openplaces.io import admin_wikidata as wd

QUERIES = {
    'country_children': lambda: wd.country_children_query('NL'),
    'parent_children': lambda: wd.parent_children_query(['Q55']),
    'item_details': lambda: wd.item_details_query(['Q9934'], 'Q55'),
    'parent_children_classes': lambda: wd.parent_children_classes_query(['Q55']),
}


@pytest.mark.parametrize('name', sorted(QUERIES))
def test_types_come_from_statements_not_the_truthy_predicate(name):
    sparql = QUERIES[name]()
    assert 'ps:P31 ?cls' in sparql, f'{name} does not read type statements'
    assert 'OPTIONAL { ?item wdt:P31 ?cls }' not in sparql, (
        f'{name} still reads types through the truthy predicate, which '
        'keeps memberships that have ended'
    )


@pytest.mark.parametrize('name', sorted(QUERIES))
def test_an_ended_type_statement_is_excluded(name):
    sparql = QUERIES[name]()
    assert 'pq:P582' in sparql, f'{name} does not exclude ended type statements'


@pytest.mark.parametrize('name', sorted(QUERIES))
def test_a_deprecated_type_statement_is_excluded(name):
    """Reading statements rather than truthy values re-admits bad ones."""
    sparql = QUERIES[name]()
    assert 'wikibase:DeprecatedRank' in sparql, (
        f'{name} reads type statements without excluding deprecated ones'
    )


def test_a_dissolved_item_is_still_dropped_outright():
    """The P576 filter stays: this rule adds to it rather than replacing it."""
    sparql = wd.country_children_query('NL')
    assert 'wdt:P576' in sparql


def test_a_replaced_unit_is_not_dropped_for_being_replaced():
    """The P1366 filter is gone, and must not come back by reflex.

    It read "replaced by, and no ISO code" as "superseded", which most
    units below level 2 satisfy without being superseded at all: it cost
    Sweden twelve current municipalities including Stockholm. What it was
    written for, Armenia's marzer against the Soviet raions they
    replaced, returns the same 11 provinces without it, because a raion's
    type statement has ended and `LIVE_TYPES` already excludes it.
    """
    for build in QUERIES.values():
        sparql = build()
        assert 'wdt:P1366' not in sparql, (
            'the replaced-by filter is back; it drops current units whose '
            'only fault is having no ISO code'
        )


def test_the_type_variable_is_still_named_cls():
    """`select_units` reads the `classes` column this binds; renaming it
    would silently empty every item's type list."""
    assert '?cls' in wd.parent_children_classes_query(['Q55'])
    assert 'AS ?classes' in wd.parent_children_classes_query(['Q55'])
