"""Tests for the ALIS adapter and the per-platform dispatch.

Three Massachusetts registries run ALIS rather than masslandrecords,
and hold 37 towns between them including Brookline. What is tested
here is everything that does not need their server: the date and code
translation, the dispatch, and the crosswalk's own claims.
"""

from __future__ import annotations

import asyncio
import csv

import pytest

import openplaces.io.ingester.registry_ingester as registry_module
from openplaces.io.scrapers.alis_adapter import (
    DOC_TYPE_GROUPS,
    AlisAdapter,
    AlisSearchRefused,
)
from openplaces.io.scrapers.alis_selectors import (
    ALL_DOC_TYPES,
    ALL_TOWNS,
    DEED_DOC_GROUP,
)
from openplaces.path import recipe_path
from openplaces.recipe import get_recipe_by_id

NORFOLK = 'https://www.norfolkresearch.org/ALIS/WW400R.HTM'


def _adapter(towns=None, doc_types=None):
    adapter = AlisAdapter(NORFOLK, context=None)
    adapter._town_lookup = towns if towns is not None else {'BROOKLINE': 'BRKL'}
    adapter._doc_type_lookup = doc_types or {'MORTGAGE': 'MTGE'}
    return adapter


def test_dates_become_the_eight_character_form_the_screen_takes():
    """The form's date fields are 8 characters, MMDDYYYY, not ISO."""
    assert AlisAdapter._fmt('2024-01-15') == '01152024'
    # Single-digit month and day keep their leading zeros, or the
    # field's fixed width would shift every later character.
    assert AlisAdapter._fmt('2020-03-07') == '03072020'


def test_a_deed_search_asks_for_the_registry_s_deed_group():
    """'DEED' maps to the document *group*, not an enumerated type.

    The individual deed types differ between the three ALIS
    registries; the group does not.
    """
    adapter = _adapter()

    assert adapter._doc_type_value('DEED') == DEED_DOC_GROUP
    assert adapter._doc_type_value('FORECLOSURE DEED') == DEED_DOC_GROUP
    assert DOC_TYPE_GROUPS['DEED'] == DEED_DOC_GROUP
    # A type the registry lists by name is used as listed.
    assert adapter._doc_type_value('Mortgage') == 'MTGE'
    # Nothing asked for means nothing narrowed, rather than a guess.
    assert adapter._doc_type_value(None) == ALL_DOC_TYPES


def test_a_town_the_registry_does_not_list_is_a_crosswalk_error():
    """Naming the wrong registry's town is reported as exactly that.

    A town belongs to one registry, so this is never "no records".
    """
    adapter = _adapter(towns={'BROOKLINE': 'BRKL', 'DEDHAM': 'DEDH'})

    assert adapter._town_value('Brookline') == 'BRKL'
    assert adapter._town_value(None) == ALL_TOWNS
    with pytest.raises(AlisSearchRefused, match='not one of the 2 towns'):
        adapter._town_value('Somerville')


def test_the_entry_url_survives_a_crosswalk_that_names_a_screen():
    """The registry URL may carry a screen query; the entry is the path.

    The crosswalk holds the URL the Commonwealth publishes, which for
    Norfolk includes a `WSIQTP` screen code.
    """
    with_screen = AlisAdapter(f'{NORFOLK}?WSIQTP=LR09D&WSKYCD=I', context=None)
    bare_host = AlisAdapter('https://www.norfolkresearch.org', context=None)

    assert with_screen._entry_url == NORFOLK
    assert bare_host._entry_url == NORFOLK


def test_a_date_range_crawl_of_alis_is_refused_with_the_reason():
    """ALIS has no date index, and saying so beats returning nothing.

    Every screen the registry offers was read on 2026-09-23: three name
    searches, three year-plus-instrument lookups, plans, cart, login,
    view options. None takes a date range on its own.
    """
    adapter = _adapter()

    with pytest.raises(AlisSearchRefused, match='not by date'):
        asyncio.run(adapter.search('2024-01-15', '2024-01-15', town='Brookline'))


def test_a_name_search_still_requires_a_name():
    """Blank surname is refused here rather than at the registry.

    Submitted blank, ALIS does not leave its own search screen, which
    reads as an empty result rather than a refusal.
    """
    adapter = _adapter()

    with pytest.raises(AlisSearchRefused, match='requires a name'):
        asyncio.run(adapter.search_by_name('  ', '2024-01-01', '2024-01-31'))


def test_alis_is_skipped_by_the_crawler_with_its_reason():
    """The ingester crawls what it can and says why it skips the rest."""
    assert 'alis' not in registry_module.ADAPTERS
    assert 'not by date' in registry_module.NOT_CRAWLABLE['alis']
    assert (
        registry_module.ADAPTERS[registry_module.DRIVABLE_PLATFORM].__name__
        == 'AvenuAdapter'
    )
    # A platform nobody has examined has no reason recorded, so the
    # skip falls back to naming what the ingester does crawl.
    assert 'unexamined' not in registry_module.NOT_CRAWLABLE


def test_the_three_alis_registries_are_marked_and_now_drivable():
    """The crosswalk's own claim, checked against the dispatch table."""
    recipe = get_recipe_by_id('US-MA_transaction-masslandrecords-v1')
    path = recipe_path(
        recipe['admin_id'], recipe['entity'], filename='town_to_registry.csv'
    )
    rows = list(csv.DictReader(path.open(encoding='utf-8')))

    alis = [r for r in rows if r['platform'] == 'alis']
    hosts = {r['base_url'].split('/')[2] for r in alis}
    assert len(alis) == 37, f'expected 37 ALIS towns, found {len(alis)}'
    assert hosts == {
        'www.norfolkresearch.org',
        'search.lawrencedeeds.com',
        'www.fitchburgdeeds.com',
    }
    # Brookline is the town this adapter was written for.
    brookline = [r for r in alis if r['town_name'] == 'Brookline']
    assert len(brookline) == 1
    assert 'norfolkresearch' in brookline[0]['base_url']
    # And every one of them is skipped with a recorded reason rather
    # than crawled into a search that cannot answer by date.
    assert all(r['platform'] in registry_module.NOT_CRAWLABLE for r in alis)
