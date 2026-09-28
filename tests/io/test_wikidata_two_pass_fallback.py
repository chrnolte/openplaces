"""The two-pass route for a parent the endpoint will not answer for.

Germany's, Australia's and India's states each hold tens of thousands of
items through "located in". Asking for their labels, codes, parents and
grandparents in one query runs past the endpoint's budget and it closes
the connection; halving the batch cannot help, because one parent is
already the smallest question that query can be asked. The route below
asks a cheaper question instead of a smaller one.

No network: the endpoint is a stub that fails the one-shot query and
answers the two cheap ones.
"""

import http.client

import pandas as pd
import pytest

from openplaces.io.admin import wikidata as wd
from openplaces.io.scrapers import wikidata_admin_scraper as scraper

ADMIN_CLASS = 'Q56061'
JUNK_CLASS = 'Q39594'
PARENT = 'Q980'


def _entity(qid):
    return f'http://www.wikidata.org/entity/{qid}'


@pytest.fixture
def endpoint(monkeypatch):
    """A Wikidata that refuses the one-shot query and answers the rest."""
    calls = []

    def _query(sparql, timeout=180, retries=None):
        calls.append(sparql)
        if 'wikibase:label' in sparql and 'VALUES ?parent' in sparql:
            raise http.client.RemoteDisconnected('closed without response')
        if 'VALUES ?parent' in sparql:
            return pd.DataFrame(
                {
                    'item': [_entity('Q1'), _entity('Q2'), _entity('Q3')],
                    'classes': [
                        _entity(ADMIN_CLASS),
                        _entity(JUNK_CLASS),
                        f'{_entity(JUNK_CLASS)}|{_entity(ADMIN_CLASS)}',
                    ],
                }
            )
        wanted = [q for q in ('Q1', 'Q2', 'Q3') if f'wd:{q}' in sparql]
        return pd.DataFrame(
            {
                'item': [_entity(q) for q in wanted],
                'parent': [_entity(PARENT)] * len(wanted),
                'itemLabel': wanted,
                'enLabel': wanted,
                'native': [''] * len(wanted),
                'iso': [''] * len(wanted),
                'classes': [_entity(ADMIN_CLASS)] * len(wanted),
                'parents': [''] * len(wanted),
                'grandparents': [''] * len(wanted),
                'country_code': [''] * len(wanted),
            }
        )

    monkeypatch.setattr(scraper, 'query', _query)
    monkeypatch.setattr(scraper, 'admin_classes', lambda: {ADMIN_CLASS})
    return calls


def test_a_parent_the_one_shot_query_fails_on_still_returns_rows(endpoint):
    frame = scraper._children_of([PARENT], 'de')
    # Q2 is a bay, which the first pass sees and the second never asks
    # about; Q1 and Q3 carry an administrative class.
    assert sorted(frame['itemLabel']) == ['Q1', 'Q3']


def test_the_costly_query_is_tried_once_before_the_cheap_pair(endpoint):
    scraper._children_of([PARENT], 'de')
    kinds = [
        'one-shot'
        if 'wikibase:label' in q and 'VALUES ?parent' in q
        else 'light'
        if 'VALUES ?parent' in q
        else 'details'
        for q in endpoint
    ]
    assert kinds == ['one-shot', 'light', 'details']


def test_details_are_asked_in_batches(endpoint, monkeypatch):
    monkeypatch.setattr(wd, 'DETAIL_BATCH_SIZE', 1)
    frame = scraper._children_of([PARENT], 'de')
    details = [q for q in endpoint if 'VALUES ?item' in q]
    assert len(details) == 2
    assert sorted(frame['itemLabel']) == ['Q1', 'Q3']


def test_a_parent_with_no_administrative_child_returns_nothing(endpoint, monkeypatch):
    monkeypatch.setattr(scraper, 'admin_classes', lambda: {'Q999'})
    assert scraper._children_of([PARENT], 'de').empty
    # and it does not go on to ask for details of nothing
    assert not [q for q in endpoint if 'VALUES ?item' in q]
