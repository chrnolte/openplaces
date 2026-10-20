"""A registry search that fails on every attempt skips its month.

Raising instead ended the whole town's run (Cambridge, 2026-10-20). The
adapter here is a stand-in; no registry is contacted.
"""

import asyncio

import pytest
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from openplaces.io.ingester.registry_ingester import RegistryIngester


class _Page:
    def on(self, event, handler):
        pass

    def remove_listener(self, event, handler):
        pass


class _Adapter:
    def __init__(self, error):
        self.page = _Page()
        self.error = error
        self.searches = 0

    async def search(self, date_from, date_to, town, doc_type):
        self.searches += 1
        raise self.error


def _ingester():
    ingester = object.__new__(RegistryIngester)
    ingester.max_attempts = 2
    ingester.backoff = (0, 0)
    ingester.rate_limit = (0, 0)
    ingester.fetch_details = False
    ingester.verbose = False
    return ingester


def _crawl(adapter):
    return asyncio.run(
        _ingester()._crawl_partition(
            adapter,
            'US-XX-EXA',
            '2021-06-01',
            '2021-06-30',
            town='EXAMPLE',
            doc_type='DEED',
            page_count=5,
        )
    )


def test_a_search_that_times_out_every_time_is_skipped():
    adapter = _Adapter(PlaywrightTimeoutError('Timeout 15000ms exceeded'))
    with pytest.warns(UserWarning, match='retried on the next run'):
        records, page_count = _crawl(adapter)
    assert records is None
    assert page_count == 5
    assert adapter.searches == 2


def test_an_error_that_is_not_the_browsers_still_raises():
    adapter = _Adapter(KeyError('town'))
    with pytest.raises(KeyError):
        _crawl(adapter)
    assert adapter.searches == 2
