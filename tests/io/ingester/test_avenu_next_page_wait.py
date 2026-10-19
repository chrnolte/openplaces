"""Next waits for the results grid to change before rows are read.

Network idle came before the grid's partial update, so the first row of a
new page was read from the old grid (Nantucket, 2026-10-19). Fake page.
"""

import asyncio

import pytest

from openplaces.io.scrapers.avenu_adapter import AvenuAdapter


class _FakePage:
    def __init__(self, first_rows):
        self._rows = list(first_rows)
        self.clicked = False

    async def evaluate(self, script, arg=None):
        return self._rows.pop(0) if len(self._rows) > 1 else self._rows[0]

    async def click(self, selector):
        self.clicked = True

    async def wait_for_load_state(self, state):
        return None


def _adapter(page, monkeypatch):
    adapter = object.__new__(AvenuAdapter)
    adapter.page = page

    async def unblocked(*args, **kwargs):
        return None

    monkeypatch.setattr(adapter, '_wait_unblocked', unblocked)
    return adapter


def test_next_returns_once_the_first_row_changes(monkeypatch):
    # Before the click, then two reads of the old grid, then the new one.
    page = _FakePage(['1/10 DEED', '1/10 DEED', '1/10 DEED', '2/20 DEED'])
    asyncio.run(_adapter(page, monkeypatch).next_page(timeout=3))
    assert page.clicked
    assert page._rows == ['2/20 DEED']


def test_next_refuses_a_grid_that_never_changes(monkeypatch):
    page = _FakePage(['1/10 DEED'])
    with pytest.raises(TimeoutError, match='did not change'):
        asyncio.run(_adapter(page, monkeypatch).next_page(timeout=0.6))
