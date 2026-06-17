"""
Playwright-based adapter for Avenu Insights WebForms registry platforms
(used by Massachusetts registries at masslandrecords.com).

The browser owns session state, cookies, and ViewState — this adapter
never touches ASP.NET internals directly.
"""

from __future__ import annotations

import asyncio
from datetime import date as _date

from openplaces.io.avenu_selectors import (
    CRITERIA_DATE_SEARCH,
    DATE_FROM,
    DATE_TO,
    DETAILS_CONTAINER,
    DETAILS_ROW_CLICK_TARGET,
    DOC_LINK,
    DOC_TYPE_DROPDOWN,
    MESSAGE_BOX_PANEL,
    NEXT_BUTTON,
    PROGRESS_BAR,
    REAL_PAGE_INDICATOR,
    RESULTS_ROW,
    RESULTS_TABLE,
    SCREEN_BLOCKER,
    SUBMIT,
    TOWNS_DROPDOWN,
)


class AvenuAdapter:
    """Async adapter for one Avenu WebForms registry session.

    Parameters
    ----------
    base_url : str
        Registry home URL, e.g. ``https://www.masslandrecords.com/MiddlesexSouth/``
    context : playwright.async_api.BrowserContext
        Playwright browser context (one per registry session).
    """

    def __init__(self, base_url: str, context) -> None:
        self.base_url = base_url
        self._context = context
        self.page = None
        self._town_lookup: dict[str, str] | None = None

    async def start_session(self) -> None:
        """Navigate to the registry home page and resolve any bot challenge."""
        self.page = await self._context.new_page()
        await self.page.goto(self.base_url)
        await self._wait_for_challenge()

    async def _wait_unblocked(self, timeout: int = 8000) -> None:
        """Wait for ASP.NET overlay controls to clear before interaction.

        First waits for the UpdateProgress bar (auto-clears after AJAX),
        then for the ScreenBlocker (may need a JS dismiss if a MessageBox
        is waiting for user interaction).
        """
        # UpdateProgress bar clears automatically — just wait for it
        try:
            await self.page.locator(PROGRESS_BAR).wait_for(
                state='hidden', timeout=timeout
            )
        except Exception:
            pass
        # ScreenBlocker: wait 3 s, then JS-dismiss the MessageBox if still up
        try:
            await self.page.locator(SCREEN_BLOCKER).wait_for(
                state='hidden', timeout=3000
            )
            return
        except Exception:
            pass
        await self.page.evaluate(f"""() => {{
            const panel = document.querySelector('{MESSAGE_BOX_PANEL}');
            if (!panel) return;
            const el = panel.querySelector(
                'input[type="button"], input[type="submit"], a'
            );
            if (el) el.click();
        }}""")
        try:
            await self.page.locator(SCREEN_BLOCKER).wait_for(
                state='hidden', timeout=max(timeout - 3000, 2000)
            )
        except Exception:
            pass

    async def _wait_for_challenge(self, timeout: int = 20) -> None:
        """Wait for Incapsula challenge to resolve and real page to load."""
        for _ in range(timeout):
            urls = [f.url for f in self.page.frames]
            if any(REAL_PAGE_INDICATOR in u for u in urls):
                await self.page.wait_for_load_state('networkidle')
                return
            await asyncio.sleep(1)
        await self.page.wait_for_load_state('networkidle')

    async def _build_town_lookup(self) -> None:
        """Populate ``_town_lookup`` from the live towns dropdown.

        Keys are the uppercase display names shown in the dropdown; values are
        the numeric ``<option value>`` used to select them. Built once per
        session so later calls to ``search()`` skip the round-trip.
        """
        opts = await self.page.eval_on_selector(
            TOWNS_DROPDOWN,
            'el => [...el.options].map(o => ({v: o.value, t: o.text.trim()}))',
        )
        self._town_lookup = {o['t']: o['v'] for o in opts}

    async def search(
        self,
        date_from: str,
        date_to: str,
        town: str | None = None,
        doc_type: str | None = None,
    ) -> None:
        """Submit a date-range search form with optional town and doc-type filters.

        Parameters
        ----------
        date_from : str
            Start date in ``YYYY-MM-DD`` format.
        date_to : str
            End date in ``YYYY-MM-DD`` format.
        town : str, optional
            Uppercase town name as displayed in the registry dropdown,
            e.g. ``'ACTON'``. Pass ``None`` to search all towns.
        doc_type : str, optional
            Document type display label, e.g. ``'DEED'``.
            Pass ``None`` to search all document types.
        """

        def _fmt(iso: str) -> str:
            d = _date.fromisoformat(iso)
            return f'{d.month}/{d.day}/{d.year}'

        # Switch to Recorded Date Search
        async with self.page.expect_response(
            lambda r: r.url.endswith('Default.aspx') and r.request.method == 'POST',
            timeout=15000,
        ):
            await self.page.evaluate(
                f"document.querySelector('{CRITERIA_DATE_SEARCH}').click()"
            )
        await self.page.wait_for_load_state('networkidle')
        await self._wait_unblocked()
        await self.page.wait_for_selector(DATE_FROM, timeout=10000)

        # Fill date fields
        await self.page.evaluate(
            f"document.querySelector('{DATE_FROM}').value = '{_fmt(date_from)}'"
        )
        await self.page.evaluate(
            f"document.querySelector('{DATE_TO}').value = '{_fmt(date_to)}'"
        )
        await self.page.evaluate(f"""
            (() => {{
                const el = document.querySelector('{DATE_TO}');
                el.dispatchEvent(new Event('input', {{bubbles: true}}));
                el.dispatchEvent(new Event('change', {{bubbles: true}}));
            }})()
        """)

        # Select town via JS (element may not be visible but is in DOM)
        if town is not None:
            if self._town_lookup is None:
                await self._build_town_lookup()
            town_value = self._town_lookup.get(town, '')
            await self.page.evaluate(f"""
                (() => {{
                    const el = document.querySelector('{TOWNS_DROPDOWN}');
                    for (const o of el.options) o.selected = false;
                    for (const o of el.options) {{
                        if (o.value === '{town_value}') {{ o.selected = true; break; }}
                    }}
                    el.dispatchEvent(new Event('change', {{bubbles: true}}));
                }})()
            """)
            await asyncio.sleep(0.2)

        # Select document type via JS (same visibility issue)
        if doc_type is not None:
            safe_label = doc_type.replace("'", "\\'")
            await self.page.evaluate(f"""
                (() => {{
                    const el = document.querySelector('{DOC_TYPE_DROPDOWN}');
                    for (const o of el.options) o.selected = false;
                    for (const o of el.options) {{
                        if (o.text.trim() === '{safe_label}') {{
                            o.selected = true; break;
                        }}
                    }}
                    el.dispatchEvent(new Event('change', {{bubbles: true}}));
                }})()
            """)
            await asyncio.sleep(0.2)

        # Submit — wait for networkidle, then wait for the AJAX-populated results
        # table (onload="ClientLoad(100);" fires ~100ms after page load).
        # Timeout is swallowed so a zero-results page doesn't raise.
        await self._wait_unblocked()
        await self.page.locator(SUBMIT).click()
        await self.page.wait_for_load_state('networkidle')
        try:
            await self.page.wait_for_selector(
                f'{RESULTS_TABLE} tr.DataGridRow',
                timeout=10000,
            )
        except Exception:
            pass

    async def parse_results(
        self, fetch_details: bool = False, debug: bool = False
    ) -> list[dict]:
        """Extract all result rows from the current results page.

        Parameters
        ----------
        fetch_details : bool
            If True, click each row to load the View Details panel and merge
            its fields (consideration, grantor/grantee, address, etc.) into
            the returned dict.
        debug : bool
            If True, print a per-row trace (index, book, page, whether details
            loaded, instrument_number) and a summary of rows-seen vs. distinct
            (book, page). Diagnostic aid for index-desync duplicates; off by
            default even in verbose ingestion runs.

        Returns
        -------
        list of dict
            One dict per data row; header rows are skipped.
        """
        # Re-query row handles on each iteration: clicking a row for details
        # triggers an AJAX UpdatePanel refresh that detaches previously-queried
        # ElementHandles, causing stale-element errors on subsequent rows.
        count = await self.page.locator(RESULTS_ROW).count()
        results = []
        seen: list[tuple] = []
        for i in range(count):
            rows = await self.page.query_selector_all(RESULTS_ROW)
            if i >= len(rows):
                break
            row = rows[i]
            base = await self._parse_row(row)
            if base is None:
                continue
            loaded = None
            if fetch_details:
                loaded = await self._click_for_details(i)
                if loaded:
                    details = await self.parse_details()
                    base.update(details)
            results.append(base)
            seen.append((base.get('book'), base.get('page')))
            if debug:
                print(
                    f'    row {i}: book={base.get("book")} '
                    f'page={base.get("page")} details_loaded={loaded} '
                    f'instrument={base.get("instrument_number", "")}'
                )
        if debug:
            print(
                f'    parse_results: rows_seen={len(seen)} '
                f'distinct_(book,page)={len(set(seen))}'
            )
        return results

    async def _wait_for_details_change(
        self, prev_text: str, timeout: float = 6.0
    ) -> bool:
        """Poll until the details panel text differs from prev_text and has
        more than 50 chars. Returns True if loaded, False on timeout."""
        read_js = (
            f'() => {{'
            f'  const el = document.querySelector("{DETAILS_CONTAINER}");'
            f'  return el ? el.innerText.trim() : "";'
            f'}}'
        )
        for _ in range(int(timeout / 0.3)):
            current = await self.page.evaluate(read_js)
            if len(current) > 50 and current != prev_text:
                return True
            await asyncio.sleep(0.3)
        return False

    async def _click_for_details(self, row_index: int) -> bool:
        """Click the first ButtonRow link in the row at the given 0-based index.

        Snapshots the details panel text before clicking, then waits until the
        panel content changes. Returns True if new content loaded within the
        timeout, False if the panel appears stale.

        Uses the Locator API (lazy re-resolution at click time) to avoid stale
        ElementHandle errors caused by AJAX panel updates between rows.
        """
        read_js = (
            f'() => {{'
            f'  const el = document.querySelector("{DETAILS_CONTAINER}");'
            f'  return el ? el.innerText.trim() : "";'
            f'}}'
        )
        prev_text = await self.page.evaluate(read_js)
        target = (
            self.page.locator(RESULTS_ROW)
            .nth(row_index)
            .locator(DETAILS_ROW_CLICK_TARGET)
            .first
        )
        await target.click()
        await self._wait_unblocked()
        return await self._wait_for_details_change(prev_text)

    async def parse_details(self) -> dict:
        """Parse the View Details panel currently visible on the right.

        Returns
        -------
        dict
            Keys: instrument_number, rec_time, n_pages, doc_status,
            consideration, street_number, street_name, description,
            grantor (';'-joined), grantee (';'-joined),
            linked_docs (';'-joined 'book/page:type' strings).
        """
        container = await self.page.query_selector(DETAILS_CONTAINER)
        if container is None:
            return {'_details_missing': True}

        tables = await container.query_selector_all('table')
        result: dict = {}
        grantors: set[str] = set()
        grantees: set[str] = set()
        linked_docs: list[str] = []

        for tbl in tables:
            rows = await tbl.query_selector_all('tr')
            if not rows:
                continue
            first_cells = await rows[0].query_selector_all('td, th')
            if not first_cells:
                continue
            first_texts = [(await c.inner_text()).strip() for c in first_cells]
            # Outer wrapper tables have cells whose inner_text spans the
            # full content of a nested table (tab/newline separated) —
            # skip them so only inner data tables are processed.
            if any('\t' in t or '\n' in t for t in first_texts):
                continue
            headers = [t.lower() for t in first_texts]

            # Metadata table: ≥5 columns, one labelled "consideration"
            if (
                len(headers) >= 5
                and any('consideration' in h for h in headers)
                and 'instrument_number' not in result
            ):
                if len(rows) >= 2:
                    data_cells = await rows[1].query_selector_all('td')
                    data = [(await c.inner_text()).strip() for c in data_cells]

                    def _get(label: str) -> str:
                        return next(
                            (d for h, d in zip(headers, data) if label in h),
                            '',
                        )

                    result['instrument_number'] = _get('doc')
                    result['rec_time'] = _get('rec time')
                    result['n_pages'] = _get('pgs')
                    result['doc_status'] = _get('status')
                    result['consideration'] = _get('consideration')
                continue

            # Address table: at least one header contains "street"
            if any('street' in h for h in headers) and 'street_number' not in result:
                if len(rows) >= 2:
                    data_cells = await rows[1].query_selector_all('td')
                    data = [(await c.inner_text()).strip() for c in data_cells]
                    result['street_number'] = data[0] if data else ''
                    result['street_name'] = data[1] if len(data) > 1 else ''
                    result['description'] = data[2] if len(data) > 2 else ''
                continue

            # All other tables: scan for grantor/grantee rows and linked docs
            for row in rows:
                cells = await row.query_selector_all('td')
                if len(cells) < 2:
                    continue
                texts = [(await c.inner_text()).strip() for c in cells]
                role = texts[-1].lower()
                name = texts[0]
                if role == 'grantor' and name:
                    grantors.add(name)
                elif role == 'grantee' and name:
                    grantees.add(name)
                elif '/' in name and role not in ('grantor', 'grantee'):
                    linked_docs.append(f'{name}:{texts[1]}')

        result.setdefault('instrument_number', '')
        result.setdefault('rec_time', '')
        result.setdefault('n_pages', '')
        result.setdefault('doc_status', '')
        result.setdefault('consideration', '')
        result.setdefault('street_number', '')
        result.setdefault('street_name', '')
        result.setdefault('description', '')
        result['grantor'] = '; '.join(sorted(grantors))
        result['grantee'] = '; '.join(sorted(grantees))
        result['linked_docs'] = '; '.join(dict.fromkeys(linked_docs))
        return result

    async def has_next_page(self) -> bool:
        """Return True if the pagination Next button is present and enabled."""
        el = await self.page.query_selector(NEXT_BUTTON)
        if el is None:
            return False
        cls = await el.get_attribute('class') or ''
        return 'disabled' not in cls.lower()

    async def next_page(self) -> None:
        """Navigate to the next results page via the Next postback link."""
        await self._wait_unblocked()
        await self.page.click(NEXT_BUTTON)
        await self.page.wait_for_load_state('networkidle')

    async def dump_details_candidates(self) -> None:
        """Print frame URLs and locate the "Consideration" label across all frames.

        Run this after clicking a row to open View Details, to discover which
        frame holds the details panel and what container ID to use.
        """
        find_js = """() => {
            const walker = document.createTreeWalker(
                document.body, NodeFilter.SHOW_TEXT
            );
            let node;
            while ((node = walker.nextNode())) {
                if (node.textContent.trim() === 'Consideration') {
                    let el = node.parentElement;
                    const path = [];
                    while (el) {
                        if (el.id) path.push('#' + el.id);
                        el = el.parentElement;
                    }
                    return path.join(' < ');
                }
            }
            return null;
        }"""

        print('Frames on page:')
        for frame in self.page.frames:
            print(f'  [{frame.name or "(unnamed)"}] {frame.url}')
            try:
                found = await frame.evaluate(find_js)
                if found:
                    print(f'    *** "Consideration" found: {found}')
            except Exception:
                pass
        print()

    async def open_document(self, row_element) -> str | None:
        """Open the document popup for a result row and return its URL.

        Parameters
        ----------
        row_element : playwright ElementHandle
            A result row element containing a document link.

        Returns
        -------
        str or None
            The URL of the document popup page, or None if no link found.
        """
        link = await row_element.query_selector(DOC_LINK)
        if link is None:
            return None
        async with self.page.expect_popup() as popup_info:
            await link.click()
        popup = await popup_info.value
        await popup.wait_for_load_state('domcontentloaded')
        url = popup.url
        await popup.close()
        return url

    async def _parse_row(self, row) -> dict | None:
        """Extract field values from a single result row.

        Columns (confirmed against Middlesex South list view):
        td[0]=checkbox, td[1]=File Date, td[2]=Book/Page, td[3]=Type Desc.,
        td[4]=Town, td[5]=View Img button, td[6]=Add to Basket button.
        Grantor/grantee/consideration are not in the list view.
        """
        cells = await row.query_selector_all('td')
        if len(cells) < 5:
            return None
        texts = [await cell.inner_text() for cell in cells]
        recorded_date = texts[1].strip()
        if not recorded_date:
            return None
        book_page = texts[2].strip().split('/', 1)
        return {
            'recorded_date': recorded_date,
            'book': book_page[0] if book_page else None,
            'page': book_page[1] if len(book_page) > 1 else None,
            'doc_type': texts[3].strip() if len(texts) > 3 else None,
            'town': texts[4].strip() if len(texts) > 4 else None,
        }
