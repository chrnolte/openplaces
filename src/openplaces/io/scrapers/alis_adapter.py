"""Drive the ALIS registry search, the one masslandrecords does not serve.

Massachusetts spreads its 21 registry districts over at least three web
applications. `masslandrecords.com` serves 11 of them and the Avenu
adapter drives those; three more run **ALIS**, a mainframe-backed
search, and hold 37 towns between them, Brookline among them:

    Norfolk          norfolkresearch.org        28 towns
    North Essex      search.lawrencedeeds.com    4 towns
    North Worcester  fitchburgdeeds.com          5 towns

**ALIS cannot be crawled by date, and that is the headline.** Every
screen it exposes was read on 2026-09-23: three name searches (`LR01D`
recorded land, `LR03D` alpha index, `LC01D` land court), three
year-plus-instrument lookups (`LR09D`, `LP09D`, `SY14D`), a plans
search, a print cart, a login and view options. **None takes a date
range without a name.** The name search's from/to dates narrow one
name's results; they do not index the day.

Submitted with a surname it works and returns exactly what is wanted
(a table of Name, Reverse Party, Town, Date Received, Document Type,
Book and page). Submitted with the name blank it does not leave the
search screen. So collecting every deed a town recorded would mean
iterating names rather than dates: enormous, incomplete against
corporate parties nobody guessed, and a crawl *of people by name*,
which is a different activity from fetching a date range of records
and a poor fit for a project that keeps personal data out of its
outputs.

This adapter therefore offers what ALIS actually supports, a lookup,
and deliberately offers no date-range crawl. `RegistryIngester` skips
ALIS towns by name for that reason rather than listing this class as
drivable.

**Why this navigates instead of building URLs.** ALIS keeps its state
server side. A submission assembled as a query string, without walking
in from the registry's own menu, returns a 655-byte page titled
"Land Recs -Error Message" carrying no message (measured on Norfolk,
2026-09-23: blank name, wildcard surname, and each range radio, all
refused alike). So the adapter opens the registry, follows the menu to
the search screen, fills the form and submits it, exactly as a person
would.

**Nothing here decides what a record means.** It returns the rows the
registry shows; the recipe's column mapping names them, and the
ingester saves them. Party names are in those rows, and are dropped
downstream by `keep_registered_columns` unless the person running it
chose otherwise (see AGENTS.md); this module neither logs nor prints a
cell.
"""

from __future__ import annotations

import asyncio
from datetime import date as _date

from openplaces.io.scrapers.alis_selectors import (
    ALL_DOC_TYPES,
    ALL_TOWNS,
    DEED_DOC_GROUP,
    ENTRY_PATH,
    ERROR_TITLE_MARKER,
    FIELD_DATE_FROM,
    FIELD_DATE_TO,
    FIELD_DOC_TYPE,
    FIELD_GIVEN_NAME,
    FIELD_SURNAME,
    FIELD_TOWN,
    SEARCH_NAME,
)

# The registry's own words for the deed-ish document groups, mapped from
# the document types a recipe asks for. A recipe names 'DEED'; ALIS
# groups deeds under '*DD', which is preferred over enumerating the
# individual types because those differ between the three registries.
DOC_TYPE_GROUPS = {
    'DEED': DEED_DOC_GROUP,
    'FORECLOSURE DEED': DEED_DOC_GROUP,
}


class AlisSearchRefused(RuntimeError):
    """The registry refused a search, with its own screen title."""


class AlisAdapter:
    """Drive one ALIS registry's search through a browser page.

    Presents the same methods as
    :class:`~openplaces.io.scrapers.avenu_adapter.AvenuAdapter` so that
    `RegistryIngester` can drive either from one loop.

    Parameters
    ----------
    base_url : str
        Registry entry URL, e.g.
        ``https://www.norfolkresearch.org/ALIS/WW400R.HTM``.
    context : playwright.async_api.BrowserContext
        Browser context (one per registry session).
    """

    def __init__(self, base_url: str, context) -> None:
        self.base_url = base_url
        self._context = context
        self.page = None
        self._town_lookup: dict[str, str] | None = None
        self._doc_type_lookup: dict[str, str] | None = None

    @property
    def _entry_url(self) -> str:
        """The application entry, whatever path the crosswalk gave."""
        if ENTRY_PATH in self.base_url:
            return self.base_url.split('?')[0]
        return self.base_url.rstrip('/') + ENTRY_PATH

    async def start_session(self) -> None:
        """Open the registry and walk to the name-search screen.

        Navigating to the search screen is what gives the application
        the server-side state a submission needs; see the module
        docstring for what a constructed URL returns instead.
        """
        self.page = await self._context.new_page()
        await self.page.goto(self._entry_url)
        await self.page.wait_for_load_state('networkidle')
        await self.page.goto(f'{self._entry_url}?WSIQTP={SEARCH_NAME}&WSKYCD=I')
        await self.page.wait_for_load_state('networkidle')
        await self._refuse_if_error('opening the name search')
        await self.page.wait_for_selector(f'select[name="{FIELD_TOWN}"]', timeout=30000)
        await self._build_lookups()

    async def _refuse_if_error(self, doing: str) -> None:
        """Raise with the registry's own title when a screen is refused.

        The error screen carries no message, so the title plus what we
        were doing is the whole of what can be reported; saying that
        plainly beats a selector timeout three steps later.
        """
        title = (await self.page.title() or '').strip()
        if ERROR_TITLE_MARKER in title.lower():
            raise AlisSearchRefused(
                f'ALIS refused the request while {doing} '
                f'({self.base_url}): the registry returned {title!r}. '
                'This screen keeps its state server side, so a search '
                'has to be reached through the menu rather than by URL.'
            )

    async def _build_lookups(self) -> None:
        """Read the town and document-type selects from the live page.

        Read rather than hardcoded for the reason the Avenu adapter
        reads its towns dropdown: the codes are per registry (Norfolk
        offers 34 towns and 118 document types), and a bundled list
        would be wrong for the other two and would rot besides.
        """
        self._town_lookup = await self._options(FIELD_TOWN)
        self._doc_type_lookup = await self._options(FIELD_DOC_TYPE)

    async def _options(self, field: str) -> dict[str, str]:
        """Map a select's option labels (uppercased) to their values."""
        options = await self.page.eval_on_selector(
            f'select[name="{field}"]',
            'el => [...el.options].map(o => ({v: o.value, t: o.text.trim()}))',
        )
        return {o['t'].strip().upper(): o['v'] for o in options if o['v']}

    def _town_value(self, town: str | None) -> str:
        """The registry's code for *town*, or every town."""
        if town is None:
            return ALL_TOWNS
        lookup = self._town_lookup or {}
        key = town.strip().upper()
        if key in lookup:
            return lookup[key]
        raise AlisSearchRefused(
            f'{town!r} is not one of the {len(lookup)} towns '
            f'{self.base_url} lists. A town belongs to exactly one '
            'registry, so this is a crosswalk error rather than a '
            'missing record.'
        )

    def _doc_type_value(self, doc_type: str | None) -> str:
        """The registry's code for *doc_type*, preferring its group."""
        if doc_type is None:
            return ALL_DOC_TYPES
        key = doc_type.strip().upper()
        if key in DOC_TYPE_GROUPS:
            return DOC_TYPE_GROUPS[key]
        lookup = self._doc_type_lookup or {}
        return lookup.get(key, ALL_DOC_TYPES)

    @staticmethod
    def _fmt(iso: str) -> str:
        """ISO date to the MMDDYYYY the form's 8-character fields take."""
        d = _date.fromisoformat(iso)
        return f'{d.month:02d}{d.day:02d}{d.year:04d}'

    async def search(
        self,
        date_from: str,
        date_to: str,
        town: str | None = None,
        doc_type: str | None = None,
    ) -> None:
        """Refuse: ALIS has no date-range index to crawl.

        Present so that driving this adapter like the Avenu one fails
        with the reason rather than with an empty result. Measured on
        Norfolk, 2026-09-23: submitted with the name blank the search
        does not leave its own screen, and no other screen the registry
        exposes takes a date range without a name.

        Use :meth:`search_by_name` for what ALIS does support.
        """
        raise AlisSearchRefused(
            f'{self.base_url} runs ALIS, which indexes recorded land by '
            'name and by instrument number, not by date: no screen it '
            'offers takes a date range on its own. A date-range crawl '
            f'of {town or "this registry"} is therefore not possible '
            'here. Search a name with search_by_name(), or look a known '
            'instrument up by year and number.'
        )

    async def search_by_name(
        self,
        surname: str,
        date_from: str,
        date_to: str,
        town: str | None = None,
        doc_type: str | None = None,
        given_name: str = '',
    ) -> None:
        """Submit the one search ALIS supports: a name, optionally bounded.

        Parameters
        ----------
        surname : str
            Surname or corporation, which the application requires.
        date_from, date_to : str
            Inclusive bounds in ``YYYY-MM-DD`` form, narrowing the name's
            results.
        town : str, optional
            Town as the registry's dropdown spells it. None searches
            every town the registry covers.
        doc_type : str, optional
            Document type label, e.g. ``'DEED'``. None searches all.
        given_name : str, optional
            Given name, to narrow a common surname.
        """
        if not surname or not surname.strip():
            raise AlisSearchRefused(
                'ALIS requires a name: a search submitted with the name '
                'blank does not leave its own screen. There is no '
                'date-only index to fall back on.'
            )
        town_value = self._town_value(town)
        doc_value = self._doc_type_value(doc_type)

        await self.page.fill(f'input[name="{FIELD_SURNAME}"]', surname.strip())
        await self.page.fill(f'input[name="{FIELD_GIVEN_NAME}"]', given_name)
        await self.page.fill(f'input[name="{FIELD_DATE_FROM}"]', self._fmt(date_from))
        await self.page.fill(f'input[name="{FIELD_DATE_TO}"]', self._fmt(date_to))
        await self.page.select_option(f'select[name="{FIELD_TOWN}"]', town_value)
        await self.page.select_option(f'select[name="{FIELD_DOC_TYPE}"]', doc_value)

        # The submit navigates, and reading the next screen without
        # awaiting that navigation raised "Execution context was
        # destroyed" from the first evaluate: the context the parse ran
        # in had already been torn down. Awaiting the navigation with
        # the click is what makes the result screen the one being read.
        async with self.page.expect_navigation(wait_until='networkidle'):
            await self.page.click('input[type="submit"]')
        await self._refuse_if_error(
            f'searching {town or "every town"} '
            f'{date_from} to {date_to} for {doc_type or "every type"}'
        )

    async def parse_results(self) -> list[dict]:
        """Return the rows of the current result screen.

        Each row is a dict keyed by the result table's own header
        text, lowercased with spaces as underscores, so the recipe's
        `columns:` block names them and nothing here decides what a
        column means.
        """
        return await self.page.evaluate(
            """() => {
                // The results table is the one whose rows carry links
                // (each row offers a View/Print of the document), not
                // the largest one: the result screen also repeats the
                // search criteria as a table, and picking by row count
                // returned the *form's* own rows, which read as three
                // plausible name columns and looked like results.
                const tables = [...document.querySelectorAll('table')];
                const table = tables
                    .map(t => [t, t.querySelectorAll('a').length])
                    .filter(pair => pair[1] > 0)
                    .sort((a, b) => b[1] - a[1])
                    .map(pair => pair[0])[0];
                if (!table) return [];
                const rows = [...table.querySelectorAll('tr')];
                if (rows.length < 2) return [];
                const head = [...rows[0].querySelectorAll('th, td')]
                    .map(c => (c.textContent || '').trim()
                        .toLowerCase().replace(/\\s+/g, '_'));
                return rows.slice(1).map(r => {
                    const cells = [...r.querySelectorAll('td')]
                        .map(c => (c.textContent || '').trim());
                    if (!cells.length) return null;
                    const out = {};
                    head.forEach((h, i) => { if (h) out[h] = cells[i] ?? ''; });
                    return out;
                }).filter(Boolean);
            }"""
        )

    async def has_next_page(self) -> bool:
        """Whether the result screen offers a further page."""
        return await self.page.evaluate(
            """() => {
                const links = [...document.querySelectorAll('a, input')];
                return links.some(el => /next|more|forward/i.test(
                    el.textContent || el.value || ''));
            }"""
        )

    async def next_page(self) -> None:
        """Advance to the next result screen."""
        await self.page.evaluate(
            """() => {
                const links = [...document.querySelectorAll('a, input')];
                const hit = links.find(el => /next|more|forward/i.test(
                    el.textContent || el.value || ''));
                if (hit) hit.click();
            }"""
        )
        await self.page.wait_for_load_state('networkidle')
        await asyncio.sleep(1)
