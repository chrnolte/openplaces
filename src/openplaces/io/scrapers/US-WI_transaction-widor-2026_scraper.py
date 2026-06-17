"""
Browser-driven scraper for the Wisconsin Real Estate Transfer Return (RETR)
public dataset.

The Wisconsin Department of Revenue publishes monthly RETR extracts through its
Tax Account Portal (TAP), a JavaScript single-page app behind a terms-of-use
gate at https://tap.revenue.wi.gov/mta/. The portal generates each month's CSV
on the fly (the download URL is a one-off token), so the file cannot be fetched
with a static URL. This module drives a real browser (via Playwright) to:

1. open the portal and accept the terms of use if prompted,
2. navigate to the requested year,
3. switch the report format to CSV ("CSV Report" button),
4. click the requested month to trigger generation of the CSV, and
5. capture the resulting download and save it to a target path.

The portal serves one file per year-month named ``RETRHistoricalReport{YYYYMM}.csv``.

Called by `openplaces.io.ingester.Ingester._run_download_scraper` when a recipe's
source sets ``download_url_scraper: US-WI_transaction-widor-2026_scraper`` (the
browser-driven peer of ``download_url_source``). The ingester loads this module
by file path and calls its standard ``fetch`` entrypoint.

Because the portal's DOM is not part of this codebase and may change, every step
uses resilient text/role locators and every selector is overridable through the
recipe's ``scraper_options`` block. On failure a screenshot and the page HTML are
written next to the target path to aid debugging.
"""

from __future__ import annotations

import asyncio
import calendar
import contextlib
import sys
import threading
import warnings
from pathlib import Path

# Direct URL of the public RETR historical-data tool (no login required).
# Navigating here lands straight on the year/month download interface, so the
# home-page entry-link step is unnecessary by default.
DEFAULT_PORTAL_URL = 'https://tap.revenue.wi.gov/RETRHistoric'

# Fallback only: link on the TAP home page that opens the RETR historical-data
# tool (in the "Real Estate Transfer Return (RETR)" panel). Used when starting
# from the home page instead of the direct tool URL.
DEFAULT_ENTRY_LINK_TEXT = 'Download Historical RETR Data'

# Candidate button labels for the terms-of-use gate, tried in order.
DEFAULT_TERMS_TEXTS = ('I Agree', 'Agree', 'Accept', 'I Accept', 'Continue')

# Label of the button that switches the report output format to CSV.
DEFAULT_CSV_REPORT_TEXT = 'CSV Report'

# Substring of the modal shown when a month's file is not published yet
# ("A file has not yet been generated for the selected year and month."). The
# modal is rendered in shadow DOM — invisible to page.content() but reachable
# via Playwright text locators, which pierce open shadow roots.
UNAVAILABLE_MODAL_TEXT = 'not yet been generated'

# Human-readable label prefixed to progress messages. `fetch` overrides this
# per run with the recipe id passed by the Ingester (e.g.
# 'US-WI_transaction-widor-2026'), so logs say which recipe is running.
_LABEL = 'Wisconsin RETR'


def _log(message: str) -> None:
    """Print an indented scraper progress line.

    The caller (Ingester) prints a single header naming the recipe/partition;
    these step lines are indented and unprefixed to read as its sub-steps.
    """
    print(f'  {message}')


def fetch(
    partition_id: str,
    target_path: str | Path,
    portal_url: str | None = None,
    *,
    headless: bool = False,
    timeout_s: int = 120,
    label: str | None = None,
    entry_link_text: str = '',
    accept_terms: bool = True,
    terms_texts: tuple[str, ...] = DEFAULT_TERMS_TEXTS,
    csv_report_text: str = DEFAULT_CSV_REPORT_TEXT,
    month_style: str = 'name',
    browser_channel: str | None = None,
    slow_mo_ms: int = 0,
    screenshot_on_error: bool = True,
    verbose: bool = False,
) -> Path:
    """Download one month of Wisconsin RETR data to *target_path*.

    Parameters
    ----------
    partition_id : str
        Year-month to download, as a six-digit ``YYYYMM`` string (e.g.
        ``'202601'``).
    target_path : str or pathlib.Path
        Where to save the downloaded CSV. Parent directories are created.
    portal_url : str, optional
        TAP portal URL. Defaults to the public RETR landing page.
    headless : bool, default False
        Run the browser without a visible window. The portal uses bot
        detection, so a visible (headed) browser is more reliable on first
        use; flip to True once a run is known to work in your environment.
    timeout_s : int, default 120
        Per-action timeout in seconds (navigation, waits, clicks, download).
    label : str, optional
        Human-readable prefix for progress messages (the Ingester passes the
        recipe id). Defaults to "Wisconsin RETR".
    entry_link_text : str, default ''
        Empty by default because the default portal_url lands directly on the
        RETR tool. Set to a link label (e.g. 'Download Historical RETR Data')
        only when starting from the TAP home page rather than the tool URL.
    accept_terms : bool, default True
        Click the terms-of-use button if the gate is shown.
    terms_texts : tuple of str
        Candidate labels for the terms-of-use button, tried in order.
    csv_report_text : str, default 'CSV Report'
        Label of the button that switches the report format to CSV.
    month_style : {'name', 'abbr', 'number'}, default 'name'
        How months are labelled on the portal: full name ('January'),
        three-letter abbreviation ('Jan'), or zero-padded number ('01').
    browser_channel : str, optional
        Playwright browser channel (e.g. ``'chrome'`` or ``'msedge'``) to use
        an installed system browser instead of the bundled Chromium.
    slow_mo_ms : int, default 0
        Slow each Playwright operation by this many milliseconds. Useful for
        debugging or to look less robotic to bot detection.
    screenshot_on_error : bool, default True
        On failure, write ``{stem}_error.png`` and ``{stem}_error.html`` next
        to *target_path*.
    verbose : bool, default False
        Print progress messages.

    Returns
    -------
    pathlib.Path or None
        Path to the downloaded CSV (equal to *target_path*), or ``None`` when
        the requested year/month is not published yet (the portal shows a
        "file has not yet been generated" modal, or the year is not listed).
        ``None`` signals the caller to skip the partition.

    Raises
    ------
    RuntimeError
        If Playwright is not installed, or a step fails (e.g. the terms gate,
        CSV-format button, year, or month could not be located).
    """
    global _LABEL
    if label:
        _LABEL = label

    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    portal_url = portal_url or DEFAULT_PORTAL_URL
    year, month = _parse_year_month(partition_id)

    # Playwright's sync API refuses to run inside a thread that already owns a
    # running asyncio loop (e.g. a Jupyter kernel). Run the browser work in a
    # dedicated worker thread — which has no running loop — so the same code
    # works from both plain scripts and notebooks.
    box: dict = {}

    def _worker() -> None:
        try:
            box['result'] = _fetch_in_browser(
                target_path,
                portal_url,
                year,
                month,
                headless=headless,
                timeout_s=timeout_s,
                entry_link_text=entry_link_text,
                accept_terms=accept_terms,
                terms_texts=terms_texts,
                csv_report_text=csv_report_text,
                month_style=month_style,
                browser_channel=browser_channel,
                slow_mo_ms=slow_mo_ms,
                screenshot_on_error=screenshot_on_error,
                verbose=verbose,
            )
        except BaseException as exc:  # noqa: BLE001
            box['exc'] = exc

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join()
    if 'exc' in box:
        raise box['exc']
    return box['result']


def _fetch_in_browser(
    target_path: Path,
    portal_url: str,
    year: int,
    month: int,
    *,
    headless: bool,
    timeout_s: int,
    entry_link_text: str,
    accept_terms: bool,
    terms_texts: tuple[str, ...],
    csv_report_text: str,
    month_style: str,
    browser_channel: str | None,
    slow_mo_ms: int,
    screenshot_on_error: bool,
    verbose: bool,
) -> Path:
    """Drive the browser to download one month's CSV (runs in a worker thread).

    Separated from `fetch` so the synchronous Playwright calls execute in a
    thread with no running asyncio loop, which is required under Jupyter.
    """
    sync_playwright = _import_playwright()
    timeout_ms = timeout_s * 1000

    # On Windows the Playwright driver runs as a subprocess that only a
    # ProactorEventLoop can spawn; _proactor_policy_on_windows() makes the loop
    # Playwright builds via asyncio.new_event_loop() a Proactor one even under
    # Jupyter's global SelectorEventLoop policy (else: NotImplementedError).
    with _proactor_policy_on_windows(), sync_playwright() as p:
        launch_kwargs: dict = {'headless': headless, 'slow_mo': slow_mo_ms}
        if not headless:
            # Run headed (the portal's bot detection rejects headless) but move
            # the window far off-screen so no visible window appears.
            launch_kwargs['args'] = ['--window-position=-32000,-32000']
        if browser_channel:
            launch_kwargs['channel'] = browser_channel
        browser = p.chromium.launch(**launch_kwargs)
        # A single context with one page: the portal forbids multiple tabs.
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()
        page.set_default_timeout(timeout_ms)

        try:
            page.goto(portal_url, wait_until='domcontentloaded')
            page.wait_for_load_state('networkidle')

            if entry_link_text:
                _open_entry(page, entry_link_text, verbose=verbose)

            if accept_terms:
                _accept_terms(page, terms_texts, timeout_ms, verbose=verbose)

            _click_csv_report(page, csv_report_text, verbose=verbose)

            # Every available year is listed at once; resolve the requested
            # year's grid column so the month click targets the right block
            # (not the topmost year). None => year not published.
            col = _year_column(page, year)
            if col is None:
                if verbose:
                    _log(f'year {year} not listed on portal, skipping')
                return None
            if verbose:
                _log(f'year {year} -> grid column {col}')

            download = _download_month(
                page, year, month, col, month_style, timeout_s, verbose=verbose
            )
            if download is None:
                return None  # month not generated yet → skip partition
            download.save_as(str(target_path))
        except Exception as exc:
            if screenshot_on_error:
                _dump_debug_artifacts(page, target_path, verbose=verbose)
            raise RuntimeError(
                f'{_LABEL} scraper failed for {year}-{month:02d}: {exc}\n'
                'The TAP portal layout may have changed. Inspect the saved '
                'screenshot/HTML and adjust scraper_options (terms_texts, '
                'csv_report_text, month_style) in the recipe.'
            ) from exc
        finally:
            context.close()
            browser.close()

    if verbose:
        _log(f'saved {target_path.name}')
    return target_path


@contextlib.contextmanager
def _proactor_policy_on_windows():
    """Temporarily install the Proactor event-loop policy on Windows.

    Playwright's sync API builds its loop with ``asyncio.new_event_loop()``,
    which uses the process-global policy. Its driver runs as a subprocess that
    only a ProactorEventLoop can spawn, but Jupyter installs a SelectorEventLoop
    policy globally. Swap to the Proactor policy for the browser session and
    restore it afterward. Safe in practice: this runs in a worker thread while
    the caller is blocked, and already-created loops keep their type. No-op off
    Windows. (The policy API is deprecated in 3.14 but remains the only lever
    over the loop Playwright's sync API creates internally.)
    """
    if sys.platform != 'win32':
        yield
        return
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        prev_policy = asyncio.get_event_loop_policy()
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    try:
        yield
    finally:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            asyncio.set_event_loop_policy(prev_policy)


def _import_playwright():
    """Return Playwright's ``sync_playwright`` or raise an install hint."""
    try:
        from playwright.sync_api import sync_playwright
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f'The {_LABEL} scraper needs Playwright, which is not installed.\n\n'
            'Install it into the openplaces environment:\n\n'
            '    pip install playwright\n'
            '    playwright install chromium\n'
        ) from exc
    return sync_playwright


def _parse_year_month(partition_id: str) -> tuple[int, int]:
    """Split a 'YYYYMM' partition id into (year, month) integers."""
    s = str(partition_id)
    if len(s) != 6 or not s.isdigit():
        raise ValueError(
            f"{_LABEL} partition_id must be 'YYYYMM' (six digits); got {s!r}."
        )
    year, month = int(s[:4]), int(s[4:6])
    if not 1 <= month <= 12:
        raise ValueError(f'Invalid month in partition_id {s!r}: {month}.')
    return year, month


def _month_label(month: int, style: str) -> str:
    """Return the on-portal label for *month* in the requested style."""
    if style == 'name':
        return calendar.month_name[month]
    if style == 'abbr':
        return calendar.month_abbr[month]
    if style == 'number':
        return f'{month:02d}'
    raise ValueError(f"month_style must be 'name', 'abbr', or 'number'; got {style!r}.")


def _open_entry(page, entry_link_text: str, verbose: bool = False) -> None:
    """Open the public RETR historical-data tool from the TAP home page.

    The TAP landing page ("My Tax Account") shows a "Real Estate Transfer
    Return (RETR)" panel whose "Download Historical RETR Data" link opens the
    public download tool without logging in.
    """
    link = page.get_by_role('link', name=entry_link_text, exact=False)
    try:
        if link.count() and link.first.is_visible():
            link.first.click()
            page.wait_for_load_state('networkidle')
            if verbose:
                _log(f'opened entry "{entry_link_text}"')
            return
    except Exception:
        pass

    text = page.get_by_text(entry_link_text, exact=False)
    if not text.count():
        raise RuntimeError(
            f'Could not find the entry link "{entry_link_text}" on the portal '
            'home page. Adjust scraper_options.entry_link_text.'
        )
    text.first.click()
    page.wait_for_load_state('networkidle')
    if verbose:
        _log(f'opened entry "{entry_link_text}" (text match)')


def _accept_terms(
    page, terms_texts: tuple[str, ...], timeout_ms: int, verbose: bool = False
) -> None:
    """Click the terms-of-use button if the gate is present.

    Tries each candidate label as a button role first, then as any clickable
    element with matching text. Absence of the gate is not an error: the
    portal only shows it on the first visit of a session.

    Matching is exact: the RETR disclaimer offers both "Agree" and "Disagree",
    and a substring match on "Agree" would also hit "Disagree".

    After a successful click, waits for the disclaimer to disappear: the SPA
    renders the next view client-side (no network), so wait_for_load_state
    alone returns before the transition completes.
    """
    for text in terms_texts:
        for locator in (
            page.get_by_role('button', name=text, exact=True),
            page.get_by_text(text, exact=True),
        ):
            try:
                if not (locator.count() and locator.first.is_visible()):
                    continue
                locator.first.click()
                _wait_for_disclaimer_gone(page, timeout_ms)
                page.wait_for_load_state('networkidle')
                if verbose:
                    _log(f'accepted terms via "{text}"')
                return
            except Exception:
                pass

    if verbose:
        _log('no terms-of-use gate shown (continuing)')


def _wait_for_disclaimer_gone(page, timeout_ms: int) -> None:
    """Wait for the disclaimer gate to detach after agreeing."""
    try:
        page.get_by_role('heading', name='Disclaimer', exact=False).first.wait_for(
            state='detached', timeout=timeout_ms
        )
    except Exception:
        # Fall back to a short settle if the heading lookup does not apply.
        page.wait_for_timeout(2000)


def _year_column(page, year: int) -> str | None:
    """Return the FastGrid column index for *year*, or None if not listed.

    The RETR tool lists every available year at once as
    ``span#caption2_Dc-c1-{col}`` headings (a sliding window of recent years,
    so the column for a given year is not fixed). Read them live and map the
    requested year to its column index. The 12 month links under that year
    share the same trailing ``-{col}`` id suffix, which `_download_month` uses
    to click the correct year's month.
    """
    captions = page.locator('span.CaptionLabel')
    for i in range(captions.count()):
        el = captions.nth(i)
        if (el.text_content() or '').strip() == str(year):
            caption_id = el.get_attribute('id') or ''  # caption2_Dc-c1-{col}
            col = caption_id.rsplit('-', 1)[-1]
            if col.isdigit():
                return col
    return None


def _click_csv_report(page, csv_report_text: str, verbose: bool = False) -> None:
    """Switch the report output format to CSV.

    Skips silently if the button is not present (some layouts default to CSV
    or expose it only after a month is selected).
    """
    button = page.get_by_role('button', name=csv_report_text, exact=False)
    try:
        if button.count() and button.first.is_visible():
            button.first.click()
            page.wait_for_load_state('networkidle')
            if verbose:
                _log(f'clicked "{csv_report_text}"')
            return
    except Exception:
        pass

    link = page.get_by_text(csv_report_text, exact=False)
    try:
        if link.count() and link.first.is_visible():
            link.first.click()
            page.wait_for_load_state('networkidle')
            if verbose:
                _log(f'clicked "{csv_report_text}" (text match)')
            return
    except Exception:
        pass

    if verbose:
        _log(f'"{csv_report_text}" button not found (continuing)')


def _download_month(page, year, month, col, month_style, timeout_s, verbose=False):
    """Click the month in year column *col*; return its Download, or None.

    Returns the Playwright ``Download`` on success, or ``None`` when the portal
    shows the "file has not yet been generated" modal (the month's CSV is not
    published yet). Clicking either triggers a download or pops that modal, so
    we race the two: a download event resolves available months quickly, while
    the (shadow-DOM) modal text marks an unavailable one without waiting out
    the full timeout.
    """
    label = _month_label(month, month_style)
    # Month links in this year share the trailing -{col} id suffix; filter to
    # the <a> whose text is the month name (no month name is a substring of
    # another, so the match is unambiguous; year captions are <span>, not
    # a.CaptionLink, so they do not collide).
    link = page.locator(f'a.CaptionLink[id$="-{col}"]').filter(has_text=label)
    if not link.count():
        raise RuntimeError(
            f'Could not find month "{label}" in year {year} (grid column {col}).'
        )

    captured: list = []

    def _on_download(download):
        captured.append(download)

    page.on('download', _on_download)
    try:
        link.first.click()
        for _ in range(max(int(timeout_s / 0.3), 1)):
            if captured:
                if verbose:
                    _log(f'captured download for {year}-{month:02d}')
                return captured[0]
            modal = page.get_by_text(UNAVAILABLE_MODAL_TEXT, exact=False)
            try:
                if modal.count() and modal.first.is_visible():
                    ok = page.get_by_role('button', name='OK', exact=True)
                    if ok.count():
                        ok.first.click()
                    if verbose:
                        _log(f'{year}-{month:02d} not yet generated, skipping')
                    return None
            except Exception:
                pass
            page.wait_for_timeout(300)
    finally:
        page.remove_listener('download', _on_download)

    raise TimeoutError(
        f'No download or "not generated" modal appeared for {year}-{month:02d} '
        f'within {timeout_s}s.'
    )


def _dump_debug_artifacts(page, target_path: Path, verbose: bool = False) -> None:
    """Write a screenshot and page HTML next to *target_path* on failure."""
    try:
        png_path = target_path.with_name(target_path.stem + '_error.png')
        page.screenshot(path=str(png_path), full_page=True)
        html_path = target_path.with_name(target_path.stem + '_error.html')
        html_path.write_text(page.content(), encoding='utf-8')
        if verbose:
            _log(f'wrote debug artifacts to {png_path.parent}')
    except Exception:
        pass
