"""
RegistryIngester — orchestrates Avenu WebForms registry crawling.

Reads a recipe with a ``scraper`` key, iterates MA towns (admin4), looks
up each town's registry base URL from the town_to_registry.csv crosswalk,
crawls monthly date partitions via AvenuAdapter, and saves parquet output
via the standard openplaces path/save machinery.
"""

from __future__ import annotations

import asyncio
import random
import sys
import threading
import warnings
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

from openplaces.io import save_parquet
from openplaces.io.aggregate import (
    aggregate_partitions,
    read_file_metadata,
    read_partition_coverage,
)
from openplaces.io.avenu_adapter import AvenuAdapter
from openplaces.io.avenu_selectors import TRANSACTION_DOC_TYPES
from openplaces.io.readers import get_admin
from openplaces.io.transform import apply_transformations
from openplaces.path import recipe_path
from openplaces.recipe import get_output_path, get_recipe_by_id
from openplaces.timing import Timer


def _month_partitions(year: int) -> list[tuple[str, str]]:
    """Return (date_from, date_to) string pairs for each month of year."""
    partitions = []
    for month in range(1, 13):
        first = date(year, month, 1)
        if month == 12:
            last = date(year, 12, 31)
        else:
            last = date(year, month + 1, 1) - timedelta(days=1)
        partitions.append((first.isoformat(), last.isoformat()))
    return partitions


def _split_date_range(date_from: str, date_to: str) -> list[tuple[str, str]]:
    """Split a date range in half for adaptive subdivision."""
    d0 = date.fromisoformat(date_from)
    d1 = date.fromisoformat(date_to)
    mid = d0 + (d1 - d0) // 2
    if mid == d0:
        return [(date_from, date_to)]
    mid1 = (mid + timedelta(days=1)).isoformat()
    return [(d0.isoformat(), mid.isoformat()), (mid1, d1.isoformat())]


class RegistryIngester:
    """Crawl registry transaction records and save as parquet.

    Parameters
    ----------
    recipe : str or dict
        Recipe ID string or loaded recipe dict. Must contain a ``scraper``
        key with Avenu WebForms configuration.
    admin_ids : str, list, or None
        Admin4 IDs (MA towns) to process. If None, processes all towns
        found in the town_to_registry crosswalk.
    years : int, list of int, or None
        Four-digit calendar year(s) (e.g. 2024) to crawl, in the order given.
        If neither years nor partition_ids is provided, crawls the largest
        available range from ``scraper.first_year`` through the current year,
        newest first.
    partition_ids : str, list of str, or None
        Specific year-months (YYYYMM, e.g. 202401) to crawl.
    reprocess : bool
        If True, re-crawl even when a checkpoint marks a partition done.
    verbose : bool
        Print progress messages.
    """

    def __init__(
        self,
        recipe: str | dict,
        admin_ids: str | list | None = None,
        years: int | list | None = None,
        reprocess: bool = False,
        verbose: bool = False,
        partition_ids: str | list | None = None,
    ) -> None:
        if isinstance(recipe, str):
            recipe = get_recipe_by_id(recipe)
        self.recipe = recipe
        self.admin_ids = [admin_ids] if isinstance(admin_ids, str) else admin_ids
        self.partition_ids = (
            [partition_ids] if isinstance(partition_ids, str) else partition_ids
        )
        self.years = [years] if isinstance(years, int) else years
        if self.partition_ids and self.years:
            raise ValueError('Pass either partition_ids or years, not both.')
        self.reprocess = reprocess
        self.verbose = verbose

        scraper = recipe['scraper']
        self.rate_limit: tuple[float, float] = tuple(
            scraper.get('rate_limit_seconds', [2, 5])
        )
        self.session_refresh_pages: int = scraper.get('session_refresh_pages', 100)
        self.max_attempts: int = scraper.get('retry', {}).get('max_attempts', 3)
        self.backoff: tuple[float, float] = tuple(
            scraper.get('retry', {}).get('backoff_seconds', [2, 10])
        )
        self.fetch_details: bool = scraper.get('fetch_details', False)
        self.timer = Timer(name='registry_ingester')
        # Per-town partition coverage of the aggregated _all file (footer
        # metadata), used by _is_done when a checkpoint file is gone.
        self._coverage_cache: dict[str, set[str]] = {}

        crosswalk_name = scraper.get('crosswalk', 'town_to_registry')
        crosswalk_path = recipe_path(
            recipe['admin_id'], recipe['entity'], filename=crosswalk_name + '.csv'
        )
        self._crosswalk = self._load_crosswalk(crosswalk_path, recipe['admin_id'])

    def ingest(self) -> None:
        """Run the full crawl (synchronous entry point).

        Runs in a dedicated thread so that Playwright's subprocess transport
        works correctly even when called from Jupyter (whose SelectorEventLoop
        does not support ``create_subprocess_exec`` on Windows).
        """
        exc: list[BaseException] = []

        def _run() -> None:
            if sys.platform == 'win32':
                loop = asyncio.ProactorEventLoop()
            else:
                loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._ingest_async())
            except BaseException as e:
                exc.append(e)
            finally:
                loop.close()
                asyncio.set_event_loop(None)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join()
        if exc:
            raise exc[0]

    async def _ingest_async(self) -> None:
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise ImportError(
                'playwright is required for RegistryIngester. '
                'Install with: pip install playwright && playwright install chromium'
            ) from exc

        today = date.today()
        partitions = [
            (df, dt)
            for df, dt in self._date_partitions()
            if date.fromisoformat(df) <= today
        ]
        admin4_ids = self._resolve_admin4_ids()

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=False,
                args=['--window-position=-32000,-32000'],
            )
            for admin4_id in admin4_ids:
                entry = self._crosswalk.get(admin4_id)
                if entry is None:
                    if self.verbose:
                        print(f'No registry found for {admin4_id}, skipping')
                    continue
                town_name = entry['town_name']
                base_url = entry['base_url']
                if self.verbose:
                    print(f'Crawling {town_name} ({admin4_id}) via {base_url}')

                page_count = 0
                context = await browser.new_context()
                adapter = AvenuAdapter(base_url, context)
                await adapter.start_session()

                for doc_type in TRANSACTION_DOC_TYPES:
                    n_checkpointed = 0
                    for date_from, date_to in partitions:
                        ck = f'{date_from[:7]}_{doc_type.replace(" ", "_")}'
                        if self._is_done(admin4_id, ck) and not self.reprocess:
                            n_checkpointed += 1
                            continue
                        records, page_count = await self._crawl_partition(
                            adapter,
                            admin4_id,
                            date_from,
                            date_to,
                            town=town_name.upper(),
                            doc_type=doc_type,
                            page_count=page_count,
                        )
                        records = self._drop_mismatched_records(
                            records, town_name, date_from, date_to, ck
                        )
                        if records is None:
                            continue  # criteria failed; retry next run
                        self._save(records, admin4_id, partition_id=ck)
                        self.timer.mark(f'{admin4_id} {ck}')

                        if page_count >= self.session_refresh_pages:
                            await context.close()
                            context = await browser.new_context()
                            adapter = AvenuAdapter(base_url, context)
                            await adapter.start_session()
                            page_count = 0

                        await asyncio.sleep(random.uniform(*self.rate_limit))

                    if self.verbose and n_checkpointed:
                        print(
                            f'  {doc_type}: skipped {n_checkpointed} '
                            'checkpointed partition(s)'
                        )

                await context.close()
                self._aggregate_town(admin4_id)

    async def _crawl_partition(
        self,
        adapter: AvenuAdapter,
        admin4_id: str,
        date_from: str,
        date_to: str,
        town: str,
        doc_type: str,
        page_count: int,
    ) -> tuple[list[dict], int]:
        """Crawl one date range for one town + doc type.

        If the site returns the ">1000 records" dialog, the date range is
        split in half and each half is crawled recursively until results fit.
        """
        for attempt in range(1, self.max_attempts + 1):
            try:
                dialog_fired = asyncio.Event()

                async def _on_dialog(dialog, _ev=dialog_fired):
                    await dialog.accept()
                    _ev.set()

                adapter.page.on('dialog', _on_dialog)
                try:
                    await adapter.search(
                        date_from, date_to, town=town, doc_type=doc_type
                    )
                    try:
                        await asyncio.wait_for(dialog_fired.wait(), timeout=3.0)
                        # Dialog fired → >1000 results → split date range
                        if date_from == date_to:
                            if self.verbose:
                                print(
                                    f'  {date_from} {doc_type}: '
                                    f'>1000 on single day, skipping'
                                )
                            return [], page_count
                        if self.verbose:
                            print(
                                f'  {date_from} – {date_to} {doc_type}:'
                                f' >1000, splitting'
                            )
                        sub: list[dict] = []
                        for sf, st in _split_date_range(date_from, date_to):
                            part, page_count = await self._crawl_partition(
                                adapter, admin4_id, sf, st, town, doc_type, page_count
                            )
                            sub.extend(part)
                        return sub, page_count
                    except TimeoutError:
                        pass  # no dialog → results fit, parse normally
                finally:
                    adapter.page.remove_listener('dialog', _on_dialog)

                records: list[dict] = []
                while True:
                    page_records = await adapter.parse_results(
                        fetch_details=self.fetch_details
                    )
                    records.extend(page_records)
                    page_count += 1
                    await asyncio.sleep(random.uniform(*self.rate_limit))
                    if not await adapter.has_next_page():
                        break
                    await adapter.next_page()

                if self.verbose:
                    print(
                        f'  {date_from} – {date_to} {doc_type}: {len(records)} records'
                    )
                return records, page_count

            except Exception as exc:
                if attempt == self.max_attempts:
                    raise
                wait = random.uniform(*self.backoff) * (2 ** (attempt - 1))
                if self.verbose:
                    print(
                        f'  attempt {attempt} failed ({exc}), retrying in {wait:.1f}s'
                    )
                await asyncio.sleep(wait)

        return [], page_count

    def _drop_mismatched_records(
        self,
        records: list[dict],
        town: str,
        date_from: str,
        date_to: str,
        partition_id: str,
    ) -> list[dict] | None:
        """Drop crawled rows that do not match the queried town and dates.

        When the registry search silently fails to apply its criteria, the
        result grid shows the portal's default listing (registry-wide,
        current date). Saving that would poison the per-town partition and
        produce duplicates across partitions. Returns the matching records,
        or None when every record mismatches (the partition is then not
        saved and not checkpointed, so the next run retries it).
        """
        if not records:
            return records

        col_map = self.recipe.get('columns', {})
        town_key = col_map.get('town', 'town')
        date_key = col_map.get('recorded_date', 'recorded_date')
        lo, hi = pd.Timestamp(date_from), pd.Timestamp(date_to)

        kept = []
        n_dropped = 0
        for rec in records:
            rec_town = rec.get(town_key)
            if rec_town is not None and str(rec_town).upper() != town.upper():
                n_dropped += 1
                continue
            rec_date = pd.to_datetime(rec.get(date_key), errors='coerce')
            if pd.notna(rec_date) and not (lo <= rec_date <= hi):
                n_dropped += 1
                continue
            kept.append(rec)

        if n_dropped and not kept:
            warnings.warn(
                f'{partition_id}: all {n_dropped} crawled row(s) are for other '
                'towns or dates - the search criteria did not apply (default '
                'registry listing?). Partition not saved; it will be retried '
                'on the next run.',
                stacklevel=2,
            )
            return None
        if n_dropped:
            warnings.warn(
                f'{partition_id}: dropped {n_dropped} crawled row(s) for other '
                f'towns or dates; saving the {len(kept)} matching row(s).',
                stacklevel=2,
            )
        return kept

    def _save(self, records: list[dict], admin4_id: str, partition_id: str) -> None:
        col_map = self.recipe.get('columns', {})
        df = pd.DataFrame(records)
        rename = {v: k for k, v in col_map.items() if v in df.columns}
        df = df.rename(columns=rename)
        for column in col_map:
            if column not in df.columns:
                df[column] = None
        df = apply_transformations(df, self.recipe)
        df['admin4_id'] = admin4_id

        out_path = get_output_path(self.recipe, admin4_id, partition_id=partition_id)

        # Diagnostic (temporary): a transaction should be unique per (book, page).
        # Rather than aborting, dump the full duplicate rows to a CSV beside the
        # parquet so the cause (crawl index desync vs. a legitimate repeat) can be
        # classified, then save the raw frame (duplicates kept) for inspection.
        # Never print the rows: they contain names and addresses.
        _book = 'book' if 'book' in df.columns else None
        _page = 'page' if 'page' in df.columns else None
        if _book and _page and len(df) > 0:
            dup_mask = df.duplicated(subset=[_book, _page], keep=False)
            if dup_mask.any():
                dups = df.loc[dup_mask].sort_values([_book, _page])
                n_pairs = dups.groupby([_book, _page]).ngroups
                dups_path = out_path.with_name(out_path.stem + '_dups.csv')
                dups_path.parent.mkdir(parents=True, exist_ok=True)
                dups.to_csv(dups_path, index=False)
                warnings.warn(
                    f'{n_pairs} duplicate (book, page) pair(s) in {partition_id}; '
                    f'full rows written to {dups_path.name}. Saving raw frame.',
                    stacklevel=2,
                )

        # Stamp the scrape time into the parquet footer so _is_done can tell
        # whether the partition's month was already over when it was scraped
        # (i.e. whether the file holds the complete month).
        save_parquet(
            df,
            out_path,
            file_metadata={'openplaces:scraped_at': datetime.now(UTC).isoformat()},
        )
        if self.verbose and len(df) > 0:
            print(f'  saved {len(df)} rows → {out_path.name}')

    def _date_partitions(self) -> list[tuple[str, str]]:
        """Month (date_from, date_to) pairs to crawl.

        Explicit years or year-month partition IDs are crawled in the order
        given. Without either filter, the largest available range is crawled
        from ``scraper.first_year`` through the current year, newest first.
        """
        if self.partition_ids:
            partitions = []
            for partition_id in dict.fromkeys(str(x) for x in self.partition_ids):
                if (
                    len(partition_id) != 6
                    or not partition_id.isdigit()
                    or not 1 <= int(partition_id[4:]) <= 12
                ):
                    raise ValueError(
                        f"Invalid partition_id '{partition_id}'. Use YYYYMM."
                    )
                year, month = divmod(int(partition_id), 100)
                partitions.append(_month_partitions(year)[month - 1])
            return partitions

        if self.years:
            years = list(dict.fromkeys(self.years))
            invalid = [
                year for year in years if len(str(year)) != 4 or not str(year).isdigit()
            ]
            if invalid:
                raise ValueError(
                    f'Invalid years {invalid}. Use four-digit YYYY values; '
                    'use partition_ids for specific YYYYMM months.'
                )
        else:
            this_year = date.today().year
            first_year = self.recipe['scraper'].get('first_year', this_year)
            years = list(range(this_year, first_year - 1, -1))
        partitions = []
        for year in years:
            partitions.extend(_month_partitions(int(year)))
        return partitions

    def _resolve_admin4_ids(self) -> list[str]:
        if self.admin_ids is not None:
            return self.admin_ids
        return list(self._crosswalk.keys())

    def _town_name_for(self, admin4_id: str) -> str:
        return self._crosswalk[admin4_id]['town_name']

    def _is_done(self, admin4_id: str, checkpoint_key: str) -> bool:
        """Whether a town × month × doc-type partition is completely scraped.

        Done when the checkpoint file exists and its footer scrape timestamp
        postdates the partition month's end — a file scraped while its month
        was still running is incomplete and gets re-scraped (this covers the
        current month without special-casing). Files without the timestamp
        (written before it was introduced) count as done; use reprocess to
        force a re-crawl. A missing file still counts as done when the
        checkpoint is recorded in the aggregated _all file's footer coverage,
        because its rows are already integrated there.
        """
        path = get_output_path(self.recipe, admin4_id, partition_id=checkpoint_key)
        if path.exists():
            scraped_at = read_file_metadata(path).get('openplaces:scraped_at')
            if scraped_at is None:
                return True
            return date.fromisoformat(scraped_at[:10]) > self._month_end(checkpoint_key)
        return checkpoint_key in self._coverage(admin4_id)

    @staticmethod
    def _month_end(checkpoint_key: str) -> date:
        """Last day of the month encoded in a 'YYYY-MM_DocType' key."""
        year, month = int(checkpoint_key[:4]), int(checkpoint_key[5:7])
        if month == 12:
            return date(year, 12, 31)
        return date(year, month + 1, 1) - timedelta(days=1)

    def _coverage(self, admin4_id: str) -> set[str]:
        """Footer partition coverage of the town's aggregated _all file."""
        if admin4_id not in self._coverage_cache:
            all_path = get_output_path(self.recipe, admin4_id, partition_id='all')
            self._coverage_cache[admin4_id] = read_partition_coverage(all_path)
        return self._coverage_cache[admin4_id]

    def _aggregate_town(self, admin4_id: str) -> None:
        """Roll the town's checkpoint files into one _all file.

        Runs when the recipe declares aggregate_by: single_file. Partition
        ids are discovered from the files on disk, so new rows in any
        retained checkpoint file — including a re-scraped current month —
        are integrated (how defaults to union, which never drops rows that
        are only present in the existing _all file).
        """
        agg = self.recipe.get('aggregate_by')
        if not agg or not agg.get('single_file'):
            return
        aggregate_partitions(
            self.recipe,
            single_file=True,
            how=agg.get('how', 'union'),
            admin_ids=[admin4_id],
            keep_original=agg.get('keep_partitions', False),
            verbose=self.verbose,
        )
        self._coverage_cache.pop(admin4_id, None)

    @staticmethod
    def _load_crosswalk(csv_path: Path, admin_id: str) -> dict[str, dict]:
        """Return mapping of admin4_id → {'base_url', 'town_name'}."""
        import csv

        raw: dict[str, str] = {}
        with open(csv_path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                raw[row['town_name']] = row['base_url']

        admin4_df = get_admin(admin_id, level=4)
        crosswalk: dict[str, dict] = {}
        for idx, row in admin4_df.iterrows():
            name = row['name']
            # Census data appends " city" (and sometimes " Town city") to
            # incorporated cities; strip those suffixes for CSV lookup.
            if str(row.get('type', '')).lower() == 'city':
                if name.lower().endswith(' city'):
                    name = name[:-5]
                if name.lower().endswith(' town'):
                    name = name[:-5]
            if name in raw:
                crosswalk[idx] = {'base_url': raw[name], 'town_name': name}
        return crosswalk
