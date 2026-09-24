"""
Pipeline step turning assessor last-sale fields into transaction rows:
  - append_last_sales: read the admin unit's ingested property and parcel
    tables and append one transaction row per recorded last sale

An assessment roll commonly carries the most recent sale of each property
(price, date, sometimes a qualification code and the deed's book and
page). Where no recorder's feed exists, those fields are the only public
record of prices, so this step reshapes them into the transaction
vocabulary and marks each row `sale_record_kind = 'assessor_last_sale'`.

What the lane is, and is not. A last-sale field holds the most recent sale
only: earlier sales of the same property are gone, and which sale is "the
last" depends on the day the roll was pulled. Rows built from it support
cross-sectional hedonic work and recent price levels. They do not support
a repeat-sales index.

Why a reshape and nothing more. The row is born on the property or parcel
record, so its parcel key is that record's own `parcel_id_local`: no
matching happens here, exact or otherwise, and nothing is scored, tuned
or unlinked. Rows are not grouped to guess which parcels shared a deed
either; a deed's book and page are carried where the roll has them and
the curate stage counts parcels per document from those alone.

Only an allow-list of columns is read from a roll. Owner names and mailing
addresses, which many rolls carry, never enter a transaction row.
"""

from __future__ import annotations

import warnings

import pandas as pd

from openplaces.io.harmonizer import (
    HarmonizeState,
    _register,
    restrict_to_admin_by_name,
)
from openplaces.io.harmonizer.spine import _expand_auto_discover
from openplaces.io.readers import get_entities

ASSESSOR_LAST_SALE = 'assessor_last_sale'
DEED = 'deed'

# Roll column to transaction column, for the columns that move unchanged.
_RENAMES = {
    'last_sale_qualification_code': 'sale_qualification_code',
    'last_sale_vacant': 'sale_vacant',
    'last_sale_doc_type': 'doc_type',
    'last_sale_book': 'sale_book',
    'last_sale_page': 'sale_page',
}

# Identifiers and location a roll row may lend its sale. No owner fields.
_CARRY = (
    'parcel_id_local',
    'parcel_id_assessor',
    'parcel_id_tax',
    'parcel_id_admin2',
    'property_id_local',
    'property_id_assessor',
    'address',
    'city',
    'use_group_code',
)

# A numeric date above this magnitude is epoch milliseconds (1e11 ms is
# 1973-03; as seconds it would be the year 5138), below it a calendar
# number such as 20120817. Esri's f=geojson serializes date fields as
# epoch milliseconds, and nothing casts them at ingest, so eleven North
# Carolina county layers store last_sale_date as floats like
# 1345161600000.0 (measured 2026-09-20).
_EPOCH_MS_MIN = 1e11

# Dated rows a table needs before "every date is the first of a month"
# is read as month-only precision (see last_sales_to_transactions).
_MONTH_ONLY_MIN_ROWS = 30


def parse_sale_dates(values: pd.Series) -> pd.Series:
    """Parse a roll's sale dates, whatever encoding ingest left them in.

    Handles the four encodings found on disk: parsed datetimes (timezone
    aware or not), strings, epoch-millisecond numbers, and eight-digit
    calendar numbers (YYYYMMDD). Zero and negative numbers are a roll's
    "no sale" placeholder and parse to missing, not to 1970-01-01.

    Parameters
    ----------
    values : pandas.Series
        Sale dates as ingested.

    Returns
    -------
    pandas.Series
        Timezone-naive datetimes, missing where the value does not parse.
    """
    if pd.api.types.is_datetime64_any_dtype(values):
        parsed = values
    elif pd.api.types.is_numeric_dtype(values):
        numbers = pd.to_numeric(values, errors='coerce').astype('float64')
        numbers = numbers.where(numbers > 0)
        is_ms = numbers >= _EPOCH_MS_MIN
        from_ms = pd.to_datetime(numbers.where(is_ms), unit='ms', errors='coerce')
        calendar = numbers.where(~is_ms & numbers.between(10000101, 99991231))
        text = calendar.map(lambda v: f'{int(v):08d}', na_action='ignore')
        from_calendar = pd.to_datetime(text, format='%Y%m%d', errors='coerce')
        parsed = from_ms.fillna(from_calendar)
    else:
        text = values.astype('string').str.strip()
        # format='mixed': pandas otherwise fixes the format from the
        # first value and turns every differently spelled date into NaT.
        parsed = pd.to_datetime(text, errors='coerce', utc=True, format='mixed')
    parsed = pd.to_datetime(parsed, errors='coerce')
    if getattr(parsed.dt, 'tz', None) is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed


def _blank_to_missing(values: pd.Series) -> pd.Series:
    """Read an empty or whitespace-only text value as missing.

    Florida's roll fills book and page with a single space where it has
    none; carried forward as text, that space reads as a value every
    row shares. Non-text columns pass through unchanged.
    """
    if not (
        pd.api.types.is_string_dtype(values) or pd.api.types.is_object_dtype(values)
    ):
        return values
    blank = values.astype('string').str.strip().eq('').fillna(False)
    return values.where(~blank)


def _split_book_page(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Split a combined book-and-page reference into its two parts.

    A reference splits only where it holds exactly two tokens separated
    by a slash, hyphen or whitespace. Anything else is left missing on
    both sides: a wrong document id would join unrelated sales in the
    curate stage's parcel count, a missing one only leaves it unknown.
    """
    text = values.astype('string').str.strip()
    parts = text.str.split(r'\s*[/\-\s]\s*', regex=True)
    two = parts.map(lambda p: isinstance(p, list) and len(p) == 2 and all(p))
    two = two.fillna(False).astype(bool)
    book = parts.where(two).map(lambda p: p[0], na_action='ignore')
    page = parts.where(two).map(lambda p: p[1], na_action='ignore')
    return book.astype('string'), page.astype('string')


def last_sales_to_transactions(
    table: pd.DataFrame,
    label: str,
    min_year: int = 1800,
) -> pd.DataFrame:
    """Reshape a roll's last-sale fields into transaction rows.

    A row is emitted where the roll records both a price and a real date.
    "Real" means a `last_sale_date` that parses, or, where the roll has
    no full date, a `last_sale_year` of at least *min_year* (the month is
    optional). Florida's parcel layer is why this is not "non-null": it
    fills every last-sale column on every row, and 89% of a sampled
    county's rows hold price 0 and year 0. A price of 0 beside a real
    date is a recorded nominal transfer and is kept; a missing price is
    not a sale this lane can use (maintainer's decision, 2026-09-20).

    Parameters
    ----------
    table : pandas.DataFrame
        An ingested parcel or property table.
    label : str
        Source label written to the `source` column.
    min_year : int, optional
        Earliest year accepted as a real sale year. Default 1800.

    Returns
    -------
    pandas.DataFrame
        One row per recorded last sale, in the transaction vocabulary,
        with a fresh integer index. Empty when the table has no price
        column or no date or year column.
    """
    has_date = 'last_sale_date' in table.columns
    has_year = 'last_sale_year' in table.columns
    if 'last_sale_price' not in table.columns or not (has_date or has_year):
        return pd.DataFrame()

    out = pd.DataFrame(index=table.index)
    out['price'] = pd.to_numeric(table['last_sale_price'], errors='coerce')

    max_year = pd.Timestamp.now().year + 1
    date = (
        parse_sale_dates(table['last_sale_date'])
        if has_date
        else pd.Series(pd.NaT, index=table.index)
    )
    date = date.where(date.dt.year.between(min_year, max_year))
    year = date.dt.year.astype('float64')
    month = date.dt.month.astype('float64')
    if has_year:
        roll_year = pd.to_numeric(table['last_sale_year'], errors='coerce')
        roll_year = roll_year.where(roll_year.between(min_year, max_year))
        year = year.fillna(roll_year)
        if 'last_sale_month' in table.columns:
            roll_month = pd.to_numeric(table['last_sale_month'], errors='coerce')
            roll_month = roll_month.where(roll_month.between(1, 12))
            # Only beside the roll's own year: a month from the roll
            # under a year from the date would mix two statements.
            month = month.fillna(roll_month.where(date.isna()))
    # A roll that records the month only reaches us as first-of-month
    # dates: all 73,253 of Pitt County NC's last-sale dates fall on day
    # 1, and its recorded deeds for the same sales follow 0 to 30 days
    # later (measured 2026-09-20). A day nobody recorded is not written:
    # year and month stay, the date is left missing. A source with real
    # days puts about 1 date in 30 on the first, so "every one of at
    # least 30" cannot happen by chance.
    dated = date.dropna()
    if len(dated) >= _MONTH_ONLY_MIN_ROWS and (dated.dt.day == 1).all():
        date = pd.Series(pd.NaT, index=table.index)
    out['recorded_date'] = date
    out['sale_year'] = year
    out['sale_month'] = month

    for src, dst in _RENAMES.items():
        if src in table.columns:
            out[dst] = _blank_to_missing(table[src])
    if 'last_sale_book_page' in table.columns and 'sale_book' not in out:
        out['sale_book'], out['sale_page'] = _split_book_page(
            table['last_sale_book_page']
        )
    for col in _CARRY:
        if col in table.columns:
            out[col] = table[col]

    keep = out['price'].notna() & out['sale_year'].notna()
    out = out.loc[keep]
    # A roll ingested for several years states a property's last sale
    # once per roll year until the property sells again: Osceola County
    # FL's multi-year DOR roll produced 1,290,606 last-sale rows beside
    # 616,551 deeds (measured 2026-09-21). The same sale of the same
    # property is one row. What is kept across years is the sequence of
    # *different* last sales, which is real history a single roll year
    # cannot give. Identity is the record's own ids plus date and
    # price; columns that drift between roll years (a use code, an
    # address spelling) are deliberately not part of it.
    identity = [
        c
        for c in (
            'parcel_id_local',
            'parcel_id_assessor',
            'property_id_local',
            'property_id_assessor',
            'parcel_id_admin2',
            'sale_year',
            'sale_month',
            'price',
            'sale_book',
            'sale_page',
        )
        if c in out.columns
    ]
    id_columns = [c for c in identity if c.endswith(('_local', '_assessor', 'admin2'))]
    if id_columns:
        # A row carrying no identifier at all is never merged: two
        # unnamed sales in one month at one price are still two sales.
        named = out[id_columns].notna().any(axis=1)
        repeat = out.duplicated(subset=identity, keep='last') & named
        out = out.loc[~repeat]
    out = out.reset_index(drop=True)
    out['sale_record_kind'] = ASSESSOR_LAST_SALE
    out['source'] = label
    return out


@_register('append_last_sales', phase='geometry')
def append_last_sales(
    state: HarmonizeState,
    entity_types: list[str] | None = None,
    min_year: int = 1800,
) -> HarmonizeState:
    """Append assessor last-sale rows to a transaction spine.

    Discovers the ingested tables of each entity type that cover the
    admin unit, exactly as `union_spine_sources` discovers transaction
    sources, and appends the rows :func:`last_sales_to_transactions`
    builds from them. Every spine row ends up with a `sale_record_kind`:
    `deed` for the rows already there, `assessor_last_sale` for these.
    The two kinds are kept apart on purpose, including where a county
    has both for the same sale; preferring one is a curate decision.

    Entity types are read in the order given, sources within a type most
    specific and newest first, and a later source adds a sale only for a
    `parcel_id_local` no earlier source supplied. With the
    default order a condominium unit's sale comes from its property row,
    and a parcel layer repeating the same roll (Florida's does) adds
    nothing twice. Rows without a `parcel_id_local` are always kept.

    Registered in the geometry phase because, like
    `union_spine_sources`, it adds spine rows.

    Parameters
    ----------
    entity_types : list of str, optional
        Entity types to read, in priority order. Default
        `['property', 'parcel']`.
    min_year : int, optional
        Earliest year accepted as a real sale year. Default 1800.
    """
    entity_types = entity_types or ['property', 'parcel']

    parts = []
    seen_keys: set = set()
    for entity_type in entity_types:
        sentinel = [{'auto_discover': True, 'entity_type': entity_type}]
        # Discovery orders sources most specific first, newest version
        # first, so a county layer wins over a statewide one and a 2026
        # layer over the 2025 layer it replaced.
        for src in _expand_auto_discover(sentinel, state):
            recipe_id = src['recipe_id']
            label = src.get('label', recipe_id)
            try:
                table = get_entities(
                    recipe_id,
                    state.admin_id,
                    layer=src.get('layer'),
                    missing='ignore',
                )
            except Exception as exc:
                warnings.warn(
                    f'append_last_sales: could not load {recipe_id} for '
                    f'{state.admin_id}: {exc}'
                )
                continue
            table = restrict_to_admin_by_name(table, recipe_id, state.admin_id)
            if table is None or len(table) == 0:
                continue
            table = pd.DataFrame(table.drop(columns='geometry', errors='ignore'))
            rows = last_sales_to_transactions(table, label, min_year=min_year)
            if rows.empty:
                continue
            if 'parcel_id_local' in rows.columns and seen_keys:
                key = rows['parcel_id_local']
                rows = rows.loc[key.isna() | ~key.isin(seen_keys)]
            if rows.empty:
                continue
            parts.append(rows)
            if 'parcel_id_local' in rows.columns:
                seen_keys.update(rows['parcel_id_local'].dropna())
            if state.verbose:
                print(f'  Last sales from {label} ({entity_type}): {len(rows):,d} rows')

    spine = state.spine
    if spine is not None and len(spine):
        spine = spine.copy()
        if 'sale_record_kind' not in spine.columns:
            spine['sale_record_kind'] = DEED
        else:
            spine['sale_record_kind'] = spine['sale_record_kind'].fillna(DEED)
        parts = [spine, *parts]
    if not parts:
        return state

    # The roll recipes read here are deliberately not added to
    # metadata['spine_source_recipe_ids']: link_by_id skips columns of a
    # recipe listed there as already carried by the spine, and these rows
    # carry none of the roll's parcel attributes.
    state.spine = pd.concat(parts, ignore_index=True, sort=False)
    if state.timer:
        state.timer.mark('Last sales')
    if state.verbose:
        n_lane = int((state.spine['sale_record_kind'] == ASSESSOR_LAST_SALE).sum())
        print(f'  append_last_sales: {n_lane:,d} assessor last-sale rows')
    return state
