"""Frame operations that put recorded sales into the transaction entity's
terms: the sale period, the document behind a row, exact repeats of one
document, the fold of a multi-parcel deed into one row, and the sale's id.

They live below the stages because two stages call them. Until
2026-09-29 they were curate steps; the transaction spine runs them now
(plans/core-schema-and-stage-contracts-review.md, section 7: "Transaction
ids move from curate to the transaction spine, and no stage re-mints an
id after de-duplication"), and the curate steps of the same names remain
as thin wrappers for a recipe that still lists them. Every function takes
and returns a plain frame and decides nothing a recipe did not state.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from openplaces.table import add_unique_suffix, normalize_issued_id

_NON_ALNUM = re.compile(r'[^0-9A-Za-z]')

# How a sale with no recorded document is fingerprinted, narrowest
# first. Each tier is added only if the ones before it leave two such
# sales sharing a fingerprint, because **every column in the hash is a
# column that can change the id**: these ids are published, and a sale
# re-ingested with a corrected use code or parcel key should keep its
# name. What a sale is, before anything else, is a price on a date.
#
# The parcel keys come next because two sales at one price in one
# month are nearly always different properties, and the record kind
# last because it separates a deed from the assessor's echo of it,
# which the duplicate drop and `flag_sales_matching_other_kind` have
# usually already dealt with.
TRANSACTION_FINGERPRINT_TIERS = (
    ('sale_year', 'sale_month', 'price'),
    ('parcel_id_local', 'parcel_id_assessor'),
    ('sale_record_kind',),
)

# Above this share of rows sharing a document with another, the column
# is not identifying sales and the ids stop being references anyone
# could look up. Matches the spirit of `NO_ACCOUNT_NUMBER_SHARE` for
# properties.
REPEATED_DOCUMENT_SHARE = 0.02


def normalized_text(series: pd.Series) -> pd.Series:
    """Strip every non-alphanumeric character, for a key compared as text."""
    return series.astype(str).str.replace(_NON_ALNUM, '', regex=True)


def derive_sale_period(frame: pd.DataFrame, date_column: str = 'recorded_date'):
    """Fill `sale_year` and `sale_month` from a date where they are missing.

    Sources state when a sale happened in one of two ways: a date (a
    recorder's table) or a year and month (Florida's roll, and any roll
    that records the month only). Every later step compares sales by
    year and month, so both spellings are brought to that form here. A
    year or month the source stated itself is never overwritten. A frame
    without *date_column* is returned as it is.
    """
    if date_column not in frame.columns:
        return frame
    date = pd.to_datetime(frame[date_column], errors='coerce')
    for column, part in (('sale_year', date.dt.year), ('sale_month', date.dt.month)):
        derived = part.astype('float64')
        if column in frame.columns:
            stated = pd.to_numeric(frame[column], errors='coerce')
            derived = stated.fillna(derived)
        frame[column] = derived
    return frame


def duplicate_document_mask(
    frame: pd.DataFrame, key_columns: list, normalize_columns: list | None = None
) -> pd.Series:
    """True on every row after the first with the same *key_columns* values.

    Sale sources published as overlapping rolling windows (FL DOR's SDF)
    carry the same legal transaction in two adjacent files. A source's
    own transaction id cannot detect this (assigned per county, not
    unique across the source's coverage); the composite key identifies
    the underlying document instead. A column the frame lacks reads as
    missing on every row, so one key serves every source. Compared as
    strings with one placeholder for missing through the frame's own
    hashed `duplicated`, which is vectorized: a row-wise join of the key
    took 83 of Lake County FL's 111 s transaction curate (2026-09-29).

    *normalize_columns* names key columns compared without punctuation
    and case (`normalized_text`, upper-cased), so that a number spelled
    `01-18-24` in one file and `011824` in the next still reads as one.
    Florida's DOR rewrote the punctuation of every parcel number between
    two adjacent SDF years in 23 counties while the clerk instrument,
    price and date stayed the same; compared as spelled, both copies of
    each deed survived, and the multi-parcel fold then blanked the
    deed's parcel number because its two copies "disagreed". Measured
    2026-10-24 on US-FL-BA: in sale year 2024, the raw spelling differed
    on 9,111 of 9,111 deeds listed in both files, the stripped number on
    0.2%. The comparison is still exact, on a deterministic rewrite of
    the string: no tolerance, no similarity score.
    """
    key = frame.reindex(columns=list(key_columns)).astype('string')
    for column in normalize_columns or []:
        if column in key.columns:
            stated = key[column].notna()
            key[column] = (
                normalized_text(key[column]).astype('string').str.upper().where(stated)
            )
    return key.fillna('NA').duplicated(keep='first')


def derive_document_id(
    frame: pd.DataFrame,
    candidates: list,
    output: str = 'sale_document_id',
    separator: str = '/',
    scope_column: str | None = None,
    unscoped_value: str | None = None,
    strip_leading_zeros: bool = False,
) -> pd.DataFrame:
    """Name the recorded document (deed) behind each sale row.

    A deed covering several parcels is published once per parcel, and a
    consumer that treats each row as a sale counts one transfer several
    times. The document identifier is what lets the rows be recognized
    as one sale, and every source spells it differently, which is why
    it is assembled here once rather than in every consumer.

    Parameters
    ----------
    candidates : list of list of str
        Column groups to try in order; the first whose columns are all
        present on a row names its document. Florida supplies a clerk
        instrument number for newer sales and an official-record book
        and page for older ones, so ``[[sale_clerk_instrument],
        [sale_book, sale_page]]``. Values are stripped of punctuation
        and joined by *separator*. A part that is empty or all zeros is
        a placeholder, not an identifier: Glades County FL's roll writes
        a single space for book and page on every row, which named one
        document for all 1,333 of its last sales.
    output : str, optional
        Column written. Missing where no candidate is complete.
    separator : str, optional
        Joins a multi-column candidate.
    scope_column : str, optional
        Column whose value is prefixed to the identifier (`value:id`),
        so that rows of different scopes never share one. With
        `sale_record_kind`, a deed and the assessor's last-sale record
        citing that deed's book and page stay two rows.
    unscoped_value : str, optional
        The *scope_column* value left unprefixed (`deed`), so identifiers
        already published for it do not change.
    strip_leading_zeros : bool, optional
        Write an all-digit part without its leading zeros (default
        False). A registry and an assessor can pad one book and page
        differently, and then the deed and the roll row citing it name
        two documents: Nantucket's crawled deeds read `646/141` where its
        MassGIS rows read `00313/0278`. Off by default because it changes
        the identifier a source's sales are published under.
        Measured 2026-10-19 on US-MA-NTC: 4,225 of 5,815 last sales then name a deed.
    """
    document = pd.Series(pd.NA, index=frame.index, dtype='string')
    for group in candidates:
        columns = [group] if isinstance(group, str) else list(group)
        if any(c not in frame.columns for c in columns):
            continue
        parts = []
        for c in columns:
            part = normalized_text(frame[c]).astype('string')
            placeholder = part.str.fullmatch('0*').fillna(True)
            if strip_leading_zeros:
                digits = part.str.fullmatch(r'\d+').fillna(False)
                part = part.where(~digits, part.str.lstrip('0'))
            parts.append(part.where(frame[c].notna() & ~placeholder))
        complete = pd.concat(parts, axis=1).notna().all(axis=1)
        joined = parts[0]
        for part in parts[1:]:
            joined = joined + separator + part
        document = document.where(document.notna() | ~complete, joined)
    if scope_column is not None and scope_column in frame.columns:
        scope = frame[scope_column].astype('string')
        scoped = document.notna() & scope.notna()
        if unscoped_value is not None:
            scoped &= (scope != unscoped_value).fillna(False)
        document = document.where(~scoped, scope + ':' + document)
    frame[output] = document
    return frame


def count_parcels_per_document(
    frame: pd.DataFrame,
    parcel_column: str | list[str],
    document_column: str = 'sale_document_id',
    output: str = 'n_parcels_per_sale',
) -> pd.DataFrame:
    """Count the distinct parcels each recorded document covered, on every row.

    *parcel_column* names the column identifying a row's parcel, or
    several in order of preference, a row taking the first it has a
    value in. Count on the normalized number before the assessor's own
    spelling: a source published over several years spells one parcel
    two ways (Lake County FL's deed feed writes `01-18-24-...` in one
    vintage and the bare digits in another), and counted on the
    spelling those rows read as a two-parcel deed. Measured 2026-10-01
    on US-FL-LA: 119,027 of 157,087 multi-parcel documents were one
    parcel under two spellings.

    Run after the source's duplicate rows are removed, or a document
    listed twice for the same parcel counts as two. A frame lacking the
    document column, or every parcel column, is returned as it is.
    """
    if isinstance(parcel_column, str):
        parcel_column = [parcel_column]
    present = [c for c in parcel_column if c in frame.columns]
    if document_column not in frame.columns or not present:
        return frame
    key = frame[present[0]].astype('string')
    for column in present[1:]:
        key = key.fillna(frame[column].astype('string'))
    counts = key.groupby(frame[document_column], dropna=True).nunique()
    frame[output] = frame[document_column].map(counts).astype('float')
    return frame


def _agree_per_group(members: pd.DataFrame, by: str) -> pd.DataFrame:
    """One row per group: a column's value where every member states the
    same one, missing where they differ.

    The rule the stacked-units split applies to a lot's units and the
    curate stage to a parcel's properties (`aggregation.AGREE`): a deed
    of three parcels has one price and one date, which every row repeats,
    and three parcel numbers, which no single cell can hold.
    """
    grouped = members.groupby(by, sort=False, dropna=True)
    first = grouped.first()
    distinct = grouped.nunique(dropna=True)
    out = first.where(distinct <= 1)
    # `first` skips missing values; a column every member leaves empty
    # has distinct 0 and stays empty, which `where` preserves.
    return out


def aggregate_multi_parcel_sales(
    frame: pd.DataFrame,
    document_column: str = 'sale_document_id',
    weight_column: str | None = None,
    count_column: str = 'n_parcels_per_sale',
) -> tuple[pd.DataFrame, int, int, pd.DataFrame]:
    """Emit one row per recorded document instead of one per parcel.

    The rows of a multi-parcel deed are right about the price and wrong
    about the parcel: the source repeats one price on every parcel the
    deed conveyed, and the parcels differ. Grouped by document, the rows
    become one sale that keeps what every member states alike (the
    price, the dates, the document, the parties' kind) and nothing a
    member states for its parcel alone: a column whose members disagree
    is left missing on the folded row. The parcels themselves become
    rows of the transaction-to-parcel link table, written by
    `link_transactions` from the *members* this function returns, which
    is where a reader finds the deed's parcels and their shares. Until
    2026-10-01 the folded row took its other columns from the member
    with the largest *weight_column*, which named one of several unlike
    parcels as if it were the deed's (the maintainer's objection).

    Rows with no document id, or alone on theirs, are kept as they are.
    The folded row keeps the index label of its first member (or of the
    heaviest, when *weight_column* is given), and *count_column* is kept
    as that member's value. Returns the frame, the number of rows
    folded, the number of documents they became, and the member rows
    themselves (every column, plus the document they fold into), empty
    where nothing folded.
    """
    empty = frame.iloc[0:0]
    if document_column not in frame.columns:
        return frame, 0, 0, empty
    has_document = frame[document_column].notna()
    shared = has_document & frame[document_column].duplicated(keep=False)
    if not shared.any():
        return frame, 0, 0, empty
    members = frame.loc[shared].copy()
    if weight_column is not None and weight_column in members.columns:
        members = members.sort_values(weight_column, ascending=False, kind='stable')
    label = members.index.to_series().groupby(members[document_column]).first()
    folded = _agree_per_group(members, document_column)
    folded[document_column] = folded.index
    if count_column in members.columns:
        # The count is the same on every member; keep it even where a
        # member's copy is missing.
        first_count = members.groupby(document_column, sort=False)[count_column].first()
        folded[count_column] = first_count.reindex(folded.index)
    folded.index = pd.Index(
        label.reindex(folded.index).to_numpy(), name=frame.index.name
    )
    kept = frame.loc[~shared]
    result = pd.concat([kept, folded.reindex(columns=kept.columns)])
    # Back in input order, so the step is stable under a later sort.
    result = result.loc[frame.index[frame.index.isin(result.index)]]
    return result, int(shared.sum()), len(folded), members


def mint_transaction_ids(
    frame: pd.DataFrame,
    admin: str,
    document_column: str = 'sale_document_id',
    label: str = 'sale',
) -> tuple[pd.DataFrame, dict]:
    """Index the sales by a stable id, scoped to the admin unit.

    The id is the recorded document, scoped by the admin unit that
    issued it (`US-FL-MD_<document>`), because document numbers repeat
    between counties. Measured 2026-09-23, a document identifies a sale
    wherever the source names one: unique on 100% of Dane County WI and
    Alachua County FL rows, and on 94.8% of Miami-Dade's, whose other
    5.2% name no document at all. Those rows are fingerprinted instead,
    the same fallback `assign_entity_ids` uses for a property whose
    assessor issued no account number, but over a deliberately minimal
    column set (`TRANSACTION_FINGERPRINT_TIERS`): price and date first,
    and a further tier only where that leaves two unnamed sales sharing
    a fingerprint. Every surviving row gets its own id (`_2`, `_3` where
    two are named alike), because an index that merged two rows would
    have the delivery's de-duplication drop one silently.

    Positional throughout: the frame can carry a duplicate index. Returns
    the re-indexed frame (index `transaction_id`) and a report with
    `n_unnamed`, `tiers_used`, `n_repeated_documents` and whether
    *document_column* was present.
    """
    positional = frame.reset_index(drop=True)
    has_document = document_column in positional.columns
    issued = (
        normalize_issued_id(positional[document_column])
        if has_document
        else pd.Series(pd.NA, index=positional.index, dtype='string')
    )
    ids = (f'{admin}_' + issued).astype('string')

    unnamed = issued.isna()
    tiers_used = 0
    if unnamed.any():
        # Widen the fingerprint only while two unnamed sales still share
        # one. Escalation is over the whole unit rather than per row, so
        # an id depends on the columns used, not on which other rows
        # happen to be present.
        columns: list[str] = []
        fingerprint = None
        for tier in TRANSACTION_FINGERPRINT_TIERS:
            present = [c for c in tier if c in positional.columns]
            if not present:
                continue
            columns += present
            tiers_used += 1
            fingerprint = pd.util.hash_pandas_object(
                positional.loc[unnamed, columns].astype('string'), index=False
            )
            if not fingerprint.duplicated().any():
                break
        if fingerprint is not None:
            named = fingerprint.map('{:016x}'.format).astype('string')
            ids.loc[unnamed] = f'{admin}_{label}:' + named

    named_rows = ~unnamed
    n_repeated = (
        int(ids[named_rows].duplicated(keep=False).sum()) if named_rows.any() else 0
    )
    ids = add_unique_suffix(ids)
    out = positional.copy()
    out.index = pd.Index(ids.to_numpy(), name='transaction_id')
    report = {
        'has_document_column': has_document,
        'n_unnamed': int(unnamed.sum()),
        'tiers_used': tiers_used,
        'n_repeated_documents': n_repeated,
        'n_rows': len(out),
    }
    return out, report


def nominal_price_floor(
    values: pd.Series,
    cap: float = 5000,
    min_share: float = 0.05,
    isolation: float = 20,
) -> float:
    """The price at or below which a source's values are a recording convention.

    A nominal value is a mass point: one price carrying at least
    *min_share* of all rows and at least *isolation* times the rows of
    every other price within ten percent of it. The floor is the largest
    mass point at or below *cap*, or 0 where there is none. One number
    per call, never a level per period: a floor that moved with time
    would drift toward a price index, which plans/design-limits (L7)
    keeps out of every keep-which-record decision. The curate indicator
    `nominal_floor` and `duplicate_sale_pairs` both read it from here,
    so the floor a pair was judged by is the one the curated table
    publishes.

    Parameters
    ----------
    values : pandas.Series
        Prices; non-numeric values read as missing.
    cap : float, optional
        Largest value that can be a floor (default 5,000).
    min_share : float, optional
        Share of all rows a mass point needs (default 0.05).
    isolation : float, optional
        How many times the rows of its neighbors a mass point needs
        (default 20).

    Returns
    -------
    float
    """
    values = pd.to_numeric(values, errors='coerce')
    low = values[(values >= 0) & (values <= cap)]
    if low.empty:
        return 0.0
    counts = low.value_counts()
    floor = 0.0
    for value, count in counts.items():
        if count < min_share * len(values):
            break  # value_counts is sorted descending
        near = low[(low > value * 0.9 - 1) & (low < value * 1.1 + 1)]
        neighbors = len(near) - count
        if neighbors * isolation <= count:
            floor = max(floor, float(value))
    return floor


# Labels of the transaction-to-transaction pair table, in the order a
# row's summary resolves them (plans/transaction-sale-deduplication.md,
# section 4). Each names one exact rule; none is a score (L17).
#: A roll's last sale citing a deed's document.
DEED_ECHO_DOCUMENT = 'deed_echo_document'
#: A last sale stating a deed's thing sold, year, month and price.
DEED_ECHO_PERIOD_PRICE = 'deed_echo_period_price'
#: Two last-sale rows stating one thing sold, year, month and price.
ROLL_RESTATED = 'roll_restated'
#: Two deeds, one thing sold, one market price, within the window.
DOUBLE_CLOSING = 'double_closing'
#: A return stating the document it corrects (Wisconsin's RETR).
RESTATES_DOCUMENT = 'restates_document'
#: Two deeds, one thing sold, different market prices, in the window.
RESALE_WITHIN_WINDOW = 'resale_within_window'

DUPLICATE_LABELS = (
    DEED_ECHO_DOCUMENT,
    DEED_ECHO_PERIOD_PRICE,
    ROLL_RESTATED,
    DOUBLE_CLOSING,
    RESTATES_DOCUMENT,
    RESALE_WITHIN_WINDOW,
)
#: Labels that do not make a row a duplicate: both sales stand.
NOT_DUPLICATE_LABELS = (RESALE_WITHIN_WINDOW,)

#: The pair table's columns, and nothing else, so it can never carry a
#: person (as the other link tables).
DUPLICATE_PAIR_COLUMNS = (
    'transaction_id',
    'other_transaction_id',
    'link_method',
    'link_source',
)

# Link methods that never make two sales "the same thing sold". A lot
# reached through one of its units is shared by every unit on it: two
# condominium units of one lot sold in one month at one price are two
# properties (plans/transaction-sale-deduplication.md, 3.6, OR 8,609
# such sales). A key shared by very many rows is left out of every
# reader of a link table for the same reason.
_NOT_THE_THING_SOLD = ('stacked_units', '_shared_key')


def sale_things(
    parcel_links: pd.DataFrame | None, property_links: pd.DataFrame | None
) -> pd.DataFrame:
    """What each sale conveyed, as one id per (sale, thing) pair.

    The thing sold is the linked property where the sale names one, else
    the linked parcel, never a lot reached through `stacked_units` and
    never a pair found on a shared key (decision D3). A property and a
    parcel are different kinds of thing, so their ids are kept apart by
    a prefix: a sale reaching only a parcel is never the same thing as
    one reaching only a property.

    Parameters
    ----------
    parcel_links, property_links : pandas.DataFrame or None
        The transaction spine's two link tables (`link_transactions`).

    Returns
    -------
    pandas.DataFrame
        Columns `transaction_id` and `thing`, both text, one row per pair.
    """
    frames = []
    has_property: set = set()
    for links, id_column, kind in (
        (property_links, 'property_id', 'property'),
        (parcel_links, 'parcel_id', 'parcel'),
    ):
        if links is None or not len(links) or id_column not in links.columns:
            continue
        links = links[['transaction_id', id_column, 'link_method']].dropna(
            subset=['transaction_id', id_column]
        )
        method = links['link_method'].astype('string').fillna('')
        excluded = pd.Series(False, index=links.index)
        for marker in _NOT_THE_THING_SOLD:
            excluded |= method.str.contains(marker, regex=False)
        links = links[~excluded.to_numpy()]
        tid = links['transaction_id'].astype('string')
        if kind == 'parcel':
            keep = ~tid.isin(has_property)
            links, tid = links[keep.to_numpy()], tid[keep.to_numpy()]
        else:
            has_property = set(tid)
        frames.append(
            pd.DataFrame(
                {
                    'transaction_id': tid.to_numpy(),
                    'thing': (
                        f'{kind}:' + links[id_column].astype('string')
                    ).to_numpy(),
                }
            )
        )
    if not frames:
        return pd.DataFrame({'transaction_id': [], 'thing': []}, dtype='string')
    out = pd.concat(frames, ignore_index=True).astype('string')
    return out.drop_duplicates(ignore_index=True)


def _pair_frame(left, right, label) -> pd.DataFrame:
    return pd.DataFrame(
        {
            'transaction_id': pd.Series(left, dtype='string'),
            'other_transaction_id': pd.Series(right, dtype='string'),
            'link_method': label,
        }
    )


def duplicate_sale_pairs(
    sales: pd.DataFrame,
    things: pd.DataFrame,
    nominal_floor: float = 0.0,
    *,
    kind_column: str = 'sale_record_kind',
    deed_kind: str = 'deed',
    last_sale_kind: str = 'assessor_last_sale',
    document_column: str = 'sale_document_id',
    date_column: str = 'recorded_date',
    previous_document_column: str | None = None,
    correction_column: str | None = None,
    correction_values: list | tuple = (),
) -> pd.DataFrame:
    """Pairs of sale rows that describe one sale, or two sales close in time.

    Every rule compares facts the documents state, exactly: a document
    reference, the year and month (and the day, where both rows state
    one), the price as stated, and the thing sold (`sale_things`). No
    name, no assessed value or ratio, no price adjusted by any index,
    and no tolerance on a price: a price enters only by equality between
    two rows or by comparison with the unit's one *nominal_floor*.
    That is what keeps the steps inside plans/design-limits L7 (an index
    in a keep-which-record decision) and L29 (sales ratios by period).

    The rules, in order of precedence, each writing one fixed label:

    - `deed_echo_document`: a roll's last sale citing a deed's document
      (the last sale's scope removed). Kept: the deed.
    - `deed_echo_period_price`: a last sale and a deed, same thing sold,
      year, month and price. Kept: the deed. This reaches what the
      document cannot: a county whose roll states book and page while its
      deeds are named by clerk instrument.
    - `roll_restated`: two last-sale rows, same thing sold, year, month
      and price: two roll lanes stating one sale, or a restatement the
      roll-year rule of `append_last_sales` could not settle. Kept: the
      row read first (the property table's).
    - `double_closing`: two deeds, same thing sold, same price above the
      nominal floor, on the same day where both state a day, else 0 or
      1 calendar month apart. Kept: the later leg, the transferee who
      ended up owning. Same-day legs are ordered by id.
    - `restates_document`: a return stating, under an exemption for a
      correction (*correction_values* of *correction_column*), the
      document it corrects (*previous_document_column*), on the same
      thing sold. Kept: the correcting return.
    - `resale_within_window`: two deeds as for `double_closing` but at
      different prices, both above the floor. Not a duplicate: two sales,
      and nothing is kept over the other.

    A nominal companion (either price at or below the floor) is not
    labeled: the floor and the arm's-length grade already separate it.
    A pair keeps the first label that found it, and no pair is removed
    once found (L28); a row may sit in several pairs.

    Parameters
    ----------
    sales : pandas.DataFrame
        The transaction spine, indexed by `transaction_id`. Only the
        columns named by the parameters and `sale_year`, `sale_month`,
        `price` and `source` are read.
    things : pandas.DataFrame
        `sale_things` of the spine's link tables.
    nominal_floor : float, optional
        The unit's nominal price floor (`nominal_price_floor`).
    kind_column, deed_kind, last_sale_kind : str, optional
        The record kind column and its two values.
    document_column, date_column : str, optional
        The document id (`derive_document_id`, scoped by kind) and the
        recorded date.
    previous_document_column, correction_column : str, optional
        A source's stated previous document and the column whose
        *correction_values* say the return corrects it. Without both,
        no `restates_document` pair is written.

    Returns
    -------
    pandas.DataFrame
        `DUPLICATE_PAIR_COLUMNS`: `transaction_id` is the row the label
        puts in question (the earlier leg, the echo, the restatement, the
        corrected document; the earlier sale of a resale) and
        `other_transaction_id` its partner, the row a consumer keeps.
    """
    allow = [
        kind_column,
        document_column,
        'sale_year',
        'sale_month',
        'price',
        date_column,
        'source',
        previous_document_column,
        correction_column,
    ]
    present = [c for c in dict.fromkeys(allow) if c and c in sales.columns]
    s = pd.DataFrame(sales[present]).copy()
    s.index = pd.Index(sales.index.astype('string'), name='transaction_id')
    s = s[~s.index.duplicated()]

    def _number(column):
        if column not in s.columns:
            return pd.Series(float('nan'), index=s.index)
        return pd.to_numeric(s[column], errors='coerce').astype('float64')

    s['_period'] = _number('sale_year') * 12 + _number('sale_month')
    s['_price'] = _number('price')
    s['_day'] = (
        pd.to_datetime(s[date_column], errors='coerce').dt.normalize()
        if date_column in s.columns
        else pd.NaT
    )
    s['_order'] = range(len(s))
    kind = (
        s[kind_column].astype('string').fillna(deed_kind)
        if kind_column in s.columns
        else pd.Series(deed_kind, index=s.index, dtype='string')
    )
    s['_kind'] = kind
    is_deed = kind.eq(deed_kind).to_numpy()
    is_last = kind.eq(last_sale_kind).to_numpy()

    found = []

    # A: the echo of a deed by its document.
    if document_column in s.columns:
        doc = s[document_column].astype('string')
        prefix = f'{last_sale_kind}:'
        cited = doc[is_last]
        cites = cited.str.startswith(prefix).fillna(False).astype(bool)
        cited = cited.where(cites).str.slice(len(prefix))
        left = cited.dropna().rename('doc').reset_index()
        right = (
            doc[is_deed]
            .dropna()
            .rename('doc')
            .rename_axis('other_transaction_id')
            .reset_index()
        )
        hit = left.merge(right, on='doc')
        found.append(
            _pair_frame(
                hit['transaction_id'], hit['other_transaction_id'], DEED_ECHO_DOCUMENT
            )
        )

    t = things.merge(
        s[['_period', '_day', '_price', '_order', '_kind']],
        left_on='transaction_id',
        right_index=True,
    )
    stated = t['_period'].notna() & t['_price'].notna()
    last_t = t[stated & t['_kind'].eq(last_sale_kind)]
    deed_t = t[stated & t['_kind'].eq(deed_kind)]
    key = ['thing', '_period', '_price']

    # A: the echo by what was sold, when and for how much.
    hit = last_t[['transaction_id', *key]].merge(
        deed_t[['transaction_id', *key]], on=key, suffixes=('', '_other')
    )
    found.append(
        _pair_frame(
            hit['transaction_id'], hit['transaction_id_other'], DEED_ECHO_PERIOD_PRICE
        )
    )

    # A2, A3: one last sale stated twice; the later-read row is in
    # question, the one read first kept.
    cols = ['transaction_id', '_order', *key]
    hit = last_t[cols].merge(last_t[cols], on=key, suffixes=('', '_other'))
    hit = hit[hit['_order'] > hit['_order_other']]
    found.append(
        _pair_frame(hit['transaction_id'], hit['transaction_id_other'], ROLL_RESTATED)
    )

    # B: two deeds close in time. Both above the floor, or the pair is a
    # nominal companion and stays unlabeled (D5). Same calendar month,
    # or the next one (D4), merged on the period so a parcel with many
    # deeds over the years is never compared across them.
    market = deed_t[deed_t['_price'] > nominal_floor]
    cols = ['transaction_id', 'thing', '_period', '_day', '_price']
    same = market[cols].merge(
        market[cols], on=['thing', '_period'], suffixes=('', '_other')
    )
    # Within one month the earlier leg is the smaller id: nothing a
    # month-only source states orders them, and a fixed order keeps
    # the label reproducible. Where both state a day, the same day is
    # required below, so a day never orders a pair either.
    same = same[same['transaction_id'] < same['transaction_id_other']]
    shifted = market[cols].assign(_period=market['_period'] + 1)
    following = shifted.merge(
        market[cols], on=['thing', '_period'], suffixes=('', '_other')
    )
    window = pd.concat([same, following], ignore_index=True)
    both_days = window['_day'].notna() & window['_day_other'].notna()
    window = window[~both_days | (window['_day'] == window['_day_other'])]
    window = window[window['transaction_id'] != window['transaction_id_other']]
    window = window.drop_duplicates(['transaction_id', 'transaction_id_other'])
    same_price = (window['_price'] == window['_price_other']).to_numpy()
    closing = window[same_price]
    found.append(
        _pair_frame(
            closing['transaction_id'], closing['transaction_id_other'], DOUBLE_CLOSING
        )
    )

    # B2: a return naming the document it corrects. Read only where the
    # source says the return is a correction: Wisconsin's RETR states a
    # previous document under several exemptions, and only "correct
    # prior" names a document the return restates (the land-contract
    # and transfer-on-death exemptions name an earlier, different
    # transfer, 1,008 and 1,961 days back at the median on US-WI-VI).
    if (
        previous_document_column
        and correction_column
        and previous_document_column in s.columns
        and correction_column in s.columns
        and document_column in s.columns
    ):
        correcting = s[
            is_deed
            & s[correction_column].astype('string').isin(list(correction_values))
            & s[previous_document_column].notna()
        ]
        named = (
            normalized_text(correcting[previous_document_column])
            .str.upper()
            .rename('doc')
            .rename_axis('other_transaction_id')
            .reset_index()
        )
        documents = (
            normalized_text(s.loc[is_deed, document_column].dropna())
            .str.upper()
            .rename('doc')
            .reset_index()
        )
        hit = named[named['doc'] != ''].merge(documents, on='doc')
        hit = hit[hit['transaction_id'] != hit['other_transaction_id']]
        # The corrected and the correcting return must convey one thing.
        shared = hit.merge(things, on='transaction_id').merge(
            things.rename(columns={'transaction_id': 'other_transaction_id'}),
            on=['other_transaction_id', 'thing'],
        )
        shared = shared.drop_duplicates(['transaction_id', 'other_transaction_id'])
        found.append(
            _pair_frame(
                shared['transaction_id'],
                shared['other_transaction_id'],
                RESTATES_DOCUMENT,
            )
        )

    resale = window[~same_price]
    found.append(
        _pair_frame(
            resale['transaction_id'],
            resale['transaction_id_other'],
            RESALE_WITHIN_WINDOW,
        )
    )

    pairs = pd.concat(found, ignore_index=True)
    pairs = pairs[pairs['transaction_id'] != pairs['other_transaction_id']]
    # One label per pair of rows, whichever way round: the first rule
    # that found it, in precedence order, and never relabeled.
    a = pairs['transaction_id'].to_numpy(dtype=object)
    b = pairs['other_transaction_id'].to_numpy(dtype=object)
    low = np.where(a < b, a, b)
    high = np.where(a < b, b, a)
    pairs = pairs[~pd.DataFrame({'l': low, 'h': high}).duplicated().to_numpy()]
    source = s['source'].astype('string') if 'source' in s.columns else None
    pairs = pairs.assign(
        link_method=pairs['link_method'].astype('string'),
        link_source=(
            pairs['transaction_id'].map(source).astype('string')
            if source is not None
            else pd.Series(pd.NA, index=pairs.index, dtype='string')
        ),
    )
    return pairs[list(DUPLICATE_PAIR_COLUMNS)].reset_index(drop=True)


def duplicate_columns(index: pd.Index, pairs: pd.DataFrame | None) -> pd.DataFrame:
    """Per-row summary of the pair table: values, not verdicts.

    Parameters
    ----------
    index : pandas.Index
        The curated sales' `transaction_id`s.
    pairs : pandas.DataFrame or None
        The pair table (`duplicate_sale_pairs`); None where the spine
        wrote none, which leaves all three columns missing.

    Returns
    -------
    pandas.DataFrame
        Indexed like *index*:

        - `sale_duplicate_of`: the kept partner of the row's pair, by the
          label precedence (`DUPLICATE_LABELS`), then by the partner's
          id; missing on a kept row and on a row in no pair.
        - `sale_duplicate_reason`: that pair's label.
        - `sale_resale_within_window`: 1 on a row with a resale partner,
          else 0. Kept apart from the two above so a consumer dropping
          duplicates never drops a resale by accident.

        A row's partner may itself be a duplicate of a third row: the
        reasons stay apart rather than chaining into one group (D10).
    """
    ids = pd.Index(index.astype('string'))
    out = pd.DataFrame(
        {
            'sale_duplicate_of': pd.Series(pd.NA, index=ids, dtype='string'),
            'sale_duplicate_reason': pd.Series(pd.NA, index=ids, dtype='string'),
            'sale_resale_within_window': pd.Series(float('nan'), index=ids),
        }
    )
    if pairs is None:
        out.index = index
        return out
    pairs = pairs[
        pairs['transaction_id'].isin(ids) & pairs['other_transaction_id'].isin(ids)
    ]
    resale = pairs['link_method'].isin(NOT_DUPLICATE_LABELS)
    in_resale = set(pairs.loc[resale, 'transaction_id']) | set(
        pairs.loc[resale, 'other_transaction_id']
    )
    out['sale_resale_within_window'] = ids.isin(in_resale).astype('float64')
    duplicates = pairs[~resale].copy()
    rank = {label: i for i, label in enumerate(DUPLICATE_LABELS)}
    duplicates['_rank'] = duplicates['link_method'].map(rank)
    first = duplicates.sort_values(
        ['transaction_id', '_rank', 'other_transaction_id'], kind='stable'
    ).drop_duplicates('transaction_id')
    first = first.set_index('transaction_id')
    out['sale_duplicate_of'] = ids.map(first['other_transaction_id']).astype('string')
    out['sale_duplicate_reason'] = ids.map(first['link_method']).astype('string')
    out.index = index
    return out
