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


def duplicate_document_mask(frame: pd.DataFrame, key_columns: list) -> pd.Series:
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
    """
    key = frame.reindex(columns=list(key_columns)).astype('string').fillna('NA')
    return key.duplicated(keep='first')


def derive_document_id(
    frame: pd.DataFrame,
    candidates: list,
    output: str = 'sale_document_id',
    separator: str = '/',
    scope_column: str | None = None,
    unscoped_value: str | None = None,
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
