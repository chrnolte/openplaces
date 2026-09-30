"""Harmonize steps that establish the transaction entity on its spine.

A transaction is one recorded sale (`core.schema.ENTITY_DEFINITIONS`).
The sources do not arrive that way: a source published in overlapping
windows lists one document twice, and a deed covering several parcels
is published once per parcel. Until 2026-09-29 the curate stage put the
rows into the entity's terms and minted the ids last; the plan's stage
contracts place both in harmonize, with property ids ("Transaction ids
are minted in the transaction spine, like property ids; curate filters
and never re-mints", plans/stage-contract-audit.md). The steps here run
at the end of `US_transaction-spine-2026`, after every parcel link, in
this order: the sale period, the exact-duplicate drop, the document id,
the parcels-per-document count, the multi-parcel fold, the ids. The
fold precedes the ids so that a deed's id is its document's, without a
suffix, exactly as the curate stage minted it after its own fold.

The frame operations live in `io.sale_records`; the curate steps of
the same names wrap them too, for a recipe that still lists them.
"""

from __future__ import annotations

import warnings

from openplaces.io import sale_records
from openplaces.io.harmonizer import HarmonizeState, _register


@_register('derive_sale_period')
def derive_sale_period(
    state: HarmonizeState, date_column: str = 'recorded_date'
) -> HarmonizeState:
    """Fill `sale_year` and `sale_month` from a date where they are missing.

    Parameters
    ----------
    date_column : str, optional
        Column holding the date (default `recorded_date`).
    """
    if state.spine is None:
        return state
    state.spine = sale_records.derive_sale_period(state.spine, date_column)
    return state


@_register('dedup_transactions')
def dedup_transactions(state: HarmonizeState, key_columns: list) -> HarmonizeState:
    """Drop rows that are the same recorded document, kept once.

    What it drops is unambiguous, an exact repeat of one document from
    overlapping source windows (15.4% of Lake County FL's rows, all
    deeds, measured 2026-09-29), which is why it belongs to the spine
    and not to a curation another project might do differently.

    Parameters
    ----------
    key_columns : list of str
        Columns whose combination identifies one recorded document. The
        first occurrence of a repeated combination is kept. A column the
        table lacks reads as missing on every row.
    """
    spine = state.spine
    if spine is None:
        return state
    repeated = sale_records.duplicate_document_mask(spine, key_columns)
    state.spine = spine.loc[~repeated.to_numpy()].copy()
    if state.verbose:
        print(f'  dedup_transactions: dropped {int(repeated.sum()):,} duplicate rows')
    return state


@_register('derive_document_id')
def derive_document_id(
    state: HarmonizeState,
    candidates: list,
    output: str = 'sale_document_id',
    separator: str = '/',
    scope_column: str | None = None,
    unscoped_value: str | None = None,
) -> HarmonizeState:
    """Name the recorded document (deed) behind each sale row.

    See `io.sale_records.derive_document_id` for the parameters.
    """
    if state.spine is None:
        return state
    state.spine = sale_records.derive_document_id(
        state.spine, candidates, output, separator, scope_column, unscoped_value
    )
    if state.verbose:
        named = int(state.spine[output].notna().sum())
        print(
            f'  derive_document_id: {named:,} of {len(state.spine):,} rows name '
            'a document'
        )
    return state


@_register('count_parcels_per_document')
def count_parcels_per_document(
    state: HarmonizeState,
    parcel_column: str,
    document_column: str = 'sale_document_id',
    output: str = 'n_parcels_per_sale',
) -> HarmonizeState:
    """Count the distinct parcels each recorded document covered.

    Written on every row before the fold below, so a consumer can tell a
    multi-parcel sale from a single-parcel one afterwards.
    """
    if state.spine is None:
        return state
    state.spine = sale_records.count_parcels_per_document(
        state.spine, parcel_column, document_column, output
    )
    return state


@_register('aggregate_multi_parcel_sales')
def aggregate_multi_parcel_sales(
    state: HarmonizeState,
    document_column: str = 'sale_document_id',
    weight_column: str | None = None,
    count_column: str = 'n_parcels_per_sale',
) -> HarmonizeState:
    """Emit one row per recorded document instead of one per parcel.

    The spine's row is the entity, one recorded sale, so the fold
    happens here; the curate stage reads deeds. See
    `io.sale_records.aggregate_multi_parcel_sales`.
    """
    if state.spine is None:
        return state
    state.spine, n_rows, n_documents = sale_records.aggregate_multi_parcel_sales(
        state.spine, document_column, weight_column, count_column
    )
    if state.verbose and n_rows:
        print(
            f'  aggregate_multi_parcel_sales: {n_rows:,} rows on {n_documents:,} '
            'multi-parcel documents became one row each'
        )
    return state


@_register('assign_transaction_ids')
def assign_transaction_ids(
    state: HarmonizeState,
    document_column: str = 'sale_document_id',
    label: str = 'sale',
) -> HarmonizeState:
    """Index the spine by a stable sale id, scoped to the admin unit.

    The id is the recorded document, scoped by the admin unit
    (`US-FL-MD_<document>`); a row without one is fingerprinted over the
    fewest columns that separate it (`sale_records.mint_transaction_ids`).
    Minted here, once: the curate stage keeps the index it loads and
    never re-mints, so a delivered sale's id is the spine's.

    Parameters
    ----------
    document_column : str, optional
        Column holding the recorded document's identifier.
    label : str, optional
        Name used for rows the source gave no document, appearing in
        their id as `{admin}_{label}:{fingerprint}`.
    """
    spine = state.spine
    if spine is None or spine.empty:
        return state
    if document_column not in spine.columns:
        warnings.warn(
            f'assign_transaction_ids: no {document_column!r} column, so every '
            'sale is named by its content fingerprint. Run derive_document_id '
            'first if the source carries a document reference.',
            stacklevel=2,
        )
    state.spine, report = sale_records.mint_transaction_ids(
        spine, str(state.admin_id), document_column, label
    )
    repeated = report['n_repeated_documents']
    if repeated > sale_records.REPEATED_DOCUMENT_SHARE * report['n_rows']:
        # A document that names many rows names none of them. Those rows
        # still get an id, by suffix, but the id stops being the
        # reference a reader could look the sale up by.
        warnings.warn(
            f'assign_transaction_ids: {repeated:,} of {report["n_rows"]:,} sales '
            f'share their {document_column!r} with another, so that column is '
            'not identifying them. They are separated by suffix; check what '
            'derive_document_id found for this source.',
            stacklevel=2,
        )
    if state.verbose:
        print(
            f'  assign_transaction_ids: {report["n_rows"]:,} sales '
            f'({report["n_unnamed"]:,} fingerprinted over '
            f'{report["tiers_used"]} tier(s))'
        )
    return state
