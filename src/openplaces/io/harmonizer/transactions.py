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

import numpy as np
import pandas as pd

from openplaces.geo.link import get_entity_link_path, get_link_owner_recipe_id
from openplaces.io import sale_records
from openplaces.io.harmonizer import HarmonizeState, _register
from openplaces.io.harmonizer.entity_links import (
    ADDRESS_CITY_METHOD,
    ADDRESS_METHOD,
    ADDRESS_RANGE_METHOD,
    ASSESSOR_LAST_SALE_METHOD,
    LINK_COLUMNS_TAIL,
    POLYGON_OVERLAP_METHOD,
    STACKED_UNITS_METHOD,
    apportion_shares,
    build_id_links,
    write_entity_link,
)
from openplaces.io.readers import get_entities
from openplaces.recipe import get_recipe_id

#: state.metadata key under which the fold leaves the rows of every
#: multi-parcel deed, each with the deed's transaction id once minted.
DEED_MEMBERS_KEY = 'deed_members'
TRANSACTION_ID = 'transaction_id'
PARCEL_KEY_METHOD = 'parcel_id_local'
#: Link label of a sale that took the ids of the roll rows citing its
#: document (`link_transactions`, *document_key*).
CITED_DOCUMENT_METHOD = 'cited_document'


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
def dedup_transactions(
    state: HarmonizeState, key_columns: list, normalize_columns: list | None = None
) -> HarmonizeState:
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
    normalize_columns : list of str, optional
        Key columns compared without punctuation and case, for a source
        that respells one number between its files
        (`io.sale_records.duplicate_document_mask`).
    """
    spine = state.spine
    if spine is None:
        return state
    repeated = sale_records.duplicate_document_mask(
        spine, key_columns, normalize_columns
    )
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
    strip_leading_zeros: bool = False,
) -> HarmonizeState:
    """Name the recorded document (deed) behind each sale row.

    See `io.sale_records.derive_document_id` for the parameters.
    """
    if state.spine is None:
        return state
    state.spine = sale_records.derive_document_id(
        state.spine,
        candidates,
        output,
        separator,
        scope_column,
        unscoped_value,
        strip_leading_zeros=strip_leading_zeros,
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
    parcel_column: str | list[str],
    document_column: str = 'sale_document_id',
    output: str = 'n_parcels_per_sale',
) -> HarmonizeState:
    """Count the distinct parcels each recorded document covered.

    Written on every row before the fold below, so a consumer can tell a
    multi-parcel sale from a single-parcel one afterwards. *parcel_column*
    may list several columns in order of preference, the normalized
    number before the assessor's spelling
    (`io.sale_records.count_parcels_per_document`).
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
    state.spine, n_rows, n_documents, members = (
        sale_records.aggregate_multi_parcel_sales(
            state.spine, document_column, weight_column, count_column
        )
    )
    # The folded rows' parcels, kept for the link table: the sale is one
    # row, its parcels are link rows (`link_transactions`).
    state.metadata[DEED_MEMBERS_KEY] = members
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
    # The index labels before minting, so the folded deeds' members can
    # be given the id their deed received.
    old_labels = spine.index.to_numpy()
    state.spine, report = sale_records.mint_transaction_ids(
        spine, str(state.admin_id), document_column, label
    )
    members = state.metadata.get(DEED_MEMBERS_KEY)
    if members is not None and len(members) and document_column in members.columns:
        # A folded row kept one member's old label; the new id of that
        # label is the deed's, and every member of the deed takes it.
        new_by_old = pd.Series(state.spine.index.to_numpy(), index=old_labels)
        folded_label_by_document = (
            pd.Series(old_labels, index=spine[document_column].to_numpy())
            .loc[lambda s: s.index.isin(members[document_column])]
            .groupby(level=0)
            .first()
        )
        new_by_document = folded_label_by_document.map(new_by_old)
        members = members.copy()
        members[TRANSACTION_ID] = (
            members[document_column].map(new_by_document).astype('string').to_numpy()
        )
        state.metadata[DEED_MEMBERS_KEY] = members
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


def _rows_to_link(state: HarmonizeState, columns: list[str]) -> pd.DataFrame:
    """One row per (sale, source record), indexed by the sale's id.

    A sale the fold left alone is its own row. A folded deed is absent
    from the spine's keys (the fold blanked what its members disagreed
    on) and is represented by its member rows instead, each carrying
    the deed's id, so a deed of three parcels tries three parcel keys.
    """
    spine = state.spine
    members = state.metadata.get(DEED_MEMBERS_KEY)
    present = [c for c in columns if c in spine.columns]
    own = spine[present].copy()
    own.index = pd.Index(spine.index.astype('string'), name=TRANSACTION_ID)
    if members is None or not len(members) or TRANSACTION_ID not in members.columns:
        return own
    folded_ids = set(members[TRANSACTION_ID].dropna().astype('string'))
    own = own[~own.index.isin(folded_ids)]
    member_rows = members[[c for c in present if c in members.columns]].copy()
    member_rows.index = pd.Index(
        members[TRANSACTION_ID].astype('string').to_numpy(), name=TRANSACTION_ID
    )
    return pd.concat([own, member_rows.reindex(columns=present)])


def _pass_links(rows, reference, row_key, ref_key, method, source_column):
    """The link rows one exact-key pass finds, labeled *method*."""
    links = build_id_links(
        rows,
        reference,
        row_key,
        ref_key,
        TRANSACTION_ID,
        'parcel_id' if reference.index.name == 'parcel_id' else reference.index.name,
        method,
        source_column=source_column,
    )
    return links


def _cited_document_rows(rows, document_key, kind_column, citing_kind, id_columns):
    """Each sale row of another kind, once per citing row naming its document.

    The returned rows keep the sale's id and source and carry the citing
    row's *id_columns*, so a deed is linked to every parcel and property
    whose roll row cites its book and page: a deed of three parcels is
    cited three times. None where no row cites a document.
    """
    if document_key not in rows.columns or kind_column not in rows.columns:
        return None
    citing_mask = rows[kind_column].astype('string').eq(citing_kind).fillna(False)
    present = [c for c in id_columns if c in rows.columns]
    if not present:
        return None
    citing = rows.loc[citing_mask, [document_key, *present]]
    citing = citing.dropna(subset=[document_key]).dropna(how='all', subset=present)
    if citing.empty:
        return None
    carried = [c for c in ('source',) if c in rows.columns]
    targets = rows.loc[
        ~citing_mask & rows[document_key].notna(), [document_key, *carried]
    ]
    merged = targets.reset_index().merge(
        citing.drop_duplicates().reset_index(drop=True), on=document_key, how='inner'
    )
    return merged.set_index(TRANSACTION_ID)


@_register('link_transactions')
def link_transactions(
    state: HarmonizeState,
    parcel_recipe_id: str = 'US_parcel-spine-2026',
    property_recipe_id: str | None = 'US_property-spine-2026',
    parcel_key: str = 'parcel_link_key',
    lot_column: str = 'lot_id_local',
    property_key: str = 'parcel_id_local',
    parcel_id_keys: tuple[str, ...] = (),
    document_key: str | None = None,
    address_keys: tuple[str, ...] = ('address_id_local_city', 'address_id_local'),
    address_ranges: bool = True,
    kind_column: str = 'sale_record_kind',
    last_sale_kind: str = 'assessor_last_sale',
    output_column: str = 'parcel_link_method',
) -> HarmonizeState:
    """Write the transaction-to-parcel and transaction-to-property link tables.

    A sale conveys real estate, and what it conveyed is recorded as link
    rows, one per parcel or property, beside the transaction spine's
    output (`geo.link.get_entity_link_path`): the spine itself carries
    what the document states and nothing it conveyed. A deed of three
    parcels is one spine row and three link rows, each with its area
    `share` of the deed; its values reach the curated sale through the
    link, summed or kept per parcel as the question requires, instead of
    one parcel's columns standing for all three (`aggregate_multi_parcel_sales`).

    The passes, each an exact match on a normalized key, tried in order
    and each only for rows no earlier pass reached:

    1. the parcel key (`parcel_link_key`, the lot where the record named
       a stacked unit): method `parcel_id_local`, or `stacked_units`
       where the row reached the lot through its unit (*lot_column* set),
       or `assessor_last_sale` where the row is a roll's last-sale field
       and so was never matched at all, the key being the roll row's own;
    1a. any further parcel ids the record carries (*parcel_id_keys*),
       each against the parcel spine's column of the same name and
       labeled with it: a Massachusetts last sale is a MassGIS assessor
       row with no `parcel_id_local` but its parcel's `parcel_id_admin2`,
       which the address passes below would only approximate;
    1b. with *document_key*, the ids of the roll rows citing the sale's
       document (`cited_document`): a registry deed names no parcel, but
       the assessor's last-sale row citing its book and page does, so the
       deed takes that row's *parcel_id_keys* (and, below, its
       *property_key*), every citing row's where several cite one deed;
    2. the address keys, town-scoped then county-scoped
       (`address_id_local_city`, `address_id_local`), for a registry that
       carries no parcel number (Massachusetts);
    3. the address range (`address_range`): a deed whose number is a
       range (`704-706`) resolves where exactly one parcel matches either
       half (`links.address_ranges.match_address_ranges`).

    The property link is one pass: the record's own number against the
    property spine's (`parcel_id_local`), which is how a condo sale
    reaches its unit while the parcel link above reaches the lot.

    Nothing here falls through to a similarity comparison, no link
    carries a score (`share` is an area fraction) and no pass removes a
    link an earlier one wrote (plans/design-limits, L17 and L28).

    Parameters
    ----------
    parcel_recipe_id, property_recipe_id : str
        The spines to link to. A property spine the unit lacks yields an
        empty property table, which is still written so a reader finds
        a file.
    parcel_key, lot_column, property_key : str
        Spine columns: the key the parcel passes join on, the column a
        unit's lot was written to, and the record's own number.
    parcel_id_keys : tuple of str
        Spine columns holding another parcel id, tried in order after
        the parcel key and before the addresses, each present on the
        parcel spine under the same name (default none).
    document_key : str, optional
        Spine column naming each row's document the same way on every
        record kind (a book and page without the kind's scope), for the
        `cited_document` pass; default none, no such pass.
    address_keys : tuple of str
        Spine columns tried in order after the parcel key, each present
        on the parcel spine under the same name.
    address_ranges : bool
        Try the range pass after the address keys.
    kind_column, last_sale_kind : str
        The record-kind column and the value that marks a roll's
        last-sale row.
    output_column : str
        Spine column written with the methods that linked each sale,
        `+`-joined where several did; missing where none did.
    """
    spine = state.spine
    if spine is None or spine.empty:
        return state
    own_id = get_recipe_id(state.recipe)
    owner_id = get_link_owner_recipe_id(state.recipe)
    admin_id = state.admin_id
    # De-duplicated: a source with no lot key links parcels on the
    # number it states (`parcel_key: parcel_id_local`), the default
    # property key too, and a column selected twice reads as a frame.
    columns = list(
        dict.fromkeys(
            [
                parcel_key,
                lot_column,
                property_key,
                *parcel_id_keys,
                *([document_key] if document_key else []),
                *address_keys,
                'address_street',
                'address_number',
                'admin4_id',
                kind_column,
                'source',
            ]
        )
    )
    rows = _rows_to_link(state, columns)
    source_column = 'source' if 'source' in rows.columns else None
    cited = (
        _cited_document_rows(
            rows,
            document_key,
            kind_column,
            last_sale_kind,
            [*parcel_id_keys, property_key],
        )
        if document_key
        else None
    )

    def _load(recipe_id):
        try:
            frame = get_entities(recipe_id, admin_id, geom=False, missing='ignore')
        except (FileNotFoundError, OSError, KeyError, ValueError):
            return None
        return frame if frame is not None and len(frame) else None

    # Parcel links.
    parcels = _load(parcel_recipe_id)
    parcel_links = pd.DataFrame(
        columns=[TRANSACTION_ID, 'parcel_id', *LINK_COLUMNS_TAIL]
    )
    if parcels is not None and 'parcel_id_local' in parcels.columns:
        parcels = parcels.copy()
        if parcels.index.name != 'parcel_id':
            parcels.index = parcels.index.rename('parcel_id')
        reached = pd.Series(False, index=range(len(rows)))
        found = []

        def _tier(mask, key, ref_key, method):
            nonlocal reached
            if key not in rows.columns or ref_key not in parcels.columns:
                return
            tried = rows[mask.to_numpy() & ~reached.to_numpy()]
            if tried.empty:
                return
            links = _pass_links(tried, parcels, key, ref_key, method, source_column)
            if links.empty:
                return
            found.append(links)
            linked_ids = set(links[TRANSACTION_ID].astype('string'))
            hit = rows.index.isin(linked_ids) & mask.to_numpy() & ~reached.to_numpy()
            reached = reached | pd.Series(hit, index=reached.index)

        everything = pd.Series(True, index=rows.index)
        if kind_column in rows.columns:
            is_last_sale = rows[kind_column].astype('string').eq(last_sale_kind)
            is_last_sale = is_last_sale.fillna(False)
        else:
            is_last_sale = pd.Series(False, index=rows.index)
        via_lot = (
            rows[lot_column].notna()
            if lot_column in rows.columns
            else pd.Series(False, index=rows.index)
        )
        _tier(is_last_sale, parcel_key, 'parcel_id_local', ASSESSOR_LAST_SALE_METHOD)
        _tier(
            via_lot & ~is_last_sale, parcel_key, 'parcel_id_local', STACKED_UNITS_METHOD
        )
        _tier(everything, parcel_key, 'parcel_id_local', PARCEL_KEY_METHOD)
        for key in parcel_id_keys:
            _tier(everything, key, key, key)
        if cited is not None:
            for key in parcel_id_keys:
                pending = set(rows.index[~reached.to_numpy()])
                tried = cited[cited.index.isin(pending)]
                if tried.empty or key not in tried.columns or key not in parcels:
                    continue
                links = _pass_links(
                    tried, parcels, key, key, CITED_DOCUMENT_METHOD, source_column
                )
                if links.empty:
                    continue
                found.append(links)
                hit = rows.index.isin(set(links[TRANSACTION_ID].astype('string')))
                reached = reached | pd.Series(hit, index=reached.index)
        for key, method in zip(address_keys, (ADDRESS_CITY_METHOD, ADDRESS_METHOD)):
            _tier(everything, key, key, method)
        if (
            address_ranges
            and 'address_number' in rows.columns
            and 'address_street' in rows.columns
            and 'address_number' in parcels.columns
            and 'address_street' in parcels.columns
        ):
            from openplaces.io.harmonizer.links.address_ranges import (
                match_address_ranges,
            )

            attempt = pd.Series(
                (~reached).to_numpy() & rows['address_number'].notna().to_numpy(),
                index=rows.index,
            )
            if attempt.any():
                matched, ambiguous, _n = match_address_ranges(
                    rows, parcels, admin_id, attempt=attempt
                )
                if ambiguous and state.verbose:
                    print(
                        f'  link_transactions: {len(ambiguous)} address ranges '
                        'match more than one parcel; left unlinked'
                    )
                if matched:
                    positions = list(matched)
                    ranged = pd.DataFrame(
                        {
                            TRANSACTION_ID: rows.index[positions].to_numpy(),
                            'parcel_id': [matched[p] for p in positions],
                            'link_method': ADDRESS_RANGE_METHOD,
                            'link_source': (
                                rows['source']
                                .iloc[positions]
                                .astype('string')
                                .to_numpy()
                                if source_column
                                else pd.NA
                            ),
                            'share': pd.Series(
                                pd.NA, index=range(len(positions)), dtype='Float64'
                            ),
                        }
                    )
                    ranged['link_method'] = ranged['link_method'].astype('string')
                    ranged['link_source'] = ranged['link_source'].astype('string')
                    found.append(ranged)
        if found:
            parcel_links = pd.concat(found, ignore_index=True)
            parcel_links = parcel_links.drop_duplicates(
                subset=[TRANSACTION_ID, 'parcel_id'], keep='first'
            )
            parcel_links = parcel_links.sort_values(
                [TRANSACTION_ID, 'parcel_id'], kind='stable'
            ).reset_index(drop=True)
        parcel_links = apportion_shares(parcel_links, parcels)
    else:
        parcel_links = apportion_shares(parcel_links, pd.DataFrame())
    write_entity_link(
        get_entity_link_path(owner_id, parcel_recipe_id, admin_id),
        parcel_links,
        writer_recipe_id=own_id,
        finer_recipe_id=owner_id,
        coarser_recipe_id=parcel_recipe_id,
        admin_id=admin_id,
        step_config={
            'parcel_key': parcel_key,
            'address_keys': list(address_keys),
            'address_ranges': address_ranges,
            'document_key': document_key,
        },
        # The spine's own output is saved after this step, so only the
        # other side is stamped (see write_entity_link).
        stamped_recipe_ids=(parcel_recipe_id,),
    )

    # Property links.
    if property_recipe_id:
        properties = _load(property_recipe_id)
        property_links = pd.DataFrame(
            columns=[TRANSACTION_ID, 'property_id', *LINK_COLUMNS_TAIL]
        )
        if (
            properties is not None
            and property_key in properties.columns
            and property_key in rows.columns
        ):
            properties = properties.copy()
            properties.index = properties.index.rename('property_id')
            property_links = build_id_links(
                rows,
                properties,
                property_key,
                property_key,
                TRANSACTION_ID,
                'property_id',
                # The label names the key that matched, as on the
                # parcel passes; the default key is PARCEL_KEY_METHOD.
                property_key,
                source_column=source_column,
            )
            if cited is not None and property_key in cited.columns:
                linked_ids = set(property_links[TRANSACTION_ID].astype('string'))
                tried = cited[~cited.index.isin(linked_ids)]
                if len(tried):
                    property_links = pd.concat(
                        [
                            property_links,
                            build_id_links(
                                tried,
                                properties,
                                property_key,
                                property_key,
                                TRANSACTION_ID,
                                'property_id',
                                CITED_DOCUMENT_METHOD,
                                source_column=source_column,
                            ),
                        ],
                        ignore_index=True,
                    )
            # A folded deed is linked through its member rows, and two
            # members naming one property (the roll's row and the parcel
            # layer's, say) would make the pair twice.
            property_links = property_links.drop_duplicates(
                subset=[TRANSACTION_ID, 'property_id'], keep='first'
            ).reset_index(drop=True)
        property_links = apportion_shares(property_links, pd.DataFrame())
        write_entity_link(
            get_entity_link_path(owner_id, property_recipe_id, admin_id),
            property_links,
            writer_recipe_id=own_id,
            finer_recipe_id=owner_id,
            coarser_recipe_id=property_recipe_id,
            admin_id=admin_id,
            step_config={'property_key': property_key, 'document_key': document_key},
            stamped_recipe_ids=(property_recipe_id,),
        )
    else:
        property_links = pd.DataFrame(columns=[TRANSACTION_ID])

    # Which rules linked each sale, on the spine, for the reader who
    # wants one column rather than the table: a fixed label per rule,
    # several joined by '+', never a score.
    if len(parcel_links):
        methods = (
            parcel_links.groupby(TRANSACTION_ID, sort=False)['link_method']
            .agg(lambda s: '+'.join(dict.fromkeys(s.astype(str))))
            .astype('string')
        )
        spine[output_column] = (
            pd.Index(spine.index.astype('string')).map(methods).astype('string')
        )
    else:
        spine[output_column] = pd.Series(pd.NA, index=spine.index, dtype='string')
    state.spine = spine
    if state.verbose:
        n_linked = spine[output_column].notna().sum()
        print(
            f'  link_transactions: {len(parcel_links):,d} parcel links for '
            f'{int(n_linked):,d} of {len(spine):,d} sales '
            f'({spine[output_column].value_counts().to_dict()}); '
            f'{len(property_links):,d} property links'
        )
    return state


@_register('link_transactions_by_overlap')
def link_transactions_by_overlap(
    state: HarmonizeState,
    parcel_recipe_id: str = 'US_parcel-spine-2026',
    min_parcel_share: float = 0.8,
    output_column: str = 'parcel_link_method',
) -> HarmonizeState:
    """Add parcels lying mostly inside a transaction's own polygon.

    For a source whose records are polygons (land purchases drawn by the
    acquiring agency) the polygon states what was conveyed as directly as
    a parcel number does. A parcel is linked when more than
    *min_parcel_share* of its area lies inside the polygon; 0.8 is the
    rule of French et al. (2026, preprint), who selected "cadaster
    parcels that share more than 80% of their area with a PLA boundary".

    The pass runs over every transaction, not only those the key passes
    of `link_transactions` missed, and merges into the same table: a
    pair both rules found keeps one row labeled
    `parcel_id_local+polygon_overlap`. It is deliberately not a tier of
    that cascade. Nothing falls through to it, no link carries a score
    (`share` remains the area apportionment of `apportion_shares`), and
    no link either rule wrote is removed (plans/design-limits, L17 and
    L28).

    Parameters
    ----------
    parcel_recipe_id : str
        The parcel spine to overlay; its geometry is read, its `area_ha`
        apportions shares.
    min_parcel_share : float
        Share of a parcel's area that must lie inside the transaction's
        polygon; strictly greater than this links.
    output_column : str
        Spine column rewritten with the methods that linked each sale.
    """
    import geopandas as gpd
    import shapely

    spine = state.spine
    if spine is None or spine.empty or not isinstance(spine, gpd.GeoDataFrame):
        return state
    own_id = get_recipe_id(state.recipe)
    owner_id = get_link_owner_recipe_id(state.recipe)
    admin_id = state.admin_id
    path = get_entity_link_path(owner_id, parcel_recipe_id, admin_id)
    columns = [TRANSACTION_ID, 'parcel_id', *LINK_COLUMNS_TAIL]
    existing = (
        pd.read_parquet(path)[columns]
        if path.exists()
        else pd.DataFrame(columns=columns)
    )
    try:
        parcels = get_entities(parcel_recipe_id, admin_id, geom=True, missing='ignore')
    except (FileNotFoundError, OSError, KeyError, ValueError):
        parcels = None
    if parcels is None or not len(parcels):
        return state

    crs = spine.estimate_utm_crs()
    sales = spine[[spine.geometry.name]].to_crs(crs)
    sales = sales[sales.geometry.notna() & ~sales.geometry.is_empty]
    sales.index = pd.Index(sales.index.astype('string'), name=TRANSACTION_ID)
    lots = gpd.GeoDataFrame(geometry=parcels.geometry, crs=parcels.crs).to_crs(crs)
    lots.index = pd.Index(parcels.index, name='parcel_id')
    pairs = gpd.sjoin(
        lots.reset_index(), sales.reset_index(), predicate='intersects', how='inner'
    )
    found = pd.DataFrame(columns=columns)
    if len(pairs):
        lot_geoms = lots.geometry.loc[pairs['parcel_id']].to_numpy()
        sale_geoms = sales.geometry.loc[pairs[TRANSACTION_ID]].to_numpy()
        inside = shapely.area(shapely.intersection(lot_geoms, sale_geoms))
        lot_area = shapely.area(lot_geoms)
        ratio = np.divide(
            inside, lot_area, out=np.zeros_like(inside), where=lot_area > 0
        )
        hits = pairs.iloc[np.flatnonzero(ratio > min_parcel_share)]
        source = (
            spine['source'].astype('string').reindex(spine.index)
            if 'source' in spine.columns
            else None
        )
        found = pd.DataFrame(
            {
                TRANSACTION_ID: hits[TRANSACTION_ID].astype('string').to_numpy(),
                'parcel_id': hits['parcel_id'].to_numpy(),
                'link_method': POLYGON_OVERLAP_METHOD,
                'link_source': (
                    source.set_axis(source.index.astype('string'))
                    .reindex(hits[TRANSACTION_ID].astype('string'))
                    .to_numpy()
                    if source is not None
                    else pd.NA
                ),
                'share': pd.Series(pd.NA, index=range(len(hits)), dtype='Float64'),
            }
        )
    links = pd.concat([existing, found], ignore_index=True)
    links[TRANSACTION_ID] = links[TRANSACTION_ID].astype('string')
    links['link_method'] = links['link_method'].astype('string')
    links['link_source'] = links['link_source'].astype('string')
    links = (
        links.groupby([TRANSACTION_ID, 'parcel_id'], sort=True, as_index=False)
        .agg(
            link_method=('link_method', lambda s: '+'.join(dict.fromkeys(s.dropna()))),
            link_source=('link_source', 'first'),
        )
        .astype({'link_method': 'string', 'link_source': 'string'})
    )
    links['share'] = pd.Series(pd.NA, index=links.index, dtype='Float64')
    links = apportion_shares(links, parcels)
    write_entity_link(
        path,
        links,
        writer_recipe_id=own_id,
        finer_recipe_id=owner_id,
        coarser_recipe_id=parcel_recipe_id,
        admin_id=admin_id,
        step_config={'min_parcel_share': min_parcel_share, 'overlap': True},
        stamped_recipe_ids=(parcel_recipe_id,),
    )

    # Sorted, so one set of rules reads one way whichever pair came first.
    methods = (
        links.groupby(TRANSACTION_ID, sort=False)['link_method']
        .agg(lambda s: '+'.join(sorted(set('+'.join(s).split('+')))))
        .astype('string')
    )
    spine[output_column] = (
        pd.Index(spine.index.astype('string')).map(methods).astype('string')
    )
    state.spine = spine
    if state.verbose:
        print(
            f'  link_transactions_by_overlap: {len(found):,d} overlap links; '
            f'{len(links):,d} parcel links for '
            f'{int(spine[output_column].notna().sum()):,d} of {len(spine):,d} '
            f'sales ({spine[output_column].value_counts().to_dict()})'
        )
    return state


#: Column `check_parcel_links_by_address_points` writes on each
#: transaction-to-parcel link row, and its fixed values.
ADDRESS_POINTS_CHECK = 'address_points_check'
ADDRESS_POINTS_AGREE = 'agree'
ADDRESS_POINTS_OTHER_PARCEL = 'other_parcel'
ADDRESS_POINTS_NO_MATCH = 'no_point_match'
ADDRESS_POINTS_NONE = 'no_points'
ADDRESS_POINTS_NO_SALE_ADDRESS = 'no_sale_address'


def address_points_check(
    links: pd.DataFrame,
    sale_keys: pd.Series,
    parcel_keys: pd.Series,
) -> pd.Series:
    """Label each sale-to-parcel link with what the address points say.

    Parameters
    ----------
    links : pandas.DataFrame
        Link rows with `transaction_id` and `parcel_id`.
    sale_keys : pandas.Series
        The sale's address keys, indexed by `transaction_id`; a folded
        deed has one entry per member row, and any of them may agree.
    parcel_keys : pandas.Series
        The keys of the address points inside each parcel, one entry per
        (parcel, point) pair, indexed by `parcel_id`.

    Returns
    -------
    pandas.Series
        One fixed label per link row, in the order of *links*:
        `no_sale_address` where the sale has no key, `no_points` where
        the parcel holds no keyed point, `agree` where a sale key is a
        key of a point inside the parcel, `other_parcel` where it is
        only a key of points in other parcels, and `no_point_match`
        where it is the key of no point at all.
    """
    sale = sale_keys.dropna().astype('string')
    sale_sets: dict = {}
    for tid, key in zip(sale.index, sale, strict=True):
        sale_sets.setdefault(str(tid), set()).add(key)
    keyed = parcel_keys.dropna().astype('string')
    parcel_sets: dict = {}
    for pid, key in zip(keyed.index, keyed, strict=True):
        parcel_sets.setdefault(pid, set()).add(key)
    any_parcel = set(keyed)
    labels = []
    for tid, pid in zip(
        links[TRANSACTION_ID].astype('string'), links['parcel_id'], strict=True
    ):
        keys = sale_sets.get(tid)
        if not keys:
            labels.append(ADDRESS_POINTS_NO_SALE_ADDRESS)
            continue
        inside = parcel_sets.get(pid)
        if not inside:
            labels.append(ADDRESS_POINTS_NONE)
        elif keys & inside:
            labels.append(ADDRESS_POINTS_AGREE)
        elif keys & any_parcel:
            labels.append(ADDRESS_POINTS_OTHER_PARCEL)
        else:
            labels.append(ADDRESS_POINTS_NO_MATCH)
    return pd.Series(labels, index=links.index, dtype='string')


def _parcel_point_keys(points, parcels, admin_id) -> pd.Series:
    """The address keys of the points inside each parcel's polygon.

    One entry per (parcel, point) pair, indexed by parcel id. Points are
    keyed as the spine keys a sale (`reconcile_addresses_df`, then
    `add_address_id_local`), so equal addresses give equal keys.
    """
    import geopandas as gpd

    from openplaces.io.harmonizer.addresses import (
        add_address_id_local,
        reconcile_addresses_df,
    )

    # Positional index: only the row order matters here, and a point
    # table's ids need not be unique (an index that repeated aborted
    # the address formatting's assignment once, on US-MA-BOS).
    frame = pd.DataFrame(points[['address_number', 'address_street']]).reset_index(
        drop=True
    )
    frame = reconcile_addresses_df(
        frame,
        admin_id,
        False,
        sources={
            'points': {
                'address_number': 'address_number',
                'address_street': 'address_street',
            }
        },
        number_output_col='address_number',
    )
    add_address_id_local(frame, admin_id, output_column_city=None)
    keyed = gpd.GeoDataFrame(
        {'key': frame['address_id_local'].to_numpy()},
        geometry=points.geometry.to_numpy(),
        crs=points.crs,
    )
    keyed = keyed[keyed['key'].notna()]
    lots = gpd.GeoDataFrame(geometry=parcels.geometry.to_numpy(), crs=parcels.crs)
    lots['parcel_id'] = parcels.index.to_numpy()
    hits = gpd.sjoin(keyed.to_crs(lots.crs), lots, predicate='within')
    return pd.Series(
        hits['key'].to_numpy(), index=hits['parcel_id'].to_numpy(), dtype='string'
    )


@_register('check_parcel_links_by_address_points')
def check_parcel_links_by_address_points(
    state: HarmonizeState,
    dwelling_recipe_ids: list[str] | tuple[str, ...] = ('dwelling-overture-2025',),
    parcel_recipe_id: str = 'US_parcel-spine-2026',
    address_key: str = 'address_id_local',
    output_column: str = ADDRESS_POINTS_CHECK,
) -> HarmonizeState:
    """State on each parcel link whether the address points confirm it.

    A sale linked to its parcel by a parcel number can also state an
    address. Each parcel takes the address keys of the address points
    lying inside its own polygon, and each row of the transaction-to-
    parcel link table gets one fixed label saying how the sale's
    address key relates to them (`address_points_check`): the link and
    the address points agree, the address belongs to another parcel's
    points, it belongs to no point, the parcel holds no point (most
    vacant land: an address point marks a building), or the sale states
    no address.

    Why a label and not a score or a new link: a link is found by one
    exact rule and keeps that rule's fixed name (`link_method`); a
    confidence on it, or a link removed when the address disagrees, is
    what plans/design-limits keeps out (L17, L28). The label is a fact
    about one pair that a reader may filter on; it enters no combined
    measure, and the step adds and removes no link. Only points inside
    the parcel's own polygon count, never a neighbor's (L27), and the
    point source is the first of *dwelling_recipe_ids* that has points
    for the unit, in the order given, never a vote between sources
    (L18).

    Run it after every step that writes the parcel link table: a later
    step that rebuilds the table drops the column.

    Parameters
    ----------
    dwelling_recipe_ids : list of str
        Address-point recipes, the first with points for the unit wins
        (a state patch puts its own source first).
    parcel_recipe_id : str
        The parcel spine whose link table and polygons are read.
    address_key : str
        Spine column holding the sale's address key.
    output_column : str
        Link-table column written.
    """
    import json

    import pyarrow.parquet as pq

    from openplaces.io.harmonizer.entity_links import ENTITY_LINK_METADATA_KEY

    spine = state.spine
    if spine is None or spine.empty:
        return state
    own_id = get_recipe_id(state.recipe)
    owner_id = get_link_owner_recipe_id(state.recipe)
    admin_id = state.admin_id
    path = get_entity_link_path(owner_id, parcel_recipe_id, admin_id)
    if not path.exists():
        return state
    links = pd.read_parquet(path)
    if links.empty:
        return state

    def _load(recipe_id):
        try:
            frame = get_entities(recipe_id, admin_id, geom=True, missing='ignore')
        except (FileNotFoundError, OSError, KeyError, ValueError):
            return None
        return frame if frame is not None and len(frame) else None

    points = next((p for p in map(_load, dwelling_recipe_ids) if p is not None), None)
    parcels = _load(parcel_recipe_id)
    parcel_keys = pd.Series(dtype='string')
    if (
        points is not None
        and parcels is not None
        and {'address_number', 'address_street'} <= set(points.columns)
    ):
        parcel_keys = _parcel_point_keys(points, parcels, admin_id)

    rows = _rows_to_link(state, [address_key])
    sale_keys = (
        rows[address_key] if address_key in rows.columns else pd.Series(dtype='string')
    )
    links[output_column] = address_points_check(links, sale_keys, parcel_keys)

    footer = pq.read_schema(path).metadata or {}
    raw = footer.get(ENTITY_LINK_METADATA_KEY.encode())
    step_config = dict(json.loads(raw).get('step_config') or {}) if raw else {}
    step_config[output_column] = {
        'dwelling_recipe_ids': list(dwelling_recipe_ids),
        'address_key': address_key,
    }
    write_entity_link(
        path,
        links,
        writer_recipe_id=own_id,
        finer_recipe_id=owner_id,
        coarser_recipe_id=parcel_recipe_id,
        admin_id=admin_id,
        step_config=step_config,
        stamped_recipe_ids=(parcel_recipe_id,),
    )
    if state.verbose:
        print(
            '  check_parcel_links_by_address_points: '
            f'{links[output_column].value_counts().to_dict()}'
        )
    return state
