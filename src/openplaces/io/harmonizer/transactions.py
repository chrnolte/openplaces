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


@_register('link_transactions')
def link_transactions(
    state: HarmonizeState,
    parcel_recipe_id: str = 'US_parcel-spine-2026',
    property_recipe_id: str | None = 'US_property-spine-2026',
    parcel_key: str = 'parcel_link_key',
    lot_column: str = 'lot_id_local',
    property_key: str = 'parcel_id_local',
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
    columns = [
        parcel_key,
        lot_column,
        property_key,
        *address_keys,
        'address_street',
        'address_number',
        'admin4_id',
        kind_column,
        'source',
    ]
    rows = _rows_to_link(state, columns)
    source_column = 'source' if 'source' in rows.columns else None

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
                PARCEL_KEY_METHOD,
                source_column=source_column,
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
            step_config={'property_key': property_key},
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
