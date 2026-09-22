"""Registered curation steps specific to the transaction entity type."""

from __future__ import annotations

import re

import pandas as pd

from openplaces.io.curator import CurateState, _register
from openplaces.io.readers import get_entities
from openplaces.table import aggregate_rows

_NON_ALNUM = re.compile(r'[^0-9A-Za-z]')


def _normalized(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(_NON_ALNUM, '', regex=True)


@_register('dedup_transactions')
def dedup_transactions(state: CurateState, key_columns: list) -> CurateState:
    """Drop rows that are the same recorded document, kept once.

    Sale sources published as overlapping rolling windows (e.g. FL DOR's
    SDF) can carry the same legal transaction in two adjacent files. A
    ``transaction_id_source`` cannot be used to detect this (assigned
    per-county, not unique across a source's full coverage); the
    composite key below identifies the underlying document instead.

    Parameters
    ----------
    key_columns : list of str
        Columns whose combination identifies one recorded document. The
        first occurrence of a repeated combination is kept. A column the
        table lacks reads as missing on every row: the key is written
        once for every source, and a clerk instrument number exists in
        Florida and nowhere else.
    """
    curated = state.curated
    key = (
        curated.reindex(columns=key_columns)
        .astype('string')
        .fillna('NA')
        .agg('|'.join, axis=1)
    )
    mask = ~key.duplicated(keep='first')
    n_dropped = int((~mask).sum())
    state.curated = curated.loc[mask].copy()
    if state.verbose:
        print(f'  dedup_transactions: dropped {n_dropped:,} duplicate rows')
    return state


@_register('derive_sale_period')
def derive_sale_period(
    state: CurateState, date_column: str = 'recorded_date'
) -> CurateState:
    """Fill `sale_year` and `sale_month` from a date where they are missing.

    Sources state when a sale happened in one of two ways: a date (a
    recorder's table) or a year and month (Florida's roll, and any roll
    that records the month only). Every later step compares sales by
    year and month, so both spellings are brought to that form here. A
    year or month the source stated itself is never overwritten.

    Parameters
    ----------
    date_column : str, optional
        Column holding the date (default `recorded_date`).
    """
    curated = state.curated
    if date_column not in curated.columns:
        return state
    date = pd.to_datetime(curated[date_column], errors='coerce')
    for column, part in (('sale_year', date.dt.year), ('sale_month', date.dt.month)):
        derived = part.astype('float64')
        if column in curated.columns:
            stated = pd.to_numeric(curated[column], errors='coerce')
            derived = stated.fillna(derived)
        curated[column] = derived
    state.curated = curated
    return state


@_register('flag_sales_matching_other_kind')
def flag_sales_matching_other_kind(
    state: CurateState,
    key_column: str,
    kind_column: str = 'sale_record_kind',
    flagged_kind: str = 'assessor_last_sale',
    reference_kind: str = 'deed',
    output: str = 'sale_matches_deed',
) -> CurateState:
    """Flag an assessor last-sale row that a recorded deed also reports.

    A county with a recorder's table and a roll reports its most recent
    sales twice, once as a deed and once as the roll's last-sale field.
    Both rows are kept, since nothing is filtered here; this flag is what
    lets a consumer keep the deed and drop its echo.

    The test is exact equality of four values: *key_column*, `sale_year`,
    `sale_month` and `price`. Month, not day, because a roll often
    records the month only (every last-sale date of Pitt County NC falls
    on the first of a month, and the matching deed follows within 30
    days). Nothing is fuzzy, scored or tuned, and no row is removed or
    relinked. Measured on Pitt County NC, 2026-09-20: 89.9% of
    last-sale rows whose parcel has any recorded deed match one this way.

    Parameters
    ----------
    key_column : str
        Column identifying the parcel both rows name.
    kind_column : str, optional
        Column holding the record kind (default `sale_record_kind`).
    flagged_kind, reference_kind : str, optional
        The kind that receives the flag and the kind it is compared to.
    output : str, optional
        Column written: 1 on a *flagged_kind* row with a match, 0 on one
        without, missing on every other row and on a *flagged_kind* row
        lacking any of the four values.
    """
    curated = state.curated
    needed = [key_column, kind_column, 'sale_year', 'sale_month', 'price']
    if any(c not in curated.columns for c in needed):
        # No comparison is possible (Massachusetts rows carry no parcel
        # key); the column is still written, so every unit has it.
        curated[output] = float('nan')
        state.curated = curated
        return state
    values = curated[[key_column, 'sale_year', 'sale_month', 'price']]
    complete = values.notna().all(axis=1)
    # Column-wise concatenation: a missing part makes the key missing.
    text = values.astype('string')
    key = text.iloc[:, 0]
    for column in text.columns[1:]:
        key = key + '|' + text[column]
    reference = set(key[curated[kind_column] == reference_kind].dropna())
    flag = pd.Series(float('nan'), index=curated.index, dtype='float64')
    target = (curated[kind_column] == flagged_kind) & complete
    flag.loc[target] = key.loc[target].isin(reference).astype('float64')
    curated[output] = flag
    state.curated = curated
    if state.verbose:
        print(
            f'  flag_sales_matching_other_kind: {int((flag == 1).sum()):,} of '
            f'{int(target.sum()):,} {flagged_kind} rows match a {reference_kind}'
        )
    return state


@_register('collapse_double_closings')
def collapse_double_closings(
    state: CurateState,
    key_column: str,
    max_gap_months: int = 1,
    keep: str = 'last',
    output: str | None = None,
    within: list[str] | None = None,
) -> CurateState:
    """Drop, or flag, the earlier leg of a double closing.

    Two genuinely different recorded documents for the same entity,
    close together in time and at an identical price, most likely
    represent one real transfer recorded as two deeds (e.g. a
    pass-through through a broker's intermediary entity) rather than two
    independent market sales. Keeping both double-counts one
    transaction. Distinct from :func:`dedup_transactions`, which removes
    the same document recorded twice, not two different documents.

    Assumes the canonical transaction columns ``sale_year``,
    ``sale_month``, ``price``, ``sale_book``, ``sale_page``.

    Parameters
    ----------
    key_column : str
        Column identifying the entity (e.g. a parcel id) the sales share.
    max_gap_months : int, optional
        Maximum gap between the two sales, in months (default 1).
    keep : {'last'}, optional
        Which leg survives. Only 'last' (the later sale) is implemented,
        matching the rationale that the later leg reflects who actually
        ended up owning the property.
    output : str, optional
        Write the judgment instead of acting on it: the named column
        gets 1 on an earlier leg and 0 elsewhere, and no row is dropped.
        Whether two documents a month apart at one price are one sale
        is a judgment a consumer may want to make with its own gap and
        price tolerance, so the canonical entity carries the flag and a
        filtered product drops the row.
    within : list of str, optional
        Columns a pair must also share, e.g. `[sale_record_kind]`. A
        deed and the assessor's last-sale record of that same deed sit
        on one parcel at one price in one month, and the assessor row
        often has no book and page, which reads as "a different
        document": without this every such pair would be flagged as a
        double closing. Columns the table lacks are ignored.
    """
    if keep != 'last':
        raise NotImplementedError("collapse_double_closings only supports keep='last'.")
    curated = state.curated
    if key_column not in curated.columns:
        # Nothing to compare by (Massachusetts rows carry no assessor
        # parcel id). The flag is still written, as missing, so that
        # every unit's table has the column and 0 keeps meaning "was
        # checked and is not an earlier leg".
        if output is not None:
            curated[output] = float('nan')
            state.curated = curated
        return state

    keys = [key_column, *[c for c in within or [] if c in curated.columns]]
    df = curated.reindex(
        columns=[*keys, 'sale_year', 'sale_month', 'price', 'sale_book', 'sale_page']
    )
    period = pd.to_numeric(df['sale_year'], errors='coerce') * 12 + pd.to_numeric(
        df['sale_month'], errors='coerce'
    )
    df = df.assign(_period=period).sort_values([*keys, '_period'])

    grouped = df.groupby(keys)
    prev_period = grouped['_period'].shift(1)
    prev_price = grouped['price'].shift(1)
    prev_book = grouped['sale_book'].shift(1)
    prev_page = grouped['sale_page'].shift(1)

    gap = df['_period'] - prev_period
    different_document = (df['sale_book'] != prev_book) | (df['sale_page'] != prev_page)
    same_price = df['price'] == prev_price
    is_later_leg = (
        gap.notna() & gap.between(0, max_gap_months) & different_document & same_price
    )

    # df is sorted by (key_column, _period), and is_later_leg is only True
    # where the shifted-within-group lookups above are non-null, so the
    # row immediately before it, by position in this sorted frame, is
    # guaranteed to be its own group's earlier leg.
    position = pd.Series(range(len(df)), index=df.index)
    drop_index = df.index[position[is_later_leg].to_numpy() - 1]

    if output is not None:
        flag = pd.Series(0, index=curated.index, dtype='int64')
        flag.loc[drop_index] = 1
        curated[output] = flag
        state.curated = curated
        if state.verbose:
            print(f'  collapse_double_closings: flagged {len(drop_index):,} rows')
        return state
    state.curated = curated.drop(index=drop_index)
    if state.verbose:
        print(f'  collapse_double_closings: dropped {len(drop_index):,} rows')
    return state


@_register('join_temporal_snapshot')
def join_temporal_snapshot(
    state: CurateState,
    recipe_id: str,
    join_key: str,
    date_column: str,
    vintage_column: str,
    columns: list,
    direction: str = 'backward',
    offset_years: int | None = None,
    only_unmatched: bool = False,
    restrict_to: dict | None = None,
    match_type_column: str | None = None,
    prefix: str = '',
    require_panel: bool = False,
    no_panel_value: str = 'no_panel',
    single_vintage_value: str = 'single_vintage',
    mark_unmatched: str | None = None,
) -> CurateState:
    """Attach a dated reference snapshot valid at (or near) each row's date.

    The one temporal (as-of) join in the repo: nothing else here uses
    ``pd.merge_asof``. *join_key* is normalized (stripped of
    non-alphanumeric characters) on both sides before matching, the same
    way :func:`~openplaces.io.curator.filters.normalize_id_column` treats
    the entity side, since the reference recipe is loaded fresh here and
    never sees that step.

    Parameters
    ----------
    recipe_id : str
        Recipe whose output supplies the snapshot (e.g. a property roll).
    join_key : str
        Column present on both the entity and the reference, identifying
        what the snapshot is *of* (e.g. a parcel id).
    date_column : str
        Entity-side column giving the date to match (e.g. a sale year).
    vintage_column : str
        Reference-side column giving each snapshot's own year.
    columns : list of str
        Reference columns to attach.
    direction : {'backward', 'forward', 'exact'}, optional
        ``'backward'``/``'forward'``: nearest reference year at or before
        / at or after *date_column* (a :func:`pandas.merge_asof` pass).
        ``'exact'``: only a reference year equal to *date_column* plus
        *offset_years* (required for this mode).
    offset_years : int, optional
        Required when *direction* is ``'exact'``; the number of years
        after *date_column* the reference vintage must equal.
    only_unmatched : bool, optional
        Restrict this pass to rows where *match_type_column* is not yet
        set (default False), so a later pass fills only what an earlier
        pass left unmatched.
    restrict_to : dict, optional
        ``{'column': ..., 'equals': ...}``. Restrict this pass to rows
        where that column equals that value.
    match_type_column : str, optional
        Column to write ``'exact'`` / ``'{direction}_fallback'`` into for
        rows this pass matched. Left untouched for rows it does not
        reach, so a later pass can still claim them.
    prefix : str, optional
        Prepended to each attached column's name (default ``''``).
    require_panel : bool, optional
        For a pipeline that runs everywhere: return quietly, rather than
        raising, where this admin unit has no reference on disk, and
        where the reference carries only one vintage. Only Florida's DOR
        roll is a panel today (24 yearly rolls); the other 104 assessor
        sources are one vintage each, so a nationwide recipe must treat
        "no panel here" as the ordinary case
        (`plans/multi-year-tax-rolls-property-panel.md`).
    no_panel_value, single_vintage_value : str, optional
        Written into *match_type_column* for every active row when
        *require_panel* turns the pass off. **A missing value would not
        say this**: it cannot distinguish a county with no panel from a
        sale a panel did not match, and the two mean very different
        things to anyone reading the output.
    mark_unmatched : str, optional
        Written into *match_type_column* for active rows this pass left
        unset. Belongs on the last temporal pass of a recipe, where a
        row still unmarked genuinely means "the panel was read and did
        not have this property".
    """
    # Validate the request before reading the reference, so a recipe
    # error is reported as one rather than as a missing data file.
    if direction not in ('backward', 'forward', 'exact'):
        raise ValueError(f'Unknown direction: {direction!r}')
    if direction == 'exact' and offset_years is None:
        raise ValueError("direction='exact' requires offset_years.")
    curated = state.curated

    active_mask = pd.Series(True, index=curated.index)
    if only_unmatched:
        active_mask &= curated[match_type_column].isna()
    if restrict_to:
        # A restriction on a column this source does not have selects no
        # rows, rather than raising: the national pipeline runs this step
        # over every state, and `sale_vacant` is Florida's word. Reached
        # before the panel check below, so it has to be survivable on its
        # own (Wisconsin has neither the column nor a panel).
        if restrict_to['column'] not in curated.columns:
            return state
        active_mask &= curated[restrict_to['column']] == restrict_to['equals']
    if not active_mask.any():
        return state

    active = curated.loc[active_mask, [join_key, date_column]].copy()
    active[join_key] = _normalized(active[join_key])
    # Both sides to one float dtype: merge_asof refuses keys of
    # different types, and a sale year with any missing value parses to
    # float64 while a complete tax_year column parses to int64, so the
    # join fails on exactly the counties whose sale dates are imperfect.
    active[date_column] = pd.to_numeric(active[date_column], errors='coerce').astype(
        'float64'
    )

    def _mark_all(value: str) -> CurateState:
        """Say why this pass did nothing, rather than leaving it blank."""
        if match_type_column:
            if match_type_column not in curated.columns:
                curated[match_type_column] = pd.NA
            curated.loc[active_mask, match_type_column] = value
            state.curated = curated
        if state.verbose:
            print(f'  join_temporal_snapshot ({recipe_id}): {value}')
        return state

    try:
        ref = get_entities(recipe_id, admin_id=state.admin_id)
    except (FileNotFoundError, OSError, KeyError, ValueError):
        if not require_panel:
            raise
        return _mark_all(no_panel_value)
    if require_panel and (
        ref is None
        or vintage_column not in ref.columns
        or pd.to_numeric(ref[vintage_column], errors='coerce').nunique() < 2
    ):
        # One vintage is a roll, not a panel: joining a sale to the only
        # year on file would read as a temporal match while telling the
        # reader nothing the cross-sectional spine did not already say.
        return _mark_all(single_vintage_value)
    # Take the columns this roll actually has. One nationwide recipe
    # names one column list, and a county's roll is free not to carry
    # every field: Lake County FL has no `land_area_sqft`, and asking
    # for it failed the whole curate rather than attaching the seven
    # columns it does have.
    available = [c for c in columns if c != vintage_column and c in ref.columns]
    if not available:
        return _mark_all(no_panel_value) if require_panel else state
    ref_columns = [join_key, vintage_column] + available
    ref = ref[ref_columns].copy()
    columns = [c for c in columns if c == vintage_column or c in available]
    ref[join_key] = _normalized(ref[join_key])
    ref[vintage_column] = pd.to_numeric(ref[vintage_column], errors='coerce').astype(
        'float64'
    )
    ref = ref.dropna(subset=[join_key, vintage_column])

    # A sale with no year cannot be placed in time at all, and
    # merge_asof refuses a null key outright. Set them aside under their
    # own label: calling them `not_in_panel` would blame the panel for a
    # gap in the sale record.
    undated = active[date_column].isna()
    if undated.any():
        if match_type_column:
            if match_type_column not in curated.columns:
                curated[match_type_column] = pd.NA
            curated.loc[active.index[undated], match_type_column] = 'no_sale_date'
        active = active[~undated]
    if active.empty:
        state.curated = curated
        return state

    if direction == 'exact':
        ref = ref.drop_duplicates(subset=[join_key, vintage_column])
        active['_target_year'] = active[date_column] + offset_years
        merged = active.merge(
            ref,
            left_on=[join_key, '_target_year'],
            right_on=[join_key, vintage_column],
            how='left',
        )
        merged.index = active.index
    elif direction in ('backward', 'forward'):
        active_sorted = active.sort_values(date_column)
        ref_sorted = ref.sort_values(vintage_column)
        merged = pd.merge_asof(
            active_sorted,
            ref_sorted,
            left_on=date_column,
            right_on=vintage_column,
            by=join_key,
            direction=direction,
        )
        merged.index = active_sorted.index

    matched = merged[vintage_column].notna()
    matched_index = merged.index[matched]

    for source_column in columns:
        dest_column = f'{prefix}{source_column}'
        if dest_column not in curated.columns:
            curated[dest_column] = pd.NA
        curated.loc[matched_index, dest_column] = merged.loc[
            matched, source_column
        ].to_numpy()

    if match_type_column:
        if match_type_column not in curated.columns:
            curated[match_type_column] = pd.NA
        is_exact = matched & (merged[date_column] == merged[vintage_column])
        curated.loc[merged.index[matched & ~is_exact], match_type_column] = (
            f'{direction}_fallback'
        )
        curated.loc[merged.index[matched & is_exact], match_type_column] = 'exact'
        if mark_unmatched is not None:
            # Every row still unset, not only this pass's active ones.
            # The last pass is usually a restricted one (a bare lot must
            # not inherit a later vintage's house), so marking only its
            # own rows leaves exactly the rows it declined to touch
            # blank: 0.4% of Volusia stayed empty that way, which is the
            # ambiguity this parameter exists to remove.
            curated.loc[curated[match_type_column].isna(), match_type_column] = (
                mark_unmatched
            )

    state.curated = curated
    if state.verbose:
        print(
            f'  join_temporal_snapshot ({recipe_id}, {direction}): '
            f'matched {int(matched.sum()):,} / {len(active):,} active rows'
        )
    return state


@_register('derive_document_id')
def derive_document_id(
    state: CurateState,
    candidates: list,
    output: str = 'sale_document_id',
    separator: str = '/',
    scope_column: str | None = None,
    unscoped_value: str | None = None,
) -> CurateState:
    """Name the recorded document (deed) behind each sale row.

    A deed covering several parcels is published once per parcel, and a
    consumer that treats each row as a sale counts one transfer several
    times. The document identifier is what lets the rows be recognized
    as one sale (:func:`count_parcels_per_document`,
    :func:`aggregate_multi_parcel_sales`), and every source spells it
    differently, which is why it is assembled here once rather than in
    every consumer.

    Parameters
    ----------
    candidates : list of list of str
        Column groups to try in order; the first whose columns are all
        present on a row names its document. Florida supplies a clerk
        instrument number for newer sales and an official-record book
        and page for older ones, so ``[[sale_clerk_instrument],
        [sale_book, sale_page]]``. Values are stripped of punctuation
        and joined by *separator*.
    output : str, optional
        Column written (default ``sale_document_id``). Missing where no
        candidate is complete.
    separator : str, optional
        Joins a multi-column candidate.
    scope_column : str, optional
        Column whose value is prefixed to the identifier (`value:id`),
        so that rows of different scopes never share one. With
        `sale_record_kind`, a deed and the assessor's last-sale record
        citing that deed's book and page stay two rows: counted or
        aggregated together they would read as one sale of two parcels.
        Assessor rows of one deed still share an identifier with each
        other, which is what lets a multi-parcel last sale be counted.
    unscoped_value : str, optional
        The *scope_column* value left unprefixed (`deed`), so identifiers
        already published for it do not change.
    """
    curated = state.curated
    document = pd.Series(pd.NA, index=curated.index, dtype='string')
    for group in candidates:
        columns = [group] if isinstance(group, str) else list(group)
        if any(c not in curated.columns for c in columns):
            continue
        parts = []
        for c in columns:
            part = _normalized(curated[c]).astype('string')
            # A part that is empty or all zeros is a placeholder, not an
            # identifier. Glades County FL's roll writes a single space
            # for book and page on every row, which named one document
            # ('/') for all 1,333 of its last sales and folded them into
            # a single "deed".
            placeholder = part.str.fullmatch('0*').fillna(True)
            parts.append(part.where(curated[c].notna() & ~placeholder))
        complete = pd.concat(parts, axis=1).notna().all(axis=1)
        joined = parts[0]
        for part in parts[1:]:
            joined = joined + separator + part
        document = document.where(document.notna() | ~complete, joined)
    if scope_column is not None and scope_column in curated.columns:
        scope = curated[scope_column].astype('string')
        scoped = document.notna() & scope.notna()
        if unscoped_value is not None:
            scoped &= (scope != unscoped_value).fillna(False)
        document = document.where(~scoped, scope + ':' + document)
    curated[output] = document
    state.curated = curated
    if state.verbose:
        print(
            f'  derive_document_id: {int(document.notna().sum()):,} of '
            f'{len(curated):,} rows name a document'
        )
    return state


@_register('count_parcels_per_document')
def count_parcels_per_document(
    state: CurateState,
    parcel_column: str,
    document_column: str = 'sale_document_id',
    output: str = 'n_parcels_per_sale',
) -> CurateState:
    """Count the distinct parcels each recorded document covered.

    Written on every row, so a consumer can tell a multi-parcel sale from
    a single-parcel one before or after :func:`aggregate_multi_parcel_sales`
    has collapsed it. Run after the source's duplicate rows are removed
    (:func:`dedup_transactions`), or a document listed twice for the same
    parcel counts as two.

    Parameters
    ----------
    parcel_column : str
        Column identifying the parcel a row is about.
    document_column : str, optional
        Column naming the document (default ``sale_document_id``).
    output : str, optional
        Column written (default ``n_parcels_per_sale``). Missing where
        the document is.
    """
    curated = state.curated
    if document_column not in curated.columns or parcel_column not in curated.columns:
        return state
    counts = curated.groupby(document_column, dropna=True)[parcel_column].nunique()
    curated[output] = curated[document_column].map(counts).astype('float')
    state.curated = curated
    return state


@_register('aggregate_multi_parcel_sales')
def aggregate_multi_parcel_sales(
    state: CurateState,
    document_column: str = 'sale_document_id',
    weight_column: str | None = None,
    count_column: str = 'n_parcels_per_sale',
) -> CurateState:
    """Emit one row per recorded document instead of one per parcel.

    The rows of a multi-parcel deed are right about the price and wrong
    about the area: the source repeats one price on every parcel the
    deed covered. Grouped by document, the parcels become one observation
    whose extensive columns (an area, a count) are summed and whose
    other columns are taken from the row with the largest *weight_column*
    (the registry's aggregation rule decides which is which, through
    :func:`openplaces.table.aggregate_rows`). The price was never wrong;
    the denominator was.

    Rows with no document id, or alone on theirs, are kept as they are.
    The aggregated row keeps the index label of its heaviest member, so
    the entity id stays a real row's id, and *count_column* (written by
    :func:`count_parcels_per_document`) says how many rows it stands
    for. A column the registry does not know cannot be aggregated and is
    left missing on the aggregated rows.

    Parameters
    ----------
    document_column : str, optional
        Column naming the document (default ``sale_document_id``).
    weight_column : str, optional
        Column whose largest value picks the representative row for
        first-aggregated columns (a parcel area). Input order without it.
    count_column : str, optional
        Column holding the parcels-per-document count, kept as the
        representative member's value (default ``n_parcels_per_sale``).
    """
    curated = state.curated
    if document_column not in curated.columns:
        return state
    has_document = curated[document_column].notna()
    shared = has_document & curated[document_column].duplicated(keep=False)
    if not shared.any():
        return state
    members = curated.loc[shared].copy()
    if weight_column is not None and weight_column in members.columns:
        members = members.sort_values(weight_column, ascending=False)
    # The representative row's label, taken before the registry-driven
    # aggregation, which keeps only columns the registry knows.
    label = members.index.to_series().groupby(members[document_column]).first()
    aggregated = aggregate_rows(
        members,
        by=document_column,
        aggregation_function={count_column: 'first'},
    )
    aggregated[document_column] = aggregated.index
    aggregated.index = pd.Index(
        label.reindex(aggregated.index).to_numpy(), name=curated.index.name
    )
    kept = curated.loc[~shared]
    result = pd.concat([kept, aggregated.reindex(columns=kept.columns)])
    # Back in input order, so the step is stable under a later sort.
    result = result.loc[curated.index[curated.index.isin(result.index)]]
    state.curated = result
    if state.verbose:
        print(
            f'  aggregate_multi_parcel_sales: {int(shared.sum()):,} rows on '
            f'{len(aggregated):,} multi-parcel documents became one row each'
        )
    return state
