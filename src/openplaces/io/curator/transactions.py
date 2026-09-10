"""Registered curation steps specific to the transaction entity type."""

from __future__ import annotations

import re

import pandas as pd

from openplaces.io.curator import CurateState, _register
from openplaces.io.readers import get_entities

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
        first occurrence of a repeated combination is kept.
    """
    curated = state.curated
    key = curated[key_columns].astype('string').fillna('NA').agg('|'.join, axis=1)
    mask = ~key.duplicated(keep='first')
    n_dropped = int((~mask).sum())
    state.curated = curated.loc[mask].copy()
    if state.verbose:
        print(f'  dedup_transactions: dropped {n_dropped:,} duplicate rows')
    return state


@_register('collapse_double_closings')
def collapse_double_closings(
    state: CurateState,
    key_column: str,
    max_gap_months: int = 1,
    keep: str = 'last',
) -> CurateState:
    """Drop the earlier leg of a double closing.

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
    """
    if keep != 'last':
        raise NotImplementedError("collapse_double_closings only supports keep='last'.")
    curated = state.curated
    if key_column not in curated.columns:
        return state

    df = curated[
        [key_column, 'sale_year', 'sale_month', 'price', 'sale_book', 'sale_page']
    ]
    period = pd.to_numeric(df['sale_year'], errors='coerce') * 12 + pd.to_numeric(
        df['sale_month'], errors='coerce'
    )
    df = df.assign(_period=period).sort_values([key_column, '_period'])

    grouped = df.groupby(key_column)
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
        reach.
    prefix : str, optional
        Prepended to each attached column's name (default ``''``).
    """
    curated = state.curated

    active_mask = pd.Series(True, index=curated.index)
    if only_unmatched:
        active_mask &= curated[match_type_column].isna()
    if restrict_to:
        active_mask &= curated[restrict_to['column']] == restrict_to['equals']
    if not active_mask.any():
        return state

    active = curated.loc[active_mask, [join_key, date_column]].copy()
    active[join_key] = _normalized(active[join_key])
    active[date_column] = pd.to_numeric(active[date_column], errors='coerce')

    ref = get_entities(recipe_id, admin_id=state.admin_id)
    ref_columns = [join_key, vintage_column] + [
        c for c in columns if c != vintage_column
    ]
    ref = ref[ref_columns].copy()
    ref[join_key] = _normalized(ref[join_key])
    ref[vintage_column] = pd.to_numeric(ref[vintage_column], errors='coerce')
    ref = ref.dropna(subset=[join_key, vintage_column])

    if direction == 'exact':
        if offset_years is None:
            raise ValueError("direction='exact' requires offset_years.")
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
    else:
        raise ValueError(f'Unknown direction: {direction!r}')

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

    state.curated = curated
    if state.verbose:
        print(
            f'  join_temporal_snapshot ({recipe_id}, {direction}): '
            f'matched {int(matched.sum()):,} / {len(active):,} active rows'
        )
    return state
