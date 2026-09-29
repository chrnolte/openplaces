"""Context an entity cannot supply about itself: the share, count
or rank of its group (any id column it already carries) that
satisfies a condition, excluding the row itself. A groupby,
deliberately not a spatial operation (AGENTS.md, patent risk,
shape 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from openplaces.io.curator import CurateState, _register


@_register('derive_group_class_share', phase='infer')
def derive_group_class_share(
    state: CurateState,
    group_column: str,
    output: str,
    evidence_columns: list[str],
    match_values: list[str],
    count_output: str | None = None,
    exclude_self: bool = True,
    min_group_size: int = 2,
) -> CurateState:
    """Share of an entity's group whose evidence carries one of *match_values*.

    Context an entity cannot supply about itself: how much of the group it
    belongs to looks like a given class, according to evidence that was
    already there. A subdivided manufactured-home community is the motivating
    case -- each home sits on its own lot, so a per-parcel count sees one home
    per parcel and the community is invisible, while the group it shares (a
    census block) is three-quarters manufactured homes.

    Emits a *share*, not a flag: per the curate stage's two-layer split, an
    indicator column holds a measurement and every cutoff lives in the vote
    that reads it.

    Method note: why not the obvious spatial shape
    ----------------------------------------------
    The grouping is a **groupby on an identifier the entity already carries**
    -- a census block, a parcel, any space-partitioning id assigned upstream
    by ``link_geographic_ids``. It performs no geometric operation of its own:
    no buffering, no unioning or dissolving of boundaries, no distance or
    nearest-neighbor search, and no interpolation between neighbors. That is a
    deliberate choice, not an incidental one. Identifying "communities" by
    enlarging and merging parcel boundaries, then reading a value off the
    neighbors, is a technique shape with active patents in the property-data
    space (see the patent-risk section of ``AGENTS.md``); aggregating a
    statistic within a published administrative unit is both mechanistically
    different and far older practice. A radius-based neighborhood and a
    block-level groupby answer the question about equally well -- measured on
    Harris County, TX, they agree closely -- so the groupby wins.

    It is also the same mechanism
    :func:`flag_manufactured_home_communities` already uses, one column over,
    which keeps two related signals on one code path.

    Feedback loops
    --------------
    *evidence_columns* must hold evidence that exists **before** the vote this
    share feeds. Computing the share from a vote's own output and then scoring
    it in that same vote would let a class reinforce itself; the recipe makes
    the same point about ``manufactured_home_community``.

    Parameters
    ----------
    group_column : str
        Column holding the group id (e.g. ``census_block_id``). Rows with no
        group id get a missing share.
    output : str
        Column to write the share (0-1) into.
    evidence_columns : list of str
        Columns to read the class from. A row counts toward the group's
        matches when **any** of them holds one of *match_values*; absent
        columns are ignored.
    match_values : list of str
        Values that count as the class.
    count_output : str, optional
        Column to also write the raw match count per group into.
    exclude_self : bool, default True
        Compute each row's share over its group *excluding that row*, so the
        signal is genuinely about the neighbors rather than partly restating
        the row's own evidence.
    min_group_size : int, default 2
        Groups smaller than this get a missing share rather than a value
        computed from too little to mean anything. With *exclude_self* the
        effective minimum is one neighbor.
    """
    curated = state.curated
    if group_column not in curated.columns:
        if state.verbose:
            print(
                f'  derive_group_class_share: {group_column} absent, '
                f'{output} not derived.'
            )
        return state

    present = [c for c in evidence_columns if c in curated.columns]
    if not present:
        if state.verbose:
            print(
                '  derive_group_class_share: none of '
                f'{evidence_columns} present, {output} not derived.'
            )
        return state

    wanted = set(match_values)
    is_match = pd.Series(False, index=curated.index)
    for column in present:
        is_match = is_match | curated[column].astype(object).isin(wanted)
    is_match = is_match.fillna(False)

    groups = curated[group_column]
    matches = is_match.astype(float).groupby(groups).transform('sum')
    sizes = groups.groupby(groups).transform('size').astype(float)

    if exclude_self:
        matches = matches - is_match.astype(float)
        sizes = sizes - 1.0

    share = matches / sizes.where(sizes > 0)
    share = share.where(sizes >= max(min_group_size - (1 if exclude_self else 0), 1))
    share = share.where(groups.notna())

    curated[output] = share.astype('Float64')
    if count_output:
        curated[count_output] = matches.where(groups.notna()).astype('Int64')
    state.curated = curated

    if state.verbose:
        described = share.describe()
        print(
            f'  derive_group_class_share: {output} over {group_column} -- '
            f'{int(share.notna().sum()):,} rows, mean '
            f'{described.get("mean", float("nan")):.3f}'
        )
    return state


def _rows_where(curated: pd.DataFrame, where: list[dict] | None) -> pd.Series:
    """Rows satisfying every indicator in *where* (all rows when empty)."""
    from openplaces.io.curator.indicators import evaluate_indicator

    matched = pd.Series(True, index=curated.index)
    for indicator in where or []:
        matched = matched & evaluate_indicator(curated, indicator)
    return matched.fillna(False).astype(bool)


@_register('derive_group_count', phase='infer')
def derive_group_count(
    state: CurateState,
    group_column: str,
    output: str,
    where: list[dict] | None = None,
) -> CurateState:
    """Count, per row, the rows of its own group that satisfy *where*.

    Writes the same count onto every row of a group, the row itself
    included when it qualifies. The motivating case is the second
    occupancy pass on small secondary footprints: how many footprints on
    this footprint's parcel are primary and were voted Manufactured Home
    by the first pass (``n_primary_manufactured_homes_per_parcel``).

    Method note: a groupby on the entity's own id
    ---------------------------------------------
    The group is an identifier the row already carries (``parcel_id``),
    and the count is a plain groupby over it. No geometry is read: no
    buffering, no union of boundaries, no nearest-neighbor or distance
    search, and nothing from neighboring groups, and no parameter is
    learned from the data. That keeps it apart from the geometric
    "community" detection shape of the patent-risk section of AGENTS.md,
    in the same way as :func:`derive_group_class_share`.

    Feedback loops
    --------------
    A count over a vote's output may feed a later vote only if that
    vote cannot change the rows counted. The second occupancy pass
    counts primaries and rewrites secondaries only, which is what makes
    reading the first pass's classes safe there.

    Parameters
    ----------
    group_column : str
        Column holding the group id. Rows without one get a missing
        count, and a missing group column leaves *output* unwritten.
    output : str
        Column to write the count into (nullable integer).
    where : list of dict, optional
        Voting indicators (see
        :func:`~openplaces.io.curator.indicators.evaluate_indicator`) a
        row must all satisfy to be counted; every row counts when
        omitted.
    """
    curated = state.curated
    if group_column not in curated.columns:
        if state.verbose:
            print(f'  derive_group_count: {group_column} absent, {output} not derived.')
        return state
    groups = curated[group_column]
    qualifies = _rows_where(curated, where) & groups.notna()
    counts = qualifies.astype('int64').groupby(groups).transform('sum')
    curated[output] = counts.where(groups.notna()).astype('Int64')
    state.curated = curated
    if state.verbose:
        print(
            f'  derive_group_count: {output} over {group_column}, '
            f'{int(qualifies.sum()):,} qualifying rows'
        )
    return state


@_register('derive_group_rank', phase='infer')
def derive_group_rank(
    state: CurateState,
    group_column: str,
    output: str,
    rank_by: str,
    where: list[dict] | None = None,
    id_column: str | None = None,
) -> CurateState:
    """Rank the qualifying rows of each group by *rank_by*, largest first.

    A qualifying row's rank is 1 plus the number of qualifying rows of
    its group that come before it: a larger *rank_by*, or an equal one
    and a smaller id. Ties are therefore broken deterministically by id,
    never by row order, so a rerun ranks a parcel's footprints the same
    way. Rows that do not qualify, or have no group, get a missing rank;
    a missing *rank_by* ranks after every present value.

    The second occupancy pass reads it against a record's unit count:
    where a parcel's roll lists two mobile homes, its two largest
    Manufactured Home footprints keep that class
    (``manufactured_home_rank_on_parcel``). Like
    :func:`derive_group_count` it is a groupby on an id the row carries,
    with no geometry and no learned parameter.

    Parameters
    ----------
    group_column : str
        Column holding the group id.
    output : str
        Column to write the rank into (nullable integer).
    rank_by : str
        Numeric column ranked in descending order (e.g. ``area_m2``).
    where : list of dict, optional
        Voting indicators a row must all satisfy to be ranked; every row
        with a group is ranked when omitted.
    id_column : str, optional
        Column breaking ties, ascending. Defaults to the frame's index
        (the entity id on a curated table).
    """
    curated = state.curated
    if group_column not in curated.columns or rank_by not in curated.columns:
        if state.verbose:
            print(
                f'  derive_group_rank: {group_column} or {rank_by} absent, '
                f'{output} not derived.'
            )
        return state
    groups = curated[group_column]
    qualifies = _rows_where(curated, where) & groups.notna()
    rank = pd.Series(pd.NA, index=curated.index, dtype='Int64')
    if qualifies.any():
        ids = (
            curated[id_column]
            if id_column and id_column in curated.columns
            else pd.Series(curated.index, index=curated.index)
        )
        frame = pd.DataFrame(
            {
                'group': groups[qualifies].to_numpy(),
                'value': pd.to_numeric(curated[rank_by], errors='coerce')[
                    qualifies
                ].to_numpy(),
                'id': ids[qualifies].astype(str).to_numpy(),
                'position': np.flatnonzero(qualifies.to_numpy()),
            }
        )
        frame = frame.sort_values(
            ['group', 'value', 'id'],
            ascending=[True, False, True],
            na_position='last',
            kind='mergesort',
        )
        frame['rank'] = frame.groupby('group', sort=False).cumcount() + 1
        values = rank.to_numpy(dtype=object)
        values[frame['position'].to_numpy()] = frame['rank'].to_numpy()
        rank = pd.Series(values, index=curated.index).astype('Int64')
    curated[output] = rank
    state.curated = curated
    if state.verbose:
        print(
            f'  derive_group_rank: {output} over {group_column} by {rank_by}, '
            f'{int(qualifies.sum()):,} ranked rows'
        )
    return state
