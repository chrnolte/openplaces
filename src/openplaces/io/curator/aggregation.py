"""Registered curation steps that reduce one entity's rows onto another.

Curate assembles one dataset per entity type by aggregating across
entities: a parcel's year built or living area comes from the property
rows on it, reduced here, not from a copy the geometry phase made. The
harmonize stage keeps each entity's core table with minimal redundancy
(see AGENTS.md, "Entity model and stage roles"), so a curate recipe that
wants a property attribute on parcels asks for it here.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from openplaces.core.attribute_registry import get_agg_func
from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState, _register
from openplaces.io.curator.provenance import record_sources, source_column
from openplaces.io.readers import describe_recipe, get_entities
from openplaces.io.stacked_units import STACKED_UNITS_LABEL_SUFFIX
from openplaces.recipe import resolve_attribute_name, source_id_from_recipe_id
from openplaces.table import _agg_func_for, _has_agg_func

#: Reference column naming the source each row came from whole, as
#: `union_spine_sources` writes it on a property spine.
ROW_SOURCE_COLUMN = 'source'


def _row_source_tokens(labels: pd.Series) -> pd.Series:
    """A spine's per-row `source` labels as provenance tokens.

    A delivery withholds a restricted source's cell by finding its
    source id among the sidecar's tokens, and reads any other token as
    a different source that exonerates the cell. So each `+` part is
    reduced to the source id: the stacked-units label
    (`edgecombecounty:units`) loses its suffix, and an explicit source
    labeled with its recipe id becomes that recipe's source id. Copied
    verbatim, either would have cleared Edgecombe County's restricted
    values for shipping. A part that is not a recipe id (a source id,
    or a token such as `not_on_roll`) is kept as it is, so a cell's own
    sidecar can pass through here too.
    """

    def normalize(label):
        parts = []
        for part in str(label).split('+'):
            part = part.removesuffix(STACKED_UNITS_LABEL_SUFFIX)
            if '_' in part and _is_admin_id(part.split('_', 1)[0]):
                part = source_id_from_recipe_id(part)
            if part and part not in parts:
                parts.append(part)
        return '+'.join(parts) or None

    values = labels.astype('string')
    lookup = {v: normalize(v) for v in values.dropna().unique()}
    return values.map(lookup).astype(object)


def _is_admin_id(text: str) -> bool:
    """True when *text* parses as an admin id (a recipe id's first part)."""
    try:
        AdminId(text)
    except (ValueError, IndexError, KeyError):
        return False
    return True


def _linked_recipe_id(recipe_id: str) -> str:
    """The recipe whose rows a link table was written for.

    A link table is written in harmonize, between spines. A curate
    recipe keeps its spine's index (`US_property-openplaces-2026`'s
    rows are `US_property-spine-2026`'s property ids), so its rows are
    reached through the spine's link.
    """
    from openplaces.recipe import get_recipe_by_id

    seen = set()
    while recipe_id not in seen:
        seen.add(recipe_id)
        recipe = get_recipe_by_id(recipe_id)
        predecessor = recipe.get('entity_recipe')
        if recipe.get('stage') != 'curate' or not predecessor:
            return recipe_id
        recipe_id = predecessor
    return recipe_id


def read_curate_link(state, recipe_id):
    """The link table between the curated entity and *recipe_id*'s rows.

    Returns the table as stored and its path, or (None, None) where the
    unit has no resolvable link (a curate recipe with no geospine, a
    reference outside ENTITY_LINK_ORDER).
    """
    from openplaces.geo.link import get_entity_link_path, get_link_owner_recipe_id
    from openplaces.io.harmonizer.entity_links import read_entity_link

    try:
        owner = get_link_owner_recipe_id(state.recipe)
        linked_id = _linked_recipe_id(recipe_id)
        # A link table is named after the spine on that side (the
        # transaction-to-parcel table names US_parcel-spine-2026) or,
        # where link_entities_by_id wrote it, after the recipe that
        # mints the rows, its geospine (the footprint-to-building table
        # names US_building-geospine-2026). The spine's name is tried
        # first, so every link read before the geospine name was
        # accepted resolves as it did.
        path = None
        for name in dict.fromkeys([linked_id, get_link_owner_recipe_id(linked_id)]):
            path = get_entity_link_path(name, owner, state.admin_id)
            if path.exists():
                break
        links = read_entity_link(path)
    except Exception:
        return None, None
    if links is None or links.empty:
        return None, None
    return links, path


def units_only_links(links: pd.DataFrame) -> np.ndarray:
    """True for links found only by the ingest-time stacked-units split.

    A link's `link_source` joins its sources by `+`; a link every part
    of which is a `<source>:units` label was found through a unit row
    split off the parcel layer, never through a roll.
    """
    tokens = links['link_source'].fillna('').astype(str).str.split('+')
    return tokens.map(
        lambda parts: all(p.endswith(STACKED_UNITS_LABEL_SUFFIX) for p in parts)
    ).to_numpy(dtype=bool)


def mask_units_rows(rows: pd.DataFrame, lot: pd.Series, units, columns) -> pd.Series:
    """Leave the units' rows out where a roll's rows describe the lot.

    Column by column: on a lot where a row other than a units row
    states a column, the units rows' value of that column is masked in
    *rows* (in place). Returns, per row, whether it is left out
    altogether: a units row that gives no column on such a lot.
    """
    units = pd.Series(np.asarray(units, dtype=bool), index=rows.index)
    lot = pd.Series(np.asarray(lot), index=rows.index)
    for column in columns:
        described = lot[~units & rows[column].notna()]
        rows[column] = rows[column].mask(units & lot.isin(set(described)))
    gives_nothing = rows[list(columns)].isna().all(axis=1)
    return units & lot.isin(set(lot[~units])) & gives_nothing


def _rows_by_link(state, recipe_id, reference, values, functions):
    """Arrange the reference's values by the link table, when there is one.

    A property reaches its parcel through a key column only where one
    key can say it. The link table (`io.harmonizer.entity_links`) also
    holds a unit on a stacked lot, whose key names no parcel row, and an
    account on several lots. Where a valid link exists for the unit
    being curated, the rows are grouped by it:

    - links found on a key shared by very many rows on both sides
      (`*_shared_key`) are left out, because summing over them is the
      harm that label records;
    - on a lot where a source other than the ingest-time split states
      a column, the split's own rows (`<source>:units`) are left out of
      that column, so a tax roll and the units split off the parcel
      layer are not summed twice; where only the units state it, they
      give the lot its value. A units row that gives no column on such
      a lot is left out altogether (and from a count). A row both
      describe is one merged row and stays;
    - a property on several lots contributes its `share` of every
      summed column to each.

    Returns None when there is no valid link or it shares no ids with
    the two tables, and the caller joins on the key column as before.
    """
    from openplaces.io.harmonizer.entity_links import SHARED_KEY_SUFFIX

    # No resolvable link for this recipe and unit: the key join of the
    # caller is the documented behavior.
    links, path = read_curate_link(state, recipe_id)
    if links is None:
        return None
    finer_id, coarser_id = links.columns[0], links.columns[1]
    # Which side the curated entity is: the property-to-parcel table
    # reduces finer rows (properties) onto the coarser curated parcels;
    # the transaction-to-parcel table reduces the coarser side (parcels)
    # onto the finer curated sales. The id columns are named for their
    # entity, and so is the curated index.
    curated_name = state.curated.index.name
    if curated_name == finer_id:
        own_id, other_id = finer_id, coarser_id
    else:
        own_id, other_id = coarser_id, finer_id
    other_ids = links[other_id].astype('string')
    own_ids = links[own_id].astype('string')
    links = links[
        other_ids.isin(pd.Index(reference.index).astype('string'))
        & own_ids.isin(pd.Index(state.curated.index).astype('string'))
    ]
    if links.empty:
        warnings.warn(
            f'aggregate_from_entities: the link {path.name} shares no ids with '
            f'the tables for {state.admin_id}; joining on the key column.',
            stacklevel=3,
        )
        return None
    links = links[~links['link_method'].str.endswith(SHARED_KEY_SUFFIX, na=False)]
    only_units = units_only_links(links)

    reference_lookup = values.copy()
    reference_lookup.index = pd.Index(reference.index).astype('string')
    rows = reference_lookup.loc[links[other_id].astype('string')].reset_index(drop=True)
    lot = links[own_id].astype('string').reset_index(drop=True)
    # Units are left out column by column: on a lot where another
    # source's rows state the column, never where only the units do.
    # A roll that describes a lot without a year built must not take
    # away the year its units state. Measured 2026-11-15 on US-NC-TYR:
    # 157 lots kept a year built only through this rule.
    left_out = mask_units_rows(rows, lot, only_units, list(functions))
    links = links[~left_out.to_numpy()]
    rows = rows[~left_out].reset_index(drop=True)
    share = links['share'].astype('Float64').reset_index(drop=True)
    for column, function in functions.items():
        if function == 'sum' and share.notna().any():
            scaled = pd.to_numeric(rows[column], errors='coerce') * share.astype(
                'float64'
            )
            rows[column] = scaled.where(share.notna(), rows[column])
    group_key = links[own_id].astype('string').reset_index(drop=True)
    index_as_string = pd.Index(state.curated.index).astype('string')
    keys = pd.Series(index_as_string, index=state.curated.index)
    if state.verbose:
        print(
            f'  aggregate_from_entities: grouped by {path.name} ({len(links):,d} links)'
        )
    return rows, group_key, keys


#: Reduction for a class a parcel can only state if its properties
#: agree on it (see `_reduce`).
AGREE = 'agree'

#: Reduction for a class an outline states for all of its buildings:
#: every distinct value, joined by `DISTINCT_SEPARATOR` (see `_reduce`).
DISTINCT = 'distinct'
DISTINCT_SEPARATOR = ' | '

#: Reduction for an `_all` list (`parcel_id_local_all`): every member
#: of the group's lists, once each, in the `_all` format (see
#: `_union_lists`).
UNION = 'union'
#: The separator the harmonize overlay joins an `_all` list with
#: (`attributes._shared._join_distinct`).
UNION_SEPARATOR = ' + '


def _union_lists(values: pd.DataFrame, group_key, column: str) -> pd.Series:
    """Per group, the members of *column*'s lists, merged into one list.

    An `_all` column is missing where its list would only repeat the
    base column (`parcel_id_local_all` beside `parcel_id_local`), so a
    row missing the list contributes its base value. Members keep the
    order of first appearance, the rows' order deciding; a group whose
    merged list has one member is left missing, as the overlay leaves
    it. A building cut into pieces at its lots needs this: each piece
    lists only its own lots, and `distinct` would join whole lists,
    repeating a lot two pieces share.
    """
    lists = values[column]
    base = column.removesuffix('_all')
    if base != column and base in values.columns:
        lists = lists.where(lists.notna(), values[base])
    # Positional throughout: rows placed through a link table can share
    # an index label.
    stated = lists.notna().to_numpy()
    frame = pd.DataFrame(
        {
            'key': pd.Series(group_key).to_numpy()[stated],
            'm': lists.to_numpy()[stated].astype(str),
        }
    )
    frame['m'] = frame['m'].str.split(UNION_SEPARATOR, regex=False)
    frame = frame.explode('m')
    frame['m'] = frame['m'].str.strip()
    frame = frame[frame['m'].ne('') & frame['m'].notna()]
    frame = frame.drop_duplicates(['key', 'm'])
    counts = frame.groupby('key', sort=False)['m'].size()
    several = counts[counts > 1].index
    joined = (
        frame[frame['key'].isin(several)]
        .groupby('key', sort=False)['m']
        .agg(UNION_SEPARATOR.join)
    )
    return joined.reindex(counts.index)


def _reduce(values: pd.DataFrame, group_key, functions: dict) -> pd.DataFrame:
    """Reduce *values* per group, one function per column.

    `agree` keeps a group's value only where every row that states one
    states the same one, and leaves the group missing otherwise. It is
    the rule the ingest-time stacked-units split applies to a lot's
    units, and it exists for a class such as a foundation type: a
    parcel with a slab house and a crawl-space house has no single
    foundation, and `first` would name whichever row came first.

    `distinct` keeps every value the group's rows state, once each in
    order of first appearance, joined by `DISTINCT_SEPARATOR`; a group
    stating one value keeps it as it is. It is the rule for an outline's
    classes (maintainer, 2026-10-26): a townhome row whose buildings are
    'Single-Family' and 'Multi-Family' reads 'Single-Family |
    Multi-Family', which says what stands there, where `agree` would
    leave it missing.

    `union` merges `_all` lists member by member (`_union_lists`).
    """
    special = (AGREE, DISTINCT, UNION)
    plain = {c: f for c, f in functions.items() if f not in special}
    grouped = values[list(functions)].groupby(group_key)
    if plain:
        reduced = grouped.agg(plain)
        # pandas sums an all-missing group to 0; a parcel none of whose
        # properties states a floor area has no floor area, not a zero one.
        reduced = reduced.where(grouped.count()[list(plain)] > 0)
    else:
        reduced = pd.DataFrame(index=grouped.size().index)
    for column in functions:
        by_group = values[column].groupby(group_key)
        if functions[column] == UNION:
            reduced[column] = _union_lists(values, group_key, column).reindex(
                reduced.index
            )
        elif functions[column] == AGREE:
            reduced[column] = by_group.first().where(by_group.nunique() == 1)
        elif functions[column] == DISTINCT:
            n_distinct = by_group.nunique()
            single = by_group.first().where(n_distinct == 1)
            several = n_distinct > 1
            if several.any():
                stated = values[column].notna()
                texts = values.loc[stated, column].astype(str)
                keys = group_key[stated]
                # Only groups of several values are read from the join,
                # so only they pay a Python call each.
                wanted = keys.isin(several.index[several]).to_numpy()
                joined = (
                    texts[wanted]
                    .groupby(keys[wanted])
                    .agg(lambda s: DISTINCT_SEPARATOR.join(dict.fromkeys(s)))
                )
                single = single.astype(object).where(~several, joined)
            reduced[column] = single
    return reduced[list(functions)]


def _union_tokens(values: pd.DataFrame, group_key, column: str, sidecar: str):
    """Per group, the provenance tokens of the rows that state *column*.

    Tokens are the `+`-separated parts of each row's sidecar, kept once
    each in order of first appearance, so a parcel whose properties
    came from two sources names both.
    """
    stating = values[column].notna() & values[sidecar].notna()
    tokens = values.loc[stating, sidecar].astype(str)
    keys = group_key[stating]
    if not keys.duplicated().any():
        # One stating row per group (one building per outline): its own
        # tokens, without a Python call per group. `columns: all` asks
        # for every sidecar of a wide table, where the call per group
        # costs minutes.
        return pd.Series(tokens.to_numpy(), index=keys.to_numpy())

    def union(series):
        parts = dict.fromkeys(t for v in series for t in v.split('+') if t)
        return '+'.join(parts) or None

    # The same result as tokens.groupby(keys).agg(union), with a Python
    # call per group only where several rows share a key: a lone row's
    # union depends on its own sidecar alone, so it is computed once per
    # distinct sidecar. Wide tables of mostly lone rows spent minutes in
    # the call per group.
    index = tokens.groupby(keys).size().index
    several = keys.duplicated(keep=False).to_numpy()
    lone = tokens[~several].to_numpy()
    memo = {value: union((value,)) for value in pd.unique(lone)}
    merged = np.empty(len(index), dtype=object)
    positions = index.get_indexer(keys[~several])
    found = positions >= 0
    merged[positions[found]] = np.array(
        [memo[value] for value in lone[found]], dtype=object
    )
    if several.any():
        shared = tokens[several].groupby(keys[several]).agg(union)
        # union returns a string or None; pandas may hold None as NaN.
        joined = np.array(shared.to_numpy(dtype=object), dtype=object)
        joined[pd.isna(joined)] = None
        merged[index.get_indexer(shared.index)] = joined
    # pandas infers the dtype of an agg from the kinds of value its
    # groups return (a string, None), so the first group of each kind
    # gives the dtype the call per group gave.
    missing = pd.isna(merged)
    firsts = [int(np.argmax(kind)) for kind in (missing, ~missing) if kind.any()]
    sample = keys.isin(index[firsts]).to_numpy()
    model = tokens[sample].groupby(keys[sample]).agg(union)
    result = pd.Series(merged, index=index, name=tokens.name)
    if model.dtype == object:
        return result
    return result.astype(model.dtype)


#: Reductions whose value is one row's value, so its token is that row's.
_PICKS = frozenset({'max', 'min', 'first', 'last'})


def _supplier_tokens(values, group_key, column, sidecar, function, reduced):
    """Per group, the tokens of the rows that supplied the reduced value.

    `max` and `min` take the rows holding the group's value, and `first`
    and `last` the first or last row stating one, in the order the
    reduction saw. Rows of several sources tied on a `max` or `min`
    each supplied it, so their tokens are joined by `+`, the separator
    of `io.curator.provenance`. Every other reduction (a sum, a mean,
    `agree`) takes something from each row stating the column, and
    names them all (`_union_tokens`).
    """
    if not isinstance(function, str) or function not in _PICKS:
        return _union_tokens(values, group_key, column, sidecar)
    key = pd.Series(pd.Series(group_key).to_numpy(), index=values.index)
    stating = values[column].notna().to_numpy()
    if function in ('max', 'min'):
        held = values[column].eq(key.map(reduced)).fillna(False)
        chosen = stating & held.to_numpy(dtype=bool)
    else:
        repeat = key[stating].duplicated(keep=function).to_numpy()
        chosen = stating.copy()
        chosen[stating] = ~repeat
    return _union_tokens(values[chosen], key[chosen], column, sidecar)


#: Working column carrying each reference row's place in *order_by*.
ROW_ORDER_COLUMN = '__row_order'

_DIRECTIONS = {'ascending': True, 'descending': False}


def _order_columns(order_by, available, recipe_id, admin_id) -> dict:
    """*order_by* as {column: ascending (bool)}, for the columns described.

    Raises ValueError for a direction other than `ascending` or
    `descending`, so a misspelled one cannot silently sort the wrong way.
    """
    order = {}
    for column, direction in (order_by or {}).items():
        if direction not in _DIRECTIONS:
            raise ValueError(
                f'aggregate_from_entities: order_by {column!r} is ascending or '
                f'descending; got {direction!r}.'
            )
        if column not in available:
            warnings.warn(
                f'aggregate_from_entities: order_by column {column!r} is not in '
                f'{recipe_id} for {admin_id}; ordering without it.',
                stacklevel=3,
            )
            continue
        order[column] = _DIRECTIONS[direction]
    return order


def _row_ranks(reference: pd.DataFrame, order: dict) -> np.ndarray:
    """Each reference row's position once sorted by *order* (0 first).

    Ties keep the reference's order (a stable sort), and a missing value
    sorts last whatever the direction. A non-numeric column is compared
    as text, so ids of mixed types still sort.
    """
    frame = pd.DataFrame(
        {
            column: reference[column].to_numpy()
            if pd.api.types.is_numeric_dtype(reference[column])
            else reference[column].astype('string').to_numpy()
            for column in order
        }
    )
    positions = frame.sort_values(
        list(order), ascending=list(order.values()), kind='stable', na_position='last'
    ).index.to_numpy()
    ranks = np.empty(len(frame), dtype=np.int64)
    ranks[positions] = np.arange(len(frame))
    return ranks


def _reduced_columns(arguments) -> list[str]:
    """The reference columns `aggregate_from_entities` reduces, or
    `*` for `columns: all` (every column of the reference)."""
    columns = arguments.get('columns')
    if columns == 'all':
        return ['*']
    if isinstance(columns, dict):
        return [c for c in columns if isinstance(c, str)]
    if isinstance(columns, list):
        return [c for c in columns if isinstance(c, str)]
    return []


def _aggregated_writes(arguments) -> list[str]:
    """The columns `aggregate_from_entities` writes, for its declaration.

    `columns: all` writes every column of the reference it does not
    keep, so it declares `*`.
    """
    rename = arguments.get('rename') or {}
    written = [rename.get(c, c) for c in _reduced_columns(arguments)]
    names = [*written, *(source_column(c) for c in written if c != '*')]
    if arguments.get('count_as'):
        names.append(arguments['count_as'])
    return names


def _aggregation_reads(arguments) -> list[str]:
    """The curated columns `aggregate_from_entities` joins on."""
    key = arguments.get('key')
    keys = [key] if isinstance(key, str) else list(key or ())
    return [k for k in keys if isinstance(k, str)]


# Aggregating carries a value; it does not decide one, so nothing is
# declared under decides (plans/account-values-one-deciding-step.md,
# V1). The recover rule, which writes only onto a missing or
# non-positive cell, chooses between two values but is still a sum.
@_register(
    'aggregate_from_entities',
    phase='gather',
    reads=[_aggregation_reads],
    writes=[_aggregated_writes],
    reads_from={'recipe_id': [_reduced_columns]},
)
def aggregate_from_entities(
    state: CurateState,
    recipe_id: str,
    columns: list | dict | str,
    key: str | list[str] | None = 'parcel_id_local',
    reference_key: str | list[str] | None = None,
    fill_only: bool = True,
    recover: list[str] | None = None,
    where: dict | None = None,
    row_minimum: dict | None = None,
    prefer: list[str] | None = None,
    aggregations: dict | None = None,
    count_as: str | None = None,
    rename: dict | None = None,
    keep: list[str] | None = None,
    order_by: dict | None = None,
) -> CurateState:
    """Reduce a reference entity's rows onto the curated rows they belong to.

    Reads the reference recipe for the unit being curated, groups its rows
    by *reference_key*, reduces each named column with the registry's
    aggregation function (or the override given for it), and writes the
    result onto the curated rows by *key*. A parcel's year built becomes
    the earliest of its properties', its living area the sum, a room count
    the sum, without the harmonize stage ever having copied them.

    Where a link table between the two entities exists for the unit
    (`io.harmonizer.entity_links`), the rows are grouped by it instead of
    by the key, whichever side of the link the curated entity is: curated
    parcels take their properties' rows, curated sales take their
    parcels' rows through the transaction-to-parcel table, each parcel's
    sums scaled by the sale's `share` of it. With *key* None the link is
    the only route, and a unit without one is skipped with a warning.

    A column named in *recover* is written under a stricter rule, made
    for a value a stacked lot lost at ingest: the lot row keeps only what
    its units agree on, so a condominium lot's improvement value is
    missing or zero while its units carry the figure. The reduction is
    written only where it is positive, onto a cell that is missing or
    not positive. Nothing is written where no reference row records a
    positive value: a lot whose roll does not value the building (a
    manufactured home on a personal-property title, an exemption,
    business personal property, agricultural use value) stays as it was,
    because filling it would be estimation, which is a declared
    imputation step or nothing (maintainer's decision, 2026-09-22).
    The recovered figure is recorded, not modeled, so it carries no
    imputed marker.
    Measured 2026-09-22 on cheer-eastern-nc: 179 of 503 such lots, $44.1M.
    Measured 2026-09-22 on cheer-coastal-tx: 447 of 15,555, $64.8M (Texas
    taxes business personal property and manufactured homes on their own
    rolls, so its gaps are mostly real).

    *recipe_id* may name a curate recipe: the footprint curate takes
    every column of the building curate through the footprint-to-building
    link (`columns: all`), so an outline carries what its buildings were
    curated to, and under one building per outline equals its building
    cell for cell. A link `link_entities_by_id` wrote is found under the
    name of the geospine that mints the rows when none exists under the
    spine's (plans/building-spine-consumers-and-footprint-projection.md).

    Parameters
    ----------
    recipe_id : str
        The entity recipe to reduce (`US_property-spine-2026`), a spine
        or a curate recipe. Named here so the recipe graph sees the
        dependency.
    columns : list of str or dict or 'all'
        Columns to reduce. A list uses the registry's aggregation for each
        column; a mapping gives a per-column override (`{year_built: min,
        n_stories: max}`), for a column whose registry rule (a mean for
        both) is not what the parcel needs. `agree` writes a value only
        where every reference row stating one states the same one, for
        a class such as `foundation_type`. Where the reference carries
        a `{column}_source` sidecar, the curated column gets one too,
        naming the tokens of the rows its value came from: for `max`,
        `min`, `first` and `last` the row holding it (ties across
        sources joined by `+`), otherwise every row stating the column.
        A reference with a per-row `source` (a property spine's) gives
        each cell its own sidecar where set and the row's `source`
        where not, reduced to source ids. A written cell no row labels
        loses its earlier label. 'all'
        reduces every column of the reference except its sidecars, its
        row `source`, the keys and the curated table's index column:
        by the registry's rule, except that `first`, and a column the
        registry does not list, become `distinct` for a class and
        `agree` for a number, because a group of rows has no first one
        to speak for it: a townhome row's buildings stating two
        occupancy types give the outline 'Single-Family |
        Multi-Family', and two disagreeing latitudes give it none. The
        sidecars come along as above; their tokens are a union over the
        supplying rows, never an agreement, so an `imputed` marker on
        any row a value came from stays on the reduced cell.
    key : str or list of str or None, optional
        Curated column the reference joins on (default `parcel_id_local`).
        A list is tried in order and the first column present on both
        sides is used, so one recipe serves a source keyed on the local
        parcel id and one keyed on a state-wide id (`parcel_id_admin2`).
        None: no key join at all; the link table alone places the rows.
    reference_key : str or list of str, optional
        Reference column to group by; defaults to *key*.
    fill_only : bool, optional
        Fill only curated cells that are missing (default). With False the
        reduced value replaces whatever the curated column held.
    recover : list of str, optional
        Columns written only where the reduction is positive, onto cells
        that are missing or not positive (see above). Takes precedence
        over *fill_only* for the columns it names.
    where : dict, optional
        Reduce only the reference rows whose ``column`` matches
        ``pattern`` in full (a regular expression), e.g. ``{column:
        use_group_code, pattern: '102.*'}`` for a roll's condominium
        units. Rows without a value in ``column`` are left out. Default
        None (every row).
    row_minimum : dict, optional
        Per column, the least value each reference row contributes,
        applied before the reduction (and before a link's ``share``
        scales it): ``{n_dwellings: 1}`` counts a row that states no
        dwellings, or zero, as one. For a row whose code defines what
        it is, so that the value follows from the record's existence:
        a residential condominium unit is one dwelling. Default None.
    prefer : list of str, optional
        Columns where the reduction comes first: written wherever it
        states a value, and the curated value kept wherever it does not
        (first non-null, reduction ahead of the curated column). Takes
        precedence over *fill_only* for the columns it names.
    aggregations : dict, optional
        With `columns: all`, per-column overrides of the rule
        (`{year_built: min}`: an outline's buildings date from the
        earliest of them, where the registry's mean would state a year
        none was built in). Besides a pandas reduction it takes
        `agree`, `distinct` and `union`; `union` merges an `_all` list
        (`parcel_id_local_all`) member by member. Default None.
    count_as : str, optional
        Also write the number of reference rows each curated row takes
        (`n_buildings_per_footprint`), 0 where it takes none. Default
        None.
    rename : dict, optional
        Reference column to the curated name it is written under, its
        sidecar following (`n_permits_per_building:
        n_permits_per_footprint`, so a count keeps the name of the
        entity it is written on). Default None.
    keep : list of str, optional
        With `columns: all`, curated columns left as they are: an
        outline's own centroid, area, geographic ids and geometry
        provenance describe the outline, not its buildings. Default
        None.
    order_by : dict, optional
        Reference columns to order each group's rows by before the
        reduction, each `ascending` or `descending`, the first column
        deciding and the next breaking its ties (`{area_m2: descending,
        parcel_id: ascending}`). Without it the rows come in the link
        table's or the reference's order, which nothing fixes, so
        `first` names an arbitrary row; with it `first` names the row
        that leads (an outline cut at its lots takes the lot of its
        largest piece, the dominant one), and `distinct` lists values in
        that order. A row missing an ordering value comes last. A column
        the reference does not describe is left out with a warning.
        Default None.
    """
    curated = state.curated
    all_columns = columns == 'all'
    if isinstance(columns, str) and not all_columns:
        raise ValueError(
            "aggregate_from_entities: columns is a list, a mapping or 'all'; "
            f'got {columns!r}.'
        )
    if aggregations and not all_columns:
        raise ValueError(
            'aggregate_from_entities: aggregations overrides the rule of '
            'columns: all; a list or mapping names its rules itself.'
        )
    link_only = key is None
    keys_ = [] if link_only else [key] if isinstance(key, str) else list(key)
    reference_keys = (
        keys_
        if reference_key is None
        else [reference_key]
        if isinstance(reference_key, str)
        else list(reference_key)
    )
    try:
        described = list(describe_recipe(recipe_id, state.admin_id).index)
    except (FileNotFoundError, OSError, ValueError) as exc:
        warnings.warn(
            f'aggregate_from_entities: could not read {recipe_id} for '
            f'{state.admin_id}: {exc}. Skipping.',
            stacklevel=2,
        )
        return state
    available = set(described)
    if all_columns:
        sidecar_names = {source_column(c) for c in described}
        left_out = {
            ROW_SOURCE_COLUMN,
            'geometry',
            curated.index.name,
            *keys_,
            *reference_keys,
            *(keep or ()),
        }
        spec = {
            c: None for c in described if c not in sidecar_names and c not in left_out
        }
    elif isinstance(columns, list):
        spec = {c: None for c in columns}
    else:
        spec = dict(columns)
    pair = next(
        (
            (k, r)
            for k, r in zip(keys_, reference_keys, strict=True)
            if k in curated.columns and r in available
        ),
        None,
    )
    if pair is None and not link_only:
        warnings.warn(
            f'aggregate_from_entities: no key of {keys_} is present on both '
            f'the curated table and {recipe_id} for {state.admin_id}; skipping.',
            stacklevel=2,
        )
        return state
    key, reference_key = pair if pair is not None else (None, None)
    # A source that states none of a column is not an error here: the
    # step fills what the unit's sources can say and leaves the rest.
    present = [c for c in spec if c in available]
    if not present:
        return state
    sidecars = {c: source_column(c) for c in present if source_column(c) in available}
    # A property spine records which source each row came from, and a
    # sidecar only on the cells another source filled (a supplement, a
    # merged row's other source): an empty sidecar means the row's
    # source. Without the row's label a curated value would carry no
    # token at all, so a delivery could withhold it only by the coarser
    # scope rules (Lake County FL sales: land_value_source on 0.1% of
    # rows).
    row_source = ROW_SOURCE_COLUMN if ROW_SOURCE_COLUMN in available else None
    where_column = None
    if where is not None:
        if set(where) != {'column', 'pattern'}:
            raise ValueError(
                "aggregate_from_entities: where takes 'column' and 'pattern'; "
                f'got {sorted(where)}.'
            )
        where_column = where['column']
        if where_column not in available:
            # No row can match a column the unit's reference lacks.
            return state
    order = _order_columns(order_by, available, recipe_id, state.admin_id)
    reference = get_entities(
        recipe_id,
        state.admin_id,
        columns=list(
            dict.fromkeys(
                [
                    *([reference_key] if reference_key else []),
                    *present,
                    *sidecars.values(),
                    *([row_source] if row_source else []),
                    *([where_column] if where_column else []),
                    *order,
                ]
            )
        ),
    )
    if reference is None or len(reference) == 0:
        return state
    if all_columns:
        # A recipe's description lists its index too (a curate recipe's
        # `building_id`), which arrives as the index, not a column.
        present = [c for c in present if c in reference.columns]
        sidecars = {c: s for c, s in sidecars.items() if c in present}
    if where_column is not None:
        matches = (
            reference[where_column]
            .astype('string')
            .str.fullmatch(where['pattern'])
            .fillna(False)
            .to_numpy(dtype=bool)
        )
        reference = reference[matches]
        if len(reference) == 0:
            return state
    if row_source and row_source in reference.columns:
        labels = reference[row_source].astype('string')
        for column in present:
            side = source_column(column)
            cell = labels
            if column in sidecars:
                own = reference[side].astype('string')
                cell = own.where(own.notna(), labels)
            sidecars[column] = side
            reference[side] = _row_source_tokens(cell)
    # Rows without the key stay: the link table may still place them (a
    # unit that names its lot), and the key join's groupby drops them.
    functions = {}
    for column in present:
        override = spec[column]
        if all_columns:
            override = (aggregations or {}).get(column)
        if override is not None:
            functions[column] = override
        else:
            canonical = resolve_attribute_name(column)
            fname = get_agg_func(canonical)
            if all_columns and (fname == 'first' or not _has_agg_func(fname)):
                # See `columns`: no row of a group speaks for it. A
                # class keeps each value, a number only an agreed one.
                numeric_column = pd.api.types.is_numeric_dtype(reference[column])
                functions[column] = AGREE if numeric_column else DISTINCT
                continue
            if not _has_agg_func(fname):
                warnings.warn(
                    f'aggregate_from_entities: {column!r} has no registry '
                    'aggregation and no override; skipped.',
                    stacklevel=2,
                )
                continue
            functions[column] = _agg_func_for(canonical, fname)
    if not functions:
        return state
    numeric = {}
    for column in functions:
        series = reference[column]
        numeric[column] = (
            pd.to_numeric(series, errors='coerce')
            if functions[column] in ('sum', 'mean', 'median', 'min', 'max')
            and not pd.api.types.is_numeric_dtype(series)
            else series
        )
    for column, minimum in (row_minimum or {}).items():
        if column in numeric:
            # A row that states less than the minimum, or nothing,
            # contributes the minimum (see row_minimum).
            stated = pd.to_numeric(numeric[column], errors='coerce')
            numeric[column] = stated.where(stated >= minimum, float(minimum))
    for column, sidecar in sidecars.items():
        if column in functions:
            numeric[sidecar] = reference[sidecar]
    values = pd.DataFrame(numeric)
    if order:
        # Each row's place in the requested order travels with it
        # through the link, which rearranges the rows.
        values[ROW_ORDER_COLUMN] = _row_ranks(reference, order)
    linked = _rows_by_link(state, recipe_id, reference, values, functions)
    if linked is not None:
        values, group_key, keys = linked
    elif link_only:
        warnings.warn(
            f'aggregate_from_entities: no link table between this entity and '
            f'{recipe_id} for {state.admin_id} and no key to join on; skipping.',
            stacklevel=2,
        )
        return state
    else:
        group_key, keys = reference[reference_key], curated[key]
    if order:
        sorted_rows = np.argsort(values[ROW_ORDER_COLUMN].to_numpy(), kind='stable')
        values = values.iloc[sorted_rows].drop(columns=ROW_ORDER_COLUMN)
        group_key = group_key.iloc[sorted_rows]
    reduced = _reduce(values, group_key, functions)
    written = []
    recover = set(recover or ())
    prefer = set(prefer or ())
    rename = dict(rename or {})
    for column in reduced.columns:
        incoming = keys.map(reduced[column])
        target = rename.get(column, column)
        if column in prefer:
            wrote = incoming.notna()
            if target in curated.columns:
                curated[target] = incoming.where(wrote, curated[target])
            else:
                curated[target] = incoming
        elif column in recover:
            positive = pd.to_numeric(incoming, errors='coerce').gt(0).fillna(False)
            if target in curated.columns:
                existing = curated[target]
                stated = pd.to_numeric(existing, errors='coerce').gt(0).fillna(False)
                wrote = positive & ~stated
                curated[target] = existing.where(~wrote, incoming)
            else:
                wrote = positive
                curated[target] = incoming.where(positive)
        elif target in curated.columns and fill_only:
            existing = curated[target]
            wrote = existing.isna() & incoming.notna()
            curated[target] = existing.where(existing.notna(), incoming)
        else:
            wrote = incoming.notna()
            curated[target] = incoming
        if wrote.any():
            tokens = pd.Series(pd.NA, index=curated.index, dtype=object)
            if column in sidecars:
                tokens = keys.map(
                    _supplier_tokens(
                        values,
                        group_key,
                        column,
                        sidecars[column],
                        functions[column],
                        reduced[column],
                    )
                )
            # A written cell without a token loses the label it had:
            # that label named the value this step replaced.
            record_sources(curated, target, tokens, mask=wrote, clear_missing=True)
        written.append(target)
    if count_as:
        counts = pd.Series(group_key).value_counts()
        curated[count_as] = keys.map(counts).fillna(0).astype('float64')
        written.append(count_as)
    # One column at a time fragments a wide table (columns: all writes
    # well over a hundred); one copy consolidates it.
    state.curated = curated.copy() if all_columns else curated
    if state.verbose:
        print(
            f'  aggregate_from_entities: {recipe_id} -> {", ".join(written)} '
            f'on {int(keys.isin(reduced.index).sum()):,} of {len(curated):,} rows'
        )
    return state
