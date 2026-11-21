"""
Registry-driven table helpers shared by the geo and io layers.

These functions were previously defined in :mod:`openplaces.io.aggregate`
and :mod:`openplaces.io.transform`, which put them above `geo/` in the
module layer hierarchy even though `geo/` is one of their main callers.
That produced a module-level import cycle between
`geo/crosswalk.py` and `io/aggregate.py`, broken only by deferring the
`geo.address` import inside :func:`join_nonnull_addresses`.

They live here instead because their real dependencies are low: the
attribute registry, the recipe attribute-name resolver, and the pure
string helper :func:`openplaces.geo.address.strip_unit_suffix`. Both
former homes re-export them, so existing import paths keep working.
"""

import warnings
from itertools import combinations

import geopandas as gpd
import numpy as np
import pandas as pd

from openplaces.core.attribute_registry import get_agg_func
from openplaces.geo.address import strip_unit_suffix
from openplaces.recipe import resolve_attribute_name


def summarize_conflicts(
    present: list[tuple[str, pd.Series]],
    index: pd.Index,
) -> pd.Series:
    """Summarize disagreeing evidence values per row as a compact string.

    *present* is a list of (label, values) pairs, each values Series
    aligned to *index*. Returns an object Series that is missing except
    where at least two present values disagree; there, sources are
    grouped by unique value (groups ordered by first-appearing label,
    labels within a group joined with '/'), e.g. 'nsi/parcel: Single
    Family | fema: Manufactured Home', so agreements and disagreements
    are both visible at a glance.

    Shared by the harmonize stage (address reconciliation) and the curate
    stage (occupancy and land-use reconciliation); it lives here because
    neither stage may import the other.
    """
    conflict = pd.Series(pd.NA, index=index, dtype=object)
    if len(present) < 2:
        return conflict

    differ = pd.Series(False, index=index)
    for (_, class_a), (_, class_b) in combinations(present, 2):
        both = class_a.notna() & class_b.notna()
        # Compare only rows where both are present: on object columns
        # a pd.NA in the comparison raises "boolean value of NA is
        # ambiguous" before the mask could drop it.
        a = class_a[both].astype(object).to_numpy()
        b = class_b[both].astype(object).to_numpy()
        differ.loc[both] = differ.loc[both].to_numpy() | (a != b)
    if not differ.any():
        return conflict

    labels = [label for label, _ in present]
    stacked = pd.concat(
        {label: values.astype(object) for label, values in present}, axis=1
    )

    def _row_summary(row) -> str:
        groups: dict[str, list[str]] = {}
        for label in labels:
            value = row[label]
            if pd.notna(value):
                groups.setdefault(str(value), []).append(label)
        return ' | '.join(f'{"/".join(who)}: {value}' for value, who in groups.items())

    conflict.loc[differ] = stacked.loc[differ].apply(_row_summary, axis=1)
    return conflict


def add_unique_suffix(s):
    """Make string Series unique by appending unique integer suffices.

    All duplicate occurrences are suffixed (`-1`, `-2`, ...), including the
    first one.  Use `make_index_unique` when operating on a DataFrame index and
    the first (or largest) occurrence should keep the unsuffixed value.

    A categorical Series is converted to object first: a suffixed value is a
    new category, and assigning it into a categorical raises instead.

    Parameters
    ----------
    s : pd.Series
        String Series containing duplicate entries
    """
    # Avoid warnings about setting slices
    s = s.copy()
    if isinstance(s.dtype, pd.CategoricalDtype):
        s = s.astype(object)
    duplicates = s.duplicated(keep=False)
    # Handle collisions with suffix
    counts = s[duplicates].groupby(s[duplicates], sort=False).cumcount() + 1
    s.loc[duplicates] = s.loc[duplicates].astype(str) + '-' + counts.astype(str)
    return s


def normalize_issued_id(values: pd.Series) -> pd.Series:
    """Reduce an issued number to what two publications of it agree on.

    Folds case and turns every run of separators into one hyphen, so
    `12_a  3`, `12-A-3` and `12 A 3` converge. The separators' positions
    are kept, unlike in `geo.ids.add_parcel_id_alnum`: an id is not a
    fallback match key, and in a map-block-lot number the positions
    carry meaning. Measured on Somerville MA, 2026-09-20: dropping
    separators made 426 of 19,013 account numbers collide (`12_A_1_2`
    with `12_A_12`) where 24 repeat as published. The cost is that a
    source publishing the same number with no separators at all does
    not converge; Boston's two sources needed no help.

    Blank and all-zero values become missing: they are placeholders,
    not numbers anyone issued. Lives here, below the stages, because the
    property spine (harmonize) and the transaction steps (harmonize and
    curate) all mint ids from an issued number.
    """
    text = (
        values.astype('string')
        .str.upper()
        .str.replace(r'[^A-Z0-9]+', '-', regex=True)
        .str.strip('-')
    )
    usable = text.notna() & text.str.replace('-', '').str.strip('0').ne('')
    return text.where(usable)


def require_unique_index(df, context: str, max_labels: int = 5) -> None:
    """Raise when the index of *df* carries repeated labels.

    An index-aligned operation (a reindex, a label join, a .loc row
    selection) either fans rows out or collapses them when a label
    repeats, and the resulting failure surfaces far from its cause. Call
    this where such an operation is about to run and duplicates cannot be
    resolved sensibly.

    Parameters
    ----------
    df : pd.DataFrame, gpd.GeoDataFrame, pd.Series, or pd.Index
        Object whose index must be unique, or the index itself.
    context : str
        What is about to run, named in the message (stage, recipe id and
        admin unit, say), so the cause reads off the traceback.
    max_labels : int, default 5
        How many of the repeated labels to list in the message.

    Raises
    ------
    ValueError
        When the index has repeated labels.
    """
    index = df if isinstance(df, pd.Index) else df.index
    if not index.has_duplicates:
        return
    repeated = index[index.duplicated(keep=False)]
    labels = list(dict.fromkeys(repeated.tolist()))
    shown = ', '.join(repr(label) for label in labels[:max_labels])
    if len(labels) > max_labels:
        shown += ', ...'
    name = index.name if index.name is not None else list(index.names)
    raise ValueError(
        f'{context}: index {name!r} carries {len(labels)} repeated '
        f'label(s) over {len(repeated)} rows: {shown}. An index-aligned '
        'operation cannot run on repeated labels, so the input has to be '
        'deduplicated or made unique first.'
    )


def join_nonnull_strings(x):
    """Join non-null values of *x* as strings with ' + '; None when all null."""
    parts = [str(v) for v in x if v is not None and pd.notna(v)]
    return ' + '.join(parts) if parts else None


def join_nonnull_addresses(x):
    """Join non-null address strings, collapsing same-building unit variants.

    Like :func:`join_nonnull_strings`, but first deduplicates by each
    value's base address (unit designator stripped via
    :func:`openplaces.geo.address.strip_unit_suffix`) -- a condo/apartment
    building's per-unit property records otherwise differ only by an
    APT/UNIT/# suffix, and joining every one of them with ' + ' produces
    a multi-address string that no downstream address parser can split back
    into a single street/number. Genuinely different base addresses (e.g. a
    parcel spanning two streets) still join with ' + ', unchanged from
    :func:`join_nonnull_strings`.
    """
    parts = [str(v) for v in x if v is not None and pd.notna(v)]
    if not parts:
        return None
    by_base: dict[str, str] = {}
    for part in parts:
        base_key = ' '.join(strip_unit_suffix(part).split()).casefold()
        by_base.setdefault(base_key, part)
    return ' + '.join(by_base.values())


def join_distinct_text(values, sep: str = '|') -> str:
    """The distinct stated values as text, in order, joined by *sep*.

    The reducer form of ``sep.join(s.dropna().astype(str).unique())``:
    each non-missing value is written with ``str`` and kept once, at its
    first position. A group stating nothing gives ''. ``str`` and
    ``astype(str)`` agree for text, integers, floats and booleans; a
    datetime is written differently by the two, so a datetime column
    must not be joined with this function.
    """
    return sep.join(dict.fromkeys(str(v) for v in values if not pd.isna(v)))


def _keeps_dtype_per_group(dtype) -> bool:
    """Whether pandas may cast an aggregate back to *dtype*.

    An aggregate over an extension column other than text (a category,
    a nullable integer, a geometry) is cast back to that dtype where
    every result fits it, a test only the whole result can pass.
    """
    return isinstance(dtype, pd.api.extensions.ExtensionDtype) and not isinstance(
        dtype, pd.StringDtype
    )


def _missing_scalar(value) -> bool:
    """True for None, NaN, pd.NA and NaT, never for a container."""
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, float) and value != value


def reduce_groups(values: pd.Series, keys, reduce, *, sort: bool = True) -> pd.Series:
    """Reduce each group of *values* with *reduce*, one call per group.

    The result is what ``values.groupby(keys, sort=sort).agg(f)``
    returns, with ``f(s) = reduce(s.to_numpy(dtype=object))``: the same
    index (the distinct non-missing keys, sorted when
    *sort*, else in order of first appearance; rows with a missing key
    are left out), the same name, the same values and the same dtype.
    *reduce* receives the group's values as an object array in row
    order, missing values included as they are stored, and its result is
    used as returned.

    Parameters
    ----------
    values : pandas.Series
        Values to reduce.
    keys : pandas.Series or array-like
        Group key of each value, matched by position.
    reduce : callable
        Takes an object array, returns one value.
    sort : bool
        Order of the result's index, as in `pandas.DataFrame.groupby`.

    Returns
    -------
    pandas.Series
        One reduced value per group.

    Notes
    -----
    pandas builds a Series for every group, which made a join over a few
    hundred thousand groups of one or two rows cost minutes; here the
    groups are slices of one sorted array. The dtype is the one pandas
    infers from the results: a few groups, one returning each kind of
    value (missing, text, number), are aggregated by pandas itself and
    their dtype is applied to all. Where pandas could cast the results
    back to the column's own dtype (an extension dtype other than text),
    or where the keys are categorical (pandas then also lists categories
    no row holds), pandas aggregates every group itself.
    """
    values = values.reset_index(drop=True)
    keys = (
        keys.reset_index(drop=True)
        if isinstance(keys, pd.Series)
        else pd.Series(np.asarray(keys))
    )

    def by_series(series):
        return reduce(series.to_numpy(dtype=object))

    grouped = values.groupby(keys, sort=sort)
    if (
        not keys.notna().any()
        or isinstance(keys.dtype, pd.CategoricalDtype)
        or _keeps_dtype_per_group(values.dtype)
    ):
        return grouped.agg(by_series)
    index = grouped.size().index
    codes, uniques = pd.factorize(keys, sort=False)
    stated = np.flatnonzero(codes >= 0)
    rows = stated[np.argsort(codes[stated], kind='stable')]
    group_codes = codes[rows]
    array = values.to_numpy(dtype=object)[rows]
    starts = np.r_[0, np.flatnonzero(np.diff(group_codes)) + 1]
    ends = np.r_[starts[1:], len(rows)]
    reduced = np.empty(len(starts), dtype=object)
    for position, (start, end) in enumerate(zip(starts, ends, strict=True)):
        reduced[position] = reduce(array[start:end])
    merged = np.empty(len(index), dtype=object)
    merged[index.get_indexer(pd.Index(uniques).take(group_codes[starts]))] = reduced

    # The first group returning each kind of value is aggregated by
    # pandas, whose result dtype then applies to every group.
    kinds: dict = {}
    for position, value in enumerate(merged):
        kinds.setdefault('missing' if _missing_scalar(value) else type(value), position)
    sample = keys.isin(index[sorted(kinds.values())]).to_numpy()
    model = values[sample].groupby(keys[sample], sort=sort).agg(by_series)
    result = pd.Series(merged, index=index, name=model.name)
    if model.dtype == object:
        return result
    return result.astype(model.dtype)


def map_distinct(series: pd.Series, function) -> pd.Series:
    """``series.map(function)``, calling *function* once per distinct value.

    Values are distinct when their type and value are (1, 1.0 and True
    stay apart, and so do None, NaN and pd.NA). The distinct values are
    mapped by `pandas.Series.map` itself, so *function* receives what it
    would, and the result's dtype is the one pandas infers from the
    same set of results. Columns of another dtype than object or text
    are mapped as they are.
    """
    if not len(series) or not (
        series.dtype == object or isinstance(series.dtype, pd.StringDtype)
    ):
        return series.map(function)
    array = series.to_numpy(dtype=object)
    if pd.api.types.infer_dtype(array, skipna=True) in ('string', 'empty'):
        # Text and missing values only: text compares exactly by hash,
        # and the missing values are told apart by their type, both
        # without a Python step per value.
        missing = pd.isna(array)
        codes = np.empty(len(array), dtype=np.int64)
        text_codes, texts = pd.factorize(array[~missing])
        codes[~missing] = text_codes
        if missing.any():
            kinds = np.frompyfunc(type, 1, 1)(array[missing])
            codes[missing] = pd.factorize(kinds)[0] + len(texts)
    else:
        tagged = np.empty(len(array), dtype=object)
        for position, value in enumerate(array):
            tagged[position] = (
                (type(value), None) if _missing_scalar(value) else (type(value), value)
            )
        codes, _ = pd.factorize(tagged)
    _, firsts = np.unique(codes, return_index=True)
    mapped = series.iloc[firsts].map(function)
    return mapped.iloc[codes].set_axis(series.index)


_AGG_ALIASES = {'join_nonnull': join_nonnull_strings}
#: Reducers a group's values can be handed as an array, so
#: `aggregate_rows` runs them through `reduce_groups`.
ARRAY_REDUCERS = (join_nonnull_strings, join_nonnull_addresses)


def _agg_func_for(canonical_name: str, fname: str):
    """Resolve a registry aggregation name to a concrete callable/name.

    Identical to a plain `_AGG_ALIASES.get(fname, fname)` lookup, except
    *address* gets :func:`join_nonnull_addresses` instead of the plain
    :func:`join_nonnull_strings` every other 'join_nonnull' column
    (e.g. `use_group`) still uses -- see that function's docstring for
    why a plain string join corrupts a multi-unit building's address.
    """
    if fname == 'join_nonnull' and canonical_name == 'address':
        return join_nonnull_addresses
    return _AGG_ALIASES.get(fname, fname)


def _has_agg_func(fname) -> bool:
    """True when *fname* is a usable aggregation name from the registry.

    The registry is read with `pd.read_csv`, so a blank aggregation cell
    arrives as float NaN rather than None; a NaN reaching `groupby.agg`
    raises a TypeError far from its cause.

    Parameters
    ----------
    fname : str, float, or None
        Value returned by
        :func:`openplaces.core.attribute_registry.get_agg_func`.
    """
    if fname is None:
        return False
    return not (isinstance(fname, float) and pd.isna(fname))


def _aggregate_columns(df, by, grouped, agg_cols: dict) -> pd.DataFrame | None:
    """``grouped.agg(agg_cols)``, the join reducers through `reduce_groups`.

    A join reducer called by pandas costs a Series per group, which on a
    reference of a hundred thousand mostly single-row groups was most of
    the link step; `reduce_groups` returns the same values and dtype.
    Grouping by several columns or an index level keeps the plain call.
    """
    if not agg_cols:
        return None
    fast = {
        column: function
        for column, function in agg_cols.items()
        if any(function is reducer for reducer in ARRAY_REDUCERS)
    }
    if not fast or not isinstance(by, str) or by not in df.columns:
        return grouped.agg(agg_cols)
    rest = {column: f for column, f in agg_cols.items() if column not in fast}
    base = grouped.agg(rest) if rest else None
    index = base.index if base is not None else grouped.size().index
    columns = {
        column: (
            reduce_groups(df[column], df[by], fast[column])
            if column in fast
            else base[column]
        )
        for column in agg_cols
    }
    return pd.DataFrame(columns, index=index)


def aggregate_rows(
    df: pd.DataFrame,
    by: str | list[str],
    aggregation_function=None,
    sort_by: str | None = None,
    list_columns: list[str] | None = None,
) -> pd.DataFrame | None:
    """Aggregate rows of *df* using per-column functions from the attribute registry.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.  Columns that appear in the attribute registry with a
        non-null aggregation function are included in the result.  The
        grouping column(s) are excluded: they are the result's index, and
        returning them as columns too breaks a downstream `reset_index`.
    by : str or list of str
        Column(s) to group by.
    aggregation_function : None, callable, or dict, optional
        Controls which aggregation function is applied to each column.

        None
            Use the function recorded in the attribute registry for each column.
        callable
            Apply this single function to all aggregatable columns.
        dict
            Map column names to callables; columns absent from the dict fall
            back to the registry default.
    sort_by : str, optional
        Column to sort *df* by descending before grouping.  When omitted and
        *df* is a GeoDataFrame, rows are sorted by geometry area descending.
        A *sort_by* naming a column *df* does not have warns and falls back
        to that area order, which changes which row every 'first'-aggregated
        column comes from.
    list_columns : list of str, optional
        Column names for which an additional `{col}_list` column is added to
        the output, collecting all values per group into a Python list.  The
        normal scalar aggregation for each column is still applied; these are
        extra columns alongside the registry-aggregated ones.

    Returns
    -------
    pd.DataFrame or None
        Aggregated DataFrame with *by* as the index, or None when no
        aggregatable columns are found in *df*.

    Raises
    ------
    ValueError
        When *aggregation_function* is not None, a callable, or a dict.

    Warns
    -----
    UserWarning
        When *sort_by* names a column that is absent, or when rows are
        dropped because their grouping key is null.
    """
    if not (
        aggregation_function is None
        or callable(aggregation_function)
        or isinstance(aggregation_function, dict)
    ):
        raise ValueError(
            'aggregation_function must be None, a callable, or a dict; '
            f'got {type(aggregation_function)}'
        )

    by_cols = [by] if isinstance(by, str) else list(by)

    if sort_by is not None and sort_by not in df.columns:
        warnings.warn(
            f'aggregate_rows: sort_by={sort_by!r} is not a column of the '
            'input, so rows are ordered by geometry area (or left in input '
            'order); a first-aggregated column then comes from a different '
            'row than intended.',
            stacklevel=2,
        )

    if sort_by is not None and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)
    elif isinstance(df, gpd.GeoDataFrame):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Geometry is in a geographic CRS')
            areas = df.geometry.area.to_numpy()
        # Positionally, never by label: re-selecting rows with .loc on a
        # non-unique index multiplies them, and the groupby below would
        # then aggregate the inflated input.
        df = df.iloc[(-areas).argsort(kind='stable')]

    agg_cols: dict = {}
    for col in df.columns:
        if col in by_cols:
            continue
        canonical_name = resolve_attribute_name(col)
        fname = get_agg_func(canonical_name)
        if not _has_agg_func(fname):
            continue
        if callable(aggregation_function):
            agg_cols[col] = aggregation_function
        elif isinstance(aggregation_function, dict):
            agg_cols[col] = aggregation_function.get(
                col, _agg_func_for(canonical_name, fname)
            )
        else:
            agg_cols[col] = _agg_func_for(canonical_name, fname)

    list_cols = [col for col in list_columns or [] if col in df.columns]
    # Categorical dtype cannot hold list values; cast to object
    # first, in one copy of the frame rather than one per column.
    categorical_lists = [
        col for col in list_cols if isinstance(df[col].dtype, pd.CategoricalDtype)
    ]
    if categorical_lists:
        df = df.copy()
        for col in categorical_lists:
            df[col] = df[col].astype(object)

    if not agg_cols and not list_cols:
        return None

    # *by* may name an index level rather than a column, which has
    # no null key to report.
    key_cols = [col for col in by_cols if col in df.columns]
    n_null_keys = int(df[key_cols].isna().any(axis=1).sum()) if key_cols else 0
    if n_null_keys:
        # groupby drops these rows. Keeping them instead (dropna=False)
        # would put a null-keyed group in the result, and a downstream
        # merge treats null as equal to null, attaching that group's
        # values to unrelated rows: a worse failure than the drop.
        warnings.warn(
            f'aggregate_rows: {n_null_keys} row(s) have a null '
            f'{by_cols} grouping key and are dropped from the aggregate.',
            stacklevel=2,
        )

    grouped = df.groupby(by)
    result = _aggregate_columns(df, by, grouped, agg_cols)

    if list_cols:
        # Named aggregation only works through keyword arguments: a
        # pd.NamedAgg placed in the dict passed to agg() raises
        # KeyError, because '{col}_list' is not a column of the input.
        lists = grouped.agg(
            **{
                f'{col}_list': pd.NamedAgg(column=col, aggfunc=list)
                for col in list_cols
            }
        )
        result = lists if result is None else result.join(lists)

    return result
