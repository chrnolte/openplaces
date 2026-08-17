"""
Registry-driven table helpers shared by the geo and io layers.

These functions were previously defined in :mod:`openplaces.io.aggregate`
and :mod:`openplaces.io.transform`, which put them above ``geo/`` in the
module layer hierarchy even though ``geo/`` is one of their main callers.
That produced a module-level import cycle between
``geo/crosswalk.py`` and ``io/aggregate.py``, broken only by deferring the
``geo.address`` import inside :func:`join_nonnull_addresses`.

They live here instead because their real dependencies are low: the
attribute registry, the recipe attribute-name resolver, and the pure
string helper :func:`openplaces.geo.address.strip_unit_suffix`. Both
former homes re-export them, so existing import paths keep working.
"""

import warnings

import geopandas as gpd
import pandas as pd

from openplaces.core.attribute_registry import get_agg_func
from openplaces.geo.address import strip_unit_suffix
from openplaces.recipe import resolve_attribute_name


def add_unique_suffix(s):
    """Make string Series unique by appending unique integer suffices.

    All duplicate occurrences are suffixed (``-1``, ``-2``, …), including the
    first one.  Use `make_index_unique` when operating on a DataFrame index and
    the first (or largest) occurrence should keep the unsuffixed value.

    Parameters
    ----------
    s : pd.Series
        String Series containing duplicate entries
    """
    # Avoid warnings about setting slices
    s = s.copy()
    duplicates = s.duplicated(keep=False)
    # Handle collisions with suffix
    counts = s[duplicates].groupby(s[duplicates], sort=False).cumcount() + 1
    s.loc[duplicates] = s.loc[duplicates].astype(str) + '-' + counts.astype(str)
    return s


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
    APT/UNIT/# suffix, and joining every one of them with ``' + '`` produces
    a multi-address string that no downstream address parser can split back
    into a single street/number. Genuinely different base addresses (e.g. a
    parcel spanning two streets) still join with ``' + '``, unchanged from
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


_AGG_ALIASES = {'join_nonnull': join_nonnull_strings}


def _agg_func_for(canonical_name: str, fname: str):
    """Resolve a registry aggregation name to a concrete callable/name.

    Identical to a plain ``_AGG_ALIASES.get(fname, fname)`` lookup, except
    *address* gets :func:`join_nonnull_addresses` instead of the plain
    :func:`join_nonnull_strings` every other ``'join_nonnull'`` column
    (e.g. ``use_group``) still uses -- see that function's docstring for
    why a plain string join corrupts a multi-unit building's address.
    """
    if fname == 'join_nonnull' and canonical_name == 'address':
        return join_nonnull_addresses
    return _AGG_ALIASES.get(fname, fname)


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
        non-null aggregation function are included in the result.
    by : str or list of str
        Column(s) to group by.
    aggregation_function : None, callable, or dict, optional
        Controls which aggregation function is applied to each column.

        ``None``
            Use the function recorded in the attribute registry for each column.
        callable
            Apply this single function to all aggregatable columns.
        dict
            Map column names to callables; columns absent from the dict fall
            back to the registry default.
    sort_by : str, optional
        Column to sort *df* by descending before grouping.  When the column is
        absent or omitted and *df* is a GeoDataFrame, rows are sorted by
        geometry area descending.
    list_columns : list of str, optional
        Column names for which an additional ``{col}_list`` column is added to
        the output, collecting all values per group into a Python list.  The
        normal scalar aggregation for each column is still applied; these are
        extra columns alongside the registry-aggregated ones.

    Returns
    -------
    pd.DataFrame or None
        Aggregated DataFrame with *by* as the index, or ``None`` when no
        aggregatable columns are found in *df*.

    Raises
    ------
    ValueError
        When *aggregation_function* is not ``None``, a callable, or a dict.
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

    if sort_by is not None and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)
    elif isinstance(df, gpd.GeoDataFrame):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Geometry is in a geographic CRS')
            df = df.loc[df.geometry.area.sort_values(ascending=False).index]

    agg_cols: dict = {}
    for col in df.columns:
        fname = get_agg_func(resolve_attribute_name(col))
        if fname is None:
            continue
        if callable(aggregation_function):
            agg_cols[col] = aggregation_function
        elif isinstance(aggregation_function, dict):
            agg_cols[col] = aggregation_function.get(
                col, _agg_func_for(resolve_attribute_name(col), fname)
            )
        else:
            agg_cols[col] = _agg_func_for(resolve_attribute_name(col), fname)

    for col in list_columns or []:
        if col not in df.columns:
            continue
        # Categorical dtype cannot hold list values; cast to object first.
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            df = df.copy()
            df[col] = df[col].astype(object)
        agg_cols[f'{col}_list'] = pd.NamedAgg(column=col, aggfunc=list)

    if not agg_cols:
        return None

    return df.groupby(by).agg(agg_cols)
