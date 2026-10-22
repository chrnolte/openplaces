"""
Harmonize steps that re-express monetary values.

The one step here, ``deflate_values``, turns nominal amounts recorded in
different years into base-year real amounts with a consumer price index
read from a sidecar CSV beside the recipe. It is an attribute-phase step:
it reads the spine and writes new columns, never geometry.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from openplaces.io.harmonizer import HarmonizeState, _register
from openplaces.io.harmonizer.addresses import _record_source
from openplaces.path import recipe_path

# Columns a price index sidecar must carry. The World Bank "Inflation,
# consumer prices (annual %)" indicator exports as one row per country
# and one column per year; the sidecar is that table melted to long
# form, which is what the file beside the recipe holds.
_INDEX_COLUMNS = ('country', 'year', 'inflation_pct')


@cache
def _load_price_index(path: str) -> pd.DataFrame:
    """Read a long-form price index CSV and validate its columns."""
    index = pd.read_csv(path)
    missing = set(_INDEX_COLUMNS) - set(index.columns)
    if missing:
        raise ValueError(
            f'Price index {path} lacks column(s) {sorted(missing)}; expected '
            f'{list(_INDEX_COLUMNS)}.'
        )
    index['year'] = pd.to_numeric(index['year'], errors='coerce').astype('Int64')
    index['inflation_pct'] = pd.to_numeric(index['inflation_pct'], errors='coerce')
    return index


def deflator_by_year(index: pd.DataFrame, country: str, base_year: int) -> pd.Series:
    """Return the multiplier that carries a nominal value of year y to base-year terms.

    The series is built by chaining annual inflation rates: a value
    observed in year y is multiplied by the product of (1 + rate) over
    every year after y up to and including *base_year*. A year later than
    the base year is divided by the same chain, so the mapping is exact in
    both directions. Years outside the index get no entry, and a caller
    treats an absent year as "cannot be deflated" rather than as 1.0.

    Parameters
    ----------
    index : pandas.DataFrame
        Long-form price index with ``country``, ``year`` and
        ``inflation_pct`` columns.
    country : str
        Country code to select rows for (e.g. ``'COL'``).
    base_year : int
        Year whose currency the output is expressed in.

    Returns
    -------
    pandas.Series
        Deflator indexed by year, covering every year the index reports
        for *country*. Missing rates inside the range make the years on
        the far side of the gap unavailable.
    """
    rows = index[index['country'] == country].dropna(subset=['year'])
    if rows.empty:
        raise ValueError(f'Price index has no rows for country {country!r}.')
    rates = rows.set_index('year')['inflation_pct'].sort_index()
    if base_year not in rates.index:
        raise ValueError(
            f'Price index for {country!r} has no row for base year {base_year}.'
        )
    factors = 1.0 + rates / 100.0
    years = rates.index.to_numpy()
    deflator = pd.Series(np.nan, index=rates.index, dtype='float64')
    # A value of year y is carried forward through the inflation of the
    # years y+1 ... base_year: the cumulative product of those factors.
    # Going the other way (y > base_year) divides by the product of
    # base_year+1 ... y.
    for year in years:
        if year == base_year:
            deflator.loc[year] = 1.0
        elif year < base_year:
            span = factors.loc[year + 1 : base_year]
            deflator.loc[year] = float(np.prod(span)) if span.notna().all() else np.nan
        else:
            span = factors.loc[base_year + 1 : year]
            deflator.loc[year] = (
                1.0 / float(np.prod(span)) if span.notna().all() else np.nan
            )
    return deflator


def _resolve_index_path(state: HarmonizeState, index: str) -> Path:
    """Resolve *index* beside the recipe unless it is already a path."""
    candidate = Path(index)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    recipe = state.recipe
    recipe_dir = recipe_path(recipe['admin_id'], recipe['entity'], as_dir=True)
    return Path(recipe_dir) / index


@_register('deflate_values')
def deflate_values(
    state: HarmonizeState,
    columns: list[str],
    year_column: str,
    index: str,
    base_year: int,
    country: str,
    output_suffix: str = '_real',
) -> HarmonizeState:
    """Re-express nominal amounts in base-year currency with a price index.

    Each of *columns* is multiplied by the deflator of the row's year (see
    :func:`deflator_by_year`) and written to ``{column}{output_suffix}``.
    A row whose year is missing, or outside the years the index covers,
    gets a missing real value: silently passing the nominal amount
    through would put values of different years on one axis, which is
    the error this step exists to prevent.

    Provenance is recorded in ``{output}_source`` as ``{token}+deflated``,
    where the token is the input column's own ``_source`` sidecar when it
    has one and the input column's name otherwise.

    Parameters
    ----------
    columns : list of str
        Nominal-value columns on the spine to deflate. A listed column
        that is absent is skipped.
    year_column : str
        Column holding the year each row's amount was observed in.
    index : str
        Price index CSV: a filename beside the recipe, or a path. Long
        form with ``country``, ``year`` and ``inflation_pct`` columns.
    base_year : int
        Year whose currency the outputs are expressed in.
    country : str
        Country code selecting the index rows (e.g. ``'COL'``).
    output_suffix : str, optional
        Suffix appended to each input column's name for its output,
        default ``'_real'``.
    """
    spine = state.spine
    if spine is None or year_column not in spine.columns:
        return state

    path = _resolve_index_path(state, index)
    deflator = deflator_by_year(_load_price_index(str(path)), country, base_year)
    years = pd.to_numeric(spine[year_column], errors='coerce')
    factor = years.map(deflator)

    for column in columns:
        if column not in spine.columns:
            continue
        output = f'{column}{output_suffix}'
        nominal = pd.to_numeric(spine[column], errors='coerce')
        spine[output] = nominal * factor
        source_column = f'{column}_source'
        if source_column in spine.columns:
            tokens = spine[source_column].astype(object).fillna(column)
        else:
            tokens = pd.Series(column, index=spine.index, dtype=object)
        spine[f'{output}_source'] = pd.NA
        written = spine[output].notna()
        for token in tokens[written].unique():
            mask = written & tokens.eq(token)
            _record_source(spine, output, mask, f'{token}+deflated')
        if state.verbose:
            n_missing = int((~written & nominal.notna()).sum())
            print(
                f'  deflate_values: {column} -> {output} in {base_year} terms; '
                f'{n_missing:,} value(s) left missing for years outside the index'
            )

    state.spine = spine
    return state
