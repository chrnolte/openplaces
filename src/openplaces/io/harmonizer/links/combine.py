"""Writing a reference's values onto the spine: align two frames for
a combine, and fill by priority without overwriting what an earlier
source supplied.
"""

from __future__ import annotations

from typing import NamedTuple

import geopandas as gpd
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_object_dtype


def _align_for_combine(existing, incoming):
    """Make two columns safe to `combine_first`, or fall back to object.

    `Series.combine_first` re-infers a dtype for the union, and on two date
    columns that disagree about time zones it raises rather than choosing
    one ("Tz-aware datetime.datetime cannot be converted to datetime64").
    That is a real mix: one source stamps its sale dates UTC and another
    writes naive local dates, and nothing upstream reconciles them. It
    surfaced the first time a statewide layer's date columns reached a
    county whose own source is naive.

    Where both sides are datetimes, the tz-aware one is converted to UTC
    and the naive one localized to it, so the union is comparable rather
    than merely combinable. Anything else that cannot be reconciled falls
    back to object dtype: losing a dtype is recoverable, losing a county's
    whole harmonize run is not.
    """
    left_dt = is_datetime64_any_dtype(existing)
    right_dt = is_datetime64_any_dtype(incoming)
    if not (left_dt or right_dt):
        return existing, incoming
    if not (left_dt and right_dt):
        # One side is datetimes and the other is not (commonly object
        # holding datetimes, which is what `to_datetime` chokes on). There
        # is no timezone to reconcile because one side has no dtype to
        # reconcile it with, so hand both over as object and let
        # combine_first do the only thing that cannot raise.
        return existing.astype(object), incoming.astype(object)
    left_tz = getattr(existing.dtype, 'tz', None)
    right_tz = getattr(incoming.dtype, 'tz', None)
    if (left_tz is None) == (right_tz is None):
        return existing, incoming
    try:
        aware = 'UTC'
        existing = (
            existing.dt.tz_localize(aware)
            if left_tz is None
            else existing.dt.tz_convert(aware)
        )
        incoming = (
            incoming.dt.tz_localize(aware)
            if right_tz is None
            else incoming.dt.tz_convert(aware)
        )
    except (AttributeError, TypeError, ValueError):
        return existing.astype(object), incoming.astype(object)
    return existing, incoming


class RowRule(NamedTuple):
    """Rows where a write overwrites or only fills, whatever its coverage.

    Built per reference by
    :func:`~openplaces.io.harmonizer.links.own_layer.own_layer_rule`.
    A row in neither mask follows the coverage rule.

    Parameters
    ----------
    overwrite : pandas.Series of bool
        Rows where every value the reference states replaces the cell.
    fill : pandas.Series of bool
        Rows where the reference only fills an empty cell.
    values : tuple of str
        Value columns a lot takes from one layer or not at all (the
        step's ``value_group``); empty for no group.
    year : str, optional
        The column holding the tax year of *values*, which follows them.
    keep : pandas.Series of bool, optional
        Set per column by
        :func:`~openplaces.io.harmonizer.links.own_layer.value_group_writes`:
        rows where the cell stays as it is, whatever the reference states.
    exact : pandas.Series of bool, optional
        Set the same way: rows where the cell takes the reference's
        statement as it is, a missing one included.
    """

    overwrite: pd.Series
    fill: pd.Series
    values: tuple[str, ...] = ()
    year: str | None = None
    keep: pd.Series | None = None
    exact: pd.Series | None = None


def _blank_as_missing(values: pd.Series) -> pd.Series:
    """*values* with empty and whitespace-only strings made missing.

    A source that writes an empty string states nothing, but pandas
    counts it as a value: a statewide layer's empty `use_group`
    replaced a county's label on every lot it covered (measured
    2026-11-18 on US-NC-ONS: 92,350 lots). Only text columns are
    touched; a category column loses only its blank categories.
    """
    if isinstance(values.dtype, pd.CategoricalDtype):
        blank_categories = [
            c for c in values.cat.categories if isinstance(c, str) and not c.strip()
        ]
        if not blank_categories:
            return values
        return values.cat.remove_categories(blank_categories)
    if isinstance(values.dtype, pd.StringDtype):
        blank = values.str.strip().eq('').fillna(False).astype(bool)
    elif is_object_dtype(values.dtype):
        blank = values.map(
            lambda v: isinstance(v, str) and not v.strip(), na_action='ignore'
        )
        blank = blank.fillna(False).astype(bool)
    else:
        return values
    if not blank.any():
        return values
    return values.mask(blank)


def _write_prioritized(
    spine: gpd.GeoDataFrame,
    name: str,
    new_vals: pd.Series,
    majority_coverage: float = 0.5,
    provenance_token: str | None = None,
    row_rule: RowRule | None = None,
) -> None:
    """Write *new_vals* into ``spine[name]``, applying the recency/coverage rule.

    A column already on the spine came from an earlier (less admin-specific,
    per :func:`_find_admin_scoped_recipe_ids`'s ordering) source. *new_vals*
    only overwrites it outright when *new_vals* itself covers at least
    *majority_coverage* of the spine; otherwise it only fills the existing
    column's gaps, so a sparse more-specific source can't blank out a more
    complete less-specific one.

    An empty or whitespace-only string counts as missing on both sides:
    it never overwrites a value, and a later source may fill it.

    Parameters
    ----------
    row_rule : RowRule, optional
        Rows that overwrite or only fill whatever the coverage. The
        coverage is still measured over every row, so the rows outside
        both masks behave as without a rule. Its *keep* rows are left
        as they are, and its *exact* rows take *new_vals* as it is,
        missing included (a cell emptied there loses its `_source`).
    provenance_token : str, optional
        When given, also updates the ``{name}_source`` categorical sidecar
        (via :func:`openplaces.io.harmonizer._record_source`) for exactly
        the cells this call wrote. On a gap fill that is every cell that
        was empty and now holds a value. On a majority-coverage overwrite
        it is every cell *new_vals* holds a value for, including cells
        whose value it replaced with an equal one. A cell *new_vals*
        leaves empty keeps its value and its label. Reserve this for a
        recipe's declared "key" columns (see ``link_by_id``'s
        ``track_provenance``) rather than every joined attribute, to
        avoid a provenance-sidecar column explosion.
    """
    from openplaces.io.harmonizer import (
        _PROVENANCE_SUFFIX,
        _record_source,
        _record_sources,
    )

    def _mark(mask) -> None:
        """Record provenance, per row where the caller supplied it."""
        if isinstance(provenance_token, pd.Series):
            _record_sources(spine, name, provenance_token, mask)
        else:
            _record_source(spine, name, mask, provenance_token)

    new_vals = _blank_as_missing(new_vals)
    if name not in spine.columns:
        if row_rule is not None and row_rule.keep is not None:
            # A kept cell stays as it is: empty, on a new column.
            new_vals = new_vals.mask(row_rule.keep)
        spine[name] = new_vals
        if provenance_token is not None:
            _mark(new_vals.notna())
        return

    before = spine[name].copy() if provenance_token is not None else None
    coverage = new_vals.notna().mean() if len(new_vals) else 0.0
    existing, incoming = _align_for_combine(_blank_as_missing(spine[name]), new_vals)
    overwrite = coverage >= majority_coverage
    replaced = incoming.combine_first(existing)
    filled = existing.combine_first(incoming)
    if row_rule is None:
        spine[name] = replaced if overwrite else filled
    else:
        # Per row: the rule's masks decide, the coverage rule the rest.
        overwrite = row_rule.overwrite | (overwrite & ~row_rule.fill)
        result = replaced.where(overwrite, filled)
        if row_rule.keep is not None:
            result = result.where(~row_rule.keep, existing)
            overwrite = overwrite & ~row_rule.keep
        if row_rule.exact is not None:
            result = result.where(~row_rule.exact, incoming)
            overwrite = overwrite | row_rule.exact
            # A cell emptied this way names no source any more.
            side = f'{name}{_PROVENANCE_SUFFIX}'
            emptied = row_rule.exact & result.isna()
            if side in spine.columns and (emptied & spine[side].notna()).any():
                spine[side] = spine[side].astype(object).mask(emptied)
        spine[name] = result

    if provenance_token is None:
        return
    after = spine[name]
    if row_rule is not None:
        # The two cases below, each on the rows written its way.
        by_overwrite = incoming.notna() & after.notna() & overwrite
        by_fill = after.notna() & (before.isna() | (before != after)) & ~overwrite
        _mark(by_overwrite | by_fill)
        return
    if overwrite:
        # The label names the source the value was taken from, not
        # the first source that happened to state it. On an overwrite
        # every non-null incoming cell is that source's value, so all
        # of them are labeled, equal ones included. Labeling only the
        # cells whose value changed (the rule before 2026-10-17) left
        # a statewide layer's label on every parcel where the county
        # roll agreed with it, so the sidecar under-counted the roll
        # it was meant to credit (US-TX-HAR, TxGIO vs the HCAD roll).
        # Values are untouched: this mask only feeds the sidecar.
        written = incoming.notna() & after.notna()
    else:
        # A gap fill writes only where the spine was empty, so the
        # cells it wrote are exactly the ones that changed.
        written = after.notna() & (before.isna() | (before != after))
    _mark(written)


#: Columns *ref_address_key* needs on the reference: keyword name of
#: add_address_id_local and its default column.
