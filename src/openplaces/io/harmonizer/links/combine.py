"""Writing a reference's values onto the spine: align two frames for
a combine, and fill by priority without overwriting what an earlier
source supplied.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


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


def _write_prioritized(
    spine: gpd.GeoDataFrame,
    name: str,
    new_vals: pd.Series,
    majority_coverage: float = 0.5,
    provenance_token: str | None = None,
) -> None:
    """Write *new_vals* into ``spine[name]``, applying the recency/coverage rule.

    A column already on the spine came from an earlier (less admin-specific,
    per :func:`_find_admin_scoped_recipe_ids`'s ordering) source. *new_vals*
    only overwrites it outright when *new_vals* itself covers at least
    *majority_coverage* of the spine; otherwise it only fills the existing
    column's gaps, so a sparse more-specific source can't blank out a more
    complete less-specific one.

    Parameters
    ----------
    provenance_token : str, optional
        When given, also updates the ``{name}_source`` categorical sidecar
        (via :func:`openplaces.io.harmonizer._record_source`) for exactly
        the cells this call actually changes -- a value written where none
        existed, or an existing value outright replaced on a
        majority-coverage overwrite. Cells left unchanged (including
        still-null cells) are never touched. Reserve this for a recipe's
        declared "key" columns (see ``link_by_id``'s ``track_provenance``)
        rather than every joined attribute, to avoid a provenance-sidecar
        column explosion.
    """
    from openplaces.io.harmonizer import _record_source, _record_sources

    def _mark(mask) -> None:
        """Record provenance, per row where the caller supplied it."""
        if isinstance(provenance_token, pd.Series):
            _record_sources(spine, name, provenance_token, mask)
        else:
            _record_source(spine, name, mask, provenance_token)

    if name not in spine.columns:
        spine[name] = new_vals
        if provenance_token is not None:
            _mark(new_vals.notna())
        return

    before = spine[name].copy() if provenance_token is not None else None
    coverage = new_vals.notna().mean() if len(new_vals) else 0.0
    existing, incoming = _align_for_combine(spine[name], new_vals)
    if coverage >= majority_coverage:
        spine[name] = incoming.combine_first(existing)
    else:
        spine[name] = existing.combine_first(incoming)

    if provenance_token is not None:
        after = spine[name]
        changed = after.notna() & (before.isna() | (before != after))
        _mark(changed)


#: Columns *ref_address_key* needs on the reference: keyword name of
#: add_address_id_local and its default column.
