"""Shared apportionment of reference-polygon values across an n:m link.

Single implementation of "spread a reference polygon's values (e.g. a
parcel's improvement value) across the spine entities overlapping it",
following Lochhead et al. (2026, Table 4). Used by both pipeline stages:

- harmonize: :func:`~openplaces.io.harmonizer.attributes.reconcile_attributes`
  feeds it the in-memory identity overlay;
- curate: the ``apportion_curated_values`` step feeds it the link sidecar
  persisted by the harmonize overlay (``save_link: true``), joined to the
  *curated* reference entity's values.

Keeping the two stages on one code path guarantees the apportionment
semantics (overlap-area shares, dwelling-linked suppression, primary-only
columns, secondary handling) can never drift between them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    'APPORTIONED_VALUE_COLUMNS',
    'PROPORTIONAL_SPLIT_COLUMNS',
    'WHOLE_VALUE_COLUMNS',
    'ValueOverAllocationError',
    'apportion_reference_values',
]

# Reference columns this module knows how to apportion; anything else is the
# caller's business (e.g. the registry-driven generic aggregation in
# reconcile_attributes).
APPORTIONED_VALUE_COLUMNS = (
    'improvement_value',
    'n_dwellings',
    'year_built',
    'land_value',
    'total_value',
    'address',
    'land_value_imputed',
    'improvement_value_imputed',
)

# Columns sharing improvement_value's overlap-fraction, secondary-masked,
# dwelling-suppressed treatment (as opposed to land_value's whole-value-on-
# principal-entity-only special case below). A shared tuple/loop means the
# equal_area_ref_ids exemption and improvement_value_imputed both apply
# uniformly, with no duplicated logic: improvement_value_imputed always
# receives the exact same per-entity relative shares improvement_value itself
# resolves to.
PROPORTIONAL_SPLIT_COLUMNS = ('improvement_value', 'improvement_value_imputed')

# Columns sharing land_value's whole-value-on-principal-entity-only treatment
# (as opposed to PROPORTIONAL_SPLIT_COLUMNS above). land_value_imputed is
# conceptually "land_value, gap-filled" -- the same whole-parcel-value concept,
# just with the source's own value passed through where present and a learned
# estimate substituted only where it was genuinely missing -- so it gets the
# identical apportionment treatment, not the area-proportional one meant for
# improvement/structure values.
# total_value joins them rather than the proportional split: it is
# land + structures, so spreading it across every footprint would divide the
# land half by building geometry, and a parcel's total is in any case a
# single figure the assessor wrote once.
# "Whole" here means whole *per parcel*, not whole per building: the value
# goes to the principal entities only, and where a parcel has several of
# them it is divided among them rather than repeated on each. Repeating it
# was the original reading, and it inflated regional sums by 1.23x-1.69x on
# the two shipped CHEER bundles, because multi-building parcels are
# disproportionately the high-value commercial tail.
# `_assert_not_over_allocated` enforces the distinction on every call.
WHOLE_VALUE_COLUMNS = ('land_value', 'land_value_imputed', 'total_value')


def apportion_reference_values(
    pairs: pd.DataFrame,
    ref_values: pd.DataFrame,
    *,
    spine_id_col: str,
    ref_id_col: str = 'parcel_id',
    area_col: str = 'area_intersection_m2',
    priority: pd.Series | None = None,
    dwelling_linked_ids: set | None = None,
    volume_weight: pd.Series | None = None,
    equal_area_ref_ids: set | None = None,
) -> pd.DataFrame:
    """Apportion reference values to spine entities over an n:m link.

    Column semantics (only columns present in *ref_values* are produced):

    - ``improvement_value``, ``improvement_value_imputed``
      (:data:`PROPORTIONAL_SPLIT_COLUMNS`): each multiplied by every pair's
      within-reference overlap fraction and summed per spine entity; missing
      (not 0) on entities whose *priority* is ``'secondary'``. A reference id
      in *equal_area_ref_ids* is exempt from both the dwelling-linked
      suppression below and this secondary masking, so every one of its
      linked entities keeps its computed area share — see *equal_area_ref_ids*.
    - ``n_dwellings``: zeroed on ``'secondary'`` entities (an accessory
      structure contains no dwellings), then fraction-split and summed.
      Unaffected by *equal_area_ref_ids*.
    - ``year_built``: 0 treated as missing, mean over the entity's linked
      references (unweighted).
    - ``land_value``, ``land_value_imputed``, ``total_value``
      (:data:`WHOLE_VALUE_COLUMNS`): the dominant (largest-overlap)
      reference's value, kept only on principal entities —
      ``priority == 'primary'`` when *priority* is given, else entities that
      are the sole spine entity on their dominant reference — and suppressed
      for entities affected by the dwelling-linked rule below. A reference
      with one qualifying entity gives it the whole value; a reference with
      several **divides** the value among them by overlap area (equally, if
      none has a recorded area), because a parcel's assessed value is one
      figure and must never be repeated per building. Unaffected by
      *equal_area_ref_ids*.
    - ``address``: the dominant reference's value, unrestricted.

    Parameters
    ----------
    pairs : pandas.DataFrame
        One row per (spine entity, reference) link, with columns
        *spine_id_col*, *ref_id_col*, and *area_col*. Rows with a missing
        reference id are ignored.
    ref_values : pandas.DataFrame
        Reference values indexed by *ref_id_col*. Only columns from
        ``APPORTIONED_VALUE_COLUMNS`` are consumed.
    spine_id_col : str
        Name of the spine-entity id column in *pairs*.
    ref_id_col : str, optional
        Name of the reference id column in *pairs* (default ``parcel_id``).
    area_col : str, optional
        Overlap-area column in *pairs* used for fractions and dominance.
    priority : pandas.Series, optional
        ``priority_on_parcel``-style role per spine id (``'primary'`` /
        ``'secondary'`` / ``'unknown'``). Drives the secondary and
        principal-only rules above.
    dwelling_linked_ids : set, optional
        Spine ids with dwelling-point evidence. Within references where at
        least one linked entity has such evidence, entities without it get a
        zero share (fractions renormalize over the rest) and their
        ``land_value`` is suppressed (Lochhead et al. 2026, Table 4,
        Cases 1–2 vs. Case 3).
    volume_weight : pandas.Series, optional
        Story count per spine id; when given, fractions weight by
        ``area × max(n_stories, 1)`` instead of area alone.
    equal_area_ref_ids : set, optional
        Reference ids (e.g. non-residential parcels, which have no single
        dwelling to anchor a residential-style split to) whose
        :data:`PROPORTIONAL_SPLIT_COLUMNS` entities all keep their computed
        overlap-area share regardless of dwelling-linked or secondary status
        — every real structure on the reference (a warehouse plus its loading
        dock, a retail strip's several units) gets its area-proportional
        piece. Has no effect on ``n_dwellings``, ``year_built``, or
        :data:`WHOLE_VALUE_COLUMNS`.

    Returns
    -------
    pandas.DataFrame
        Indexed by spine id (only ids appearing in *pairs*), with one column
        per apportioned value.
    """
    value_cols = [c for c in APPORTIONED_VALUE_COLUMNS if c in ref_values.columns]
    empty = pd.DataFrame(index=pd.Index([], name=spine_id_col))
    if not value_cols:
        return empty

    pairs = (
        pairs[[spine_id_col, ref_id_col, area_col]]
        .dropna(subset=[ref_id_col])
        .reset_index(drop=True)
    )
    if pairs.empty:
        return empty

    spine_ids = pairs[spine_id_col]
    ref_ids = pairs[ref_id_col]

    weight = pd.to_numeric(pairs[area_col], errors='coerce').astype(float)
    if volume_weight is not None:
        n_eff = volume_weight.reindex(spine_ids).fillna(1.0).clip(lower=1.0).to_numpy()
        weight = weight * n_eff
    fraction = weight / weight.groupby(ref_ids).transform('sum')

    def _zero_and_renormalize(base: pd.Series, mask: pd.Series) -> pd.Series:
        if not mask.any():
            return base
        adjusted = base.mask(mask, 0.0)
        adjusted_sum = adjusted.groupby(ref_ids).transform('sum')
        return (adjusted / adjusted_sum.replace(0, np.nan)).fillna(0.0)

    suppressed_ids: set = set()
    suppress = pd.Series(False, index=pairs.index)
    if dwelling_linked_ids:
        has_dwelling = spine_ids.isin(dwelling_linked_ids)
        ref_has_dwelling = has_dwelling.groupby(ref_ids).transform('any')
        suppress = ref_has_dwelling & ~has_dwelling
        if suppress.any():
            suppressed_ids = set(spine_ids[suppress])

    # n_dwellings/land_value keep the plain dwelling-linked suppression,
    # unaffected by equal_area_ref_ids (see docstring).
    fraction_suppressed = _zero_and_renormalize(fraction, suppress)

    # PROPORTIONAL_SPLIT_COLUMNS get the equal_area_ref_ids-exempted variant:
    # a pair whose reference is in equal_area_ref_ids is never suppressed,
    # regardless of dwelling-linked status.
    equal_area_pair = (
        ref_ids.isin(equal_area_ref_ids)
        if equal_area_ref_ids
        else pd.Series(False, index=pairs.index)
    )
    fraction_proportional = _zero_and_renormalize(fraction, suppress & ~equal_area_pair)
    equal_area_spine_ids = (
        set(spine_ids[equal_area_pair]) if equal_area_ref_ids else set()
    )

    attrs = pairs.join(ref_values[value_cols], on=ref_id_col)

    if priority is not None:
        is_secondary_pair = (
            priority.reindex(spine_ids).eq('secondary').fillna(False).to_numpy()
        )
    else:
        is_secondary_pair = np.zeros(len(pairs), dtype=bool)

    out: dict[str, pd.Series] = {}

    for col in PROPORTIONAL_SPLIT_COLUMNS:
        if col not in value_cols:
            continue
        split = (attrs[col] * fraction_proportional).round(2)
        summed = split.groupby(spine_ids).sum()
        if priority is not None:
            secondary = priority.reindex(summed.index).eq('secondary').fillna(False)
            if equal_area_spine_ids:
                secondary = secondary & ~summed.index.isin(equal_area_spine_ids)
            summed = summed.mask(secondary)
        out[col] = summed

    if 'n_dwellings' in value_cols:
        dwellings = attrs['n_dwellings'].mask(is_secondary_pair, 0.0)
        out['n_dwellings'] = (
            (dwellings * fraction_suppressed).round(2).groupby(spine_ids).sum()
        )

    if 'year_built' in value_cols:
        out['year_built'] = (
            attrs['year_built'].replace(0, np.nan).groupby(spine_ids).mean()
        )

    dominant = None
    whole_value_cols = [c for c in WHOLE_VALUE_COLUMNS if c in value_cols]
    if whole_value_cols or 'address' in value_cols:
        dominant = (
            attrs.sort_values(area_col, ascending=False)
            .drop_duplicates(spine_id_col)
            .set_index(spine_id_col)
        )
        if 'address' in value_cols:
            out['address'] = dominant['address']
        if whole_value_cols:
            if priority is not None:
                is_principal = (
                    priority.reindex(dominant.index).eq('primary').fillna(False)
                )
            else:
                n_per_ref = attrs.groupby(ref_id_col)[spine_id_col].size()
                is_principal = (
                    n_per_ref.reindex(dominant[ref_id_col])
                    .eq(1)
                    .set_axis(dominant.index)
                )
            if suppressed_ids:
                is_principal = is_principal & ~dominant.index.isin(suppressed_ids)
            # A parcel may carry several principal structures (an apartment
            # complex, a school, a refinery), so "the value goes whole to the
            # principal entity" cannot mean "to each of them" -- that prints
            # the same dollar once per building and inflates every regional
            # sum. Where more than one entity qualifies, the parcel's value is
            # divided among them by overlap area, the same weight the
            # proportional columns use; where exactly one qualifies it still
            # receives the whole value, so the single-building case (the
            # overwhelming majority) is unchanged.
            share_weight = (
                pd.to_numeric(dominant[area_col], errors='coerce')
                .astype(float)
                .where(is_principal, 0.0)
                .clip(lower=0.0)
            )
            weight_per_ref = share_weight.groupby(dominant[ref_id_col]).transform('sum')
            share = share_weight / weight_per_ref.replace(0.0, np.nan)
            # Equal division is the fallback when a reference's qualifying
            # entities all have zero recorded overlap area, which a degenerate
            # or point-derived geometry can produce.
            n_principal = (
                is_principal.astype(float)
                .groupby(dominant[ref_id_col])
                .transform('sum')
            )
            share = share.where(
                weight_per_ref > 0, is_principal.astype(float) / n_principal
            )
            for col in whole_value_cols:
                out[col] = (dominant[col] * share).where(is_principal)

    result = pd.DataFrame(out)
    result.index.name = spine_id_col
    _assert_not_over_allocated(
        result,
        pairs=pairs,
        ref_values=ref_values,
        value_cols=value_cols,
        spine_id_col=spine_id_col,
        ref_id_col=ref_id_col,
        dominant_ref=(
            dominant[ref_id_col] if whole_value_cols and dominant is not None else None
        ),
    )
    return result


class ValueOverAllocationError(AssertionError):
    """More value was handed to spine entities than the reference holds.

    Raised by :func:`apportion_reference_values` when a reference's value
    reaches its overlapping entities more than once -- the failure mode where
    a parcel with several principal buildings pays its assessed value to each
    of them, so the column looks right per building and over-counts in every
    sum. An `AssertionError` subclass because it reports a broken invariant
    in this module, not a bad argument from the caller.
    """


def _assert_not_over_allocated(
    result: pd.DataFrame,
    *,
    pairs: pd.DataFrame,
    ref_values: pd.DataFrame,
    value_cols: list[str],
    spine_id_col: str,
    ref_id_col: str,
    dominant_ref: pd.Series | None,
) -> None:
    """Fail if any reference's value was allocated more than once.

    The invariant every apportioned money column must satisfy: what the
    entities receive from a reference never exceeds what that reference
    holds. Under-allocation is legitimate and unchecked -- a parcel with no
    buildings, a suppressed accessory structure, or a secondary-masked
    column all leave value unassigned on purpose.

    Checked here, in the one function both the harmonize and the curate
    stage apportion through, so the guarantee covers every recipe and every
    dataset rather than one pipeline's configuration.

    Whole-value columns are checked per reference, which is exact: each
    entity draws them from exactly one dominant reference. Proportional
    columns are checked on the total instead, because an entity sums shares
    from several references and no per-reference attribution survives into
    *result*; duplication still shows up there as a total above the sum of
    the references actually linked.

    Raises
    ------
    ValueOverAllocationError
        Naming the column, the worst-offending reference, and both amounts.
    """
    if result.empty:
        return

    linked_refs = pairs[ref_id_col].dropna().unique()
    available = ref_values.reindex(linked_refs)

    for col in value_cols:
        if col not in result.columns or col not in available.columns:
            continue
        if col == 'address' or col == 'year_built':
            continue  # not money, and not divided
        allocated = pd.to_numeric(result[col], errors='coerce')
        source = pd.to_numeric(available[col], errors='coerce')
        if col in WHOLE_VALUE_COLUMNS and dominant_ref is not None:
            per_ref = allocated.groupby(dominant_ref.reindex(allocated.index)).sum()
            source_per_ref = source.reindex(per_ref.index)
            # Per-pair rounding can add at most half a cent per entity.
            tolerance = (
                0.005 * allocated.groupby(dominant_ref.reindex(allocated.index)).size()
                + source_per_ref.abs() * 1e-9
            )
            excess = per_ref - source_per_ref - tolerance
            if (excess > 0).any():
                worst = excess.idxmax()
                raise ValueOverAllocationError(
                    f"'{col}' was over-allocated: reference {worst!r} holds "
                    f'{source_per_ref[worst]:,.2f} but {per_ref[worst]:,.2f} '
                    "was distributed to its entities. A reference's value "
                    'must be split among the entities that share it, never '
                    'repeated on each of them.'
                )
        else:
            total_allocated = allocated.sum()
            total_source = source.sum()
            tolerance = 0.005 * len(allocated) + abs(total_source) * 1e-9
            if total_allocated - total_source > tolerance:
                raise ValueOverAllocationError(
                    f"'{col}' was over-allocated: the linked references hold "
                    f'{total_source:,.2f} in total but {total_allocated:,.2f} '
                    'was distributed to spine entities.'
                )
