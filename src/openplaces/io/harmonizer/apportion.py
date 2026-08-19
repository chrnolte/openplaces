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
# land + structures, so splitting it by footprint area would divide the
# land half by building geometry, and a parcel's total is in any case a
# single figure the assessor wrote once. Whole-on-principal keeps the
# regional sum exact while never printing the same dollar twice across a
# parcel's footprints.
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
    - ``land_value``, ``land_value_imputed`` (:data:`WHOLE_VALUE_COLUMNS`):
      the dominant (largest-overlap) reference's whole value, kept only on
      principal entities — ``priority == 'primary'`` when *priority* is
      given, else entities that are the sole spine entity on their dominant
      reference — and suppressed for entities affected by the dwelling-linked
      rule below. Unaffected by *equal_area_ref_ids*.
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
            for col in whole_value_cols:
                whole_value = dominant[col].where(is_principal)
                if suppressed_ids:
                    whole_value[whole_value.index.isin(suppressed_ids)] = np.nan
                out[col] = whole_value

    result = pd.DataFrame(out)
    result.index.name = spine_id_col
    return result
