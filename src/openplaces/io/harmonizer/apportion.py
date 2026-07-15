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

__all__ = ['APPORTIONED_VALUE_COLUMNS', 'apportion_reference_values']

# Reference columns this module knows how to apportion; anything else is the
# caller's business (e.g. the registry-driven generic aggregation in
# reconcile_attributes).
APPORTIONED_VALUE_COLUMNS = (
    'improvement_value',
    'n_dwellings',
    'year_built',
    'land_value',
    'address',
)


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
) -> pd.DataFrame:
    """Apportion reference values to spine entities over an n:m link.

    Column semantics (only columns present in *ref_values* are produced):

    - ``improvement_value``: multiplied by each pair's within-reference
      overlap fraction and summed per spine entity; missing (not 0) on
      entities whose *priority* is ``'secondary'``.
    - ``n_dwellings``: zeroed on ``'secondary'`` entities (an accessory
      structure contains no dwellings), then fraction-split and summed.
    - ``year_built``: 0 treated as missing, mean over the entity's linked
      references (unweighted).
    - ``land_value``: the dominant (largest-overlap) reference's whole value,
      kept only on principal entities — ``priority == 'primary'`` when
      *priority* is given, else entities that are the sole spine entity on
      their dominant reference — and suppressed for entities affected by the
      dwelling-linked rule below.
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

    suppressed_ids: set = set()
    if dwelling_linked_ids:
        has_dwelling = spine_ids.isin(dwelling_linked_ids)
        ref_has_dwelling = has_dwelling.groupby(ref_ids).transform('any')
        suppress = ref_has_dwelling & ~has_dwelling
        if suppress.any():
            suppressed_ids = set(spine_ids[suppress])
            fraction = fraction.mask(suppress, 0.0)
            fraction_sum = fraction.groupby(ref_ids).transform('sum')
            fraction = (fraction / fraction_sum.replace(0, np.nan)).fillna(0.0)

    attrs = pairs.join(ref_values[value_cols], on=ref_id_col)

    if priority is not None:
        is_secondary_pair = (
            priority.reindex(spine_ids).eq('secondary').fillna(False).to_numpy()
        )
    else:
        is_secondary_pair = np.zeros(len(pairs), dtype=bool)

    out: dict[str, pd.Series] = {}

    if 'improvement_value' in value_cols:
        split = (attrs['improvement_value'] * fraction).round(2)
        summed = split.groupby(spine_ids).sum()
        if priority is not None:
            secondary = priority.reindex(summed.index).eq('secondary').fillna(False)
            summed = summed.mask(secondary)
        out['improvement_value'] = summed

    if 'n_dwellings' in value_cols:
        dwellings = attrs['n_dwellings'].mask(is_secondary_pair, 0.0)
        out['n_dwellings'] = (dwellings * fraction).round(2).groupby(spine_ids).sum()

    if 'year_built' in value_cols:
        out['year_built'] = (
            attrs['year_built'].replace(0, np.nan).groupby(spine_ids).mean()
        )

    if 'land_value' in value_cols or 'address' in value_cols:
        dominant = (
            attrs.sort_values(area_col, ascending=False)
            .drop_duplicates(spine_id_col)
            .set_index(spine_id_col)
        )
        if 'address' in value_cols:
            out['address'] = dominant['address']
        if 'land_value' in value_cols:
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
            land_value = dominant['land_value'].where(is_principal)
            if suppressed_ids:
                land_value[land_value.index.isin(suppressed_ids)] = np.nan
            out['land_value'] = land_value

    result = pd.DataFrame(out)
    result.index.name = spine_id_col
    return result
