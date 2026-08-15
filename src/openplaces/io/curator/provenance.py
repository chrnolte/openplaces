"""Per-variable provenance sidecars for the curate stage.

Each reconciled/inferred canonical column gets a categorical ``{column}_source``
sidecar recording which data source, model, or decision determined the value.
The token is a short string (e.g. ``parcel``, ``nsi``, ``imputed``, ``geometry``,
``keyword``); steps that re-decide a value overwrite the earlier token.
"""

from __future__ import annotations

import pandas as pd

from openplaces.core.attribute_registry import (
    PROVENANCE_SOURCE_SUFFIX as SOURCE_SUFFIX,
)


def source_column(column: str) -> str:
    """Return the provenance sidecar name for *column*."""
    return f'{column}{SOURCE_SUFFIX}'


def _apply_source_mask(
    existing: pd.Series | None, index: pd.Index, mask, token: str
) -> pd.Series:
    """Return a ``_source`` sidecar series with *token* set for *mask* rows.

    Pure, allocation-only core shared by :func:`record_source` (writes
    in-place onto a live DataFrame) and any caller that must defer the
    actual DataFrame write itself (e.g.
    :func:`~openplaces.io.curator.evidence.merge_enrichments`, batching
    many columns into a single concat to avoid fragmenting a wide frame
    -- see its docstring). Creates the sidecar (object dtype) when
    *existing* is ``None``; casts an existing Categorical sidecar back to
    object first (``cast_categoricals`` re-applies the Categorical dtype
    at format time); otherwise copies *existing* so the caller's own
    series is never mutated.
    """
    if existing is None:
        existing = pd.Series(pd.NA, index=index, dtype=object)
    elif isinstance(existing.dtype, pd.CategoricalDtype):
        existing = existing.astype(object)
    else:
        existing = existing.copy()
    existing.loc[mask] = token
    return existing


def record_source(curated, column: str, mask, token: str):
    """Set the ``{column}_source`` sidecar to *token* for the *mask* rows.

    Creates the sidecar (object dtype) when absent; later calls overwrite earlier
    decisions for the rows they touch. *mask* may be a boolean Series aligned to
    ``curated.index`` or an index of rows. The sidecar is cast to Categorical by
    ``cast_categoricals`` at format time.
    """
    side = source_column(column)
    curated[side] = _apply_source_mask(
        curated[side] if side in curated.columns else None,
        curated.index,
        mask,
        token,
    )
    return curated
