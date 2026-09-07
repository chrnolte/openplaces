"""Per-variable provenance sidecars for the curate stage.

Each reconciled/inferred canonical column gets a categorical ``{column}_source``
sidecar recording which data source, model, or decision determined the value.
The token is a short string (e.g. ``parcel``, ``nsi``, ``imputed``, ``geometry``,
``keyword``); steps that re-decide a value overwrite the earlier token.

**A value that was not read off an original source must say so.** Any token
naming a derived value carries the ``imputed`` marker, appended after a
``+`` (``parcel+imputed``), so the distinction survives every downstream
step that only ever copies the token forward. :func:`mark_imputed` is the
one place that marker is spelled and :func:`is_imputed` the one place it is
read, because the whole guarantee rests on the two agreeing. The ``+``
separator is shared with the harmonize stage's composite tokens
(``parcel+usaddress``), so a token may carry both a route and the marker.

**The marker means openplaces filled this cell, not that the number is
modeled.** A value read from a dataset keeps that dataset's name even when
the dataset is itself a model: ``nsi`` is a FEMA-modeled structure value and
stays plain ``nsi``, because the token already names exactly what produced
it and a reader can judge it from there. What the marker exists to catch is
the opposite case -- a cell no source ever filled, that openplaces computed
and then labeled with the *route* it arrived by, so it reads as though a
source had supplied it. That was the defect: a delivered
``structure_value_source`` of ``parcel`` on a dollar figure no assessor
wrote.

Two consequences of drawing the line there. A classification vote is not
marked -- ``nsi``/``keyword``/``classifier`` each name the evidence that
won, which is already the honest answer. And apportioning a reference's
value across the entities on it is not marked either: the parcel's total is
a real assessed figure and apportionment divides it, rather than inventing a
value where none existed. Only estimation of a genuinely missing value
counts.
"""

from __future__ import annotations

import pandas as pd

from openplaces.core.attribute_registry import (
    PROVENANCE_SOURCE_SUFFIX as SOURCE_SUFFIX,
)

# Spelled at layer 0 (openplaces.core.provenance) so the harmonize stage,
# which may not import this module, writes the same marker rather than a
# second copy of it. Re-exported here, where the rule is documented.
from openplaces.core.provenance import (  # noqa: F401
    IMPUTED_MARKER,
    TOKEN_SEPARATOR,
    is_imputed,
    mark_imputed,
)


def source_column(column: str) -> str:
    """Return the provenance sidecar name for *column*."""
    return f'{column}{SOURCE_SUFFIX}'


def _object_sidecar(existing: pd.Series | None, index: pd.Index) -> pd.Series:
    """Return a writable object-dtype sidecar series for *index*.

    Creates one when *existing* is ``None``; casts an existing Categorical
    sidecar back to object first (``cast_categoricals`` re-applies the
    Categorical dtype at format time); otherwise copies *existing* so the
    caller's own series is never mutated.
    """
    if existing is None:
        return pd.Series(pd.NA, index=index, dtype=object)
    if isinstance(existing.dtype, pd.CategoricalDtype):
        return existing.astype(object)
    return existing.copy()


def _apply_source_mask(
    existing: pd.Series | None, index: pd.Index, mask, token: str
) -> pd.Series:
    """Return a ``_source`` sidecar series with *token* set for *mask* rows.

    Pure, allocation-only core shared by :func:`record_source` (writes
    in-place onto a live DataFrame) and any caller that must defer the
    actual DataFrame write itself (e.g.
    :func:`~openplaces.io.curator.evidence.merge_enrichments`, batching
    many columns into a single concat to avoid fragmenting a wide frame
    -- see its docstring).
    """
    existing = _object_sidecar(existing, index)
    existing.loc[mask] = token
    return existing


def record_source(curated, column: str, mask, token: str, imputed: bool = False):
    """Set the ``{column}_source`` sidecar to *token* for the *mask* rows.

    Creates the sidecar (object dtype) when absent; later calls overwrite earlier
    decisions for the rows they touch. *mask* may be a boolean Series aligned to
    ``curated.index`` or an index of rows. The sidecar is cast to Categorical by
    ``cast_categoricals`` at format time.

    Parameters
    ----------
    imputed : bool, optional
        Mark *token* as a derived value (default False). Prefer this over
        writing the marker into *token* by hand -- see :func:`mark_imputed`.
    """
    side = source_column(column)
    curated[side] = _apply_source_mask(
        curated[side] if side in curated.columns else None,
        curated.index,
        mask,
        mark_imputed(token) if imputed else token,
    )
    return curated


def record_sources(curated, column: str, tokens, mask=None, imputed: bool = False):
    """Set the ``{column}_source`` sidecar from a per-row *tokens* series.

    The row-varying counterpart of :func:`record_source`, for a step that
    carries an upstream sidecar forward rather than deciding one token for
    a whole group of rows. Rows whose token is missing are left untouched,
    so a partially-known provenance never blanks what an earlier step
    recorded.

    Parameters
    ----------
    tokens : Series or array-like
        Per-row provenance token, aligned to ``curated.index``.
    mask : Series or array-like of bool, optional
        Restrict the write to these rows (default: every row with a token).
    imputed : bool, optional
        Mark every written token as a derived value (default False).
    """
    tokens = pd.Series(tokens, index=curated.index).astype(object)
    if imputed:
        tokens = tokens.map(mark_imputed)
    write = tokens.notna()
    if mask is not None:
        write &= pd.Series(mask, index=curated.index).fillna(False).astype(bool)
    if not write.any():
        return curated

    side = source_column(column)
    existing = _object_sidecar(
        curated[side] if side in curated.columns else None, curated.index
    )
    existing.loc[write] = tokens.loc[write]
    curated[side] = existing
    return curated
