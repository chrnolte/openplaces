"""Per-variable provenance sidecars for the curate stage.

Each reconciled/inferred canonical column gets a categorical ``{column}_source``
sidecar recording which data source, model, or decision determined the value.
The token is a short string (e.g. ``parcel``, ``nsi``, ``imputed``, ``geometry``,
``keyword``); steps that re-decide a value overwrite the earlier token.
"""

from __future__ import annotations

import pandas as pd

SOURCE_SUFFIX = '_source'


def source_column(column: str) -> str:
    """Return the provenance sidecar name for *column*."""
    return f'{column}{SOURCE_SUFFIX}'


def record_source(curated, column: str, mask, token: str):
    """Set the ``{column}_source`` sidecar to *token* for the *mask* rows.

    Creates the sidecar (object dtype) when absent; later calls overwrite earlier
    decisions for the rows they touch. *mask* may be a boolean Series aligned to
    ``curated.index`` or an index of rows. The sidecar is cast to Categorical by
    ``cast_categoricals`` at format time.
    """
    side = source_column(column)
    if side not in curated.columns:
        curated[side] = pd.Series(pd.NA, index=curated.index, dtype=object)
    elif isinstance(curated[side].dtype, pd.CategoricalDtype):
        curated[side] = curated[side].astype(object)
    curated.loc[mask, side] = token
    return curated
