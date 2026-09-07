"""The provenance-token vocabulary shared by every pipeline stage.

A ``{column}_source`` sidecar records what determined a column's value. The
tokens themselves are stage-agnostic, so the marker that says openplaces
filled a cell (rather than reading it off a source) is spelled here, at
layer 0, and every stage above uses these two functions rather than its own
copy: the guarantee rests on the writer and the reader agreeing.

:mod:`openplaces.io.curator.provenance` re-exports both and documents the
rule they encode, including why a modeled dataset's own value is not
marked and an estimated one is.
"""

from __future__ import annotations

import pandas as pd

# The marker every non-original value's token must contain, and the
# separator joining it to the route token it qualifies.
IMPUTED_MARKER = 'imputed'
TOKEN_SEPARATOR = '+'


def mark_imputed(token) -> str:
    """Return *token* marked as a derived, non-original value.

    A missing or empty *token* becomes the bare marker, so a derived value
    with no known route still reports itself as derived rather than as
    nothing. Idempotent: a token that already carries the marker is
    returned unchanged, which is what lets a chain of steps each mark what
    they pass along without accumulating ``imputed+imputed``.

    Parameters
    ----------
    token : str or None
        Route token naming how the value was arrived at.

    Returns
    -------
    str
        The token carrying the imputed marker.
    """
    if token is None or pd.isna(token) or str(token) == '':
        return IMPUTED_MARKER
    text = str(token)
    if IMPUTED_MARKER in text.split(TOKEN_SEPARATOR):
        return text
    return f'{text}{TOKEN_SEPARATOR}{IMPUTED_MARKER}'


def is_imputed(values) -> pd.Series:
    """Return a boolean Series flagging tokens that carry the marker.

    Matches the marker as a whole ``+``-separated part, never as a
    substring, so a source legitimately named e.g. ``imputed_rates`` is not
    mistaken for one. Missing tokens are False -- unknown provenance is not
    a claim that the value was derived.

    Parameters
    ----------
    values : array-like
        Provenance tokens.

    Returns
    -------
    pandas.Series
        True where the token carries the marker.
    """
    text = pd.Series(values).astype(object).astype('string')
    parts = text.str.split(TOKEN_SEPARATOR)
    return parts.map(
        lambda p: IMPUTED_MARKER in p if isinstance(p, list) else False
    ).astype(bool)
