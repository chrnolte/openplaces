"""
Pipeline step recording which rule linked a row:
  - record_link_method: label the rows a link pass reached, once

A transaction finds its parcel in tiers: first by the standardized parcel
number, and only where that finds nothing by an exact address key. The
two routes are not equally sure (a parcel number names one parcel, an
address can sit on a lot that was since split), so a consumer needs to
know which one a row came by. `link_by_id` writes the linked values and
no record of the route; this step writes the record.

The label names a rule, never a strength. It is a fixed word per pass
(`parcel_id_local`, `address_id_local`), it is written once, on the first
pass that reaches the row, and no later pass changes or removes it. That
is deliberate: a per-link score, a scorer tuned on the links it made, or
a step that takes an earlier link back is the shape of a patented method
in this domain (see the patent-risk section of AGENTS.md), and an exact
key tried after another exact key is none of those.

The step runs after the link pass it describes and reads the same
reference, with the same guard against placeholder keys, so it cannot
claim a match the link pass refused.
"""

from __future__ import annotations

import warnings

import pandas as pd

from openplaces.io.harmonizer import HarmonizeState, _register
from openplaces.io.harmonizer.links import (
    _discover_link_sources,
    _neutralize_degenerate_keys,
)
from openplaces.io.readers import get_entities


def _reference_keys(state, recipe_id, ref_key, layer=None) -> set:
    """Usable key values of one reference table for this admin unit."""
    try:
        ref = get_entities(recipe_id, state.admin_id, layer=layer, missing='ignore')
    except Exception as exc:
        warnings.warn(f'record_link_method: could not read {recipe_id}: {exc}')
        return set()
    if ref is None or ref_key not in getattr(ref, 'columns', ()):
        return set()
    ref = pd.DataFrame(ref[[ref_key]])
    ref = _neutralize_degenerate_keys(ref, ref_key, recipe_id)
    return set(ref[ref_key].dropna().astype('string'))


@_register('record_link_method')
def record_link_method(
    state: HarmonizeState,
    label: str,
    spine_key: str,
    ref_key: str | None = None,
    recipe_id: str | None = None,
    auto_discover: bool = False,
    entity_type: str = 'parcel',
    output_column: str = 'parcel_link_method',
) -> HarmonizeState:
    """Label rows the preceding link pass reached and no earlier one did.

    Parameters
    ----------
    label : str
        The rule's name, written as is (`parcel_id_local`).
    spine_key : str
        Spine column the pass joined on.
    ref_key : str, optional
        Reference column it joined to. Default: *spine_key*.
    recipe_id : str, optional
        The reference recipe of the pass.
    auto_discover : bool, optional
        True for a pass that joined every discovered source of
        *entity_type* instead of one recipe; the same discovery is run.
    entity_type : str, optional
        Entity type discovered when *auto_discover* is set.
    output_column : str, optional
        Column written. A row already labeled keeps its label.
    """
    spine = state.spine
    if spine is None or spine_key not in spine.columns:
        return state
    ref_key = ref_key or spine_key

    keys: set = set()
    if auto_discover:
        for match in _discover_link_sources(state, entity_type):
            # A source joined on another column (a layer's own key, a
            # supplement's) was not linked by the rule this pass names.
            if (match.get('key') or ref_key) != ref_key:
                continue
            keys |= _reference_keys(
                state, match['recipe_id'], ref_key, match.get('layer')
            )
    elif recipe_id:
        keys = _reference_keys(state, recipe_id, ref_key)
    else:
        raise ValueError('record_link_method needs recipe_id or auto_discover.')

    reached = spine[spine_key].astype('string').isin(keys)
    existing = (
        spine[output_column].astype('string')
        if output_column in spine.columns
        else pd.Series(pd.NA, index=spine.index, dtype='string')
    )
    spine[output_column] = existing.where(existing.notna() | ~reached, label)
    state.spine = spine
    if state.verbose:
        n_new = int((existing.isna() & reached).sum())
        print(f'  record_link_method: {n_new:,d} rows labeled {label!r}')
    return state
