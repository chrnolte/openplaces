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
pass that reaches the row, and no later pass changes or removes it.

That is deliberate, and the reason is narrower than "we add no score".
US10606854B2's software claim (17) recites no scoring, no calibration
and no unlinking at all, so the absence of those does not by itself
distinguish a tiered match from it. What does is that **every tier here
compares exact keys**: a parcel number, then an address key built from
normalized components. Nothing falls through to a similarity
comparison. Do not add a fuzzy tier behind these passes without the
maintainer's decision (see the patent-risk section of AGENTS.md); the
labels this step writes would be the natural place to hang one, which
is exactly why the warning belongs here.

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


def _reference_keys(state, recipe_id, ref_key, layer=None, spine_key=None) -> set:
    """Usable key values of one reference table for this admin unit.

    *spine_key* is the spine's key series, passed on to the placeholder
    guard exactly as `link_by_id` passes it, so both spare the same keys.
    """
    try:
        ref = get_entities(recipe_id, state.admin_id, layer=layer, missing='ignore')
    except Exception as exc:
        warnings.warn(f'record_link_method: could not read {recipe_id}: {exc}')
        return set()
    if ref is None or ref_key not in getattr(ref, 'columns', ()):
        return set()
    ref = pd.DataFrame(ref[[ref_key]])
    ref = _neutralize_degenerate_keys(ref, ref_key, recipe_id, spine_key=spine_key)
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
    via_column: str | None = None,
    via_label: str | None = None,
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
    via_column, via_label : str, optional
        A reached row with a value in *via_column* is labeled
        *via_label* instead: a sale that named a stacked unit and found
        the parcel through the unit's lot (`lot_id_local`,
        `stacked_units`) came by a different rule than one whose own
        number is a parcel's.
    """
    spine = state.spine
    if spine is None or spine_key not in spine.columns:
        return state
    ref_key = ref_key or spine_key
    skey = spine[spine_key]

    keys: set = set()
    if auto_discover:
        for match in _discover_link_sources(state, entity_type):
            # A source joined on another column (a layer's own key, a
            # supplement's) was not linked by the rule this pass names.
            if (match.get('key') or ref_key) != ref_key:
                continue
            keys |= _reference_keys(
                state, match['recipe_id'], ref_key, match.get('layer'), skey
            )
    elif recipe_id:
        keys = _reference_keys(state, recipe_id, ref_key, spine_key=skey)
    else:
        raise ValueError('record_link_method needs recipe_id or auto_discover.')

    reached = skey.astype('string').isin(keys)
    existing = (
        spine[output_column].astype('string')
        if output_column in spine.columns
        else pd.Series(pd.NA, index=spine.index, dtype='string')
    )
    labels = pd.Series(label, index=spine.index, dtype='string')
    if via_column and via_label and via_column in spine.columns:
        labels = labels.mask(spine[via_column].notna(), via_label)
    spine[output_column] = existing.where(existing.notna() | ~reached, labels)
    state.spine = spine
    if state.verbose:
        n_new = int((existing.isna() & reached).sum())
        print(f'  record_link_method: {n_new:,d} rows labeled {label!r}')
    return state
