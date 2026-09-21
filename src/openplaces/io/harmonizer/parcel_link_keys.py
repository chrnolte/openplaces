"""
Pipeline step choosing the key a row finds its parcel by:
  - derive_parcel_link_key: the row's lot where the row names a unit

Every join from a transaction to a parcel goes through the one column
this step writes (`parcel_link_key`), so the rule for "which parcel does
this record's number mean" lives in one place.

Why a rule is needed at all. Since the ingest-time split of stacked
parcel layers (`io/stacked_units.py`), a lot that carries several
ownership records is one parcel row, keyed on the lot, and each record
is a property row keeping its own number and naming its lot in
`lot_id_local`. A deed for a condo unit states the unit's number. That
number is on no parcel row any more, so a join on `parcel_id_local`
alone would lose every condo sale in a re-ingested county, silently.

The rule is a lookup in the pairs the split itself recorded (unit key,
lot key): an exact match on the unit's key, or the row keeps its own
number. A unit the split saw on several lots is left on its own number,
not assigned to one of them: that would state which parcel was sold when
the record does not say. Nothing is compared for
similarity, nothing is scored, and no later step revisits the choice
(see the patent-risk section of AGENTS.md, shape 4).
"""

from __future__ import annotations

import pandas as pd

from openplaces.io.harmonizer import HarmonizeState, _register
from openplaces.io.harmonizer.entity_links import load_unit_lot_pairs


def unit_lots(pairs: pd.DataFrame) -> pd.Series:
    """Lot key per unit key, for units the split saw on exactly one lot.

    Parameters
    ----------
    pairs : pd.DataFrame
        Columns `unit_key` and `lot_key`, one row per pair.

    Returns
    -------
    pd.Series
        `lot_key` indexed by `unit_key`; units on several lots are absent.
    """
    pairs = pairs.dropna().drop_duplicates()
    n_lots = pairs.groupby('unit_key')['lot_key'].transform('nunique')
    single = pairs[n_lots == 1]
    return single.set_index('unit_key')['lot_key']


@_register('derive_parcel_link_key')
def derive_parcel_link_key(
    state: HarmonizeState,
    key_column: str = 'parcel_id_local',
    lot_column: str = 'lot_id_local',
    output_column: str = 'parcel_link_key',
) -> HarmonizeState:
    """Write the key each row joins parcels on.

    Parameters
    ----------
    key_column : str, optional
        The number the record states.
    lot_column : str, optional
        Column written with the lot of a row whose number is a unit's;
        missing elsewhere. An existing value is kept.
    output_column : str, optional
        The join key: *lot_column* where known, else *key_column*.
    """
    spine = state.spine
    if spine is None or key_column not in spine.columns:
        return state
    key = spine[key_column].astype('string')
    lots = unit_lots(load_unit_lot_pairs(state.admin_id))
    if len(lots):
        found = key.map(lots).astype('string')
    else:
        found = pd.Series(pd.NA, index=spine.index, dtype='string')
    if lot_column in spine.columns:
        found = spine[lot_column].astype('string').fillna(found)
    spine[lot_column] = found
    spine[output_column] = found.fillna(key)
    state.spine = spine
    if state.verbose:
        print(
            f'  derive_parcel_link_key: {int(found.notna().sum()):,d} of '
            f'{int(key.notna().sum()):,d} keyed rows name a stacked unit'
        )
    return state
