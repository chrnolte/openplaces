"""Decide a building's dwelling count from its preliminary count and
its final class (plans/occupancy-and-dwellings-one-decider.md, O2 to
O4)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from openplaces.io.curator import CurateState, _register

#: Tokens of the counts this step sets from the class, not from a
#: source. Each is written with the imputed marker.
SECONDARY_TOKEN = 'secondary'
SINGLE_FAMILY_TOKEN = 'single_family'


def _has_token(tokens: pd.Series, token: str) -> pd.Series:
    """Whether each provenance token has *token* as a whole part."""
    text = tokens.astype('string').fillna('')
    return text.map(lambda value: token in value.split('+')).astype(bool)


def _below_lot_largest(
    curated: pd.DataFrame, lot_column: str, area_column: str
) -> pd.Series:
    """Rows smaller than the largest building on their lot.

    A row without a lot or an area is never below: the rule that reads
    this has nothing to compare it with. Rows tied for the largest area
    are all the largest.
    """
    if lot_column not in curated.columns or area_column not in curated.columns:
        return pd.Series(False, index=curated.index)
    area = pd.to_numeric(curated[area_column], errors='coerce')
    lot = curated[lot_column].astype(object)
    largest = area.groupby(lot, dropna=True).transform('max')
    return (area < largest).fillna(False).astype(bool)


@_register(
    'decide_dwellings',
    phase='infer',
    reads=[
        '{preliminary}',
        '{preliminary}_source',
        '{class_column}',
        '{lot_column}',
        '{area_column}',
        '{stated_column}',
        '{lot_class_column}',
    ],
    writes=['{output}', '{output}_source'],
    decides=['{output}'],
)
def decide_dwellings(
    state: CurateState,
    preliminary: str = 'n_dwellings_preliminary',
    output: str = 'n_dwellings',
    class_column: str = 'occupancy_type',
    lot_column: str = 'parcel_id',
    area_column: str = 'area_m2',
    stated_column: str = 'n_dwellings_parcel_whole',
    lot_class_column: str = 'land_use_class_parcel',
    stated_zero_classes: list[str] | tuple[str, ...] = (),
    secondary_class: str | None = 'Secondary',
    single_family_class: str | None = 'Single-Family',
    roll_token: str = 'parcel',
) -> CurateState:
    """Decide each building's dwelling count, after its class.

    The one step that sets *output* for buildings. It starts from
    *preliminary*, the count the evidence lanes chose before any class
    existed (the roll's apportioned share, then Overture's address
    points, then NSI, then a class lookup), and applies three rules, the
    first that holds deciding:

    1. A stated zero holds. Where the lot's own count (*stated_column*)
       is 0 and the lot's class is one of *stated_zero_classes*, the
       building has 0 dwellings, credited to *roll_token*: it is the
       roll's value. On any other lot a stated 0 stays what it was
       before, a statement the lanes pass over.
    2. A Secondary building that is not the largest on its lot (by
       *area_column* within *lot_column*) has 0 dwellings, marked
       imputed. The largest building of a lot keeps its preliminary
       count whatever its class.
    3. A Single-Family building whose preliminary count is the roll's
       share and rounds to 2 or more has 1 dwelling, marked imputed.
       The roll's share stays in its own column as evidence.

    Every other row keeps its preliminary count and its token.

    Parameters
    ----------
    preliminary : str, optional
        The count before the class (default `n_dwellings_preliminary`)
        and, with `_source`, its token.
    output : str, optional
        The decided count (default `n_dwellings`).
    class_column : str, optional
        The building's final class (default `occupancy_type`).
    lot_column, area_column : str, optional
        The building's lot and its outline area, for rule 2.
    stated_column : str, optional
        The lot's own count as the curated parcel states it, undivided
        (default `n_dwellings_parcel_whole`). Rule 1 is skipped where it
        is absent.
    lot_class_column : str, optional
        The lot's land-use class (default `land_use_class_parcel`).
    stated_zero_classes : list of str, optional
        Lot classes on which the roll's 0 dwellings holds. Empty (the
        default) switches rule 1 off; the recipe names them.
    secondary_class, single_family_class : str or None, optional
        The class names rules 2 and 3 read; None switches a rule off.
    roll_token : str, optional
        The token of the roll's lane (default `parcel`).
    """
    from openplaces.io.curator.provenance import (
        mark_imputed,
        record_source,
        source_column,
    )

    curated = state.curated
    index = curated.index
    if preliminary in curated.columns:
        counts = pd.to_numeric(curated[preliminary], errors='coerce').astype(float)
    else:
        counts = pd.Series(np.nan, index=index, dtype=float)
    prelim_side = source_column(preliminary)
    tokens = (
        curated[prelim_side].astype(object)
        if prelim_side in curated.columns
        else pd.Series(pd.NA, index=index, dtype=object)
    )
    classes = (
        curated[class_column].astype(object)
        if class_column in curated.columns
        else pd.Series(pd.NA, index=index, dtype=object)
    )

    # Rule 1. The lot's own statement, never the building's share: the
    # share is 0 both where the roll states 0 and where it states
    # nothing, so only the lot's count tells the two apart.
    stated_zero = pd.Series(False, index=index)
    if (
        stated_zero_classes
        and stated_column in curated.columns
        and lot_class_column in curated.columns
    ):
        stated = pd.to_numeric(curated[stated_column], errors='coerce')
        lot_class = curated[lot_class_column].astype(object)
        stated_zero = (stated.eq(0) & lot_class.isin(set(stated_zero_classes))).fillna(
            False
        )
    stated_zero = stated_zero.astype(bool)

    # Rule 2. Only the lot's other buildings: which building of a lot is
    # the house when its largest is Secondary is open (question 5 of
    # the plan), so the largest keeps today's count.
    secondary = pd.Series(False, index=index)
    if secondary_class is not None:
        secondary = classes.eq(secondary_class).fillna(False).astype(bool)
        secondary &= _below_lot_largest(curated, lot_column, area_column)
        secondary &= ~stated_zero

    # Rule 3. A share that ships as 2 or more: cast_integers rounds
    # half to even, as numpy's round does, so the test is the shipped
    # value, not the raw share.
    single = pd.Series(False, index=index)
    if single_family_class is not None:
        single = (
            classes.eq(single_family_class).fillna(False).astype(bool)
            & _has_token(tokens, roll_token)
            & pd.Series(np.round(counts.to_numpy()) >= 2, index=index)
            & ~stated_zero
            & ~secondary
        )

    decided = counts.copy()
    decided[stated_zero | secondary] = 0.0
    decided[single] = 1.0
    curated[output] = decided
    side = source_column(output)
    curated[side] = tokens.copy()
    record_source(curated, output, stated_zero, roll_token)
    record_source(curated, output, secondary, mark_imputed(SECONDARY_TOKEN))
    record_source(curated, output, single, mark_imputed(SINGLE_FAMILY_TOKEN))
    state.curated = curated

    if state.verbose:
        print(
            f'  decide_dwellings: {int(stated_zero.sum()):,} stated zeros held, '
            f'{int(secondary.sum()):,} secondary buildings set to 0, '
            f'{int(single.sum()):,} single-family buildings set to 1.'
        )
    return state
