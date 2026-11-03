"""Why a building's structure value is 0.

A tax roll writes 0 for reasons that say nothing about the structure:
the home is personal property and off the real-property roll, the
building is exempt and the roll carries only the taxable part, the
roll books the value on another building of the lot, or the roll lists
the lot as vacant. Only a shed or garage valued with the main house,
and a structure the roll means to value at nothing, are zeros about
the building itself. `structure_value_zero_reason` says which applies,
so imputation can refill the first kind and learn only from the second
(plans/zero-structure-value-reasons.md, parts A2 and B; maintainer,
2026-10-31).

The reasons, in the order they are tested (the first that holds
wins):

- `secondary`: a shed or garage (occupancy class);
- `personal_property`: a manufactured home the roll does not list
  (`building_imputation.manufactured_home_listing`, the same
  definition imputation uses), or a dwelling on a manufactured-home
  lot, park or community, or on a lot with a manufactured home;
- `exempt`: a public or institutional occupancy, an institutional land
  use, an exempt use group, or a recorded exempt improvement value;
- `included_elsewhere`: a dwelling whose lot's roll value another
  building holds (`building_imputation.lot_value_elsewhere`);
- `secondary` again: any other building ranked secondary on its lot;
- `not_on_roll`: a house on a lot the roll calls vacant that a second
  source also shows (an NSI point or the roll's own building record).
  Both sources can be older than the roll, and most such houses that
  can be dated were removed since: a teardown or a storm loss, not a
  building the roll missed (plans/zero-structure-value-reasons.md,
  research of 2026-11-02);
- `no_value`: none of these; the roll's statement about the building.

There is no `unassessed` reason: a 0 on a lot whose land is also 0, or
whose total contradicts it, no longer reaches `structure_value` as 0
(`land_value.py`), and a unit the roll leaves unvalued takes its values
from NSI (`select_value_source_by_admin_unit`).

The occupancy vote must never read this column: `personal_property`
is derived from the occupancy class, and a vote reading it would
confirm itself.

Measured 2026-11-02 on US-TX-HAR: 56,655 zeros, 12,296 no_value, 15,944 exempt.
Measured 2026-11-02 on US-NC-BRU: 12,649 zeros, 10,935 personal_property.
Measured 2026-11-02 on US-FL-LE: 77% of not_on_roll lots lost a prior value.
"""

from __future__ import annotations

import pandas as pd

from openplaces.io.curator import CurateState, _register

#: The reasons, in the order they are tested.
REASONS = (
    'secondary',
    'personal_property',
    'exempt',
    'included_elsewhere',
    'not_on_roll',
    'no_value',
)


def _column(frame: pd.DataFrame, column: str, dtype: str = 'string') -> pd.Series:
    if column not in frame.columns:
        missing = pd.NA if dtype == 'string' else float('nan')
        return pd.Series(missing, index=frame.index, dtype=dtype)
    if dtype == 'string':
        return frame[column].astype('string')
    return pd.to_numeric(frame[column], errors='coerce')


def _any_part_in(values: pd.Series, names) -> pd.Series:
    """True where a ' | '-joined part of *values* is one of *names*."""
    wanted = {str(n).strip().lower() for n in names}
    if not wanted:
        return pd.Series(False, index=values.index)
    # Positions, not labels: the curated index need not be unique.
    parts = (
        values.reset_index(drop=True)
        .str.lower()
        .str.split('|', regex=False)
        .explode()
        .str.strip()
    )
    hit = parts.isin(wanted).fillna(False).groupby(level=0).any()
    return pd.Series(hit.to_numpy(dtype=bool), index=values.index)


def structure_value_zero_reasons(
    curated: pd.DataFrame,
    dwelling_classes=('Single-Family', 'Low-Rise Multi-Family', 'Multi-Family'),
    secondary_classes=('Secondary', 'Garage'),
    exempt_classes=(),
    exempt_land_use_classes=('Institutional',),
    exempt_use_groups=(),
    manufactured_home_land_use_classes=('Manufactured Home', 'Manufactured Home Park'),
    vacant_land_use_classes=('Vacant', 'Mostly Vacant'),
    not_on_roll_min_m2: float = 50.0,
) -> pd.Series:
    """Name the reason for each structure value of 0.

    Parameters
    ----------
    curated : pandas.DataFrame
        The building curate, after `structure_value` and
        `occupancy_type` are final.
    dwelling_classes : sequence of str
        Occupancy classes of site-built dwellings.
    secondary_classes : sequence of str
        Occupancy classes of sheds and garages.
    exempt_classes : sequence of str
        Public and institutional occupancy classes.
    exempt_land_use_classes : sequence of str
        Parcel land-use classes (`land_use_class_parcel`) of exempt
        owners.
    exempt_use_groups : sequence of str
        Use groups (`use_group_combined_parcel`, matched per
        ' | '-joined part, ignoring case) the crosswalks give exempt
        parcels.
    manufactured_home_land_use_classes : sequence of str
        Parcel land-use classes of manufactured-home lots and parks.
    vacant_land_use_classes : sequence of str
        Parcel land-use classes of a lot the roll calls vacant.
    not_on_roll_min_m2 : float
        Outline area from which a house on a vacant lot counts as a
        house; a smaller one is a shed standing alone.

    Returns
    -------
    pandas.Series
        One of `REASONS` where `structure_value` is 0, else missing.
    """
    # Imported here so the curate package does not load the model
    # stack (sklearn) for a step that only classifies.
    from openplaces.io.building_imputation import (
        lot_value_elsewhere,
        manufactured_home_listing,
    )

    out = pd.Series(pd.NA, index=curated.index, dtype='string')
    if 'structure_value' not in curated.columns:
        return out
    zero = _column(curated, 'structure_value', 'float').eq(0).fillna(False)
    if not zero.any():
        return out
    occupancy = _column(curated, 'occupancy_type')
    land_use = _column(curated, 'land_use_class_parcel')
    dwelling = occupancy.isin(dwelling_classes).fillna(False)

    homes = _column(curated, 'n_manufactured_homes_per_parcel', 'float')
    community = _column(curated, 'manufactured_home_community').str.lower()
    unlisted = manufactured_home_listing(curated).eq('unlisted').fillna(False)
    on_home_lot = (
        land_use.isin(manufactured_home_land_use_classes).fillna(False)
        | community.eq('true').fillna(False)
        | homes.ge(1).fillna(False)
    )
    exempt_value = _column(curated, 'improvement_value_exempt', 'float')
    use_group = _column(curated, 'use_group_combined_parcel')
    nsi_point = _column(curated, 'building_id_nsi').notna()
    roll_record = (
        _column(curated, 'year_built_parcel', 'float').notna()
        | _column(curated, 'living_area_sqft_parcel', 'float').notna()
    )
    area = _column(curated, 'area_m2', 'float')
    priority = _column(curated, 'priority_on_parcel')

    tests = [
        ('secondary', occupancy.isin(secondary_classes).fillna(False)),
        ('personal_property', unlisted | (dwelling & on_home_lot)),
        (
            'exempt',
            occupancy.isin(exempt_classes).fillna(False)
            | land_use.isin(exempt_land_use_classes).fillna(False)
            | _any_part_in(use_group, exempt_use_groups)
            | exempt_value.gt(0).fillna(False),
        ),
        ('included_elsewhere', lot_value_elsewhere(curated)),
        ('secondary', priority.eq('secondary').fillna(False)),
        (
            'not_on_roll',
            dwelling
            & land_use.isin(vacant_land_use_classes).fillna(False)
            & area.ge(not_on_roll_min_m2).fillna(False)
            & (nsi_point | roll_record),
        ),
    ]
    left = zero.copy()
    for reason, holds in tests:
        hit = left & holds.astype(bool)
        out[hit] = reason
        left &= ~hit
    out[left] = 'no_value'
    return out


@_register('derive_structure_value_zero_reason', phase='infer')
def derive_structure_value_zero_reason(state: CurateState, **params) -> CurateState:
    """Write `structure_value_zero_reason` on buildings valued at 0.

    Runs after the last step that sets `structure_value` or
    `occupancy_type`; changes neither. See
    `structure_value_zero_reasons` for the reasons and *params*.

    Parameters
    ----------
    state : CurateState
        The curate state.
    **params
        Passed to `structure_value_zero_reasons`.
    """
    reasons = structure_value_zero_reasons(state.curated, **params)
    state.curated['structure_value_zero_reason'] = reasons
    counts = reasons.value_counts()
    summary = ', '.join(f'{r} {int(counts.get(r, 0)):,d}' for r in REASONS)
    print(f'  derive_structure_value_zero_reason: {summary}')
    return state
