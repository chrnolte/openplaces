"""Standing check: is a lot's roll value spread over its buildings?

A tax roll states one improvement value per lot, and the building
curate places it on the lot's buildings. Two ways this goes wrong were
found by hand, one county at a time
(plans/lot-value-sharing-among-buildings.md):

- **unshared lots**: a lot with several large buildings, such as an
  apartment complex, whose roll value sits on a minority of their floor
  area, so the valued buildings carry the complex and the rest look
  empty;
- **unvalued buildings beside a valued one**: a building with an
  occupancy of its own and no roll value on a lot where another
  building holds the roll value. A model that fills buildings without
  a value counts the lot's structure value twice.

This step counts both for every unit on every run and never changes a
value. It prints one line and writes the counts to the cache, so a
regression shows up in the run that caused it rather than in a county
someone happens to inspect.

Measured 2026-11-01 on US: 29,000 unshared lots, 55,000 unvalued beside valued.
Measured 2026-11-01 on US-TX-HAR: 1,685 unshared lots, 15,258 unvalued beside valued.
"""

from __future__ import annotations

import pandas as pd

from openplaces.io.curator import CurateState, _register

#: File the counts are written to, in the unit's building cache folder.
REPORT_FILENAME = 'lot-value-sharing.csv'


def lot_value_sharing_counts(
    curated: pd.DataFrame,
    lot_column: str = 'parcel_id',
    value_column: str = 'structure_value',
    source_column: str = 'structure_value_source',
    lot_value_column: str = 'improvement_value_parcel',
    occupancy_column: str = 'occupancy_type',
    area_column: str = 'area_m2',
    roll_source_prefix: str = 'parcel',
    exclude_classes: tuple[str, ...] = ('Secondary',),
    exempt_classes: tuple[str, ...] = ('Manufactured Home',),
    min_m2: float = 100.0,
) -> dict:
    """Count lots whose roll value is not shared among their buildings.

    A building *holds a roll value* when its value is above zero and its
    source names the parcel roll (starts with *roll_source_prefix*); a
    value from any other source (NSI, a model) is not the roll's. A
    building *has an occupancy of its own* when its occupancy is set and
    not in *exclude_classes*.

    Parameters
    ----------
    curated : pandas.DataFrame
        The building curate, one row per building.
    lot_column : str
        Column naming the building's lot.
    value_column, source_column : str
        The building's value and its provenance.
    lot_value_column : str
        The lot's improvement value as the roll states it.
    occupancy_column, area_column : str
        Occupancy class and outline area (m2).
    roll_source_prefix : str
        Start of a source label that names the parcel roll.
    exclude_classes : tuple of str
        Occupancies that are not a building of their own (sheds, garages).
    exempt_classes : tuple of str
        Occupancies a roll may rightly leave unvalued beside a valued
        building (a manufactured home listed as personal property), left
        out of the unvalued count.
    min_m2 : float
        Outline area from which a building counts as large.

    Returns
    -------
    dict
        `n_buildings`, `n_lots_multi` (lots with 2 or more buildings),
        `n_lots_unshared`, `unshared_unvalued_m2` (outline area of the
        large buildings without a roll value on those lots),
        `n_unvalued_beside_valued` and `n_lots_unvalued_beside_valued`.
        All zero where a needed column is missing.
    """
    counts = dict.fromkeys(
        (
            'n_buildings',
            'n_lots_multi',
            'n_lots_unshared',
            'unshared_unvalued_m2',
            'n_unvalued_beside_valued',
            'n_lots_unvalued_beside_valued',
        ),
        0,
    )
    needed = (lot_column, value_column, source_column, occupancy_column, area_column)
    if curated is None or any(c not in curated.columns for c in needed):
        return counts
    frame = curated[curated[lot_column].notna()]
    counts['n_buildings'] = len(frame)
    if frame.empty:
        return counts
    lot = frame[lot_column].astype('string')
    occupancy = frame[occupancy_column].astype('string')
    area = pd.to_numeric(frame[area_column], errors='coerce').fillna(0.0)
    value = pd.to_numeric(frame[value_column], errors='coerce')
    from_roll = (
        frame[source_column].astype('string').str.startswith(roll_source_prefix)
    ).fillna(False)
    holds_roll = value.gt(0) & from_roll
    own = occupancy.notna() & ~occupancy.isin(exclude_classes)
    own = own.fillna(False)
    large = own & area.ge(min_m2)
    lot_holds_roll = holds_roll.groupby(lot).transform('any')

    counts['n_lots_multi'] = int(lot.value_counts().gt(1).sum())

    # Unshared: the valued large buildings cover under half of the
    # lot's large-building outline area. Half, not "any unvalued":
    # a complex whose clubhouse goes unvalued is shared well enough.
    if lot_value_column in frame.columns:
        lot_value = pd.to_numeric(frame[lot_value_column], errors='coerce')
        lot_value_positive = lot_value.groupby(lot).transform('max').gt(0)
    else:
        lot_value_positive = lot_holds_roll
    large_area = area.where(large, 0.0).groupby(lot).transform('sum')
    valued_area = area.where(large & holds_roll, 0.0).groupby(lot).transform('sum')
    n_large = large.groupby(lot).transform('sum')
    unshared = (
        n_large.ge(2)
        & lot_value_positive
        & lot_holds_roll
        & valued_area.lt(0.5 * large_area)
    )
    counts['n_lots_unshared'] = int(lot[unshared].nunique())
    counts['unshared_unvalued_m2'] = round(
        float(area.where(unshared & large & ~holds_roll, 0.0).sum()), 1
    )

    unvalued = (
        own
        & ~occupancy.isin(exempt_classes).fillna(False)
        & ~holds_roll
        & lot_holds_roll
    )
    counts['n_unvalued_beside_valued'] = int(unvalued.sum())
    counts['n_lots_unvalued_beside_valued'] = int(lot[unvalued].nunique())
    return counts


@_register('report_lot_value_sharing', phase='infer')
def report_lot_value_sharing(state: CurateState, **params) -> CurateState:
    """Report how a lot's roll value is spread over its buildings.

    Counts what `lot_value_sharing_counts` describes, prints them on one
    line and writes them to the unit's building cache, in a file ending
    in `_lot-value-sharing.csv`.
    The curated table is not changed. Runs after occupancy is resolved,
    since both counts read it.

    Parameters
    ----------
    state : CurateState
        The curate state.
    **params
        Passed to `lot_value_sharing_counts` (column names, classes,
        `min_m2`).
    """
    from openplaces.path import cache_path

    counts = lot_value_sharing_counts(state.curated, **params)
    state.metadata['lot_value_sharing'] = counts
    print(
        f'  report_lot_value_sharing: {counts["n_lots_unshared"]:,d} lots '
        f'with large buildings left unvalued '
        f'({counts["unshared_unvalued_m2"]:,.0f} m2), '
        f'{counts["n_unvalued_beside_valued"]:,d} unvalued buildings '
        f'beside a valued one on {counts["n_lots_unvalued_beside_valued"]:,d} '
        f'of {counts["n_lots_multi"]:,d} lots with several buildings'
    )
    path = cache_path(
        state.admin_id, state.recipe.get('entity'), filename=REPORT_FILENAME
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{'admin_id': str(state.admin_id), **counts}]).to_csv(
        path, index=False
    )
    return state


def collect_lot_value_sharing(paths) -> pd.DataFrame:
    """Stack the per-unit reports at *paths* into one table.

    Parameters
    ----------
    paths : iterable of path-like
        `lot-value-sharing.csv` files written by `report_lot_value_sharing`.
    """
    frames = [pd.read_csv(p) for p in paths]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
