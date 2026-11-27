"""Estimate and decide a building's number of stories.

`decide_stories` is the one step that sets the canonical story count of
a building; `derive_stories_from_height` converts a measured height for
a table that only needs the conversion (the parcel curate's lot figure).
Both convert at the same floor height and round the same way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from openplaces.io.curator import CurateState, _register

#: Meters of measured height per story. The height includes the roof, so
#: a pitched one-story house reads as two stories at the 3.05 m of a
#: floor-to-floor rule of thumb.
FLOOR_HEIGHT_M = 3.5

#: Square feet to square meters.
_SQFT_TO_M2 = 0.09290304


def round_half_up(values: pd.Series) -> pd.Series:
    """Round to whole numbers, sending a half up (1.5 to 2, 2.5 to 3).

    pandas rounds half to even (2.5 to 2, 3.5 to 4), which is harder to
    explain for a count and treats a half differently depending on the
    whole number below it.
    """
    return np.floor(pd.to_numeric(values, errors='coerce').astype(float) + 0.5)


def stories_from_height(height: pd.Series, floor_height_m: float) -> pd.Series:
    """Story count from a measured height: height over floor height,
    rounded half up, at least one. A missing or non-positive height
    gives a missing count, never a fabricated minimum."""
    height = pd.to_numeric(height, errors='coerce').astype(float)
    stories = round_half_up(height / floor_height_m).clip(lower=1)
    return stories.where(height > 0)


def _first_in_range(
    curated, columns: list[str], low: float, high: float
) -> tuple[pd.Series, dict[str, pd.Series]]:
    """The first value within [low, high] across *columns*, per row, and
    for each column the rows whose value it supplied."""
    sub = curated[columns].apply(pd.to_numeric, errors='coerce').astype(float)
    values = sub.to_numpy()
    usable = (values >= low) & (values <= high)
    has_value = usable.any(axis=1)
    first = usable.argmax(axis=1)
    chosen = values[np.arange(len(values)), first]
    value = pd.Series(np.where(has_value, chosen, np.nan), index=curated.index)
    supplied = {
        column: pd.Series(has_value & (first == i), index=curated.index)
        for i, column in enumerate(columns)
    }
    return value, supplied


def _roll_fills(
    state: CurateState,
    current: pd.Series,
    roll_column: str,
    floor_area_column: str,
    area_column: str,
    group_column: str,
    priority_column: str,
    min_share: float,
    min_rows: int,
    admin_level: int,
    max_roll_stories: float,
) -> tuple[pd.Series, pd.Series]:
    """The roll's count as the roll states it, and the rows where it
    fills a missing *current*.

    The roll is adopted per admin unit, never per row: a unit qualifies
    when, on at least *min_rows* rows that are the only non-secondary
    building of their lot and where the roll's count differs from
    *current*, the roll brings outline area times stories closer to the
    roll's floor area on at least *min_share* of them.

    The test is partly self-referential (roll floor area against roll
    stories), which is why the bar is high: on 2026-09-17 the roll was
    closer on 42.9% to 83.5% of disagreeing rows in four regions, and
    on 58.7% against independent permit floor areas where those exist.
    """
    from openplaces.io.curator.reconcilers import _resolve_admin_group

    curated = state.curated
    # Not rounded: the fit test scores the count a fill would write.
    roll = pd.to_numeric(curated[roll_column], errors='coerce').astype(float)
    roll = roll.where((roll >= 1) & (roll <= max_roll_stories))
    # The incumbent can be a mean over several points (1.33): it only
    # disagrees with the roll if the count step 2 delivers (half up)
    # differs, but the fit is scored on the value itself.
    current_count = round_half_up(current)
    floor_m2 = pd.to_numeric(curated[floor_area_column], errors='coerce') * _SQFT_TO_M2
    area = pd.to_numeric(curated[area_column], errors='coerce')

    if priority_column in curated.columns:
        counts = curated[priority_column].astype('string') != 'secondary'
    else:
        counts = pd.Series(True, index=curated.index)
    n_counted = counts.groupby(curated[group_column]).transform('sum')
    sole = counts & (n_counted == 1) & curated[group_column].notna()

    eligible = sole & roll.notna()
    testable = eligible & current.notna() & (roll != current_count)
    testable &= (floor_m2 > 0) & (area > 0) & (current > 0)
    # Rows outside `testable` can hold zeros; they are masked below, so
    # their log warnings carry no information.
    with np.errstate(divide='ignore', invalid='ignore'):
        fit_roll = np.abs(np.log(floor_m2 / (area * roll)))
        fit_current = np.abs(np.log(floor_m2 / (area * current)))
    roll_closer = (fit_roll < fit_current) & testable

    chunk = pd.Series(str(state.admin_id), index=curated.index)
    group = _resolve_admin_group(state, curated, admin_level)
    units = chunk if group is None else group.fillna(str(state.admin_id))
    n_test = testable.groupby(units).transform('sum')
    n_closer = roll_closer.groupby(units).transform('sum')
    adopt_unit = (n_test >= min_rows) & (n_closer >= min_share * n_test)

    if state.verbose:
        tested = units[testable].value_counts()
        won = units[roll_closer].value_counts().reindex(tested.index).fillna(0)
        share = (won / tested).round(3).to_dict()
        print(f'  decide_stories: roll closer share by unit {share}.')
    return roll, eligible & adopt_unit & current.isna()


@_register(
    'derive_stories_from_height',
    phase='infer',
    reads=['{height_column}'],
    writes=['{column}'],
)
def derive_stories_from_height(
    state: CurateState,
    column: str = 'n_stories_footprint_fema',
    height_column: str = 'height_footprint_fema',
    floor_height_m: float = FLOOR_HEIGHT_M,
) -> CurateState:
    """Derive a story count from a measured building height.

    Height over *floor_height_m*, rounded half up, at least one story. A
    missing or non-positive height yields a missing story count rather
    than a fabricated minimum. The building curate does this inside
    `decide_stories`; this step serves a table that needs the converted
    count only.

    Parameters
    ----------
    state : CurateState
        The curation state with the target GeoDataFrame in state.curated.
    column : str, optional
        Output column name for the derived story count.
    height_column : str, optional
        Source column holding measured building height (meters). No-op if
        absent from the curated table.
    floor_height_m : float, optional
        Assumed height per story, in meters (default 3.5, see
        `decide_stories` for why).
    """
    curated = state.curated
    if height_column not in curated.columns:
        return state
    curated[column] = stories_from_height(curated[height_column], floor_height_m)
    state.curated = curated
    return state


@_register(
    'decide_stories',
    phase='infer',
    reads=[
        '{evidence}',
        '{height_column}',
        '{roll_column}',
        '{floor_area_column}',
        '{area_column}',
        '{group_column}',
        '{priority_column}',
    ],
    writes=['{output}', '{output}_source', '{height_stories_column}'],
    decides=['{output}'],
)
def decide_stories(
    state: CurateState,
    evidence: list[str],
    output: str = 'n_stories',
    height_column: str | None = 'height_footprint_fema',
    height_stories_column: str = 'n_stories_footprint_fema',
    floor_height_m: float = FLOOR_HEIGHT_M,
    valid_range: list[float] | tuple[float, float] = (1, 110),
    roll_column: str | None = 'n_stories_parcel',
    floor_area_column: str = 'living_area_sqft_parcel',
    area_column: str = 'area_m2',
    group_column: str = 'parcel_id',
    priority_column: str = 'priority_on_parcel',
    min_share: float = 0.7,
    min_rows: int = 100,
    admin_level: int = 3,
    max_roll_stories: float = 60,
    roll_token: str = 'parcel',
) -> CurateState:
    """Decide a building's number of stories from ordered evidence.

    The one step that sets *output*. In order:

    1. Convert *height_column* to a count in *height_stories_column*
       (height over *floor_height_m*, half up, at least one), so a
       measured height can be listed as evidence.
    2. Take the first value of *evidence* that lies within
       *valid_range*. A value outside it (a modeled count of 219 floors,
       a sentinel) falls through to the next column. No weights, no
       blending: the first available value wins.
    3. Round the value of step 2 half up (see Rounding below).
    4. Where nothing qualified, take the roll's count as the roll
       states it (*roll_column*, 1 to *max_roll_stories*, a 1.5 stays
       1.5) in admin units where the roll's counts fit the roll's own
       floor area: on at least *min_rows* buildings that are their
       lot's only non-secondary building and where the roll and step 3
       disagree, outline area times the roll's count must come closer
       to the roll's floor area on at least *min_share* of them. A
       count from step 2 is never overwritten.
    5. Write *output* as a float: whole numbers, except a roll's
       fraction.

    *output*_source names the column that supplied each value by its
    provenance suffix (`nsi`, `fema`, `cheer`), or *roll_token*
    for a roll fill; a row left empty has no source.

    The roll's count is not a lane of step 2 because rolls define a
    story differently (one counts a story NSI does not, another codes
    something other than a count), so whether it is used is decided per
    unit. Overwriting a present count with it
    was measured on 2026-09-17 and mostly lowered counts with no
    independent floor area to say which was right.

    Why 3.5 m per story, not 3.05: the measured height includes the
    roof. Lidar counts at 3.5 m agree better with roll counts in every
    state that has both, scored on 1,132,920 buildings that are their
    lot's only one:
    Measured 2026-11-14 on US: lidar equals roll count 80.9% at 3.5 m, 75.9% at 3.05 m.

    Rounding, by evidence:

    - The roll's count is used as the roll states it, in the gap fill
      and in its fit test alike, so a delivered count can be 1.5 or
      2.75. A roll's half story is a real half (a floor under the roof,
      about half a floor of area), and the assessor states it
      (maintainer, 2026-11-15).
    - A lidar count is whole by its conversion (step 1), and CHEER's
      counts are whole.
    - An NSI value is a mean over the NSI points linked to a building,
      and it is fractional where those points disagree: an averaging
      artifact, not a half story. Until NSI arrives as its tallest
      point (a separate change), it is rounded half up here, an
      interim rule. Half up, not half to even (the pandas default),
      because half to even sends 2.5 to 2 and 3.5 to 4, which a
      reader cannot predict.

    Measured 2026-11-15 on US: rounding decides 143,299 NSI means, 177 lidar heights.
    Measured 2026-11-15 on US: roll fills 25,349 counts, 5,753 of them a fraction.

    Parameters
    ----------
    state : CurateState
        The curation state with the target GeoDataFrame in state.curated.
    evidence : list of str
        Evidence columns in order of preference. Absent columns are
        skipped; *height_stories_column* may be listed to admit the
        converted height.
    output : str, optional
        Canonical story column to write (default `n_stories`).
    height_column : str or None, optional
        Measured height in meters. None, or a column absent from the
        table, skips the conversion.
    height_stories_column : str, optional
        Column the converted height is written to.
    floor_height_m : float, optional
        Meters of height per story (default 3.5).
    valid_range : sequence of two floats, optional
        Inclusive bounds an evidence value must lie within (default 1 to
        110; the tallest US building has 108 floors).
    roll_column : str or None, optional
        The roll's story count. None, or an absent column (or an absent
        *floor_area_column*, *area_column* or *group_column*), skips the
        gap fill.
    floor_area_column : str, optional
        The roll's floor area in square feet.
    area_column : str, optional
        Outline area in square meters.
    group_column : str, optional
        Lot id grouping buildings for the sole-building condition.
    priority_column : str, optional
        A building's role on its lot; secondary rows never count.
    min_share : float, optional
        Share of disagreeing rows the roll must fit better (default 0.7).
    min_rows : int, optional
        Disagreeing rows a unit needs before it can adopt (default 100).
    admin_level : int, optional
        Level of the unit the roll decision is made for (default 3); at
        or above the processing level, the whole chunk is one unit.
    max_roll_stories : float, optional
        Largest plausible roll count (default 60); larger values and
        zeros are treated as missing. *valid_range* does not apply to
        the roll, so this is the only cap on it: a sentinel such as 99
        would pass without it. No recipe sets it, and no roll on the 13
        units of plans/rule-thresholds-inventory.md states more than 60
        in a filled county:

        Measured 2026-11-24 on US-PA-HU: no cap, 0 of 54,297 rows change.

        To settle: a national count of roll values above 60.
    roll_token : str, optional
        Source token recorded on roll fills (default parcel).
    """
    from openplaces.io.curator.provenance import record_source, source_column
    from openplaces.io.curator.reconcilers import _source_token

    curated = state.curated
    if height_column and height_column in curated.columns:
        curated[height_stories_column] = stories_from_height(
            curated[height_column], floor_height_m
        )

    columns = [c for c in evidence if c in curated.columns]
    low, high = float(valid_range[0]), float(valid_range[1])
    if columns:
        value, supplied = _first_in_range(curated, columns, low, high)
    else:
        value, supplied = pd.Series(np.nan, index=curated.index), {}

    # Step 3: lidar and CHEER are already whole; this rounds NSI's mean.
    stated = value
    value = round_half_up(value)

    gap_inputs = [roll_column, floor_area_column, area_column, group_column]
    filled = pd.Series(False, index=curated.index)
    if roll_column and all(c in curated.columns for c in gap_inputs):
        roll, filled = _roll_fills(
            state,
            stated,
            roll_column,
            floor_area_column,
            area_column,
            group_column,
            priority_column,
            min_share,
            min_rows,
            admin_level,
            max_roll_stories,
        )
        value = value.where(~filled, roll)
    elif state.verbose:
        print('  decide_stories: roll inputs missing; no gap fill.')

    curated[output] = value.astype(float)
    side = source_column(output)
    if side in curated.columns:
        curated[side] = curated[side].astype(object)
        curated.loc[curated[output].isna(), side] = pd.NA
    for column, rows in supplied.items():
        if rows.any():
            record_source(curated, output, rows, _source_token(column))
    if filled.any():
        record_source(curated, output, filled, roll_token)
    state.curated = curated

    if state.verbose:
        counts = curated[side].value_counts().to_dict() if side in curated else {}
        print(f'  decide_stories: {output} by source {counts}.')
    return state
