"""The dominant parcel of a building: largest overlap, and a fixed
rule where two overlaps tie.

Every step that picks one parcel for a building or footprint from an
overlay (the spine's `parcel_id`, the parcel that gives a building its
land value and address, the parcel a point must share with its
building, the dominant piece of a cut outline, the curate stage's
provenance token) ranks the pairs here, so that no two of them can pick
differently.

The largest intersection area wins. A tie is real and common, not a
rounding curiosity: a building lies whole on two parcels where a lot is
drawn inside a larger parcel (an exempt county tract or common area
drawn under a whole subdivision, a small polygon drawn around one
building on a larger tract). Both overlaps are then the building's own
area. Until 2026-11-10 the pick was whichever row came first after an
unstable sort, so any step that reordered the overlay moved buildings
between parcels.

Measured 2026-11-10 on US-FL-LA: 711 buildings on a lot nested in a larger parcel
tie on overlap; the inner lot records an improvement value on most, the
outer exempt tract on none.

A tie goes, in order (maintainer's decision, 2026-11-10):

1. to a parcel with an improvement value above 0 over one without
   (missing or 0): the improved lot is where the roll records the
   building. Geometry alone gets the reverse case wrong, an exempt
   half-lot drawn inside the residential parcel that holds the house.
2. else to the parcel of smaller area: for a nested pair this is the
   inner lot, drawn around the building on purpose; for any other tie
   it is a fixed geometric key.
3. else to the lower parcel id, so the pick never depends on row order.

The rule reads no owner, removes no parcel (the outer parcel keeps its
links; only which one is a building's dominant parcel changes) and
applies no distance or contiguity tolerance beyond the overlap
tolerance below.
"""

from __future__ import annotations

import pandas as pd

__all__ = [
    'OVERLAP_TIE_M2',
    'dominant_overlaps',
    'rank_overlaps',
    'tie_breakers',
]

# Two overlaps within this many m2 of each other are a tie. A building
# lying whole on two parcels intersects both along the same edges, but
# the overlay computes each intersection separately, so the two areas
# differ by float noise (well under a square centimeter on a house);
# exact equality would miss most real ties. 0.1 m2 stays far below any
# real difference in coverage a building could show.
OVERLAP_TIE_M2 = 0.1

# Area columns a prepared or curated reference may carry, in the order
# looked for. Only the order within one reference matters, so the unit
# does not.
_AREA_COLUMNS = (
    'area_ha',
    'area_ha_geometry',
    'area_m2',
    'area_m2_geometry',
    'area_km2',
    'area_km2_geometry',
)
IMPROVEMENT_VALUE_COLUMN = 'improvement_value'


def _unique(series: pd.Series | None) -> pd.Series | None:
    if series is None:
        return None
    return series[~series.index.duplicated(keep='first')]


def tie_breakers(
    ref: pd.DataFrame | None,
) -> tuple[pd.Series | None, pd.Series | None]:
    """The improvement value and area a reference table states per id.

    Parameters
    ----------
    ref : pandas.DataFrame or None
        The reference (parcel) table, indexed by its id: the prepared
        reference of a harmonize overlay or a curated parcel table.

    Returns
    -------
    tuple of (pandas.Series or None, pandas.Series or None)
        `improvement_value` and the first area column found
        (`area_ha`, then the other units), indexed by reference id;
        None for whichever the table does not carry. A geometry column
        is not measured here: every prepared reference carries its
        area already.
    """
    if ref is None:
        return None, None
    improvement = None
    if IMPROVEMENT_VALUE_COLUMN in ref.columns:
        improvement = ref[IMPROVEMENT_VALUE_COLUMN]
    area_col = next((c for c in _AREA_COLUMNS if c in ref.columns), None)
    area = ref[area_col] if area_col is not None else None
    return improvement, area


def rank_overlaps(
    pairs: pd.DataFrame,
    *,
    spine_id_col: str,
    ref_id_col: str = 'parcel_id',
    area_col: str = 'area_intersection_m2',
    improvement_value: pd.Series | None = None,
    ref_area: pd.Series | None = None,
) -> pd.DataFrame:
    """Order each building's overlaps, its dominant parcel first.

    Within each spine id the overlaps within `OVERLAP_TIE_M2` of the
    largest come first, ordered by the tie rule of the module
    (improvement value above 0, then smaller parcel area, then lower
    parcel id); the rest follow by area, largest first, with the same
    keys behind it. The order depends on the values only, never on the
    order of the rows in *pairs*.

    Parameters
    ----------
    pairs : pandas.DataFrame
        One row per (spine id, reference id) overlap, with columns
        *spine_id_col*, *ref_id_col* and *area_col*; other columns are
        kept.
    spine_id_col : str
        The building or footprint id column.
    ref_id_col : str, optional
        The parcel id column (default 'parcel_id').
    area_col : str, optional
        The intersection area column (default 'area_intersection_m2').
    improvement_value : pandas.Series, optional
        Improvement value per parcel id. Without it, step 1 of the rule
        is skipped.
    ref_area : pandas.Series, optional
        Parcel area per parcel id, in any one unit. Without it, step 2
        of the rule is skipped.

    Returns
    -------
    pandas.DataFrame
        *pairs* reordered (spine ids ascending), with its original
        index.
    """
    if len(pairs) == 0:
        return pairs
    spine = pairs[spine_id_col]
    refs = pairs[ref_id_col]
    area = pd.to_numeric(pairs[area_col], errors='coerce').astype(float)
    largest = area.groupby(spine.to_numpy(), dropna=False).transform('max')
    top = (area >= largest - OVERLAP_TIE_M2).to_numpy()
    improvement_value = _unique(improvement_value)
    ref_area = _unique(ref_area)
    if improvement_value is not None:
        value = pd.to_numeric(improvement_value.reindex(refs), errors='coerce')
        unimproved = ~(value > 0).to_numpy()
    else:
        unimproved = False
    if ref_area is not None:
        parcel_area = pd.to_numeric(ref_area.reindex(refs), errors='coerce').to_numpy()
    else:
        parcel_area = 0.0
    keys = pd.DataFrame(
        {
            '_spine': spine.to_numpy(),
            '_rest': ~top,
            # Inside the tie the area no longer orders; outside it the
            # larger overlap still comes first.
            '_area': (-area).where(~top, 0.0).to_numpy(),
            '_unimproved': unimproved,
            '_parcel_area': parcel_area,
            '_ref': refs.to_numpy(),
        }
    )
    order = keys.sort_values(list(keys.columns), na_position='last').index
    return pairs.iloc[order.to_numpy()]


def dominant_overlaps(
    pairs: pd.DataFrame,
    *,
    spine_id_col: str,
    ref_id_col: str = 'parcel_id',
    area_col: str = 'area_intersection_m2',
    improvement_value: pd.Series | None = None,
    ref_area: pd.Series | None = None,
) -> pd.DataFrame:
    """Each building's dominant overlap: one row of *pairs* per spine id.

    The first row of `rank_overlaps` per spine id; see there for the
    parameters. Rows with a missing *ref_id_col* take part like any
    other, so a caller that wants only real parcels drops them first.
    """
    ranked = rank_overlaps(
        pairs,
        spine_id_col=spine_id_col,
        ref_id_col=ref_id_col,
        area_col=area_col,
        improvement_value=improvement_value,
        ref_area=ref_area,
    )
    return ranked[~ranked[spine_id_col].duplicated(keep='first').to_numpy()]
