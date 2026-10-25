"""Registered curation step giving a plat's lots their share of its common land.

A townhouse lot drawn around its house is a few hundred square meters,
while its plat's common land (the green between the rows, the parking
courts) is recorded as a separate account with no value: the roll
counts its worth in the lots' land values. Divided by the lot's own
area, the lot's land reads as the dearest land in the neighborhood.
This step gives each lot the common area its plat assigns it, as an
area, so that a reader can divide the lot's recorded land value by the
lot plus its share. No value moves: the lots keep their recorded land
values and the common area keeps none (maintainer's decisions 96 to
99a, 2026-10-24; plans/findings/suitland-condo-common-land.md).
"""

from __future__ import annotations

import warnings

import pandas as pd

from openplaces.io.curator import CurateState, _register

#: Columns the step writes; every other column is left as it was.
SHARE_COLUMN = 'common_land_share_ha'
LOTS_AREA_COLUMN = 'common_land_lots_area_ha'
LOT_COUNT_COLUMN = 'n_lots_per_plat'
PLAT_ID_COLUMN = 'common_land_plat_id'


def apportion_common_land_area(
    links: pd.DataFrame, area: pd.Series
) -> tuple[pd.Series, pd.DataFrame]:
    """Divide each common area among its linked lots in proportion to lot area.

    Equal per unit of area (maintainer's decision 98, 2026-10-24): a
    lot of twice the area receives twice the common land. A lot linked
    to several common areas of its plat receives a share of each.

    Parameters
    ----------
    links : pandas.DataFrame
        Plat link rows with `parcel_id` (the lot) and
        `common_area_parcel_id`.
    area : pandas.Series
        Area in hectares, indexed by parcel id (as strings).

    Returns
    -------
    shares : pandas.Series
        Common land per lot in hectares, indexed by the lot's id.
    commons : pandas.DataFrame
        Per common area, indexed by its id: `n_lots_per_plat` and
        `common_land_lots_area_ha`.
    """
    from openplaces.io.harmonizer.entity_links import COMMON_AREA_ID

    pairs = pd.DataFrame(
        {
            'lot': links['parcel_id'].astype('string').to_numpy(),
            'common': links[COMMON_AREA_ID].astype('string').to_numpy(),
        }
    )
    pairs['lot_area'] = pairs['lot'].map(area).astype('float64').to_numpy()
    pairs['common_area'] = pairs['common'].map(area).astype('float64').to_numpy()
    by_common = pairs.groupby('common')
    lots_area = by_common['lot_area'].sum(min_count=1)
    commons = pd.DataFrame(
        {
            LOT_COUNT_COLUMN: by_common.size().astype('float64'),
            LOTS_AREA_COLUMN: lots_area,
        }
    )
    total = pairs['common'].map(lots_area).astype('float64')
    pairs['share_ha'] = pairs['common_area'] * pairs['lot_area'] / total
    shares = pairs.groupby('lot')['share_ha'].sum(min_count=1)
    return shares, commons


@_register('apportion_common_land', phase='gather')
def apportion_common_land(
    state: CurateState,
    area_column: str = 'area_ha',
    subdivision_column: str = 'subdivision_code',
    plat_column: str = 'plat_id_local',
) -> CurateState:
    """Give each linked lot its share of its plat's common areas, as an area.

    Reads the plat link table the parcel spine wrote
    (`io.harmonizer.entity_links.link_common_areas_by_plat`, method
    `plat key`) and writes, on curated parcels:

    - on each lot: `common_land_share_ha`, the plat's common area
      divided among its linked lots in proportion to lot area;
    - on each common area: `n_lots_per_plat` and
      `common_land_lots_area_ha`, the number and summed area of the
      lots it is linked to;
    - on both: `common_land_plat_id`, the plat key (admin unit,
      subdivision code, plat reference), so a reader can find a common
      area's lots among the rows it already holds.

    It writes no value. The lots' land values stay as recorded, and
    nothing is written into a common area's value columns: the roll
    counts the common land's worth in the lots (maintainer's decision
    99a, 2026-10-24). The columns are areas and counts, not estimates,
    so they carry no `imputed` marker. A unit without a valid link
    table gets the columns all missing.

    Design limits checked: L25 (no missing value is filled from rows
    sharing the key; only areas are divided), L22 (no source column is
    overwritten), L2 and L27 (the link is the roll's plat key, not an
    owner or a geometric grouping).

    Parameters
    ----------
    area_column : str
        Parcel area in hectares (default `area_ha`).
    subdivision_column, plat_column : str
        The two key parts, read only to name the plat in
        `common_land_plat_id`.
    """
    from openplaces.io.harmonizer.entity_links import (
        COMMON_AREA_ID,
        PLAT_KEY_METHOD,
        get_plat_link_path,
        read_entity_link,
    )

    curated = state.curated
    for column in (SHARE_COLUMN, LOTS_AREA_COLUMN, LOT_COUNT_COLUMN):
        curated[column] = pd.Series(pd.NA, index=curated.index, dtype='Float64')
    curated[PLAT_ID_COLUMN] = pd.Series(pd.NA, index=curated.index, dtype='string')
    try:
        path = get_plat_link_path(state.recipe, state.admin_id)
        links = read_entity_link(path)
    except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
        warnings.warn(
            f'apportion_common_land: no plat link for {state.admin_id} ({exc}).',
            stacklevel=2,
        )
        links = None
    if links is None:
        warnings.warn(
            f'apportion_common_land: no valid plat link table for '
            f'{state.admin_id}; the common land columns stay empty.',
            stacklevel=2,
        )
        state.curated = curated
        return state

    ids = pd.Index(curated.index).astype('string')
    links = links[links['link_method'].astype('string').eq(PLAT_KEY_METHOD)]
    links = links[
        links['parcel_id'].astype('string').isin(ids)
        & links[COMMON_AREA_ID].astype('string').isin(ids)
    ]
    area = pd.Series(
        pd.to_numeric(curated[area_column], errors='coerce').to_numpy(), index=ids
    )
    shares, commons = apportion_common_land_area(links, area)

    keys = pd.Series(ids, index=curated.index)
    curated[SHARE_COLUMN] = keys.map(shares).astype('Float64')
    curated[LOT_COUNT_COLUMN] = keys.map(commons[LOT_COUNT_COLUMN]).astype('Float64')
    curated[LOTS_AREA_COLUMN] = keys.map(commons[LOTS_AREA_COLUMN]).astype('Float64')

    linked = keys.isin(shares.index) | keys.isin(commons.index)
    if subdivision_column in curated.columns and plat_column in curated.columns:
        plat_id = (
            f'{state.admin_id}_'
            + curated[subdivision_column].astype('string').str.strip()
            + '/'
            + curated[plat_column].astype('string').str.strip()
        )
        curated[PLAT_ID_COLUMN] = plat_id.where(linked).astype('string')
    state.curated = curated
    if state.verbose:
        print(
            f'  apportion_common_land: {len(shares):,d} lots share '
            f'{len(commons):,d} common areas '
            f'({float(area.reindex(commons.index).sum()):,.1f} ha)'
        )
    return state
