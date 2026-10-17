"""Evidence-based choice between overlapping outlines of two sources.

`resolve_spine` merges footprint sources by rank: a later source's
outline enters only where it overlaps nothing already accepted. Any
outline of an earlier source therefore blocks every later source at its
spot, however small or out of date it is: a 14 m2 roof piece or the
outline of a building since replaced blocks the 4,000 m2 building
another source draws around it. Measured 2026-10-16 on
boston-inner-core: against OBM, the ranked MassGIS layer held 113 such
fragments in Boston.

This module looks at the outlines of the accepted spine and of the next
source that overlap each other (the intersection of a pair is at least
half the smaller outline, the pair test of
`geo.polygon.resolve_overlapping_polygons`), groups them into clusters
(connected components), and switches a cluster to the later source in
one shape, a **fragment**: one later outline holds accepted pieces that
together cover under a third of it. The switch is refused when the
later outline crosses more lots than the accepted pieces do. No outline
is created, enlarged, unioned or clipped, and each switch carries a
fixed label (`spine_resolution`), never a score.

**Off by default, because its test failed.** Measured 2026-10-16 on
US-NC-CAR, against layers that never enter the spine: of the 261
outlines the rule switched to (mostly NC DPS over OBM), Microsoft's
global layer matches 2.2% of the area at IoU >= 0.5 and OpenStreetMap
4.4%, against 82.2% and 7.4% of the outlines it dropped. The larger
outline is not reliably the truer one where the earlier source is
current. A second rule, splitting one outline over several lots into a
later source's per-lot pieces, fared worse (Microsoft global 19.4% vs
98.7%) and was removed: a building may span lots, and the footprint
entity keeps a townhome row as one outline. The roll's stated building
area, the test that would separate an outdated outline from a current
one, is not available here, because the footprint geospine runs before
the parcel attributes exist.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# Equal-area CRS, the one resolve_spine's keep-out buffers use.
METRIC_CRS = 'epsg:6933'
FRAGMENT_SHARE_MAX = 1 / 3
PAIR_SHARE_MIN = 0.5
# An outline sits on a lot when at least this share of it, and at least
# this many m2, falls inside the lot: a sliver over a lot line is not a
# second lot.
LOT_SHARE_MIN = 0.1
LOT_AREA_MIN_M2 = 10.0

FRAGMENT_REPLACED = 'fragment replaced'


def _lot_counts(outlines: gpd.GeoSeries, lots: gpd.GeoSeries | None) -> pd.Series:
    """Number of lots each outline sits on (0 where no lot layer)."""
    if lots is None or lots.empty or outlines.empty:
        return pd.Series(0, index=outlines.index)
    left = gpd.GeoDataFrame(geometry=outlines.reset_index(drop=True))
    right = gpd.GeoDataFrame(geometry=lots.reset_index(drop=True))
    j = gpd.sjoin(left, right, predicate='intersects')
    if j.empty:
        return pd.Series(0, index=outlines.index)
    inter = shapely.area(
        shapely.intersection(
            j.geometry.to_numpy(),
            right.geometry.to_numpy()[j['index_right'].to_numpy()],
        )
    )
    area = shapely.area(left.geometry.to_numpy()[j.index.to_numpy()])
    on = (inter >= LOT_AREA_MIN_M2) & (inter >= LOT_SHARE_MIN * area)
    counts = pd.Series(on.astype(int), index=j.index).groupby(level=0).sum()
    counts = counts.reindex(range(len(outlines)), fill_value=0)
    counts.index = outlines.index
    return counts


def choose_between_sources(
    spine: gpd.GeoDataFrame,
    candidate: gpd.GeoDataFrame,
    lots: gpd.GeoSeries | None = None,
) -> tuple[pd.Index, pd.Series]:
    """Clusters where the later source's outlines replace the spine's.

    Parameters
    ----------
    spine : GeoDataFrame
        Outlines accepted so far (the ranked source).
    candidate : GeoDataFrame
        The next source's outlines.
    lots : GeoSeries, optional
        Lot polygons for the lot-line test, in any CRS. Without them a
        fragment switch is never refused.

    Returns
    -------
    tuple of (pandas.Index, pandas.Series)
        Index labels of the spine rows to drop, and a Series indexed by
        the candidate rows to take, holding each one's
        `spine_resolution` label.
    """
    empty = (spine.index[:0], pd.Series(dtype=object, index=candidate.index[:0]))
    if spine.empty or candidate.empty:
        return empty
    a = spine.geometry.to_crs(METRIC_CRS)
    b = candidate.geometry.to_crs(METRIC_CRS)
    lots_m = lots.to_crs(METRIC_CRS) if lots is not None and len(lots) else None
    area_a = a.area.to_numpy()
    area_b = b.area.to_numpy()

    left = gpd.GeoDataFrame(geometry=a.reset_index(drop=True))
    right = gpd.GeoDataFrame(geometry=b.reset_index(drop=True))
    j = gpd.sjoin(left, right, predicate='intersects')
    if j.empty:
        return empty
    ia = j.index.to_numpy()
    ib = j['index_right'].to_numpy()
    inter = shapely.area(
        shapely.intersection(
            left.geometry.to_numpy()[ia], right.geometry.to_numpy()[ib]
        )
    )
    keep = inter >= PAIR_SHARE_MIN * np.minimum(area_a[ia], area_b[ib])
    ia, ib = ia[keep], ib[keep]
    if not len(ia):
        return empty

    na, nb = len(a), len(b)
    graph = coo_matrix((np.ones(len(ia)), (ia, ib + na)), shape=(na + nb, na + nb))
    _, label = connected_components(graph, directed=False)
    in_a = np.zeros(na, dtype=bool)
    in_a[ia] = True
    in_b = np.zeros(nb, dtype=bool)
    in_b[ib] = True
    ca = pd.DataFrame({'cluster': label[:na], 'area': area_a})[in_a]
    cb = pd.DataFrame({'cluster': label[na:], 'area': area_b})[in_b]

    lots_a = _lot_counts(a.iloc[ca.index], lots_m).to_numpy()
    lots_b = _lot_counts(b.iloc[cb.index], lots_m).to_numpy()
    ca['lots'] = lots_a
    cb['lots'] = lots_b
    ga = ca.groupby('cluster').agg(area_a=('area', 'sum'), lots_a=('lots', 'sum'))
    gb = cb.groupby('cluster').agg(
        n_b=('area', 'size'), area_b=('area', 'sum'), lots_b=('lots', 'sum')
    )
    c = ga.join(gb, how='inner')

    fragment = (
        (c['n_b'] == 1)
        & (c['area_a'] < FRAGMENT_SHARE_MAX * c['area_b'])
        & ~(c['lots_b'] > c['lots_a'])
    )
    labels = pd.Series(FRAGMENT_REPLACED, index=c.index[fragment], dtype=object)
    if labels.empty:
        return empty
    drop_pos = ca.index[ca['cluster'].isin(labels.index)]
    take = cb[cb['cluster'].isin(labels.index)]
    drop = spine.index[drop_pos]
    taken = pd.Series(
        labels.reindex(take['cluster']).to_numpy(),
        index=candidate.index[take.index],
        dtype=object,
    )
    return drop, taken
