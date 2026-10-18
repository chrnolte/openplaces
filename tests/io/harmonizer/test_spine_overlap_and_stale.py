"""The overlap-share test and the stale-neighborhood test of resolve_spine.

Fabricated outlines in an equal-area CRS (meters).
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from openplaces.io.harmonizer.spine import (
    _overlap_share_reject_ids,
    _stale_neighborhood_reject_ids,
)

CRS = 'epsg:6933'


def _frame(rows):
    return gpd.GeoDataFrame(
        geometry=[g for _, g in rows],
        index=pd.Index([i for i, _ in rows], name='footprint_id'),
        crs=CRS,
    )


def test_a_sliver_along_a_large_building_is_rejected_a_neighbor_is_not():
    spine = _frame([('big', box(0, 0, 100, 60))])
    candidates = _frame(
        [
            ('sliver', box(95, 0, 105, 60)),  # half inside the large building
            ('wall', box(99.5, 0, 112, 10)),  # 4% over a shared wall
            ('apart', box(103, 20, 113, 30)),  # 3 m away, overlaps nothing
        ]
    )
    rejected = _overlap_share_reject_ids(spine, candidates, 0.1)
    assert list(rejected) == ['sliver']


def _grid(prefix, n, x0, offset=0.0):
    """n 10 m outlines spaced 15 m apart, shifted by *offset*."""
    return [
        (f'{prefix}{i}', box(x0 + 15 * i + offset, 0, x0 + 15 * i + 10 + offset, 10))
        for i in range(n)
    ]


def test_a_clear_outline_where_its_source_disagrees_locally_is_left_out():
    # The accepted layer draws the block one way; the candidate source
    # draws it shifted by 9 m (rebuilt since), plus one extra outline.
    accepted = _frame(_grid('a', 5, 0))
    stale = _frame(_grid('m', 5, 0, offset=9.0) + [('extra', box(30, 20, 40, 30))])
    assert 'extra' in _stale_neighborhood_reject_ids(accepted, stale, 50.0, 4, 0.5)


def test_a_clear_outline_where_its_source_agrees_is_kept():
    accepted = _frame(_grid('a', 5, 0))
    agreeing = _frame(_grid('m', 5, 0, offset=0.5) + [('shed', box(30, 20, 40, 30))])
    assert list(_stale_neighborhood_reject_ids(accepted, agreeing, 50.0, 4, 0.5)) == []


def test_too_few_neighbors_never_reject():
    accepted = _frame(_grid('a', 5, 0))
    sparse = _frame(_grid('m', 2, 0, offset=9.0) + [('lone', box(30, 20, 40, 30))])
    assert list(_stale_neighborhood_reject_ids(accepted, sparse, 50.0, 4, 0.5)) == []
