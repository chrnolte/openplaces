"""The overlap-share test of resolve_spine.

Fabricated outlines in an equal-area CRS (meters).
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from openplaces.io.harmonizer.spine import _overlap_share_reject_ids

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
