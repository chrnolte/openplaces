"""Geometry helpers must not leave a warning filter behind.

`warnings.filterwarnings('ignore', ...)` followed by
`filterwarnings('default', ...)` does not restore the caller's filter: it
prepends a second global entry that outranks whatever the caller set, for
the rest of the process.
"""

from __future__ import annotations

import warnings

import geopandas as gpd
from shapely.geometry import box

from openplaces.geo.polygon import get_lat_long_centroids


def test_centroids_leave_the_callers_filter_alone():
    gdf = gpd.GeoDataFrame({'geometry': [box(0, 0, 1, 1)]}, crs='epsg:4326')

    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        get_lat_long_centroids(gdf)
        # The caller asked for UserWarning to be an error, and still
        # expects that on the next line.
        try:
            warnings.warn('after the call', UserWarning)
        except UserWarning:
            return
    raise AssertionError('the callers UserWarning filter was overridden')
