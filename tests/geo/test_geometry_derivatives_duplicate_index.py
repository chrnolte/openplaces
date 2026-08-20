"""Geometry derivatives must survive a spine index with duplicate labels.

A spine index can legitimately carry duplicate ids (identical source
geometries share a geo_id). Seen in Fort Bend, TX: the centroid join
multiplied 6 duplicated rows past the precomputed area mask's length, and
the masked-area reindex raised on the duplicate labels -- either way the
whole county's harmonize crashed. Both paths now place results
positionally.
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from openplaces.geo.polygon import add_geometry_derivatives, get_areas


def _dup_index_gdf():
    return gpd.GeoDataFrame(
        {'geometry': [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)]},
        index=pd.Index(['a', 'a', 'b'], name='parcel_id'),
        crs='epsg:4326',
    )


def test_add_geometry_derivatives_keeps_row_count_and_mask_alignment():
    out = add_geometry_derivatives(
        _dup_index_gdf(), None, area_unit='ha', area_mask=[True, True, False]
    )
    assert len(out) == 3
    assert out['lat'].notna().all()
    assert out['area_ha'].notna().sum() == 2
    assert pd.isna(out['area_ha'].iloc[2])


def test_get_areas_masked_with_duplicate_labels():
    areas = get_areas(_dup_index_gdf(), unit='m2', mask=[False, True, True])
    assert len(areas) == 3
    assert pd.isna(areas.iloc[0])
    assert (areas.iloc[1:] > 0).all()
