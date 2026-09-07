"""Label-keyed polygon operations refuse a duplicated identifier.

Two rows sharing an id cannot be told apart by the label the operation
keys on, so both the identity overlay and the overlap resolver stop with
a message naming the id rather than returning a quietly wrong answer.
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from openplaces.geo.polygon import _coverage_fractions, resolve_overlapping_polygons


def _pieces(ids):
    return gpd.GeoDataFrame(
        {
            'parcel_id': ids,
            'geometry': [box(i, 0, i + 1, 1) for i in range(len(ids))],
        },
        crs='epsg:6933',
    )


def test_coverage_fractions_refuses_a_repeated_id():
    gdf = _pieces(['a', 'a', 'b'])
    with pytest.raises(ValueError, match='identity overlay on parcel_id'):
        _coverage_fractions(gdf, 'parcel_id', gdf)


def test_coverage_fractions_reports_full_coverage_when_ids_are_unique():
    gdf = _pieces(['a', 'b', 'c'])
    fractions = _coverage_fractions(gdf, 'parcel_id', gdf)
    assert fractions.tolist() == [1.0, 1.0, 1.0]
    # A partially covered polygon is the only case allowed to read below 1.
    half = gdf.copy()
    half['geometry'] = [box(i, 0, i + 0.5, 1) for i in range(3)]
    assert _coverage_fractions(half, 'parcel_id', gdf).tolist() == [0.5, 0.5, 0.5]


def _overlapping(index):
    return gpd.GeoDataFrame(
        {
            'use_group': ['residential', 'residential', 'commercial'],
            'geometry': [
                box(0, 0, 1, 1),
                box(0, 0, 1, 1),
                box(10, 10, 11, 11),
            ],
        },
        index=pd.Index(index, name='parcel_id'),
        crs='epsg:6933',
    )


def test_resolve_overlapping_polygons_refuses_a_repeated_label():
    with pytest.raises(ValueError, match='resolve_overlapping_polygons'):
        resolve_overlapping_polygons(_overlapping(['a', 'a', 'b']), keep=False)


def test_resolve_overlapping_polygons_still_drops_an_exact_duplicate():
    out = resolve_overlapping_polygons(_overlapping(['a', 'b', 'c']), keep=False)
    assert len(out) == 2
    assert 'c' in out.index
