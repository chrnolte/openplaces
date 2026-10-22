"""Pin the condo-cluster lookup helpers to the pandas calls they replace.

`consolidate_condo_cluster_footprints` must not change its output when
it gets faster, and a geometry union or an area can move in its last
bits when its inputs arrive in another order. Each helper is therefore
checked for the same values *and* the same order as the call it
replaces. All geometries are fabricated.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import pytest
import shapely
from shapely.geometry import box

from openplaces.geo.polygon import get_areas, local_metric_crs
from openplaces.io.harmonizer.links.condo_clusters import (
    _crosswalk_rows_by_parcel,
    _fast_source_crs,
    _first_area_m2_by_label,
    _linked_spine_ids,
    _take_by_labels,
    _to_crs_together,
)


def _boxes(n, x0=-80.6, y0=28.1, size=2e-4):
    return [box(x0 + i * size, y0, x0 + (i + 0.9) * size, y0 + size) for i in range(n)]


@pytest.mark.parametrize('shuffle', [False, True])
def test_take_by_labels_matches_index_intersection(shuffle):
    labels = [f'P{i:03d}' for i in range(40)]
    if shuffle:
        labels = list(np.random.default_rng(1).permutation(labels))
    series = pd.Series(range(40), index=pd.Index(labels, name='parcel_id'))
    wanted = ['P031', 'P007', 'NOPE', 'P007', 'P012', 'P000']
    expected = series.loc[series.index.intersection(wanted)]
    pd.testing.assert_series_equal(_take_by_labels(series, wanted), expected)


def test_take_by_labels_keeps_every_row_of_a_repeated_label():
    series = pd.Series([1, 2, 3, 4], index=['A', 'B', 'A', 'C'])
    expected = series.loc[series.index.intersection(['A', 'C'])]
    pd.testing.assert_series_equal(_take_by_labels(series, ['A', 'C']), expected)


def test_take_by_labels_with_no_match_is_empty():
    series = pd.Series([1, 2], index=['A', 'B'])
    assert _take_by_labels(series, ['Z']).empty


def test_linked_spine_ids_match_an_isin_mask():
    pairs = [
        ('F9', 'P1'),
        ('F2', 'P2'),
        ('F9', 'P3'),
        ('F4', 'P1'),
        ('F2', 'P5'),
        ('F7', 'P3'),
    ]
    crosswalk = pd.DataFrame(
        {'area_intersection_m2': range(len(pairs))},
        index=pd.MultiIndex.from_tuples(pairs, names=['footprint_id', 'parcel_id']),
    )
    rows = _crosswalk_rows_by_parcel(crosswalk)
    spine_ids = crosswalk.index.get_level_values('footprint_id')
    for cluster in (['P3', 'P1'], ['P5', 'P3', 'P3'], ['P8'], ['P2', 'P8']):
        mask = crosswalk.index.get_level_values('parcel_id').isin(cluster)
        expected = crosswalk[mask].index.get_level_values('footprint_id').unique()
        pd.testing.assert_index_equal(
            _linked_spine_ids(cluster, rows, spine_ids), expected
        )


def test_to_crs_together_matches_separate_reprojections():
    parcels = gpd.GeoSeries(_boxes(5), index=list('abcde'), crs=4326)
    union = parcels.union_all()
    parts = shapely.get_parts(
        gpd.GeoSeries(_boxes(3, y0=28.1002), crs=4326).union_all()
    )
    crs_m = local_metric_crs(gpd.GeoSeries([union], crs=4326))

    union_m, parts_m, parcels_m = _to_crs_together(4326, crs_m, [union], parts, parcels)

    expected = [
        gpd.GeoSeries([union], crs=4326).to_crs(crs_m),
        gpd.GeoSeries(parts, crs=4326).to_crs(crs_m),
        parcels.to_crs(crs_m),
    ]
    for got, want in zip((union_m, parts_m, parcels_m), expected):
        assert list(shapely.to_wkb(got.values)) == list(shapely.to_wkb(want.values))
    pd.testing.assert_index_equal(parcels_m.index, parcels.index)


def test_fast_source_crs_reprojects_a_geoparquet_crs_identically():
    # A GeoParquet file stores its CRS as PROJJSON; reading one back
    # yields a CRS whose source string is that JSON, not 'EPSG:4326'.
    stored = pyproj.CRS.from_json(pyproj.CRS.from_epsg(4326).to_json())
    fast = _fast_source_crs(stored)
    assert fast.srs == 'EPSG:4326'

    parcels = gpd.GeoSeries(_boxes(5), crs=stored)
    crs_m = local_metric_crs(parcels)
    (got,) = _to_crs_together(fast, crs_m, parcels)
    want = parcels.to_crs(crs_m)
    assert list(shapely.to_wkb(got.values)) == list(shapely.to_wkb(want.values))


def test_fast_source_crs_keeps_any_other_crs():
    utm = pyproj.CRS.from_epsg(32617)
    assert _fast_source_crs(utm) is utm
    assert _fast_source_crs(None) is None


def test_first_area_by_label_matches_per_label_get_areas():
    geoms = _boxes(4)
    geoms[2] = box(-80.6, 28.1, -80.599, 28.101)
    gdf = gpd.GeoDataFrame(
        {'geometry': geoms}, index=pd.Index(['A', 'B', 'A', 'C']), crs=4326
    )
    got = _first_area_m2_by_label(gdf, {'A', 'C'})
    for label in ('A', 'C'):
        assert got[label] == get_areas(gdf.loc[[label]], 'm2').iloc[0]
    assert set(got) == {'A', 'C'}
