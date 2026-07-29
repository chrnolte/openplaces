"""Tests for the `area_intersection` overlay option (a lighter alternative to
`iou=True` for callers that only ever read `area_intersection_m2` -- see
`build_id_or_overlay_crosswalk`'s overlay stage, the motivating caller).
"""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import box

from openplaces.geo.overlay import overlay_polygons_with_duckdb
from openplaces.geo.polygon import overlay_polygons

FUNCS = [overlay_polygons, overlay_polygons_with_duckdb]


def _gdfs():
    gdf1 = gpd.GeoDataFrame(
        {'id_1': ['A'], 'geometry': [box(0, 0, 2, 2)]}, crs='epsg:4326'
    ).set_index('id_1')
    gdf2 = gpd.GeoDataFrame(
        {'id_2': ['B'], 'geometry': [box(1, 1, 3, 3)]}, crs='epsg:4326'
    ).set_index('id_2')
    return gdf1, gdf2


@pytest.mark.parametrize('fn', FUNCS)
def test_area_intersection_matches_iou_value(fn):
    gdf1, gdf2 = _gdfs()

    from_iou = fn(gdf1, gdf2, iou=True, how='intersection')
    from_area_intersection = fn(gdf1, gdf2, area_intersection=True, how='intersection')

    assert from_area_intersection['area_intersection_m2'].to_numpy() == pytest.approx(
        from_iou['area_intersection_m2'].to_numpy(), rel=1e-6
    )


@pytest.mark.parametrize('fn', FUNCS)
def test_area_intersection_omits_iou_and_per_side_columns(fn):
    gdf1, gdf2 = _gdfs()

    result = fn(gdf1, gdf2, area_intersection=True, how='intersection')

    assert 'area_intersection_m2' in result.columns
    assert 'iou' not in result.columns
    assert not any(c.startswith('area') and c != 'area_intersection_m2' for c in result)


@pytest.mark.parametrize('fn', FUNCS)
def test_neither_flag_has_no_area_columns(fn):
    gdf1, gdf2 = _gdfs()

    result = fn(gdf1, gdf2, how='intersection')

    assert not any(c.startswith('area') or c == 'iou' for c in result)
