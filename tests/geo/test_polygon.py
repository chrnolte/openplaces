"""Tests for `get_areas`'s generic `mask` parameter."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from openplaces.geo import polygon as polygon_module
from openplaces.geo.polygon import get_areas


def _gdf(n=2):
    # 10m x 10m squares in an equal-area CRS: area = 100 m2 each.
    geometry = [box(20 * i, 0, 20 * i + 10, 10) for i in range(n)]
    return gpd.GeoDataFrame({'geometry': geometry}, crs='epsg:6933')


def test_mask_none_computes_every_row():
    gdf = _gdf(2)
    areas = get_areas(gdf, unit='m2')
    assert list(areas) == [100.0, 100.0]


def test_mask_restricts_computation_and_masked_rows_are_nan():
    gdf = _gdf(3)
    mask = [True, False, True]
    areas = get_areas(gdf, unit='m2', mask=mask)
    assert areas.iloc[0] == 100.0
    assert pd.isna(areas.iloc[1])
    assert areas.iloc[2] == 100.0


def test_mask_skips_computation_for_excluded_rows_not_just_discards(monkeypatch):
    # A spy on GeoDataFrame.to_crs (used internally by get_areas to reproject
    # before measuring) confirms the masked-out row's geometry never reaches
    # the reprojection/measurement step at all, not just that its result is
    # discarded afterward. Input CRS deliberately differs from the target
    # ('epsg:6933', get_areas's default) so to_crs is actually invoked --
    # small lon/lat offsets are still a valid, transformable region.
    geometry = [box(20 * i, 0, 20 * i + 10, 10) for i in range(3)]
    gdf = gpd.GeoDataFrame({'geometry': geometry}, crs='epsg:4326')
    seen_lengths = []
    original_to_crs = gpd.GeoDataFrame.to_crs

    def spy_to_crs(self, *args, **kwargs):
        seen_lengths.append(len(self))
        return original_to_crs(self, *args, **kwargs)

    monkeypatch.setattr(polygon_module.gpd.GeoDataFrame, 'to_crs', spy_to_crs)

    mask = [True, False, True]
    get_areas(gdf, unit='m2', mask=mask)

    assert seen_lengths == [2]
