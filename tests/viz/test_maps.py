"""Tests for `_has_geometry_output`'s two save-layout detection paths."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from openplaces import cfg
from openplaces.io import save_parquet
from openplaces.recipe import get_output_path, get_recipe_by_id
from openplaces.viz.maps import _has_geometry_output

ADMIN_ID = 'US-MA-MI'


@pytest.fixture
def share_to_tmp(tmp_path, monkeypatch):
    """Redirect the 'share' data dir so tests never touch real outputs."""
    monkeypatch.setitem(cfg.config['directories'], 'share', tmp_path)
    return tmp_path


def _gdf():
    return gpd.GeoDataFrame(
        {'name': ['a']}, geometry=[box(0, 0, 1, 1)], crs='EPSG:4326'
    )


def test_combined_layout_has_geometry_without_a_sidecar(share_to_tmp):
    # US_footprint-cheer-2026 saves with save_to.combined: true -- geometry
    # merged into the one file, no _geo sidecar at all (see read_parquet's
    # own schema-peek handling of this layout).
    recipe = get_recipe_by_id('US_footprint-cheer-2026')
    out_path = get_output_path(recipe, ADMIN_ID)
    save_parquet(_gdf(), out_path, combined=True)

    assert not get_output_path(recipe, ADMIN_ID, geo=True).exists()
    assert _has_geometry_output(recipe, ADMIN_ID, None) is True


def test_split_layout_has_geometry_via_sidecar(share_to_tmp):
    recipe = get_recipe_by_id('US_footprint-cheer-2026')
    out_path = get_output_path(recipe, ADMIN_ID)
    save_parquet(_gdf(), out_path)  # default split layout: writes a _geo sidecar

    assert get_output_path(recipe, ADMIN_ID, geo=True).exists()
    assert _has_geometry_output(recipe, ADMIN_ID, None) is True


def test_attribute_only_output_has_no_geometry(share_to_tmp):
    recipe = get_recipe_by_id('US_footprint-cheer-2026')
    out_path = get_output_path(recipe, ADMIN_ID)
    save_parquet(pd.DataFrame({'name': ['a']}), out_path)

    assert _has_geometry_output(recipe, ADMIN_ID, None) is False


def test_missing_output_has_no_geometry(share_to_tmp):
    recipe = get_recipe_by_id('US_footprint-cheer-2026')
    assert _has_geometry_output(recipe, ADMIN_ID, None) is False
