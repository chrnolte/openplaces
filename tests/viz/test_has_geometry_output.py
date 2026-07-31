from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from openplaces.viz import maps
from openplaces.viz.maps import _has_geometry_output


def _make_output_paths(tmp_path):
    return tmp_path / 'main.parquet', tmp_path / 'main_geo.parquet'


def test_has_geometry_output_true_for_geo_sidecar(tmp_path, monkeypatch):
    main_path, geo_path = _make_output_paths(tmp_path)
    geo_path.touch()

    monkeypatch.setattr(
        maps,
        'get_output_path',
        lambda recipe, admin_id, partition_id=None, geo=False: (
            geo_path if geo else main_path
        ),
    )

    assert _has_geometry_output({}, 'US', None) is True


def test_has_geometry_output_true_for_combined_file(tmp_path, monkeypatch):
    # No `_geo` sidecar: geometry is embedded directly in the main file, as
    # written by save_parquet(..., combined=True).
    main_path, geo_path = _make_output_paths(tmp_path)
    gdf = gpd.GeoDataFrame({'value': [1]}, geometry=[Point(0, 0)], crs='EPSG:4326')
    gdf.to_parquet(main_path)

    monkeypatch.setattr(
        maps,
        'get_output_path',
        lambda recipe, admin_id, partition_id=None, geo=False: (
            geo_path if geo else main_path
        ),
    )

    assert _has_geometry_output({}, 'US', None) is True


def test_has_geometry_output_false_for_tabular_only_file(tmp_path, monkeypatch):
    main_path, geo_path = _make_output_paths(tmp_path)
    pd.DataFrame({'value': [1]}).to_parquet(main_path)

    monkeypatch.setattr(
        maps,
        'get_output_path',
        lambda recipe, admin_id, partition_id=None, geo=False: (
            geo_path if geo else main_path
        ),
    )

    assert _has_geometry_output({}, 'US', None) is False


def test_has_geometry_output_false_when_nothing_written(tmp_path, monkeypatch):
    main_path, geo_path = _make_output_paths(tmp_path)

    monkeypatch.setattr(
        maps,
        'get_output_path',
        lambda recipe, admin_id, partition_id=None, geo=False: (
            geo_path if geo else main_path
        ),
    )

    assert _has_geometry_output({}, 'US', None) is False
