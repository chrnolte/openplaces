"""Tests for `resolve_raster_path`.

Recipes name a raster by a path relative to the configured `rasters`
directory so they stay portable; an absolute path has to keep working for
ad-hoc callers and one-off notebooks.
"""

from __future__ import annotations

from pathlib import Path

from openplaces.config import cfg
from openplaces.path import resolve_raster_path


def test_relative_path_is_joined_to_the_configured_root():
    result = resolve_raster_path('DEM/copernicus_30m/example.tif')

    assert result == cfg.rasters_dir / 'DEM' / 'copernicus_30m' / 'example.tif'
    assert result.is_absolute()


def test_absolute_path_passes_through_unchanged(tmp_path):
    absolute = tmp_path / 'somewhere' / 'example.tif'

    assert resolve_raster_path(absolute) == absolute
    assert resolve_raster_path(str(absolute)) == absolute


def test_accepts_str_and_path_alike():
    assert resolve_raster_path('a/b.tif') == resolve_raster_path(Path('a/b.tif'))


def test_rasters_root_is_configured_and_absolute():
    assert cfg.rasters_dir.is_absolute()
