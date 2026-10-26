"""Tests for `resolve_raster_path`.

Recipes name a raster by a path relative to the configured `external`
directory so they stay portable; an absolute path has to keep working for
ad-hoc callers and one-off notebooks.
"""

from __future__ import annotations

from pathlib import Path

from openplaces.config import cfg
from openplaces.core.constants import STANDARD_DIRS
from openplaces.path import resolve_raster_path


def test_relative_path_is_joined_to_the_external_directory():
    result = resolve_raster_path('_all/population/ghsl/r2023a/example.tif')

    assert result == cfg.external_dir / '_all' / 'population' / 'ghsl' / 'r2023a' / (
        'example.tif'
    )
    assert result.is_absolute()


def test_absolute_path_passes_through_unchanged(tmp_path):
    absolute = tmp_path / 'somewhere' / 'example.tif'

    assert resolve_raster_path(absolute) == absolute
    assert resolve_raster_path(str(absolute)) == absolute


def test_accepts_str_and_path_alike():
    assert resolve_raster_path('a/b.tif') == resolve_raster_path(Path('a/b.tif'))


def test_there_is_no_separate_rasters_bucket():
    # Rasters are downloads: they live under `external`, admin-scoped.
    assert 'rasters' not in STANDARD_DIRS
    assert not hasattr(cfg, 'rasters_dir')
