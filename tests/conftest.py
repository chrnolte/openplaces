"""Shared pytest fixtures for the openplaces test suite."""

import geopandas as gpd
import pytest
from shapely.geometry import box

from openplaces.config import cfg
from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.harmonizer import HarmonizeState


@pytest.fixture
def mock_data_root(tmp_path, monkeypatch):
    """Directs openplaces directories to a temporary synthetic root for tests."""
    dirs = dict(cfg.config['directories'])
    dirs['data_root'] = tmp_path
    for name in ('core', 'external', 'raw', 'cache', 'out', 'share'):
        dirs[name] = tmp_path / 'data' / name
    dirs['heap'] = tmp_path / 'data/cache/_heap'
    dirs['logs'] = tmp_path / 'data/cache/_logs'
    monkeypatch.setitem(cfg.config, 'directories', dirs)
    # Ensure subprocess / snakemake environmental overrides are clean
    monkeypatch.delenv('SNAKEMAKE', raising=False)
    monkeypatch.delenv('OPENPLACES_ORCHESTRATED', raising=False)
    return tmp_path


@pytest.fixture
def empty_curate_state():
    """Returns a standard empty CurateState container with EPSG:6933 CRS."""
    geometry = [box(0, 0, 10, 10)]
    curated = gpd.GeoDataFrame({'geometry': geometry}, crs='epsg:6933')
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US'),
        verbose=False,
        timer=None,
        curated=curated,
    )


@pytest.fixture
def empty_harmonize_state():
    """Returns a baseline HarmonizeState container for step testing."""
    geometry = [box(0, 0, 10, 10)]
    spine = gpd.GeoDataFrame({'geometry': geometry}, crs='epsg:6933')
    return HarmonizeState(
        recipe={},
        admin_id=AdminId('US'),
        verbose=False,
        timer=None,
        spine=spine,
        references={},
        crosswalks={},
        overlays={},
        metadata={},
    )
