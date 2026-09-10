"""Shared pytest fixtures for the openplaces test suite.

The suite runs against an empty data root. No test may depend on a
dataset having been downloaded or built on the machine running it: a
test that needs data writes a fabricated table into the isolated root
(see `mock_data_root`) or fakes the reader. The redirect happens here,
before any other openplaces module is imported, because `openplaces.path`
binds its directory defaults at import time.
"""

import tempfile
from pathlib import Path

from openplaces.config import cfg

_ISOLATED_ROOT = Path(tempfile.mkdtemp(prefix='openplaces-tests-'))


def _isolate_data_root() -> None:
    dirs = dict(cfg.config['directories'])
    dirs['data_root'] = _ISOLATED_ROOT
    for name in ('core', 'external', 'raw', 'cache', 'out', 'share'):
        dirs[name] = _ISOLATED_ROOT / 'data' / name
    dirs['heap'] = _ISOLATED_ROOT / 'data/cache/_heap'
    dirs['logs'] = _ISOLATED_ROOT / 'data/cache/_logs'
    cfg.config['directories'] = dirs


_isolate_data_root()

import geopandas as gpd  # noqa: E402
import pytest  # noqa: E402
from shapely.geometry import box  # noqa: E402

from openplaces.core.schema import AdminId  # noqa: E402
from openplaces.io.curator import CurateState  # noqa: E402
from openplaces.io.harmonizer import HarmonizeState  # noqa: E402


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
