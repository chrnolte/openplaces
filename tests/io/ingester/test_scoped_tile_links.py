"""A tile grid is linked to admin units one unit at a time, when needed.

The tile grid is global, but the question a download asks is local: which
tiles touch this county. The link used to be one global crosswalk built by
the tile recipe over every admin unit in the world, so a county build waited
on the world admin layers. It now lives beside the admin unit's own file, is
built from that unit's polygons and the grid the first time a download needs
it, and is rebuilt only when either input changes on disk.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

import openplaces.geo.link as link_module
from openplaces.core.schema import AdminId
from openplaces.geo.link import ensure_scoped_tile_link, get_scoped_tile_link_path
from openplaces.io import read_parquet, save_parquet

TILES = 'tile-test-2026'
ADMIN = 'admin-test-2026_admin3'


@pytest.fixture
def layers(tmp_path, monkeypatch):
    """A three-tile grid and one state's two counties, on disk."""
    tiles = gpd.GeoDataFrame(
        {'n_buildings': [1, 2, 3]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(5, 5, 6, 6)],
        crs='EPSG:4326',
        index=pd.Index(['t1', 't2', 't3'], name='tile_id'),
    )
    admin = gpd.GeoDataFrame(
        {'name': ['A', 'B']},
        geometry=[box(0.2, 0.2, 0.8, 0.8), box(0.9, 0.2, 1.5, 0.8)],
        crs='EPSG:4326',
        index=pd.Index(['XX-AA-AA', 'XX-AA-AB'], name='admin3_id'),
    )
    tiles_path = tmp_path / 'tiles.parquet'
    admin_path = tmp_path / 'XX-AA_admin-test-2026_admin3.parquet'
    save_parquet(tiles, tiles_path)
    save_parquet(admin, admin_path)

    recipes = {
        TILES: {'recipe_id': TILES, 'admin_id': None},
        ADMIN: {'recipe_id': ADMIN, 'admin_id': None, 'save_to': {'admin_level': 2}},
    }
    monkeypatch.setattr(link_module, 'get_recipe_by_id', lambda rid: recipes[rid])
    monkeypatch.setattr(
        link_module, 'get_save_admin_level', lambda r: 2 if r is recipes[ADMIN] else 0
    )

    def output_path(recipe, admin_id=None):
        if recipe is recipes[TILES]:
            return tiles_path
        assert str(admin_id) == 'XX-AA'
        return admin_path

    monkeypatch.setattr(link_module, 'get_output_path', output_path)
    return tiles_path, admin_path


def test_link_sits_beside_the_admin_units_file(layers):
    _tiles, admin_path = layers
    path = get_scoped_tile_link_path(TILES, ADMIN, 'XX-AA-AB')
    assert path.parent == admin_path.parent
    assert path.name == f'XX-AA_admin-test-2026_admin3_{TILES}.parquet'


def test_link_is_built_from_the_units_polygons_only(layers):
    link = ensure_scoped_tile_link(TILES, ADMIN, AdminId('XX-AA-AA'))
    pairs = set(link.index)
    assert ('t1', 'XX-AA-AA') in pairs
    assert ('t1', 'XX-AA-AB') in pairs
    assert ('t2', 'XX-AA-AB') in pairs
    # The far tile touches no county and is absent.
    assert not any(tile == 't3' for tile, _admin in pairs)
    assert get_scoped_tile_link_path(TILES, ADMIN, 'XX-AA-AA').exists()


def test_a_current_link_is_reused_and_a_changed_input_rebuilds_it(layers, monkeypatch):
    ensure_scoped_tile_link(TILES, ADMIN, 'XX-AA-AA')
    calls = []
    real = link_module.overlay_polygons_with_duckdb

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(link_module, 'overlay_polygons_with_duckdb', counting)
    ensure_scoped_tile_link(TILES, ADMIN, 'XX-AA-AA')
    assert calls == []
    # Re-ingest the admin layer: a different size on disk.
    _tiles, admin_path = layers
    admin = read_parquet(admin_path, geom=True)
    admin['name'] = admin['name'] + ' county'
    save_parquet(admin, admin_path)
    ensure_scoped_tile_link(TILES, ADMIN, 'XX-AA-AA')
    assert calls == [1]
