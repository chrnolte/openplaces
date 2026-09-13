"""A recipe's `on_invalid` reaches the geometry reader.

pyogrio raises on a whole layer when one shape cannot be built (a ring
with fewer than four points). The recipe key lets a source with one bad
shape be read, keeping that row with an empty geometry.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

from openplaces.io.ingester import table_ingester
from openplaces.io.ingester.table_ingester import TableIngester


class _Timer:
    def mark(self, *args, **kwargs):
        return None


def _ingester(recipe, data_path):
    ti = TableIngester.__new__(TableIngester)
    ti.recipe = recipe
    ti.download_partition = {'data_path': Path(data_path)}
    ti.timer = _Timer()
    return ti


@pytest.mark.parametrize('suffix', ['.gdb', '.gpkg'])
def test_on_invalid_is_passed_to_the_reader(tmp_path, monkeypatch, suffix):
    seen = {}

    def fake_read(path, **kwargs):
        seen.update(kwargs)
        return gpd.GeoDataFrame({'pid': ['1']}, geometry=[Point(0, 0)])

    monkeypatch.setattr(table_ingester, 'read_gdb_with_domains', fake_read)
    monkeypatch.setattr(table_ingester.gpd, 'read_file', fake_read)

    ti = _ingester({'on_invalid': 'warn', 'layer': 'parcels'}, tmp_path / f'x{suffix}')
    ti._read_recipe_data()

    assert seen['on_invalid'] == 'warn'


def test_without_the_key_nothing_is_passed(tmp_path, monkeypatch):
    seen = {}

    def fake_read(path, **kwargs):
        seen.update(kwargs)
        return gpd.GeoDataFrame({'pid': ['1']}, geometry=[Point(0, 0)])

    monkeypatch.setattr(table_ingester.gpd, 'read_file', fake_read)

    _ingester({}, tmp_path / 'x.gpkg')._read_recipe_data()

    assert 'on_invalid' not in seen
