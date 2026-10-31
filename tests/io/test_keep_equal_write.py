"""Under OPENPLACES_KEEP_EQUAL a write of the same table keeps the old file.

The sample rerun that checks a code change (`flow.code_staleness
--verify`) relies on it: an output the new code writes again unchanged
keeps its mtime, so nothing downstream reruns. Every value below is
fabricated.
"""

import os

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

import openplaces.io.tables as tables


def _frame(values=(1.0, 2.0)):
    return pd.DataFrame({'value': list(values)}, index=pd.Index(['a', 'b'], name='id'))


def _age(path, seconds=1000):
    stamp = os.path.getmtime(path) - seconds
    os.utime(path, (stamp, stamp))
    return os.path.getmtime(path)


def test_the_same_table_keeps_the_old_file(tmp_path, monkeypatch):
    target = tmp_path / 'x.parquet'
    tables.to_parquet(_frame(), target)
    before = _age(target)
    monkeypatch.setenv(tables.KEEP_EQUAL_ENV, '1')

    tables.to_parquet(_frame(), target, file_metadata={'other': 'footer'})

    assert os.path.getmtime(target) == before
    assert [p.name for p in tmp_path.iterdir()] == ['x.parquet']


def test_a_different_table_replaces_it(tmp_path, monkeypatch):
    target = tmp_path / 'x.parquet'
    tables.to_parquet(_frame(), target)
    before = _age(target)
    monkeypatch.setenv(tables.KEEP_EQUAL_ENV, '1')

    tables.to_parquet(_frame((1.0, 3.0)), target)

    assert os.path.getmtime(target) > before
    assert pd.read_parquet(target)['value'].tolist() == [1.0, 3.0]


def test_without_the_variable_every_write_replaces(tmp_path, monkeypatch):
    monkeypatch.delenv(tables.KEEP_EQUAL_ENV, raising=False)
    target = tmp_path / 'x.parquet'
    tables.to_parquet(_frame(), target)
    before = _age(target)

    tables.to_parquet(_frame(), target)

    assert os.path.getmtime(target) > before


def test_a_geometry_sidecar_is_compared_on_its_own(tmp_path, monkeypatch):
    gdf = gpd.GeoDataFrame(
        {'value': [1.0, 2.0]},
        geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2)],
        crs='EPSG:4326',
    )
    target = tmp_path / 'x.parquet'
    tables.save_parquet(gdf, target)
    geo = tmp_path / 'x_geo.parquet'
    attrs_before, geo_before = _age(target), _age(geo)
    monkeypatch.setenv(tables.KEEP_EQUAL_ENV, '1')

    moved = gdf.copy()
    moved.geometry = [box(0, 0, 1, 1), box(5, 5, 6, 6)]
    tables.save_parquet(moved, target)

    assert os.path.getmtime(target) == attrs_before
    assert os.path.getmtime(geo) > geo_before


def test_same_table_ignores_footer_metadata_but_not_column_order(tmp_path):
    a, b, c = (tmp_path / f'{n}.parquet' for n in 'abc')
    tables.to_parquet(_frame(), a)
    tables.to_parquet(_frame(), b, file_metadata={'other': 'footer'})
    tables.to_parquet(_frame().assign(more=[0, 0])[['more', 'value']], c)
    assert tables.same_table(a, b)
    assert not tables.same_table(a, c)
