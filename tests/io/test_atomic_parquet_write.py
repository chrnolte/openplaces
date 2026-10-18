"""A parquet write lands under a temporary name and is moved into place,
so a failed or interrupted write never leaves a half-written file where
a downstream job reads it. Every value below is fabricated."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

import openplaces.io.tables as tables


def _frame():
    return pd.DataFrame({'value': [1.0, 2.0]}, index=pd.Index(['a', 'b'], name='id'))


def test_a_write_leaves_only_the_target(tmp_path):
    target = tmp_path / 'x.parquet'
    tables.to_parquet(_frame(), target)
    assert [p.name for p in tmp_path.iterdir()] == ['x.parquet']
    assert pd.read_parquet(target)['value'].tolist() == [1.0, 2.0]


def test_a_failed_write_keeps_the_previous_file(tmp_path, monkeypatch):
    target = tmp_path / 'x.parquet'
    tables.to_parquet(_frame(), target)

    def broken(df, filepath, **kwargs):
        filepath.write_bytes(b'half a file')
        raise RuntimeError('killed mid-write')

    monkeypatch.setattr(tables, '_write_parquet', broken)
    with pytest.raises(RuntimeError):
        tables.to_parquet(_frame().assign(value=[9.0, 9.0]), target)
    assert [p.name for p in tmp_path.iterdir()] == ['x.parquet']
    assert pd.read_parquet(target)['value'].tolist() == [1.0, 2.0]


def test_a_held_target_is_retried(tmp_path, monkeypatch):
    target = tmp_path / 'x.parquet'
    real_replace = tables.os.replace
    calls = []

    def held_twice(src, dst):
        calls.append(dst)
        if len(calls) < 3:
            raise PermissionError('held by another process')
        real_replace(src, dst)

    monkeypatch.setattr(tables, 'REPLACE_BACKOFF_S', (0.0, 0.0, 0.0))
    monkeypatch.setattr(tables.os, 'replace', held_twice)
    tables.to_parquet(_frame(), target)
    assert len(calls) == 3
    assert target.exists()


def test_a_target_held_throughout_raises_and_cleans_up(tmp_path, monkeypatch):
    target = tmp_path / 'x.parquet'

    def always_held(src, dst):
        raise PermissionError('held by another process')

    monkeypatch.setattr(tables, 'REPLACE_BACKOFF_S', (0.0, 0.0))
    monkeypatch.setattr(tables.os, 'replace', always_held)
    with pytest.raises(PermissionError, match='x.parquet'):
        tables.to_parquet(_frame(), target)
    assert list(tmp_path.iterdir()) == []


def test_the_temporary_name_escapes_parquet_globs(tmp_path):
    temp = tables._temp_path(tmp_path / 'x.parquet')
    assert temp.parent == tmp_path
    assert not temp.name.endswith('.parquet')


def test_save_parquet_writes_the_pair_and_no_temps(tmp_path):
    gdf = gpd.GeoDataFrame(
        {'value': [1.0, 2.0]},
        geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2)],
        crs='EPSG:4326',
    )
    tables.save_parquet(gdf, tmp_path / 'x.parquet')
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        'x.parquet',
        'x_geo.parquet',
    ]
