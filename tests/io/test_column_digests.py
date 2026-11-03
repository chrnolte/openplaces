"""Column digests in table footers, and the log of what a job reads."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from openplaces.io import tables


@pytest.fixture
def frame():
    return pd.DataFrame({'a': [1, 2, 3], 'b': ['x', None, 'z']})


@pytest.fixture
def read_log():
    log = tables.start_read_log()
    yield log
    tables.stop_read_log()


def test_a_write_records_its_column_digests(tmp_path, frame):
    path = tmp_path / 't.parquet'
    tables.to_parquet(frame, path, index=False)
    recorded = tables.recorded_digests(path)
    assert set(recorded) == {'a', 'b'}
    assert recorded == tables.frame_digests(frame, index=False)


def test_a_digest_tracks_values_order_and_dtype(frame):
    base = tables.column_digest(frame['a'])
    assert tables.column_digest(frame['a'].copy()) == base
    assert tables.column_digest(frame['a'][::-1].reset_index(drop=True)) != base
    assert tables.column_digest(frame['a'].astype(float)) != base
    assert tables.column_digest(pd.Series([1, 2, 4])) != base


def test_long_arrays_in_a_cell_are_digested_in_full():
    a = np.zeros(2000)
    b = a.copy()
    b[1000] = 1
    # str() of both arrays is the same abbreviated text
    assert str(a) == str(b)
    assert tables.column_digest(pd.Series([a])) != tables.column_digest(pd.Series([b]))


def test_footer_and_recomputed_digests_agree(tmp_path, frame):
    path = tmp_path / 't.parquet'
    tables.to_parquet(frame, path, index=False)
    recorded = tables.recorded_digests(path)
    # Written without io.tables, so the footer records nothing
    frame.to_parquet(tmp_path / 'plain.parquet', index=False)
    assert tables.recorded_digests(tmp_path / 'plain.parquet') == {}
    computed = tables.file_digests(tmp_path / 'plain.parquet', ['a', 'b'])
    assert computed == recorded


def test_a_geometry_sidecar_is_digested_through_its_attribute_file(tmp_path):
    gdf = gpd.GeoDataFrame(
        {'v': [1, 2]}, geometry=[Point(0, 0), Point(1, 1)], crs='EPSG:4326'
    )
    path = tmp_path / 'g.parquet'
    tables.save_parquet(gdf, path)
    sidecar = tmp_path / 'g_geo.parquet'
    recorded = tables.recorded_digests(sidecar)
    assert set(recorded) == {'_join_id', 'geometry'}
    moved = gdf.copy()
    moved.geometry = [Point(0, 0), Point(2, 2)]
    tables.save_parquet(moved, path)
    assert tables.recorded_digests(sidecar)['geometry'] != recorded['geometry']
    assert tables.file_digests(sidecar, ['geometry']) == {
        'geometry': tables.recorded_digests(sidecar)['geometry']
    }


def test_the_schema_digest_tracks_column_names(tmp_path, frame):
    path = tmp_path / 't.parquet'
    tables.to_parquet(frame, path, index=False)
    before = tables.file_digests(path, [tables.SCHEMA])
    tables.to_parquet(frame.assign(c=0), path, index=False)
    after = tables.file_digests(path, ['a', tables.SCHEMA])
    assert after[tables.SCHEMA] != before[tables.SCHEMA]
    assert after['a'] == tables.frame_digests(frame, index=False)['a']


def test_reads_are_logged_by_column(tmp_path, frame, read_log):
    path = tmp_path / 't.parquet'
    tables.to_parquet(frame, path, index=False)
    tables.read_parquet(path, columns=['a'])
    assert read_log == {str(path.resolve()): {'a'}}
    tables.read_parquet(path, columns='b', filters=[('a', '>', 1)])
    assert read_log[str(path.resolve())] == {'a', 'b'}
    tables.parquet_columns(path)
    assert tables.SCHEMA in read_log[str(path.resolve())]
    tables.read_parquet(path)
    assert read_log[str(path.resolve())] is None
    tables.read_parquet(path, columns=['a'])
    assert read_log[str(path.resolve())] is None


def test_a_geometry_read_logs_the_whole_sidecar(tmp_path, read_log):
    gdf = gpd.GeoDataFrame({'v': [1]}, geometry=[Point(0, 0)], crs='EPSG:4326')
    path = tmp_path / 'g.parquet'
    tables.save_parquet(gdf, path)
    tables.read_parquet(path, columns=['v'], geom=True)
    assert read_log[str(path.resolve())] == {'v', '_join_id'}
    assert read_log[str((tmp_path / 'g_geo.parquet').resolve())] is None


def test_reads_past_io_tables_are_logged(tmp_path, frame, read_log):
    import pyarrow.parquet as pq

    path = tmp_path / 't.parquet'
    tables.to_parquet(frame, path, index=False)
    key = str(path.resolve())
    pd.read_parquet(path, columns=['a'])
    assert read_log == {key: {'a'}}
    pq.read_metadata(path)
    assert read_log[key] == {'a', tables.SCHEMA, tables.METADATA}
    pq.ParquetFile(path)
    assert read_log[key] is None


def test_io_tables_own_reads_are_logged_once(tmp_path, frame, read_log):
    path = tmp_path / 't.parquet'
    tables.to_parquet(frame, path, index=False)
    tables.read_parquet(path, columns=['a'])
    tables.file_digests(path, ['b'])
    assert read_log == {str(path.resolve()): {'a'}}


def test_the_metadata_digest_ignores_the_build_commit(tmp_path, frame):
    path = tmp_path / 't.parquet'
    tables.to_parquet(frame, path, index=False, file_metadata={'k': 'v'})
    before = tables.file_digests(path, [tables.METADATA])
    tables.to_parquet(
        frame,
        path,
        index=False,
        file_metadata={'k': 'v', 'openplaces:recipe_roots': '[]'},
    )
    assert tables.file_digests(path, [tables.METADATA]) == before
    tables.to_parquet(frame, path, index=False, file_metadata={'k': 'w'})
    assert tables.file_digests(path, [tables.METADATA]) != before


def test_nothing_is_logged_outside_a_job(tmp_path, frame):
    path = tmp_path / 't.parquet'
    tables.to_parquet(frame, path, index=False)
    tables.read_parquet(path)
    assert tables._READ_LOG is None
