"""Writer guards: footer metadata must not change a file's schema.

`to_parquet` writes through pyarrow when footer metadata is asked for,
and pyarrow takes none of pandas' keywords. `to_csv` drops geometry,
which is not always the column named 'geometry'.
"""

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import pytest
from shapely.geometry import box

from openplaces.io import to_csv, to_parquet


def _df():
    # A named, non-range index: pyarrow stores a default RangeIndex in
    # metadata alone, so only a real index exposes a schema difference.
    return pd.DataFrame({'value': [1, 2]}, index=pd.Index(['a', 'b'], name='geo_id'))


def test_file_metadata_does_not_change_the_written_schema(tmp_path):
    # index=False was dropped on the metadata path, so the same call
    # wrote an extra index level column.
    plain = tmp_path / 'plain.parquet'
    stamped = tmp_path / 'stamped.parquet'

    to_parquet(_df(), plain, index=False)
    to_parquet(_df(), stamped, index=False, file_metadata={'a': 'b'})

    assert (
        pq.ParquetFile(stamped).schema_arrow.names
        == pq.ParquetFile(plain).schema_arrow.names
    )
    assert pq.read_metadata(stamped).metadata[b'a'] == b'b'


def test_file_metadata_keeps_the_index_when_asked(tmp_path):
    path = tmp_path / 'indexed.parquet'
    to_parquet(_df(), path, index=True, file_metadata={'a': 'b'})
    assert 'geo_id' in pq.ParquetFile(path).schema_arrow.names


def test_file_metadata_rejects_a_keyword_it_cannot_honor(tmp_path):
    with pytest.raises(TypeError, match='compression'):
        to_parquet(
            _df(),
            tmp_path / 'out.parquet',
            compression='snappy',
            file_metadata={'a': 'b'},
        )


def test_to_csv_drops_a_geometry_column_under_another_name(tmp_path):
    gdf = gpd.GeoDataFrame(
        {'value': [1], 'shape': [box(0, 0, 1, 1)]}, geometry='shape', crs='EPSG:4326'
    )
    assert gdf.active_geometry_name == 'shape'

    path = tmp_path / 'out.csv'
    to_csv(gdf, path)

    assert pd.read_csv(path).columns.tolist() == ['value']


def test_to_csv_still_drops_a_plain_geometry_column(tmp_path):
    gdf = gpd.GeoDataFrame({'value': [1]}, geometry=[box(0, 0, 1, 1)], crs='EPSG:4326')
    path = tmp_path / 'out.csv'
    to_csv(gdf, path)
    assert pd.read_csv(path).columns.tolist() == ['value']
