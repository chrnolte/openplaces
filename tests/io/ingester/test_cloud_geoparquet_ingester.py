import io

import geopandas as gpd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
import shapely

from openplaces.io.ingester import cloud_geoparquet_ingester as cgi
from openplaces.io.ingester.cloud_geoparquet_ingester import (
    MIN_TILE_SIZE_DEG,
    tile_bounds,
    tile_ids_for_admin,
    tile_str,
)


def test_tile_str_integer_format_has_no_leading_zeros():
    assert tile_str(35, -78, tile_size_deg=1.0) == 'lat+35_lon-078'
    assert tile_str(-2, 100, tile_size_deg=1.0) == 'lat-02_lon+100'


@pytest.mark.parametrize('tile_size_deg', [1.0, 0.5, 0.1, 0.09])
def test_tile_str_tile_bounds_round_trip(tile_size_deg):
    lat_deg, lon_deg = 35 * tile_size_deg, -78 * tile_size_deg
    tile_id = tile_str(lat_deg, lon_deg, tile_size_deg)

    minx, miny, maxx, maxy = tile_bounds(tile_id, tile_size_deg)

    assert minx == pytest.approx(lon_deg)
    assert miny == pytest.approx(lat_deg)
    assert maxx == pytest.approx(lon_deg + tile_size_deg)
    assert maxy == pytest.approx(lat_deg + tile_size_deg)


def test_finer_tile_size_deg_uses_more_precision_than_integer_size():
    coarse_id = tile_str(35, -78, tile_size_deg=1.0)
    fine_id = tile_str(35.4, -78.3, tile_size_deg=0.1)

    assert coarse_id == 'lat+35_lon-078'
    assert fine_id == 'lat+354_lon-0783'


def test_tile_size_deg_below_minimum_raises():
    with pytest.raises(ValueError):
        tile_str(0, 0, tile_size_deg=MIN_TILE_SIZE_DEG / 10)
    with pytest.raises(ValueError):
        tile_ids_for_admin('US-MA-MI', tile_size_deg=MIN_TILE_SIZE_DEG / 10)


def test_tile_size_deg_at_minimum_is_accepted():
    lat_deg, lon_deg = 3 * MIN_TILE_SIZE_DEG, -7 * MIN_TILE_SIZE_DEG
    tile_id = tile_str(lat_deg, lon_deg, tile_size_deg=MIN_TILE_SIZE_DEG)

    minx, miny, _, _ = tile_bounds(tile_id, MIN_TILE_SIZE_DEG)

    assert minx == pytest.approx(lon_deg)
    assert miny == pytest.approx(lat_deg)


def test_tile_ids_for_admin_excludes_tiles_outside_polygon(monkeypatch):
    # Coverage inset within 3 of a 2x2 grid's 4 candidate tiles, each piece
    # kept strictly inside its cell (not touching cell boundaries). The 4th
    # (top-right) cell has zero overlap with the polygon -- not even a
    # shared boundary point -- so it must be excluded, unlike a plain
    # bbox-vs-bbox test which would include all 4.
    covered = shapely.union_all(
        [
            shapely.box(20.1, 10.1, 20.9, 10.9),  # inside tile (lon=20, lat=10)
            shapely.box(21.1, 10.1, 21.9, 10.9),  # inside tile (lon=21, lat=10)
            shapely.box(20.1, 11.1, 20.9, 11.9),  # inside tile (lon=20, lat=11)
        ]
    )
    fake_admin = gpd.GeoDataFrame(geometry=[covered], crs='EPSG:4326')

    def fake_get_admin(admin_id, geom=True):
        return fake_admin

    monkeypatch.setattr(cgi, 'get_admin', fake_get_admin)

    tiles = cgi.tile_ids_for_admin('TEST-ADMIN', tile_size_deg=1.0)

    assert set(tiles) == {
        tile_str(10, 20, 1.0),
        tile_str(10, 21, 1.0),
        tile_str(11, 20, 1.0),
    }
    assert tile_str(11, 21, 1.0) not in tiles


def test_list_s3_parquet_files_caches_across_calls(monkeypatch):
    monkeypatch.setattr(cgi, '_s3_listing_cache', {})

    xml_body = b"""<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Contents><Key>prefix/a.parquet</Key><Size>111</Size></Contents>
  <Contents><Key>prefix/b.parquet</Key><Size>222</Size></Contents>
  <Contents><Key>prefix/readme.txt</Key><Size>5</Size></Contents>
</ListBucketResult>"""

    class FakeResponse:
        content = xml_body

        def raise_for_status(self):
            pass

    calls = []

    def fake_get(url):
        calls.append(url)
        return FakeResponse()

    monkeypatch.setattr(cgi.requests, 'get', fake_get)

    first = cgi._list_s3_parquet_files('https://bucket.s3.amazonaws.com', 'prefix/')
    second = cgi._list_s3_parquet_files('https://bucket.s3.amazonaws.com', 'prefix/')

    assert first == [('prefix/a.parquet', 111), ('prefix/b.parquet', 222)]
    assert second == first
    assert len(calls) == 1


def test_get_s3_parquet_fragment_reuses_cached_fragment(monkeypatch, tmp_path):
    # A cache hit must skip re-reading the source entirely (footer parsing
    # is the expensive, redundant-per-tile part this cache exists to avoid),
    # so this returns the identical Fragment object rather than an equal one.
    monkeypatch.setattr(cgi, '_s3_fragment_cache', {})
    monkeypatch.setattr(cgi, '_shared_s3_session', None)

    table = pa.table({'x': list(range(100))})
    path = tmp_path / 'test.parquet'
    pq.write_table(table, path, row_group_size=10)
    data = path.read_bytes()

    monkeypatch.setattr(cgi, '_HTTPRangeFile', lambda url, session: io.BytesIO(data))

    url = 'https://bucket.s3.amazonaws.com/prefix/file.parquet'
    fragment_first = cgi._get_s3_parquet_fragment(url, size=len(data))
    fragment_second = cgi._get_s3_parquet_fragment(url, size=len(data))

    assert fragment_first is fragment_second

    result_low = fragment_first.to_table(filter=pc.field('x') < 5)
    result_high = fragment_first.to_table(filter=pc.field('x') >= 95)

    assert sorted(result_low['x'].to_pylist()) == [0, 1, 2, 3, 4]
    assert sorted(result_high['x'].to_pylist()) == [95, 96, 97, 98, 99]
