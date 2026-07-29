import pytest

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
