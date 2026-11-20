"""Ids of cut pieces: a centroid clash is broken by geometry, never a number."""

import geopandas as gpd
import pytest
from shapely.geometry import box

from openplaces.geo.ids import distinct_centroid_codes, get_openlocationcodes

OUTER = box(-77.4000, 34.7500, -77.3996, 34.7504)
INNER = box(-77.3999, 34.7501, -77.3997, 34.7503)


def _frame(geoms):
    return gpd.GeoDataFrame(geometry=list(geoms), crs=4326)


def test_a_ring_around_a_piece_of_the_same_centroid_gets_its_own_id():
    ring = OUTER.difference(INNER)
    ids = distinct_centroid_codes(_frame([ring, INNER]), set())
    assert ids.nunique() == 2


def test_a_piece_whose_finest_centroid_code_is_taken_gets_a_corner_code():
    taken = set(
        get_openlocationcodes(_frame([INNER]), codelength=c, handle_duplicates=False)[0]
        for c in (11, 13, 15)
    )
    ids = distinct_centroid_codes(_frame([INNER]), taken)
    assert ids.iloc[0] not in taken


def test_pieces_without_a_clash_keep_their_eleven_character_codes():
    far = box(-77.3000, 34.7500, -77.2996, 34.7504)
    ids = distinct_centroid_codes(_frame([INNER, far]), set())
    expected = get_openlocationcodes(_frame([INNER, far]), handle_duplicates=False)
    assert list(ids) == list(expected)


def test_the_same_polygon_twice_raises_and_names_the_rows():
    with pytest.raises(ValueError, match='2 rows'):
        distinct_centroid_codes(_frame([INNER, INNER]), set())
