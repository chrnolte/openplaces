"""Evidence-based choice between overlapping outlines of two sources."""

from __future__ import annotations

import geopandas as gpd
from shapely.geometry import box

from openplaces.io.harmonizer.spine_resolution import (
    FRAGMENT_REPLACED,
    choose_between_sources,
)

CRS = 'epsg:6933'  # meters, so box sizes read as m


def _gdf(geoms, ids):
    return gpd.GeoDataFrame({'fid': ids}, geometry=geoms, crs=CRS).set_index('fid')


def _lots(*geoms):
    return gpd.GeoSeries(list(geoms), crs=CRS)


def test_a_small_ranked_piece_gives_way_to_the_building_around_it():
    spine = _gdf([box(10, 10, 14, 14)], ['ROOF'])  # 16 m2
    candidate = _gdf([box(0, 0, 60, 60)], ['BUILDING'])  # 3,600 m2
    lots = _lots(box(-5, -5, 65, 65))
    drop, taken = choose_between_sources(spine, candidate, lots)
    assert list(drop) == ['ROOF']
    assert taken.to_dict() == {'BUILDING': FRAGMENT_REPLACED}


def test_a_larger_outline_over_more_lots_than_the_pieces_is_refused():
    # The later outline spills over a lot line the ranked piece does not
    # cross: more likely a merged block itself, so the ranking stands.
    spine = _gdf([box(10, 10, 14, 14)], ['PIECE'])
    candidate = _gdf([box(0, 0, 60, 60)], ['SPILL'])
    lots = _lots(box(-5, -5, 30, 65), box(30, -5, 65, 65))
    drop, taken = choose_between_sources(spine, candidate, lots)
    assert drop.empty and taken.empty


def test_one_ranked_outline_over_two_lots_is_not_split():
    # A building may span lots, and a townhome row is one footprint:
    # per-lot pieces from a later source do not replace it.
    spine = _gdf([box(0, 0, 40, 20)], ['ROW'])
    candidate = _gdf([box(1, 1, 19, 19), box(21, 1, 39, 19)], ['WEST', 'EAST'])
    lots = _lots(box(-5, -5, 20, 25), box(20, -5, 45, 25))
    drop, taken = choose_between_sources(spine, candidate, lots)
    assert drop.empty and taken.empty


def test_matching_outlines_keep_the_ranked_source():
    spine = _gdf([box(0, 0, 20, 20)], ['RANKED'])
    candidate = _gdf([box(1, 1, 21, 21)], ['LATER'])
    drop, taken = choose_between_sources(spine, candidate, _lots(box(-5, -5, 30, 30)))
    assert drop.empty and taken.empty
