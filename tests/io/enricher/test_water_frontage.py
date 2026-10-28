"""Shoreline allocation in the lake enrich step, on synthetic geometry.

A square lake 100 m on a side, lots along its south shore. The rules
the allocation promises: a first-row lot holds the shoreline in front
of it, a second-row lot behind it holds none, no shoreline meter is
counted twice, and a lot drawn into the water still wins its shore.
"""

import geopandas as gpd
import pytest
import shapely

from openplaces.io.enricher.water import (
    _access_across_open_land,
    _access_through_shore_parcels,
    _allocate_frontage,
    _shore_land,
    _waterbody_holders,
)
from openplaces.recipe import get_recipe_dependencies

CRS = 'EPSG:32617'


def _lake():
    return gpd.GeoDataFrame(
        {'is_lake': [True]},
        geometry=[shapely.box(0, 0, 100, 100)],
        index=['lake'],
        crs=CRS,
    ).rename_axis('waterbody_id_nhd')


def _parcels(boxes):
    return gpd.GeoDataFrame(
        geometry=[shapely.box(*b) for b in boxes.values()],
        index=list(boxes),
        crs=CRS,
    )


def test_first_row_takes_the_shore_and_second_row_none():
    parcels = _parcels(
        {
            'front': (0, -30, 100, -2),  # 2 m gap to the shore
            'back': (0, -60, 100, -32),  # behind it, 32 m from the shore
        }
    )
    frontage = _allocate_frontage(_lake(), parcels, 15, 2, None)

    # 100 m in front, plus the side shore within the tolerance of each
    # corner, which no other parcel holds in this scene (at most 15 m a
    # side: a known, bounded property of nearest-parcel allocation)
    assert 100 <= frontage[('front', 'lake')] <= 100 + 2 * 15
    assert 'back' not in frontage.index.get_level_values(0)


def test_shoreline_is_never_counted_twice():
    parcels = _parcels({f'lot{i}': (i * 20, -20, i * 20 + 20, 0) for i in range(5)})
    frontage = _allocate_frontage(_lake(), parcels, 15, 2, None)

    shoreline = _lake().geometry.iloc[0].length
    assert frontage.sum() <= shoreline + 1e-6
    per_lot = frontage.groupby(level=0).sum()
    # Interior lots hold exactly the 20 m in front of them; the two end
    # lots also take up to 15 m of the side shore no parcel holds
    assert per_lot[['lot1', 'lot2', 'lot3']].between(19, 21).all()
    assert per_lot[['lot0', 'lot4']].between(19, 20 + 15 + 1).all()


def test_a_lot_drawn_into_the_water_wins_its_shore():
    parcels = _parcels(
        {
            'into_water': (0, -20, 50, 10),  # reaches 10 m into the lake
            'neighbor': (50, -20, 100, -5),
        }
    )
    frontage = _allocate_frontage(_lake(), parcels, 15, 2, None)

    assert frontage[('into_water', 'lake')] > 45
    assert frontage[('neighbor', 'lake')] > 45


def test_beyond_the_tolerance_there_is_no_frontage():
    parcels = _parcels({'far': (0, -60, 100, -20)})  # 20 m from the shore

    assert _allocate_frontage(_lake(), parcels, 15, 2, None).empty


def _allocate_on_land(parcels, tolerance=15, buildings=None, yield_holders=True):
    """The step's order: land parts, lake holders, then allocation."""
    land, share = _shore_land(
        parcels, _lake(), sliver_m=5, sliver_water_share=0.5, buildings=buildings
    )
    recipients = gpd.GeoDataFrame(geometry=land, crs=CRS)
    holders = _waterbody_holders(parcels, _lake(), 0.5) if yield_holders else None
    frontage = _allocate_frontage(
        _lake(), recipients, tolerance, 2, None, holders=holders, buildings=buildings
    )
    return frontage, share


def _by_parcel(frontage):
    return frontage.groupby(level=0).sum() if len(frontage) else frontage


def test_a_lake_drawn_as_a_parcel_claims_no_shore():
    parcels = _parcels({'lake_parcel': (0, 0, 100, 100), 'lot': (0, -20, 100, 0)})
    frontage, share = _allocate_on_land(parcels)
    per = _by_parcel(frontage)

    assert share['lake_parcel'] == pytest.approx(1)
    assert 'lake_parcel' not in per.index
    assert per['lot'] >= 100


def test_a_lake_parcel_drawn_past_the_shore_loses_its_sliver():
    """The lake parcel extends 3 m past the NHD shoreline all round; the
    lot starts 3 m back from the shore. Without the opening, the 3 m ring
    of "land" would sit at distance 0 from every segment."""
    parcels = _parcels({'lake_parcel': (-3, -3, 103, 103), 'lot': (0, -30, 100, -3)})
    frontage, _ = _allocate_on_land(parcels)
    per = _by_parcel(frontage)

    assert per.get('lake_parcel', 0) == 0
    assert per['lot'] >= 100


def test_a_lake_parcel_with_a_wide_rim_yields_to_the_lots():
    """The lake parcel runs 15 m past the NHD shoreline all round, wider
    than the opening removes (seen around large Florida lakes); the lot
    starts behind that rim. The lake parcel holds the lake, so it yields
    the south shore to the lot and keeps the shore no lot reaches."""
    parcels = _parcels({'lake_parcel': (-15, -15, 115, 115), 'lot': (0, -40, 100, -15)})
    unyielding, _ = _allocate_on_land(parcels, tolerance=20, yield_holders=False)
    frontage, _ = _allocate_on_land(parcels, tolerance=20)

    assert _by_parcel(unyielding).get('lot', 0) == 0  # the failure, shown
    assert _by_parcel(frontage)['lot'] >= 100
    assert 'lake_parcel' in _by_parcel(frontage).index  # north, east, west


def test_a_lake_parcel_keeps_the_shore_by_its_house():
    """As above, but the lake parcel's house stands on its south rim: the
    shore within 20 m of the house stays with the lake parcel."""
    parcels = _parcels({'lake_parcel': (-15, -15, 115, 115), 'lot': (0, -40, 100, -15)})
    house = gpd.GeoSeries(
        [shapely.box(45, -12, 55, -3)], index=['lake_parcel'], crs=CRS
    )
    frontage, _ = _allocate_on_land(parcels, tolerance=20, buildings=house)
    south = frontage.groupby(level=0).sum()

    assert 'lot' in south.index
    assert south['lot'] < 100 - 30  # the stretch by the house is gone


def test_a_lake_parcel_with_a_house_strip_keeps_its_shore():
    """One parcel holds the lake and a 40 m strip of dry land on the west;
    the south shore belongs to a separate lot."""
    parcels = _parcels({'lake_and_house': (-40, 0, 100, 100), 'lot': (0, -30, 100, 0)})
    frontage, _ = _allocate_on_land(parcels)
    per = _by_parcel(frontage)

    assert 95 <= per['lake_and_house'] <= 115  # the west shore
    assert per['lot'] >= 95  # the south shore


def test_a_lake_bed_subdivided_among_lots():
    """Each lot runs from dry land to the middle of the lake."""
    parcels = _parcels(
        {'west_half': (-30, -30, 50, 50), 'east_half': (50, -30, 130, 50)}
    )
    frontage, share = _allocate_on_land(parcels)
    per = _by_parcel(frontage)

    assert share['west_half'] < 0.5  # mostly land: no opening applied
    assert per['west_half'] > 90 and per['east_half'] > 90


def test_the_enrich_recipe_declares_its_inputs_to_the_dag():
    """Inputs named in step parameters must be visible as edges, or the
    enrich job is neither ordered after the NHD ingest nor rerun when it
    changes."""
    edges = get_recipe_dependencies('US_parcel_water-waterbody-usgs-nhdhr2023')
    upstream = {e.upstream_recipe_id for e in edges}

    assert {
        'US_water-waterbody-usgs-nhdhr2023',
        'US_water-waterbody-usgs-nhdhr2023_area',
        'US_building-spine-2026',
        'US_parcel-spine-2026',
    } <= upstream
    # Buildings, not their outlines, carry the parcel id (b3 step 7).
    assert 'US_footprint-spine-2026' not in upstream


def test_a_mostly_submerged_lot_keeps_its_dry_corner_shore():
    """74% water; the house sits on a 12 m x 20 m dry strip: riparian."""
    parcels = _parcels({'submerged_lot': (0, -12, 20, 30)})
    frontage, share = _allocate_on_land(parcels)

    assert share['submerged_lot'] > 0.5
    assert _by_parcel(frontage)['submerged_lot'] >= 19


def test_a_building_keeps_a_dry_strip_the_opening_would_remove():
    """86% water, a 5 m dry strip: narrower than the opening removes, but a
    building stands on it, so the strip and its shore stay."""
    parcels = _parcels({'submerged_lot': (0, -5, 20, 30)})
    house = gpd.GeoSeries(
        [shapely.box(5, -4.5, 15, -0.5)], index=['submerged_lot'], crs=CRS
    )
    without, _ = _allocate_on_land(parcels)
    with_house, _ = _allocate_on_land(parcels, buildings=house)

    assert 'submerged_lot' not in _by_parcel(without).index
    assert _by_parcel(with_house)['submerged_lot'] >= 15


def _segments(parcels, tolerance=15):
    _, segments = _allocate_frontage(
        _lake(), parcels, tolerance, 2, None, return_segments=True
    )
    return segments


def test_a_lot_behind_a_common_strip_reaches_the_water_through_it():
    """A 6 m common-area strip along the south shore; two lots behind it
    touch it. The strip holds the frontage; each lot gets access to the
    shore in front of it, through the strip."""
    parcels = _parcels(
        {
            'strip': (0, -6, 100, 0),
            'west_lot': (0, -40, 50, -6),
            'east_lot': (50, -40, 100, -6),
            'back_lot': (0, -80, 100, -40),  # touches the lots, not the strip
        }
    )
    access = _access_through_shore_parcels(_segments(parcels), parcels, 150)
    per = access.groupby('parcel')['seg'].size() * 2  # 2 m segments

    assert set(access['via']) == {'strip'}
    assert per['west_lot'] >= 45 and per['east_lot'] >= 45
    assert 'back_lot' not in per.index


def test_a_lot_across_an_unparceled_beach_reaches_the_shore():
    """A 35 m beach with no parcel lies between the lot and the water;
    the frontage tolerance (15 m) does not reach across it."""
    parcels = _parcels({'lot': (0, -60, 100, -35)})
    segments = _segments(parcels)
    across = _access_across_open_land(segments, parcels, 100)

    assert segments['parcel'].isna().all()  # no frontage
    assert across['parcel'].eq('lot').all()
    assert across['distance'].between(35, 100).all()
    assert len(across) * 2 >= 100


def test_a_shore_road_blocks_access():
    """A road runs along the shore, 15 m out, with no parcel of its own;
    the lot behind it does not reach the water across open land, and a
    road running across a common strip passes no access on either."""
    road = gpd.GeoSeries([shapely.LineString([(-50, -15), (150, -15)])], crs=CRS)
    open_land = _parcels({'lot': (0, -60, 100, -35)})
    across = _access_across_open_land(_segments(open_land), open_land, 100, roads=road)
    assert across.empty

    strip = _parcels({'strip': (0, -30, 100, 0), 'lot': (0, -60, 100, -30)})
    through = _access_through_shore_parcels(_segments(strip), strip, 150, roads=road)
    assert 'lot' not in set(through['parcel'])


def test_open_land_access_is_blocked_by_another_parcel():
    """A parcel lies across the path, 20 m off the shore: the lot behind
    it is not reached across open land."""
    parcels = _parcels({'blocker': (0, -30, 100, -20), 'lot': (0, -70, 100, -40)})
    across = _access_across_open_land(_segments(parcels), parcels, 100)

    assert 'lot' not in set(across['parcel'])
