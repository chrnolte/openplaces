"""summarize_footprint_morphology on a building spine cut at parcel lines.

plans/buildings-from-parcel-pieces.md, step 3: the per-parcel building
counts and areas come from the pieces, and the span of the drawn
outline (`max_parcels_per_footprint`, read by the parcel curate's
Townhome rule) stays the outline's. The building rows carry what the
building spine's parcel attribution writes from the cut crosswalk
(`_attribute_polygon_reference`): the dominant parcel, every linked
parcel, the dominant overlap and the count of linked parcels. Every
geometry, id and value is fabricated
(docs/5_contribute/no-personal-data.rst).
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from shapely.geometry import box

import openplaces.io.harmonizer.attributes as attrs
import openplaces.io.readers as readers
from openplaces.io.harmonizer import HarmonizeState
from tests.io.harmonizer.test_parcel_pieces import (
    CRS,
    _lots,
    _row_of_four,
    _spine,
    _split,
)


def _building_spine(spine, flat):
    """The building spine's parcel columns, from the cut crosswalk."""
    kept = flat[flat['link'].notna()].sort_values(
        'area_intersection_m2', ascending=False, kind='stable'
    )
    by_building = kept.groupby('building_id', sort=False)
    dominant = kept.drop_duplicates('building_id').set_index('building_id')
    rows = spine.assign(
        parcel_id=dominant['parcel_id'].reindex(spine.index),
        parcel_id_all=by_building['parcel_id']
        .agg(lambda s: '|'.join(s.astype(str).unique()))
        .reindex(spine.index),
        area_intersection_m2_parcel=dominant['area_intersection_m2'].reindex(
            spine.index
        ),
        n_parcels_per_building=by_building.size().reindex(spine.index, fill_value=0),
        priority_on_parcel='primary',
        n_dwellings_overture=1,
    )
    return gpd.GeoDataFrame(rows.reset_index(), geometry='geometry', crs=CRS)


def _morphology(monkeypatch, cut):
    spine, lots, points, values = _row_of_four()
    spine, flat, stats = _split(spine, lots, points if cut else [], values)
    assert stats['outlines_cut'] == (1 if cut else 0)
    buildings = _building_spine(spine, flat)
    monkeypatch.setattr(readers, 'get_entities', lambda *a, **k: buildings)
    state = HarmonizeState(
        recipe={},
        admin_id='US-XX-YY',
        verbose=False,
        timer=None,
        spine=gpd.GeoDataFrame(geometry=lots.geometry, crs=CRS),
    )
    out = attrs.summarize_footprint_morphology(state, footprint_recipe_id='bldg')
    return out.spine


def test_counts_and_areas_per_parcel_come_from_the_pieces(monkeypatch):
    uncut = _morphology(monkeypatch, cut=False)
    cut = _morphology(monkeypatch, cut=True)
    # One building on each lot either way.
    assert cut['n_buildings_per_parcel'].tolist() == [1, 1, 1, 1]
    assert uncut['n_buildings_per_parcel'].tolist() == [1, 1, 1, 1]
    # A lot's building area is now its own unit's, not the whole row's.
    assert np.allclose(cut['footprint_area_m2_dominant'], 100.0, rtol=1e-6)
    assert np.allclose(uncut['footprint_area_m2_dominant'], 400.0, rtol=1e-6)
    assert np.allclose(cut['max_footprint_area_m2'], 100.0, rtol=1e-6)
    # The clipped area is the same polygon area on each lot.
    assert np.allclose(
        cut['footprint_area_m2_in_parcel'], uncut['footprint_area_m2_in_parcel']
    )


def test_the_outline_span_stays_the_outlines(monkeypatch):
    """The Townhome rule reads max_parcels_per_footprint >= 2: a row of
    four is drawn over four lots whether or not it is cut."""
    uncut = _morphology(monkeypatch, cut=False)
    cut = _morphology(monkeypatch, cut=True)
    assert uncut['max_parcels_per_footprint'].tolist() == [4, 4, 4, 4]
    assert cut['max_parcels_per_footprint'].tolist() == [4, 4, 4, 4]


def test_an_uncut_building_keeps_its_own_span(monkeypatch):
    """Two outlines, one over two lots, one on a lot of its own: the
    span is the outline's count of linked parcels, as before."""
    spine = _spine(a=box(0, 0, 20, 10), b=box(30, 0, 40, 10))
    lots = _lots(L1=box(0, -5, 10, 15), L2=box(10, -6, 20, 15), L3=box(25, -5, 45, 15))
    spine, flat, _ = _split(spine, lots)
    buildings = _building_spine(spine, flat)
    monkeypatch.setattr(readers, 'get_entities', lambda *a, **k: buildings)
    state = HarmonizeState(
        recipe={},
        admin_id='US-XX-YY',
        verbose=False,
        timer=None,
        spine=gpd.GeoDataFrame(geometry=lots.geometry, crs=CRS),
    )
    out = attrs.summarize_footprint_morphology(state, footprint_recipe_id='bldg')
    assert out.spine['max_parcels_per_footprint'].to_dict() == {
        'L1': 2,
        'L2': 2,
        'L3': 1,
    }
