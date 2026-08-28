"""Tests for `resolve_spine`'s size-scaled exclusion buffer.

IoU alone is insensitive to size mismatch: a small lower-priority candidate
that only clips the edge of a large already-accepted footprint can have a
tiny IoU and still get merged in, producing a spurious duplicate next to the
large building. The buffer thresholds add an independent, size-scaled
keep-out zone around already-accepted footprints, gated by a minimum area so
only large (likely multi-unit) buildings get one.
"""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import box

import openplaces.io.harmonizer.spine as spine_mod
from openplaces.io.harmonizer import HarmonizeState

CRS = 'epsg:6933'  # equal-area, meters -- distances in the test are exact.

SOURCES = [
    {'recipe_id': 'big', 'label': 'big'},
    {'recipe_id': 'small', 'label': 'small'},
]


# resolve_spine drops rows whose representative point falls outside the
# admin unit it is harmonizing, so the boxes below are laid out around a
# point in central North Carolina (-78.0, 35.5 -> EPSG:6933) rather than
# at the origin, which is in the Gulf of Guinea. Offsets stay in meters,
# so every distance in this file is still exact.
ORIGIN_X, ORIGIN_Y = -7_525_930.0, 4_251_000.0


def _gdf(boxes: list[tuple[float, float, float, float]]) -> gpd.GeoDataFrame:
    geoms = [
        box(ORIGIN_X + x0, ORIGIN_Y + y0, ORIGIN_X + x1, ORIGIN_Y + y1)
        for x0, y0, x1, y1 in boxes
    ]
    gdf = gpd.GeoDataFrame({'geometry': geoms}, crs=CRS)
    gdf.index.name = 'footprint_id'
    return gdf


def _state() -> HarmonizeState:
    return HarmonizeState(
        recipe={'admin_id': 'US-NC'},
        admin_id='US-NC',
        verbose=False,
        timer=None,
        spine=None,
    )


def _resolve(monkeypatch, big, small, thresholds):
    gdfs = {'big': big, 'small': small}
    monkeypatch.setattr(spine_mod, 'get_entities', lambda rid, *a, **k: gdfs[rid])
    return spine_mod.resolve_spine(
        _state(), sources=SOURCES, thresholds=thresholds
    ).spine


# A 50x50m (2500 m2) "big" footprint at the reference point.
BIG = _gdf([(0, 0, 50, 50)])
# A small footprint 5m from the big footprint's edge -- no overlap (IoU 0).
NEARBY = _gdf([(55, 0, 60, 5)])
# A small footprint 150m away -- well outside any reasonable buffer.
DISTANT = _gdf([(200, 0, 205, 5)])

BUFFERED = {
    'buffer_min_area_m2': 1000.0,
    'buffer_base_m': 2.0,
    'buffer_area_scale': 0.5,
}


def test_default_thresholds_do_not_reject_nearby_non_overlapping_candidate(monkeypatch):
    out = _resolve(monkeypatch, BIG, NEARBY, thresholds={})
    assert len(out) == 2


def test_buffer_rejects_nearby_non_overlapping_candidate(monkeypatch):
    out = _resolve(monkeypatch, BIG, NEARBY, thresholds=BUFFERED)
    assert len(out) == 1
    assert out['geometry_source'].iloc[0] == 'big'


def test_buffer_does_not_reject_distant_candidate(monkeypatch):
    out = _resolve(monkeypatch, BIG, DISTANT, thresholds=BUFFERED)
    assert len(out) == 2


def test_buffer_min_area_gates_protection(monkeypatch):
    # buffer_min_area_m2 above the big footprint's own area (2500 m2): it
    # gets no keep-out zone, so the nearby candidate is not rejected.
    thresholds = {**BUFFERED, 'buffer_min_area_m2': 5000.0}
    out = _resolve(monkeypatch, BIG, NEARBY, thresholds=thresholds)
    assert len(out) == 2


def test_overlapping_candidate_still_rejected_by_iou_with_buffer_off(monkeypatch):
    overlapping = _gdf([(10, 10, 40, 40)])  # fully inside BIG
    out = _resolve(monkeypatch, BIG, overlapping, thresholds={})
    assert len(out) == 1


@pytest.mark.parametrize(
    ('field', 'value'),
    [('buffer_base_m', 10.0), ('buffer_area_scale', 2.0)],
)
def test_either_buffer_param_alone_enables_the_feature(monkeypatch, field, value):
    # A 5m-gap candidate: buffer_base_m=10 alone, or buffer_area_scale=2
    # alone (2*sqrt(2500)=100m), each independently reaches past the gap.
    thresholds = {'buffer_min_area_m2': 1000.0, field: value}
    out = _resolve(monkeypatch, BIG, NEARBY, thresholds=thresholds)
    assert len(out) == 1
