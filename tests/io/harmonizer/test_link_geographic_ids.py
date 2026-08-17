"""Tests for the `link_geographic_ids` harmonize step."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

import openplaces.io.harmonizer.spine as spine_module
from openplaces.io.harmonizer import HarmonizeState

# Two disjoint reference squares (A: x in [0,1], B: x in [2,3]) with a gap
# between them (x in [1,2]) representing an uncovered sliver -- the same
# shape as a boundary/rounding mismatch between a spine geometry and a real
# reference layer.
_REF = gpd.GeoDataFrame(
    geometry=[box(0, 0, 1, 1), box(2, 0, 3, 1)],
    crs='epsg:4326',
    index=pd.Index(['A', 'B'], name='the_id'),
)


def _spine(rows: dict[str, tuple]) -> gpd.GeoDataFrame:
    """rows: {parcel_id: (geometry, lat, long)}"""
    ids = list(rows)
    geoms = [v[0] for v in rows.values()]
    lats = [v[1] for v in rows.values()]
    longs = [v[2] for v in rows.values()]
    return gpd.GeoDataFrame(
        {'lat': lats, 'long': longs},
        geometry=geoms,
        crs='epsg:4326',
        index=pd.Index(ids, name='parcel_id'),
    )


def _state(spine, admin_id='US-NC-CE'):
    return HarmonizeState(
        recipe={'entity': {'entity_type': 'parcel'}},
        admin_id=admin_id,
        verbose=False,
        timer=None,
        spine=spine,
    )


def _links():
    return [{'recipe_id': 'US_tile-census-2025_tract', 'output_column': 'the_id'}]


def test_pass1_within_match(monkeypatch):
    monkeypatch.setattr(spine_module, 'get_entities', lambda *a, **k: _REF)
    spine = _spine({'P1': (box(0.2, 0.2, 0.4, 0.4), 0.3, 0.3)})
    state = spine_module.link_geographic_ids(_state(spine), links=_links())
    assert state.spine.loc['P1', 'the_id'] == 'A'


def test_pass2_overlay_fallback_when_centroid_misses(monkeypatch):
    monkeypatch.setattr(spine_module, 'get_entities', lambda *a, **k: _REF)
    # Spans the gap (x: 0.9-1.5), centroid at x=1.2 falls in the gap (misses
    # both references under 'within'), but the only real overlap is with A.
    spine = _spine({'P2': (box(0.9, 0.2, 1.5, 0.4), 0.3, 1.2)})
    state = spine_module.link_geographic_ids(_state(spine), links=_links())
    assert state.spine.loc['P2', 'the_id'] == 'A'


def test_unmatched_row_stays_null(monkeypatch):
    monkeypatch.setattr(spine_module, 'get_entities', lambda *a, **k: _REF)
    # Entirely inside the gap: no overlap with either reference polygon.
    spine = _spine({'P3': (box(1.2, 0.2, 1.4, 0.4), 0.3, 1.3)})
    state = spine_module.link_geographic_ids(_state(spine), links=_links())
    assert pd.isna(state.spine.loc['P3', 'the_id'])


def test_conflicting_existing_value_warns_then_overwrites(monkeypatch):
    monkeypatch.setattr(spine_module, 'get_entities', lambda *a, **k: _REF)
    spine = _spine({'P4': (box(2.2, 0.2, 2.4, 0.4), 0.3, 2.3)})
    spine['the_id'] = 'WRONG'
    with pytest.warns(UserWarning, match='disagree'):
        state = spine_module.link_geographic_ids(_state(spine), links=_links())
    assert state.spine.loc['P4', 'the_id'] == 'B'


def test_inherit_from_unanimous_group_skips_direct_join(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError('should not need the direct reference for P1')

    monkeypatch.setattr(spine_module, 'get_admin', _boom)
    footprints = pd.DataFrame({'parcel_id': ['P1', 'P1'], 'admin4_id': ['A', 'A']})
    monkeypatch.setattr(spine_module, 'get_entities', lambda *a, **k: footprints)

    # No real geometry needed for P1 -- it should be fully resolved by
    # inheritance, never touching get_admin.
    spine = _spine({'P1': (box(50, 50, 51, 51), 50.5, 50.5)})
    state = spine_module.link_geographic_ids(
        _state(spine),
        links=[{'admin_level': 4, 'output_column': 'admin4_id'}],
        inherit_from={'recipe_id': 'US_footprint-spine-2026', 'group_by': 'parcel_id'},
    )
    assert state.spine.loc['P1', 'admin4_id'] == 'A'


def test_inherit_from_disagreeing_group_falls_back_and_is_reported(monkeypatch):
    monkeypatch.setattr(spine_module, 'get_admin', lambda *a, **k: _REF)
    footprints = pd.DataFrame({'parcel_id': ['P2', 'P2'], 'admin4_id': ['A', 'B']})
    monkeypatch.setattr(spine_module, 'get_entities', lambda *a, **k: footprints)

    spine = _spine({'P2': (box(0.2, 0.2, 0.4, 0.4), 0.3, 0.3)})
    state = spine_module.link_geographic_ids(
        _state(spine),
        links=[{'admin_level': 4, 'output_column': 'admin4_id'}],
        inherit_from={'recipe_id': 'US_footprint-spine-2026', 'group_by': 'parcel_id'},
    )
    # Falls back to the direct join, which correctly resolves P2 against A.
    assert state.spine.loc['P2', 'admin4_id'] == 'A'
    conflicts = state.metadata['geographic_id_inheritance_conflicts']
    assert conflicts == [
        {'output_column': 'admin4_id', 'group_key': 'P2', 'values': ['A', 'B']}
    ]


def test_noop_when_spine_is_none():
    state = spine_module.link_geographic_ids(_state(None), links=_links())
    assert state.spine is None


def test_noop_when_no_links():
    spine = _spine({'P1': (box(0.2, 0.2, 0.4, 0.4), 0.3, 0.3)})
    state = spine_module.link_geographic_ids(_state(spine), links=None)
    assert 'the_id' not in state.spine.columns
