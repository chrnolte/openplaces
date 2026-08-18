"""Tests for `set_camera_pitch`'s persistence across camera interaction."""

from __future__ import annotations

import pytest
from lonboard import Map
from lonboard.view_state import GlobeViewState, MapViewState

from openplaces.viz.interactive import set_camera_pitch


def _map():
    m = Map([], height='100vh')
    m.view_state = MapViewState(longitude=-71.68, latitude=42.47, zoom=11)
    return m


def _drag(m, pitch, zoom=11.4, bearing=15.0):
    """What deck.gl echoes back after a camera move: the pose, no constraints."""
    m.view_state = {
        'longitude': -71.68,
        'latitude': 42.47,
        'zoom': zoom,
        'pitch': pitch,
        'bearing': bearing,
    }


def test_sets_initial_pitch_and_ceiling():
    m = set_camera_pitch(_map(), pitch=75, max_pitch=85)
    assert m.view_state.pitch == 75
    assert m.view_state.max_pitch == 85


def test_ceiling_survives_camera_interaction():
    m = set_camera_pitch(_map(), pitch=75, max_pitch=85)
    # Without the observer, the first echo would reset max_pitch to
    # MapViewState's dataclass default of 60 and cap the tilt for good.
    for pitch in (62.0, 80.0, 84.5):
        _drag(m, pitch)
        assert m.view_state.max_pitch == 85


def test_user_pose_is_not_overwritten():
    m = set_camera_pitch(_map(), pitch=75, max_pitch=85)
    _drag(m, 62.0, zoom=12.3, bearing=45.0)
    # Only the constraint is re-applied; the starting `pitch` is a one-shot
    # so re-asserting it would fight the user's own dragging.
    assert m.view_state.pitch == 62.0
    assert m.view_state.zoom == 12.3
    assert m.view_state.bearing == 45.0


def test_min_pitch_is_unmanaged_unless_requested():
    m = set_camera_pitch(_map(), max_pitch=85)
    _drag(m, 40.0)
    assert m.view_state.min_pitch == 0

    m2 = set_camera_pitch(_map(), max_pitch=85, min_pitch=10)
    _drag(m2, 40.0)
    assert m2.view_state.min_pitch == 10
    assert m2.view_state.max_pitch == 85


def test_pitch_none_leaves_current_tilt_alone():
    m = _map()
    m.view_state = MapViewState(longitude=0, latitude=0, zoom=5, pitch=33)
    set_camera_pitch(m, max_pitch=85)
    assert m.view_state.pitch == 33
    assert m.view_state.max_pitch == 85


def test_non_map_view_state_warns_and_is_left_alone():
    m = Map([], height='100vh')
    m.view_state = GlobeViewState(longitude=0, latitude=0, zoom=5)
    with pytest.warns(UserWarning, match='Cannot set camera pitch'):
        set_camera_pitch(m, pitch=75, max_pitch=85)
    assert isinstance(m.view_state, GlobeViewState)
