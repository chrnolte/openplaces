"""The point link's keep-first drops report how many rows they discarded.

A reference point that falls inside two overlapping footprints, or two
overlapping parcels, is resolved by keeping the first match. That is a
reasonable rule, but it was applied in silence, so a run gave no sign
that overlapping inputs had cost it links.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box

import openplaces.io.harmonizer.links as links
from openplaces.io.harmonizer import HarmonizeState

CRS = 32617  # UTM 17N (metric)

# Overlapping between x=50 and x=100, so a point at x=75 is inside both.
_SPINE = gpd.GeoDataFrame(
    geometry=[box(0, 0, 100, 100), box(50, 0, 150, 100)],
    crs=CRS,
    index=pd.Index(['F1', 'F2'], name='footprint_id'),
)
_PARCELS = gpd.GeoDataFrame(
    geometry=[box(0, 0, 100, 100), box(50, 0, 150, 100)],
    crs=CRS,
    index=pd.Index(['P1', 'P2'], name='parcel_id'),
)
_CROSSWALK = pd.DataFrame(
    index=pd.MultiIndex.from_tuples(
        [('F1', 'P1'), ('F2', 'P2')], names=['footprint_id', 'parcel_id']
    )
)


def _state(verbose, with_parcels):
    state = HarmonizeState(
        recipe={}, admin_id='US-NC-WAK', verbose=verbose, timer=None, spine=_SPINE
    )
    if with_parcels:
        state.overlays['parcel_ref'] = _PARCELS
        state.references['parcel_ref'] = _PARCELS
        state.crosswalks['parcel_ref'] = _CROSSWALK
    return state


def _run(monkeypatch, verbose=True, with_parcels=True):
    points = gpd.GeoDataFrame({'source': ['Parcel']}, geometry=[Point(75, 50)], crs=CRS)
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: points)
    return links._link_spatial_point(
        _state(verbose, with_parcels), 'nsi_ref', 'building', None, {}, save_link=False
    )


def test_point_matching_two_footprints_is_reported(monkeypatch, capsys):
    state = _run(monkeypatch, with_parcels=False)
    out = capsys.readouterr().out
    assert 'Deduplicate: 1 reference point(s) matched more than one' in out
    assert len(state.crosswalks['nsi_ref']) == 1


def test_point_in_two_parcels_is_reported(monkeypatch, capsys):
    _run(monkeypatch, with_parcels=True)
    out = capsys.readouterr().out
    assert 'Cross-parcel filter: 1 reference point(s) fell in more than one' in out
    assert 'Reference join (nsi_ref): 1 point(s) fell in more than one' in out


def test_nothing_is_printed_without_verbose(monkeypatch, capsys):
    _run(monkeypatch, verbose=False)
    out = capsys.readouterr().out
    assert 'Deduplicate:' not in out
    assert 'Reference join' not in out
