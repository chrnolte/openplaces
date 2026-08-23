"""Tests for `_link_spatial_point`'s ESRI-colocation exclusion flag.

An NSI point sourced `'ESRI'` that shares a location with another,
differently-sourced point (e.g. a home-office duplicate of a Parcel-sourced
record at the same address) gets `exclude_from_upward_correction = True` in the
resulting crosswalk, so a later count-driven upward Single->Multi-Family
correction (e.g. `n_dwellings` summing) can exclude it. Uses a projected metric
CRS throughout so "same location" / "3 m apart" are unambiguous.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box

import openplaces.io.harmonizer.links as links
from openplaces.io.harmonizer import HarmonizeState

CRS = 32617  # UTM 17N (metric)


def _state():
    spine = gpd.GeoDataFrame(
        geometry=[box(0, 0, 100, 100)],
        crs=CRS,
        index=pd.Index(['F1'], name='footprint_id'),
    )
    return HarmonizeState(
        recipe={}, admin_id='US-NC-AR', verbose=False, timer=None, spine=spine
    )


def _run(monkeypatch, points_df):
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: points_df)
    state = _state()
    # save_link=False: persistence needs a real spine recipe for the
    # sidecar path and fingerprint; these tests exercise the flag logic.
    state = links._link_spatial_point(
        state, 'nsi_ref', 'building', None, {}, save_link=False
    )
    return state.crosswalks['nsi_ref']


def test_esri_point_colocated_with_other_source_is_flagged(monkeypatch):
    points = gpd.GeoDataFrame(
        {'source': ['Parcel', 'ESRI']},
        geometry=[Point(50, 50), Point(50, 50)],  # identical location
        crs=CRS,
    )
    linked = _run(monkeypatch, points)
    flags = linked.set_index('source')['exclude_from_upward_correction']
    assert bool(flags.loc['ESRI']) is True
    assert bool(flags.loc['Parcel']) is False


def test_two_esri_points_colocated_with_no_other_source_not_flagged(monkeypatch):
    # Nothing to defer to -- matches "IF they share the same location [as
    # another point]", not "if there are 2+ ESRI points".
    points = gpd.GeoDataFrame(
        {'source': ['ESRI', 'ESRI']},
        geometry=[Point(50, 50), Point(50, 50)],
        crs=CRS,
    )
    linked = _run(monkeypatch, points)
    assert not linked['exclude_from_upward_correction'].any()


def test_esri_point_at_different_location_not_flagged(monkeypatch):
    points = gpd.GeoDataFrame(
        {'source': ['Parcel', 'ESRI']},
        geometry=[Point(10, 10), Point(90, 90)],
        crs=CRS,
    )
    linked = _run(monkeypatch, points)
    assert not linked['exclude_from_upward_correction'].any()


def test_esri_point_within_snapping_tolerance_still_flagged(monkeypatch):
    # 1 m apart: not bit-identical coordinates, but well within the ~2.5-3 m
    # OLC codelength=11 snapping cell -- proves the "slight snapping" ask.
    points = gpd.GeoDataFrame(
        {'source': ['Parcel', 'ESRI']},
        geometry=[Point(50.0, 50.0), Point(50.0, 51.0)],
        crs=CRS,
    )
    linked = _run(monkeypatch, points)
    flags = linked.set_index('source')['exclude_from_upward_correction']
    assert bool(flags.loc['ESRI']) is True
