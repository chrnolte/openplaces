"""Tests that spatial linking skips an admin unit the reference does not cover.

A reference recipe can genuinely have zero rows for an admin unit -- many
rural U.S. counties hold no Overture address points at all -- and that is an
expected coverage gap, not a failed ingest. Both spatial link routes must
return the state untouched instead of raising, so the rest of the harmonize
pipeline still runs for that unit.

Reported from a Vilas County, WI pilot, where `dwelling-overture-2025`
ingested cleanly but produced no points inside the county.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

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
        recipe={}, admin_id='US-WI-VI', verbose=False, timer=None, spine=spine
    )


def _empty_points():
    return gpd.GeoDataFrame({'source': []}, geometry=[], crs=CRS)


@pytest.mark.parametrize(
    'reference',
    [None, 'empty'],
    ids=['no_output_written', 'empty_table'],
)
def test_spatial_point_skips_uncovered_admin_unit(monkeypatch, reference):
    """A missing or empty reference leaves the spine untouched, no raise."""
    ref = None if reference is None else _empty_points()
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: ref)

    state = _state()
    before = len(state.spine)
    result = links._link_spatial_point(state, 'dwelling_ref', 'dwelling', None, {})

    assert result is state
    assert len(result.spine) == before
    assert 'dwelling_ref' not in result.crosswalks


@pytest.mark.parametrize(
    'reference',
    [None, 'empty'],
    ids=['no_output_written', 'empty_table'],
)
def test_spatial_overlay_skips_uncovered_admin_unit(monkeypatch, reference):
    """The overlay route needs the same guard as the point route."""
    ref = None if reference is None else _empty_points()
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: ref)

    state = _state()
    before = len(state.spine)
    result = links._link_spatial_overlay(state, 'parcel_ref', 'parcel', {})

    assert result is state
    assert len(result.spine) == before
    assert 'parcel_ref' not in result.crosswalks
