"""A complete-coverage reference without rows raises and says why.

The parcel geospine of three New England towns stopped on FEMA's
complete-coverage guard (US-CT-COK, US-ME-PEM, US-VT-CON, 2026-11).
Their FEMA tables existed but held no rows: older code had ingested
them with the national recipe, whose county FIPS crosswalk matches no
town. The guard's message read as if nothing had been ingested. It now
tells a table saved without rows (rerun the unit's ingest with
--reprocess, since an ingest skips a unit whose file exists) from a
missing one (ingest the unit). Fabricated recipe ids and geometry
throughout.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

import openplaces.io.harmonizer.links as links
from openplaces import recipe as recipe_module
from openplaces.io.harmonizer import HarmonizeState
from openplaces.recipe import raise_if_coverage_complete
from tests.links_patching import patch_links

REFERENCE = 'US_footprint-fabricated-2023'
CRS = 32617  # UTM 17N, metric


def _state():
    spine = gpd.GeoDataFrame(
        geometry=[box(0, 0, 100, 100)],
        crs=CRS,
        index=pd.Index(['P1'], name='parcel_id'),
    )
    return HarmonizeState(
        recipe={}, admin_id='US-XX-TWN', verbose=False, timer=None, spine=spine
    )


def _declare(monkeypatch, coverage):
    # coverage_is_complete reads the recipe through the package.
    monkeypatch.setattr(
        recipe_module,
        'get_recipe_by_id',
        lambda recipe_id, **kwargs: {'coverage': coverage},
    )


@pytest.fixture
def complete_reference(monkeypatch):
    _declare(monkeypatch, 'complete')
    empty = gpd.GeoDataFrame({'height': []}, geometry=[], crs=CRS)
    patch_links(monkeypatch, 'get_entities', lambda *a, **k: empty)


@pytest.mark.parametrize(
    ('built', 'message'),
    [(True, 'saved without rows.*--reprocess'), (False, 'no output file exists')],
    ids=['saved_empty', 'never_saved'],
)
def test_overlay_link_raises_and_names_the_case(
    monkeypatch, complete_reference, built, message
):
    patch_links(monkeypatch, 'entity_outputs_exist', lambda *a, **k: built)

    with pytest.raises(RuntimeError, match=message):
        links._link_spatial_overlay(_state(), REFERENCE, 'footprint', {})


def test_point_link_raises_and_names_the_case(monkeypatch, complete_reference):
    patch_links(monkeypatch, 'entity_outputs_exist', lambda *a, **k: True)

    with pytest.raises(RuntimeError, match='saved without rows'):
        links._link_spatial_point(_state(), REFERENCE, 'footprint', None, {})


def test_the_file_check_runs_only_for_complete_coverage(monkeypatch):
    _declare(monkeypatch, None)

    def _unexpected():
        raise AssertionError('checked files for a tolerant recipe')

    raise_if_coverage_complete(REFERENCE, 'US-XX-TWN', saved_empty=_unexpected)


def test_an_unchecked_caller_keeps_the_general_message(monkeypatch):
    _declare(monkeypatch, 'complete')

    with pytest.raises(RuntimeError, match='ingest it for this admin unit'):
        raise_if_coverage_complete(REFERENCE, 'US-XX-TWN')
