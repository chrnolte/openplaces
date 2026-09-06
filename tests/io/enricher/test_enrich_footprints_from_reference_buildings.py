"""Tests for the reference-building footprint enrichment step."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

import openplaces.io.enricher.buildings as buildings_mod
from openplaces.core.schema import AdminId
from openplaces.io.enricher import EnrichState
from openplaces.io.enricher.buildings import (
    enrich_footprints_from_reference_buildings,
)


def _gdf(boxes, index_name, extra=None):
    data = {index_name: list(boxes)}
    if extra:
        data.update(extra)
    return gpd.GeoDataFrame(
        data, geometry=[boxes[k] for k in boxes], crs='EPSG:4326'
    ).set_index(index_name)


def _make_state(spine):
    return EnrichState(
        recipe={'reference_building_recipe_id': 'FAKE_building-ref'},
        entity_recipe={},
        admin_id=AdminId('US-NC-BR'),
        verbose=False,
        timer=None,
        spine=spine,
        evidence=pd.DataFrame(index=spine.index),
    )


@pytest.fixture
def reference_recipe(monkeypatch):
    recipe = {}
    monkeypatch.setattr(buildings_mod, 'get_recipe_by_id', lambda rid: recipe)
    return recipe


def test_attaches_best_overlap(monkeypatch, reference_recipe):
    reference = _gdf(
        {'b1': box(0, 0, 0.001, 0.001)},
        'building_id',
        extra={'roof_shape': ['Gable']},
    )
    monkeypatch.setattr(buildings_mod, 'get_entities', lambda *a, **k: reference)
    spine = _gdf({'f1': box(0, 0, 0.001, 0.001)}, 'footprint_id')

    result = enrich_footprints_from_reference_buildings(
        _make_state(spine), suffix='_building_ref'
    )

    assert result.evidence.loc['f1', 'roof_shape_building_ref'] == 'Gable'


def test_no_overlap_with_default_columns_still_writes_the_schema(
    monkeypatch, reference_recipe
):
    """When the reference loads but nothing overlaps, its columns are known.

    A recipe relying on the default column list would otherwise get a
    column-less evidence file for this unit, which curate reads as
    though the evidence had never been computed.
    """
    reference = _gdf(
        {'b1': box(0, 0, 0.001, 0.001)},
        'building_id',
        extra={'roof_shape': ['Gable'], 'n_stories': [2]},
    )
    monkeypatch.setattr(buildings_mod, 'get_entities', lambda *a, **k: reference)
    spine = _gdf({'f1': box(0.5, 0.5, 0.501, 0.501)}, 'footprint_id')

    result = enrich_footprints_from_reference_buildings(
        _make_state(spine), suffix='_building_ref'
    )

    assert set(result.evidence.columns) == {
        'roof_shape_building_ref',
        'n_stories_building_ref',
    }
    assert result.evidence.isna().all().all()


def test_missing_reference_with_declared_columns_writes_them_null(
    monkeypatch, reference_recipe
):
    def raise_not_found(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(buildings_mod, 'get_entities', raise_not_found)
    spine = _gdf({'f1': box(0, 0, 0.001, 0.001)}, 'footprint_id')

    result = enrich_footprints_from_reference_buildings(
        _make_state(spine), columns=['roof_shape'], suffix='_building_ref'
    )

    assert list(result.evidence.columns) == ['roof_shape_building_ref']
    assert result.evidence['roof_shape_building_ref'].isna().all()
