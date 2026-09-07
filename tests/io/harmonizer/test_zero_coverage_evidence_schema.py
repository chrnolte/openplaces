"""A reference with no rows for an admin unit still owes its columns.

`reconcile_attributes` used to emit nothing for a point reference whose
link step returned early on zero coverage, so a rural county's spine
lacked evidence columns its neighbors carried. The columns are now
written null, following the enricher's contract that a declared column is
always present and a missing one is a recipe error, not a coverage gap.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

import openplaces.io.harmonizer.attributes as attrs
import openplaces.io.harmonizer.links as links
from openplaces.core.schema import SourceGeometryType
from openplaces.io.harmonizer import HarmonizeState

OVERTURE = 'dwelling-overture-2025'
COLUMNS = ['n_dwellings', 'address_street', 'postal_code']


class _Entity:
    def __init__(self, entity_type):
        self.entity_type = entity_type


def _state():
    spine = gpd.GeoDataFrame(
        {'geometry_source': ['obm', 'obm']},
        geometry=[box(0, 0, 1, 1), box(2, 2, 3, 3)],
        crs='EPSG:4326',
        index=pd.Index(['f1', 'f2'], name='footprint_id'),
    )
    return HarmonizeState(
        recipe={'entity': _Entity('footprint')},
        admin_id='US-NC-BR',
        verbose=False,
        timer=None,
        spine=spine,
    )


def _sources():
    return [{'recipe_id': OVERTURE, 'columns': COLUMNS}]


def test_zero_coverage_point_reference_still_writes_its_columns(monkeypatch):
    # The link step finds no reference rows for this unit and returns
    # early, leaving no crosswalk behind.
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: pd.DataFrame())
    state = links.link_to_reference(
        _state(),
        recipe_id=OVERTURE,
        join='spatial_point',
        source_geometry_type='single_dwelling_point',
        save_link=False,
    )
    assert OVERTURE not in state.crosswalks
    assert state.reference_types[OVERTURE] == 'dwelling'

    state = attrs.reconcile_attributes(state, sources=_sources())
    out = state.spine

    # The same names a covered unit gets, all null, plus a zero count.
    assert out['n_dwellings_overture'].isna().all()
    assert out['address_street_dwelling_overture'].isna().all()
    assert out['postal_code_dwelling_overture'].isna().all()
    assert out['n_dwellings_overture'].shape[0] == 2


def test_declared_column_missing_from_a_covered_reference_is_still_written(
    monkeypatch,
):
    # The reference covers the unit but carries no postal_code at all.
    ref = gpd.GeoDataFrame(
        {'n_dwellings': [1.0], 'address_street': ['SAMPLE AVE']},
        geometry=[box(0.2, 0.2, 0.4, 0.4).centroid],
        crs='EPSG:4326',
    )
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: ref)
    state = links.link_to_reference(
        _state(),
        recipe_id=OVERTURE,
        join='spatial_point',
        source_geometry_type='single_dwelling_point',
        save_link=False,
    )
    assert OVERTURE in state.crosswalks

    state = attrs.reconcile_attributes(state, sources=_sources())
    out = state.spine

    assert 'postal_code_dwelling_overture' in out.columns
    assert out['postal_code_dwelling_overture'].isna().all()
    assert out.loc['f1', 'n_dwellings_overture'] == 1.0


def test_absent_non_point_reference_warns_rather_than_inventing_columns():
    state = _state()
    state.source_geometry_types['some_parcels'] = SourceGeometryType(
        'single_building_footprint'
    )
    before = list(state.spine.columns)
    import pytest

    with pytest.warns(UserWarning, match='not in state'):
        state = attrs.reconcile_attributes(
            state, sources=[{'recipe_id': 'some_parcels', 'columns': ['land_value']}]
        )
    assert list(state.spine.columns) == before
