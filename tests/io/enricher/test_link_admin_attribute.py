"""Tests for the enrich step link_admin_attribute."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from openplaces.core.schema import AdminId
from openplaces.io.enricher import EnrichState
from openplaces.io.enricher import admin as admin_module
from openplaces.io.enricher.admin import link_admin_attribute


@pytest.fixture
def state():
    spine = pd.DataFrame(
        {'admin3_id': ['CO-AN-AAA', 'CO-AN-AAA', 'CO-AN-BBB', 'CO-AN-ZZZ', None]},
        index=list('pqrst'),
    )
    return EnrichState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('CO-AN'),
        verbose=False,
        timer=None,
        spine=spine,
        evidence=pd.DataFrame(index=spine.index),
    )


@pytest.fixture
def stub_readers(monkeypatch):
    table = pd.DataFrame(
        {'population': [1000.0, 250.0], 'n_expelled': [100.0, 0.0]},
        index=pd.Index(['CO-AN-AAA', 'CO-AN-BBB'], name='admin3_id'),
    )
    # Two square units, 1 km2 and 4 km2, in an equal-area metric CRS.
    units = gpd.GeoDataFrame(
        {'name': ['A', 'B']},
        geometry=[box(0, 0, 1000, 1000), box(0, 0, 2000, 2000)],
        index=pd.Index(['CO-AN-AAA', 'CO-AN-BBB'], name='admin3_id'),
        crs='EPSG:6933',
    )
    monkeypatch.setattr(admin_module, 'get_dataset', lambda recipe_id, admin_id: table)
    monkeypatch.setattr(
        admin_module, 'get_admin', lambda admin_id, level, geom=False: units
    )


def test_plain_join_and_missing_units(state, stub_readers):
    out = link_admin_attribute(
        state, output='population', recipe_id='x', attribute='population'
    )
    assert out.evidence['population'].tolist()[:3] == [1000.0, 1000.0, 250.0]
    assert out.evidence['population'].isna().tolist()[3:] == [True, True]


def test_share_of(state, stub_readers):
    out = link_admin_attribute(
        state,
        output='displaced_share',
        recipe_id='x',
        attribute='n_expelled',
        share_of='population',
    )
    assert out.evidence['displaced_share'].tolist()[:3] == [0.1, 0.1, 0.0]


def test_per_km2(state, stub_readers):
    out = link_admin_attribute(
        state,
        output='population_density',
        recipe_id='x',
        attribute='population',
        per_km2=True,
    )
    density = out.evidence['population_density']
    assert density.iloc[0] == pytest.approx(1000.0)
    assert density.iloc[2] == pytest.approx(250.0 / 4.0)


def test_exclusive_options_and_missing_column(state, stub_readers):
    with pytest.raises(ValueError, match='exclusive'):
        link_admin_attribute(
            state,
            output='o',
            recipe_id='x',
            attribute='population',
            per_km2=True,
            share_of='population',
        )
    with pytest.raises(ValueError, match='no column'):
        link_admin_attribute(state, output='o', recipe_id='x', attribute='nope')
