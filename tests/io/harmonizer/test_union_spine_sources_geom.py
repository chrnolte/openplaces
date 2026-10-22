"""Tests for the geom option of union_spine_sources."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from openplaces.core.schema import AdminId
from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer import spine as spine_module
from openplaces.io.harmonizer.spine import union_spine_sources


def _state():
    return HarmonizeState(
        recipe={'admin_id': AdminId('CO'), 'entity': None},
        admin_id=AdminId('CO'),
        verbose=False,
        timer=None,
        spine=None,
    )


def test_union_with_geometry_returns_geodataframe(monkeypatch):
    frames = {
        'a': gpd.GeoDataFrame(
            {'price': [1.0]}, geometry=[box(0, 0, 1, 1)], crs='EPSG:4326'
        ),
        'b': gpd.GeoDataFrame(
            {'price': [2.0]}, geometry=[box(2, 2, 3, 3)], crs='EPSG:4326'
        ),
    }
    monkeypatch.setattr(
        spine_module, '_expand_auto_discover', lambda sources, state: sources
    )
    monkeypatch.setattr(
        spine_module,
        'get_entities',
        lambda rid, admin_id, layer=None, missing='raise', geom=False: (
            frames[rid] if geom else pd.DataFrame(frames[rid].drop(columns='geometry'))
        ),
    )
    monkeypatch.setattr(
        spine_module, 'restrict_to_admin_by_name', lambda df, rid, admin_id: df
    )
    sources = [{'recipe_id': 'a', 'label': 'a'}, {'recipe_id': 'b', 'label': 'b'}]

    out = union_spine_sources(_state(), sources=sources, geom=True).spine
    assert isinstance(out, gpd.GeoDataFrame)
    assert str(out.crs) == 'EPSG:4326'
    assert out['source'].tolist() == ['a', 'b']
    assert out.geometry.notna().all()

    plain = union_spine_sources(_state(), sources=sources).spine
    assert not isinstance(plain, gpd.GeoDataFrame)
    assert 'geometry' not in plain.columns
