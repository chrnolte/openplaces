"""Regression: spatial-overlay link tolerates references without `geo_id`.

`geo_id` is generated at ingest only for parcel entities, so a footprint
reference (e.g. US_footprint-fema-2023) reaches `_link_spatial_overlay` without
it. The step must derive `geo_id` from geometry instead of raising KeyError.
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

import openplaces.io.harmonizer.links as links
import openplaces.io.harmonizer.spine as spine_mod
from openplaces.io.harmonizer import HarmonizeState


class _StopAtOverlay(Exception):
    """Sentinel: raised once execution reaches overlay_polygons."""


def test_overlay_generates_geo_id_for_reference_without_it(monkeypatch):
    # Footprint-like reference with NO geo_id column (the ingest-time parcel-only
    # column), as FEMA footprints arrive.
    ref = gpd.GeoDataFrame(
        {'occupancy_type': ['A', 'B']},
        geometry=[box(0, 0, 1, 1), box(2, 2, 3, 3)],
        crs='EPSG:4326',
    )
    assert 'geo_id' not in ref.columns
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: ref)

    captured = {}

    def _fake_overlay(spine, ref_polys, **kwargs):
        # Reaching here means the geo_id-dependent lines (247-269) ran cleanly.
        captured['ref_polys'] = ref_polys
        raise _StopAtOverlay

    monkeypatch.setattr(links, 'overlay_polygons', _fake_overlay)

    spine = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs='EPSG:4326')
    spine.index.name = 'footprint_id'
    state = HarmonizeState(
        recipe={}, admin_id='US-MA-MI', verbose=False, timer=None, spine=spine
    )

    with pytest.raises(_StopAtOverlay):
        links._link_spatial_overlay(state, 'US_footprint-fema-2023', 'footprint', {})

    # The deduplicated reference polygons are keyed by the generated geo_id.
    assert captured['ref_polys'].index.name == 'parcel_id'
    assert len(captured['ref_polys']) == 2


def test_parcel_spine_overlay_does_not_collide_with_ref_level(monkeypatch):
    # A parcel spine's index is itself named 'parcel_id', which is also the
    # spatial-overlay reference-id level — they collided in the overlay
    # MultiIndex. resolve_spine renames the spine index to avoid the clash.
    parcels = gpd.GeoDataFrame(
        {'parcel_id_local': ['p1', 'p2']},
        geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)],
        crs='EPSG:4326',
        index=pd.Index(['A', 'B'], name='parcel_id'),
    )
    monkeypatch.setattr(spine_mod, 'get_entities', lambda *a, **k: parcels)
    state = HarmonizeState(
        recipe={'admin_id': 'US-MA-MI'}, admin_id='US-MA-MI', verbose=False, timer=None
    )
    state = spine_mod.resolve_spine(
        state,
        sources=[{'recipe_id': 'parcel', 'label': 'parcel'}],
        keep_columns=['parcel_id_local'],
    )
    assert state.spine.index.name == 'spine_id'
    assert state.metadata.get('spine_index_name') == 'parcel_id'

    # FEMA-like footprint reference (no geo_id) overlaying the parcel spine.
    ref = gpd.GeoDataFrame(
        {'occupancy_type': ['Res', 'Com']},
        geometry=[box(1, 1, 3, 3), box(11, 1, 13, 3)],
        crs='EPSG:4326',
    )
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: ref)
    state = links._link_spatial_overlay(
        state, 'US_footprint-fema-2023', 'footprint', {'area_intersection_m2_min': 0}
    )
    crosswalk = state.crosswalks['US_footprint-fema-2023']
    assert set(crosswalk.index.names) == {'spine_id', 'parcel_id'}


def test_resolve_spine_keep_columns_carries_value_fields(monkeypatch):
    # Value-bearing attributes listed in keep_columns are carried 1:1 off
    # each source's own row (no key-based lookup, so no duplicate-key risk),
    # and are recorded in state.metadata['spine_keep_columns'] so a later
    # link_by_id(auto_discover=True) self-join can skip re-deriving them.
    parcels = gpd.GeoDataFrame(
        {
            'parcel_id_local': ['p1', 'p2'],
            'year_built': [1964.0, 1998.0],
            'improvement_value': [100000.0, 150000.0],
            'land_value': [30000.0, 40000.0],
        },
        geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)],
        crs='EPSG:4326',
        index=pd.Index(['A', 'B'], name='parcel_id'),
    )
    monkeypatch.setattr(spine_mod, 'get_entities', lambda *a, **k: parcels)
    state = HarmonizeState(
        recipe={'admin_id': 'US-NC-CE'}, admin_id='US-NC-CE', verbose=False, timer=None
    )
    state = spine_mod.resolve_spine(
        state,
        sources=[{'recipe_id': 'nconemap', 'label': 'nconemap'}],
        keep_columns=[
            'parcel_id_local',
            'year_built',
            'improvement_value',
            'land_value',
        ],
    )

    assert state.spine['year_built'].tolist() == [1964.0, 1998.0]
    assert state.spine['improvement_value'].tolist() == [100000.0, 150000.0]
    assert state.spine['land_value'].tolist() == [30000.0, 40000.0]
    assert state.metadata['spine_keep_columns'] == {
        'parcel_id_local',
        'year_built',
        'improvement_value',
        'land_value',
    }
    assert state.metadata['spine_source_recipe_ids'] == {'nconemap'}


def test_resolve_spine_falls_back_when_highest_priority_source_has_no_geometry(
    monkeypatch,
):
    # Regression test for a real bug: Craven County's parcel roll is a
    # geometry-less attribute table by design (its companion property
    # recipe carries the geometry instead) -- when it's the highest-priority
    # (first) source in `sources`, resolve_spine used to assume sources[0]
    # always loaded and crashed with KeyError building the primary spine
    # frame. It must instead fall back to the next source that actually has
    # geometry for this admin unit.
    fallback = gpd.GeoDataFrame(
        {'parcel_id_local': ['p1']},
        geometry=[box(0, 0, 10, 10)],
        crs='EPSG:4326',
        index=pd.Index(['A'], name='parcel_id'),
    )

    def _get_entities(recipe_id, admin_id, geom=True):
        if recipe_id == 'cravencounty':
            raise FileNotFoundError(f'no _geo.parquet for {recipe_id}')
        return fallback

    monkeypatch.setattr(spine_mod, 'get_entities', _get_entities)
    state = HarmonizeState(
        recipe={'admin_id': 'US-NC-CN'}, admin_id='US-NC-CN', verbose=False, timer=None
    )
    state = spine_mod.resolve_spine(
        state,
        sources=[
            {'recipe_id': 'cravencounty', 'label': 'cravencounty'},
            {'recipe_id': 'nconemap', 'label': 'nconemap'},
        ],
        keep_columns=['parcel_id_local'],
    )

    assert state.spine['geometry_source'].tolist() == ['nconemap']
    assert state.metadata['spine_source_recipe_ids'] == {'cravencounty', 'nconemap'}
