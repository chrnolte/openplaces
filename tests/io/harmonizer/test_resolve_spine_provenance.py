"""Tests for `resolve_spine`'s `track_provenance` parameter: seeds a
`{column}_source` sidecar recording which source's own row supplied each
keep_columns cell, for both the primary source's rows and any later
non-overlapping `to_add` source's rows.
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

import openplaces.io.harmonizer.spine as spine_mod
from openplaces.io.harmonizer import HarmonizeState


def test_single_source_seeds_every_row_with_its_own_label(monkeypatch):
    parcels = gpd.GeoDataFrame(
        {
            'parcel_id_local': ['p1', 'p2'],
            'year_built': [1964.0, None],
        },
        geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)],
        crs='EPSG:4326',
        index=pd.Index(['A', 'B'], name='parcel_id'),
    )
    # Synthetic coordinates: keep the stray-row guard out of the way.
    monkeypatch.setattr(spine_mod, '_drop_out_of_unit_rows', lambda gdf, *a, **k: gdf)
    monkeypatch.setattr(spine_mod, 'get_entities', lambda *a, **k: parcels)
    state = HarmonizeState(
        recipe={'admin_id': 'US-NC-BL'}, admin_id='US-NC-BL', verbose=False, timer=None
    )
    state = spine_mod.resolve_spine(
        state,
        sources=[{'recipe_id': 'bladenco', 'label': 'bladenco'}],
        keep_columns=['parcel_id_local', 'year_built'],
        track_provenance=['year_built'],
    )

    # Populated cell is stamped with the source label; a genuinely missing
    # value (p2's year_built) is not attributed to anyone.
    year_built = [None if pd.isna(v) else v for v in state.spine['year_built']]
    assert year_built == [1964.0, None]
    got = [None if pd.isna(v) else v for v in state.spine['year_built_source']]
    assert got == ['bladenco', None]


def test_two_non_overlapping_sources_each_seed_their_own_rows(monkeypatch):
    primary = gpd.GeoDataFrame(
        {'parcel_id_local': ['p1'], 'year_built': [1964.0]},
        geometry=[box(0, 0, 10, 10)],
        crs='EPSG:4326',
        index=pd.Index(['A'], name='parcel_id'),
    )
    secondary = gpd.GeoDataFrame(
        {'parcel_id_local': ['p2'], 'year_built': [1998.0]},
        # Far away: no IoU overlap with the primary source's geometry.
        geometry=[box(100, 100, 110, 110)],
        crs='EPSG:4326',
        index=pd.Index(['B'], name='parcel_id'),
    )
    sources = {'bladenco': primary, 'nconemap': secondary}
    # Synthetic coordinates: keep the stray-row guard out of the way.
    monkeypatch.setattr(spine_mod, '_drop_out_of_unit_rows', lambda gdf, *a, **k: gdf)
    monkeypatch.setattr(
        spine_mod, 'get_entities', lambda recipe_id, *a, **k: sources[recipe_id]
    )
    state = HarmonizeState(
        recipe={'admin_id': 'US-NC-BL'}, admin_id='US-NC-BL', verbose=False, timer=None
    )
    state = spine_mod.resolve_spine(
        state,
        sources=[
            {'recipe_id': 'bladenco', 'label': 'bladenco'},
            {'recipe_id': 'nconemap', 'label': 'nconemap'},
        ],
        keep_columns=['parcel_id_local', 'year_built'],
        track_provenance=['year_built'],
    )

    by_local_id = dict(
        zip(
            state.spine['parcel_id_local'],
            state.spine['year_built_source'],
            strict=True,
        )
    )
    assert by_local_id == {'p1': 'bladenco', 'p2': 'nconemap'}
    assert state.spine['geometry_source'].tolist() == ['bladenco', 'nconemap']
