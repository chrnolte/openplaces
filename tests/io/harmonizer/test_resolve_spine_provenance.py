"""Tests for `resolve_spine`'s `track_provenance` parameter: seeds a
`{column}_source` sidecar recording which source's own row supplied each
keep_columns cell, for both the primary source's rows and any later
non-overlapping `to_add` source's rows.
"""

import geopandas as gpd
import pandas as pd
import pytest
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


# The parcel geospine's lists. The sidecars once followed a set of
# names, so their order changed with the process's hash seed (Fort
# Bend, 2026-11-21: land_value_source and improvement_value_source
# swapped places between two runs of the same code on the same data).
KEEP = [
    'parcel_id_local',
    'use_group',
    'improvement_value',
    'land_value',
    'year_built',
    'n_dwellings',
]
TRACKED = ['use_group', 'improvement_value', 'land_value', 'year_built', 'n_dwellings']


@pytest.mark.parametrize('tracked', [TRACKED, TRACKED[::-1]])
def test_sidecars_follow_the_order_track_provenance_names(monkeypatch, tracked):
    first = gpd.GeoDataFrame(
        {
            'parcel_id_local': ['p1'],
            'use_group': ['residential'],
            'improvement_value': [1.0],
            'year_built': [1964.0],
        },
        geometry=[box(0, 0, 10, 10)],
        crs='EPSG:4326',
        index=pd.Index(['A'], name='parcel_id'),
    )
    # The second source alone brings land_value and n_dwellings, so
    # their sidecars arrive through the rows it adds.
    second = gpd.GeoDataFrame(
        {
            'parcel_id_local': ['p2'],
            'land_value': [2.0],
            'n_dwellings': [1.0],
            'year_built': [1998.0],
        },
        geometry=[box(100, 100, 110, 110)],
        crs='EPSG:4326',
        index=pd.Index(['B'], name='parcel_id'),
    )
    sources = {'first': first, 'second': second}
    monkeypatch.setattr(spine_mod, '_drop_out_of_unit_rows', lambda gdf, *a, **k: gdf)
    monkeypatch.setattr(
        spine_mod, 'get_entities', lambda recipe_id, *a, **k: sources[recipe_id]
    )
    state = HarmonizeState(
        recipe={'admin_id': 'US-ZZ-AAA'},
        admin_id='US-ZZ-AAA',
        verbose=False,
        timer=None,
    )
    state = spine_mod.resolve_spine(
        state,
        sources=[
            {'recipe_id': 'first', 'label': 'first'},
            {'recipe_id': 'second', 'label': 'second'},
        ],
        keep_columns=KEEP,
        track_provenance=[*tracked, 'not_kept'],
    )
    in_first = [c for c in tracked if c in first.columns]
    in_second = [c for c in tracked if c not in first.columns]
    sidecars = [c for c in state.spine.columns if c.endswith('_source')]
    assert sidecars == [
        'geometry_source',
        *(f'{c}_source' for c in in_first),
        *(f'{c}_source' for c in in_second),
    ]
