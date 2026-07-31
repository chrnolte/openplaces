"""Tests for `consolidate_condo_cluster_footprints`'s link-sidecar re-persist.

Regression coverage for a real bug: `link_to_reference`'s `save_link: true`
persists the raw footprint<->parcel overlay to a parquet sidecar *before*
`consolidate_condo_cluster_footprints` runs. Without re-persisting the
sidecar after consolidation, the new consolidated footprint row has no
rows in that sidecar at all, so curate-stage readers
(`apportion_curated_values`, `collect_link_ids`) never see its cluster
parcels and the footprint resolves to a $0 structure value despite real,
positive `improvement_value` on every parcel underneath it.
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from openplaces.config import cfg
from openplaces.core.schema import AdminId
from openplaces.geo.link import get_entity_link_path
from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer import links as links_mod
from openplaces.recipe import get_recipe_by_id

SPINE = 'US_footprint-spine-2026'
PARCEL = 'US-NC_parcel-nconemap-2025'
COUNTY = 'US-NC-BS'


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    dirs = dict(cfg.config['directories'])
    dirs['data_root'] = tmp_path
    for name in ('core', 'external', 'raw', 'cache', 'out', 'share'):
        dirs[name] = tmp_path / 'data' / name
    dirs['heap'] = tmp_path / 'data/cache/_heap'
    dirs['logs'] = tmp_path / 'data/cache/_logs'
    monkeypatch.setitem(cfg.config, 'directories', dirs)
    return tmp_path


def _spine_gdf():
    # One ordinary footprint, unrelated to and far from the condo cluster
    # below -- mirrors test_link_sidecar.py's disconnected-geometry case,
    # confirming the overlay/sidecar machinery tolerates it.
    return gpd.GeoDataFrame(
        {'geometry': [box(100, 100, 101, 101)]},
        index=pd.Index(['f_other'], name='footprint_id'),
        crs='epsg:4326',
    )


def _condo_unit_parcels_gdf():
    # Two tiny (~123 m^2 = 0.0123 ha, under the 0.02 ha unit threshold)
    # touching parcels near the equator, each with a real, positive
    # improvement_value and no land_value -- the textbook per-unit condo
    # signature _cluster_condo_parcels looks for.
    return gpd.GeoDataFrame(
        {
            'geo_id': ['U1', 'U2'],
            'land_value': [0.0, 0.0],
            'improvement_value': [300_000.0, 250_000.0],
            'geometry': [
                box(0, 0, 0.0001, 0.0001),
                box(0.0001, 0, 0.0002, 0.0001),
            ],
        },
        crs='epsg:4326',
    )


@pytest.fixture
def state(data_root, monkeypatch):
    monkeypatch.setattr(
        links_mod, 'get_entities', lambda *a, **k: _condo_unit_parcels_gdf()
    )
    recipe = get_recipe_by_id(SPINE)
    return HarmonizeState(
        recipe=recipe,
        admin_id=AdminId(COUNTY),
        verbose=False,
        timer=None,
        spine=_spine_gdf(),
    )


def _sidecar_path():
    return get_entity_link_path(SPINE, PARCEL, admin_id=COUNTY)


def test_consolidation_re_persists_sidecar_with_all_cluster_parcels(state):
    state = links_mod._link_spatial_overlay(
        state,
        PARCEL,
        'parcel',
        {'min_fraction_of_largest': 0.1667, 'area_intersection_m2_min': 10},
        save_link=True,
    )
    sidecar = _sidecar_path()
    assert sidecar.exists()
    before = pd.read_parquet(sidecar)
    assert set(before['parcel_id'].dropna()) == set()  # neither unit parcel yet

    state = links_mod.consolidate_condo_cluster_footprints(state)

    # One new consolidated footprint row, crosswalked to both cluster parcels.
    new_ids = [
        i
        for i in state.spine.index
        if str(state.spine.at[i, 'geometry_source']).startswith('condo_cluster.')
    ]
    assert len(new_ids) == 1
    new_id = new_ids[0]

    crosswalk = state.crosswalks[PARCEL]
    linked_units = set(
        crosswalk.xs(new_id, level='footprint_id').index.get_level_values('parcel_id')
    )
    assert linked_units == {'U1', 'U2'}

    after = pd.read_parquet(sidecar)
    cluster_rows = after[after['footprint_id'] == new_id]
    assert set(cluster_rows['parcel_id']) == {'U1', 'U2'}
    assert (cluster_rows['area_intersection_m2'] > 0).all()
    assert (cluster_rows['link'] == 'condo cluster').all()

    # The dummy footprint's own pre-existing sidecar row is untouched.
    assert (after.loc[after['footprint_id'] == 'f_other', 'parcel_id'].isna()).all()


def _condo_cluster_with_hub_gdf():
    # Two tiny touching units (as _condo_unit_parcels_gdf above) plus a
    # much larger (~1.35 ha, comfortably under the 2.0 ha hub threshold),
    # square (aspect ratio ~1.1, under the 2.5 threshold), zero-value
    # parcel touching both from below -- the common-area/hub signature
    # real_data confirmed (Carteret County, NC): near-zero land AND
    # improvement value, compact shape, an order of magnitude larger than
    # a real unit.
    return gpd.GeoDataFrame(
        {
            'geo_id': ['U1', 'U2', 'H1'],
            'land_value': [0.0, 0.0, 0.0],
            'improvement_value': [300_000.0, 250_000.0, 0.0],
            'geometry': [
                box(0, 0, 0.0001, 0.0001),
                box(0.0001, 0, 0.0002, 0.0001),
                box(0, -0.001, 0.0011, 0),
            ],
        },
        crs='epsg:4326',
    )


def test_hub_geometry_excluded_from_consolidated_footprint(data_root, monkeypatch):
    monkeypatch.setattr(
        links_mod, 'get_entities', lambda *a, **k: _condo_cluster_with_hub_gdf()
    )
    recipe = get_recipe_by_id(SPINE)
    state = HarmonizeState(
        recipe=recipe,
        admin_id=AdminId(COUNTY),
        verbose=False,
        timer=None,
        spine=_spine_gdf(),
    )
    state = links_mod._link_spatial_overlay(
        state,
        PARCEL,
        'parcel',
        {'min_fraction_of_largest': 0.1667, 'area_intersection_m2_min': 10},
        save_link=True,
    )
    state = links_mod.consolidate_condo_cluster_footprints(state)

    new_ids = [
        i
        for i in state.spine.index
        if str(state.spine.at[i, 'geometry_source']).startswith('condo_cluster.')
    ]
    assert len(new_ids) == 1
    new_id = new_ids[0]
    consolidated = state.spine.at[new_id, 'geometry']

    # Covers both real units...
    from shapely.geometry import Point

    assert consolidated.intersects(box(0, 0, 0.0002, 0.0001))
    # ...but not the hub-exclusive area well below the units.
    assert not consolidated.intersects(Point(0.0005, -0.0005))
    assert consolidated.area < box(0, -0.001, 0.0011, 0).area

    # The hub still keeps its crosswalk link to the consolidated footprint
    # (only the *shape* excludes it -- linkage/apportionment is unaffected,
    # and its own $0 value contributes nothing either way).
    crosswalk = state.crosswalks[PARCEL]
    linked = set(
        crosswalk.xs(new_id, level='footprint_id').index.get_level_values('parcel_id')
    )
    assert linked == {'U1', 'U2', 'H1'}


def test_no_sidecar_is_a_no_op_write(state):
    state = links_mod._link_spatial_overlay(
        state,
        PARCEL,
        'parcel',
        {'min_fraction_of_largest': 0.1667, 'area_intersection_m2_min': 10},
        save_link=False,
    )
    assert not _sidecar_path().exists()

    state = links_mod.consolidate_condo_cluster_footprints(state)
    assert not _sidecar_path().exists()
