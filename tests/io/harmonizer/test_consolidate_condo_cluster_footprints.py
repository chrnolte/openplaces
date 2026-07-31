"""Tests for `consolidate_condo_cluster_footprints`'s link-sidecar re-persist.

Regression coverage for a real bug: `link_to_reference`'s `save_link: true`
persists the raw footprint<->parcel overlay to a parquet sidecar *before*
`consolidate_condo_cluster_footprints` runs. Without re-persisting the
sidecar after consolidation, the new consolidated footprint row has no
rows in that sidecar at all, so curate-stage readers
(`apportion_curated_values`, `collect_link_ids`) never see its cluster
parcels and the footprint resolves to a $0 structure value despite real,
positive `improvement_value` on every parcel underneath it.

Also covers the geometry-source selection logic: a real footprint linked
to a cluster's parcels is used as-is only if it adequately covers them,
never unioned together with the parcel boundaries, and any part of it
that belongs to an unrelated, non-adjacent cluster is clipped out first.
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, box

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


def _new_cluster_rows(state):
    new_ids = [
        i
        for i in state.spine.index
        if str(state.spine.at[i, 'geometry_source']).startswith('condo_cluster.')
    ]
    return new_ids


def _new_cluster_geometry(state):
    new_ids = _new_cluster_rows(state)
    assert len(new_ids) == 1
    return state.spine.at[new_ids[0], 'geometry']


def _state_with_real_footprint(data_root, monkeypatch, real_geom):
    # A real footprint alongside the ordinary, unrelated dummy footprint --
    # overlaps the two condo unit parcels well enough (>=50% of their
    # combined area, >=33% of each individually) to be linked to both by
    # _link_spatial_overlay before consolidation runs.
    monkeypatch.setattr(
        links_mod, 'get_entities', lambda *a, **k: _condo_unit_parcels_gdf()
    )
    recipe = get_recipe_by_id(SPINE)
    spine = gpd.GeoDataFrame(
        {'geometry': [box(100, 100, 101, 101), real_geom]},
        index=pd.Index(['f_other', 'f_real'], name='footprint_id'),
        crs='epsg:4326',
    )
    return HarmonizeState(
        recipe=recipe,
        admin_id=AdminId(COUNTY),
        verbose=False,
        timer=None,
        spine=spine,
    )


def test_real_footprint_covering_cluster_is_chosen_over_parcel_union(
    data_root, monkeypatch
):
    # Inset within the U1+U2 parcel union (box(0, 0, 0.0002, 0.0001)) --
    # 72% of the combined area and 72% of each unit individually, well
    # above both coverage thresholds, and visibly distinct from the raw
    # parcel boundaries so the assertion can't pass by coincidence.
    real_geom = box(0.00001, 0.00001, 0.00019, 0.00009)
    state = _state_with_real_footprint(data_root, monkeypatch, real_geom)
    state = links_mod._link_spatial_overlay(
        state,
        PARCEL,
        'parcel',
        {'min_fraction_of_largest': 0.1667, 'area_intersection_m2_min': 10},
        save_link=True,
    )
    state = links_mod.consolidate_condo_cluster_footprints(state)

    consolidated = _new_cluster_geometry(state)
    assert consolidated.equals(real_geom)
    assert not consolidated.equals(box(0, 0, 0.0002, 0.0001))


def test_real_footprint_missing_one_unit_falls_back_to_parcel_union(
    data_root, monkeypatch
):
    # Fully covers U1 but only 20% of U2 -- group coverage (60%) alone
    # would pass, but the per-parcel guard (20% < 33% for U2) must still
    # reject it, falling back to the plain parcel union rather than
    # unioning the real fragment with the parcel boundary.
    real_geom = box(0, 0, 0.00012, 0.0001)
    state = _state_with_real_footprint(data_root, monkeypatch, real_geom)
    state = links_mod._link_spatial_overlay(
        state,
        PARCEL,
        'parcel',
        {'min_fraction_of_largest': 0.1667, 'area_intersection_m2_min': 10},
        save_link=True,
    )
    state = links_mod.consolidate_condo_cluster_footprints(state)

    consolidated = _new_cluster_geometry(state)
    parcel_union = box(0, 0, 0.0002, 0.0001)
    assert consolidated.equals(parcel_union)
    assert not consolidated.equals(real_geom)


def test_real_footprint_fragment_shared_with_unrelated_cluster_is_clipped(
    data_root, monkeypatch
):
    # A messy real footprint spanning both this cluster's own parcels
    # (72% inset coverage, same shape as the "chosen as-is" case) and a
    # totally disjoint, unrelated location far away -- as if its spine id
    # were also crosswalk-linked to a different, non-adjacent cluster.
    # Only the part touching this cluster's own parcels may survive.
    near_part = box(0.00001, 0.00001, 0.00019, 0.00009)
    far_part = box(1, 1, 1.0001, 1.0001)
    real_geom = MultiPolygon([near_part, far_part])
    state = _state_with_real_footprint(data_root, monkeypatch, real_geom)
    state = links_mod._link_spatial_overlay(
        state,
        PARCEL,
        'parcel',
        {'min_fraction_of_largest': 0.1667, 'area_intersection_m2_min': 10},
        save_link=True,
    )
    state = links_mod.consolidate_condo_cluster_footprints(state)

    consolidated = _new_cluster_geometry(state)
    assert consolidated.equals(near_part)
    assert not consolidated.intersects(far_part)
    assert consolidated.bounds[2] < 1  # max x nowhere near the far part


def _ten_unit_parcels_gdf():
    # 10 touching unit parcels in a row, equal area -- U0-U8 fully real-
    # footprint-covered, U9 only 20% covered (real data: Carteret County's
    # 8764JWX6+3PV, 11 of 12 parcels 97.9-100% covered, one at 30.9%, well
    # under the old 33% per-parcel floor).
    geo_ids = [f'U{i}' for i in range(10)]
    return gpd.GeoDataFrame(
        {
            'geo_id': geo_ids,
            'land_value': [0.0] * 10,
            'improvement_value': [300_000.0] * 10,
            'geometry': [
                box(i * 0.0001, 0, (i + 1) * 0.0001, 0.0001) for i in range(10)
            ],
        },
        crs='epsg:4326',
    )


def test_one_straggler_far_below_old_floor_still_passes_smooth_score(
    data_root, monkeypatch
):
    # Real footprint covers U0-U8 fully and only 20% of U9 -- under the
    # old hard per-parcel floor (33%) this cluster would have fallen back
    # to the parcel union despite 92% overall coverage; the smooth,
    # area-weighted score must let U9's shortfall get outvoted by the
    # other 9 parcels.
    real_geom = box(0, 0, 0.00092, 0.0001)
    monkeypatch.setattr(
        links_mod, 'get_entities', lambda *a, **k: _ten_unit_parcels_gdf()
    )
    recipe = get_recipe_by_id(SPINE)
    spine = gpd.GeoDataFrame(
        {'geometry': [box(100, 100, 101, 101), real_geom]},
        index=pd.Index(['f_other', 'f_real'], name='footprint_id'),
        crs='epsg:4326',
    )
    state = HarmonizeState(
        recipe=recipe,
        admin_id=AdminId(COUNTY),
        verbose=False,
        timer=None,
        spine=spine,
    )
    state = links_mod._link_spatial_overlay(
        state,
        PARCEL,
        'parcel',
        {'min_fraction_of_largest': 0.1667, 'area_intersection_m2_min': 10},
        save_link=True,
    )
    state = links_mod.consolidate_condo_cluster_footprints(state)

    consolidated = _new_cluster_geometry(state)
    assert consolidated.equals(real_geom)


def _two_far_apart_unit_pairs_gdf():
    # Two 2-unit pairs, far enough apart (0.01 deg, ~1 km) that they never
    # touch and _cluster_condo_parcels keeps them as separate components.
    return gpd.GeoDataFrame(
        {
            'geo_id': ['U1', 'U2', 'U3', 'U4'],
            'land_value': [0.0, 0.0, 0.0, 0.0],
            'improvement_value': [300_000.0, 250_000.0, 280_000.0, 260_000.0],
            'geometry': [
                box(0, 0, 0.0001, 0.0001),
                box(0.0001, 0, 0.0002, 0.0001),
                box(0.01, 0, 0.0101, 0.0001),
                box(0.0101, 0, 0.0102, 0.0001),
            ],
        },
        crs='epsg:4326',
    )


def test_two_clusters_sharing_one_real_footprint_are_merged(data_root, monkeypatch):
    # One real footprint spans both far-apart pairs' full extent -- both
    # clusters independently pass coverage against the SAME spine id, so
    # they must merge into one row rather than each claiming the whole
    # shared polygon (which real Carteret County data showed produces
    # duplicate rows a later dedup step then silently drops entirely).
    monkeypatch.setattr(
        links_mod, 'get_entities', lambda *a, **k: _two_far_apart_unit_pairs_gdf()
    )
    recipe = get_recipe_by_id(SPINE)
    real_geom = box(0, 0, 0.0102, 0.0001)
    spine = gpd.GeoDataFrame(
        {'geometry': [box(100, 100, 101, 101), real_geom]},
        index=pd.Index(['f_other', 'f_real'], name='footprint_id'),
        crs='epsg:4326',
    )
    state = HarmonizeState(
        recipe=recipe,
        admin_id=AdminId(COUNTY),
        verbose=False,
        timer=None,
        spine=spine,
    )
    state = links_mod._link_spatial_overlay(
        state,
        PARCEL,
        'parcel',
        {'min_fraction_of_largest': 0.1667, 'area_intersection_m2_min': 10},
        save_link=True,
    )
    state = links_mod.consolidate_condo_cluster_footprints(state)

    new_ids = _new_cluster_rows(state)
    assert len(new_ids) == 1
    crosswalk = state.crosswalks[PARCEL]
    linked = set(
        crosswalk.xs(new_ids[0], level='footprint_id').index.get_level_values(
            'parcel_id'
        )
    )
    assert linked == {'U1', 'U2', 'U3', 'U4'}


def test_two_clusters_with_different_real_footprints_are_not_merged(
    data_root, monkeypatch
):
    # Same two far-apart pairs, but each covered by its OWN separate real
    # footprint -- no shared spine id, so they must stay as two distinct
    # consolidated rows.
    monkeypatch.setattr(
        links_mod, 'get_entities', lambda *a, **k: _two_far_apart_unit_pairs_gdf()
    )
    recipe = get_recipe_by_id(SPINE)
    real_geom_1 = box(0, 0, 0.0002, 0.0001)
    real_geom_2 = box(0.01, 0, 0.0102, 0.0001)
    spine = gpd.GeoDataFrame(
        {'geometry': [box(100, 100, 101, 101), real_geom_1, real_geom_2]},
        index=pd.Index(['f_other', 'f_real_1', 'f_real_2'], name='footprint_id'),
        crs='epsg:4326',
    )
    state = HarmonizeState(
        recipe=recipe,
        admin_id=AdminId(COUNTY),
        verbose=False,
        timer=None,
        spine=spine,
    )
    state = links_mod._link_spatial_overlay(
        state,
        PARCEL,
        'parcel',
        {'min_fraction_of_largest': 0.1667, 'area_intersection_m2_min': 10},
        save_link=True,
    )
    state = links_mod.consolidate_condo_cluster_footprints(state)

    new_ids = _new_cluster_rows(state)
    assert len(new_ids) == 2
    crosswalk = state.crosswalks[PARCEL]
    memberships = {
        frozenset(
            crosswalk.xs(nid, level='footprint_id').index.get_level_values('parcel_id')
        )
        for nid in new_ids
    }
    assert memberships == {frozenset({'U1', 'U2'}), frozenset({'U3', 'U4'})}
