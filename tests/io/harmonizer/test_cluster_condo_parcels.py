"""Tests for `_cluster_condo_parcels` (attributes.py)."""

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from openplaces.io.harmonizer.attributes import _cluster_condo_parcels


def _parcels_with_hub():
    return gpd.GeoDataFrame(
        {
            'land_value': [0.0, 0.0, 0.0],
            'improvement_value': [300_000.0, 250_000.0, 0.0],
            'area_ha': [0.012, 0.011, 1.35],
            'geometry': [
                box(0, 0, 0.0001, 0.0001),
                box(0.0001, 0, 0.0002, 0.0001),
                box(0, -0.001, 0.0011, 0),
            ],
        },
        index=pd.Index(['U1', 'U2', 'H1'], name='parcel_id'),
        crs='epsg:4326',
    )


def test_hub_ids_identifies_the_common_area_parcel():
    result = _cluster_condo_parcels(_parcels_with_hub(), min_group_size=2)
    assert result is not None
    component, hub_ids = result
    assert set(component.index) == {'U1', 'U2', 'H1'}
    # All three resolve to the same cluster id.
    assert component.nunique() == 1
    assert hub_ids == {'H1'}


def test_no_hub_when_all_members_are_real_units():
    parcels = gpd.GeoDataFrame(
        {
            'land_value': [0.0, 0.0],
            'improvement_value': [300_000.0, 250_000.0],
            'area_ha': [0.012, 0.011],
            'geometry': [
                box(0, 0, 0.0001, 0.0001),
                box(0.0001, 0, 0.0002, 0.0001),
            ],
        },
        index=pd.Index(['U1', 'U2'], name='parcel_id'),
        crs='epsg:4326',
    )
    result = _cluster_condo_parcels(parcels, min_group_size=2)
    assert result is not None
    component, hub_ids = result
    assert set(component.index) == {'U1', 'U2'}
    assert hub_ids == set()


def test_returns_none_when_no_candidates():
    parcels = gpd.GeoDataFrame(
        {
            'land_value': [50_000.0],
            'improvement_value': [200_000.0],
            'area_ha': [0.5],
            'geometry': [box(0, 0, 0.001, 0.001)],
        },
        index=pd.Index(['P1'], name='parcel_id'),
        crs='epsg:4326',
    )
    assert _cluster_condo_parcels(parcels, min_group_size=2) is None


def test_hub_does_not_bridge_disconnected_unit_groups():
    # Two 2-unit groups, ~55 m apart and never touching each other
    # directly, both touching the bottom edge of one shared hub -- the
    # exact shape of the real bug found in Carteret County, NC, where a
    # single shared COMMON AREA parcel bridged 3 physically disjoint
    # townhouse-unit groups into one bogus component.
    parcels = gpd.GeoDataFrame(
        {
            'land_value': [0.0, 0.0, 0.0, 0.0, 0.0],
            'improvement_value': [300_000.0, 250_000.0, 280_000.0, 260_000.0, 0.0],
            'area_ha': [0.012, 0.011, 0.012, 0.011, 0.6],
            'geometry': [
                box(0, 0, 0.0001, 0.0001),
                box(0.0001, 0, 0.0002, 0.0001),
                box(0.0007, 0, 0.0008, 0.0001),
                box(0.0008, 0, 0.0009, 0.0001),
                box(0, -0.001, 0.0009, 0),
            ],
        },
        index=pd.Index(['U1', 'U2', 'U3', 'U4', 'H1'], name='parcel_id'),
        crs='epsg:4326',
    )
    result = _cluster_condo_parcels(parcels, min_group_size=2)
    assert result is not None
    component, hub_ids = result
    assert hub_ids == {'H1'}

    groups = component.groupby(component).groups
    assert len(groups) == 2
    memberships = {frozenset(v) for v in groups.values()}
    assert memberships == {
        frozenset({'U1', 'U2', 'H1'}),
        frozenset({'U3', 'U4', 'H1'}),
    }
    # H1 is attached to both clusters, not merged into one.
    assert (component.index == 'H1').sum() == 2


def test_unit_candidacy_raised_to_new_default_area_threshold():
    # ~215 m^2 (0.0215 ha) each -- just over the old 0.02 ha cutoff,
    # comfortably under the new 0.05 ha default. Matches the real edge-
    # unit parcels found in Carteret County, NC that fell out of
    # clustering entirely under the old threshold.
    parcels = gpd.GeoDataFrame(
        {
            'land_value': [0.0, 0.0],
            'improvement_value': [247_572.0, 247_572.0],
            'area_ha': [0.0215, 0.0215],
            'geometry': [
                box(0, 0, 0.0001, 0.0001),
                box(0.0001, 0, 0.0002, 0.0001),
            ],
        },
        index=pd.Index(['E1', 'E2'], name='parcel_id'),
        crs='epsg:4326',
    )
    assert (
        _cluster_condo_parcels(parcels, min_group_size=2, max_unit_area_ha=0.02) is None
    )

    result = _cluster_condo_parcels(parcels, min_group_size=2)
    assert result is not None
    component, _hub_ids = result
    assert set(component.index) == {'E1', 'E2'}


def test_use_subgroup_blocklist_excludes_single_family_and_mobile_home():
    # SF1 is small and touches U2, so it would join the cluster under the
    # area/value test alone -- the use_subgroup blocklist ('RESIDENTIAL
    # PRIMARY', Carteret County's own single-family label, not the literal
    # text 'single family') must still keep it out.
    parcels = gpd.GeoDataFrame(
        {
            'land_value': [0.0, 0.0, 35_000.0],
            'improvement_value': [247_572.0, 247_572.0, 150_000.0],
            'area_ha': [0.012, 0.011, 0.015],
            'use_subgroup': [
                'RESIDENTIAL TOWNHOUSE',
                'RESIDENTIAL TOWNHOUSE',
                'RESIDENTIAL PRIMARY',
            ],
            'geometry': [
                box(0, 0, 0.0001, 0.0001),
                box(0.0001, 0, 0.0002, 0.0001),
                box(0.0002, 0, 0.0003, 0.0001),
            ],
        },
        index=pd.Index(['U1', 'U2', 'SF1'], name='parcel_id'),
        crs='epsg:4326',
    )
    result = _cluster_condo_parcels(parcels, min_group_size=2)
    assert result is not None
    component, _hub_ids = result
    assert set(component.index) == {'U1', 'U2'}


def _large_resort_row_gdf(n=100):
    # n touching unit parcels in a row -- a large but legitimate resort/
    # motel-style complex (real data: Carteret County, NC's "1505 Salter
    # Path Rd" has a 96-parcel hub-anchored cluster), between the old
    # (60) and new (200) max_group_size cap.
    geo_ids = [f'U{i}' for i in range(n)]
    return gpd.GeoDataFrame(
        {
            'land_value': [0.0] * n,
            'improvement_value': [300_000.0] * n,
            'area_ha': [0.003] * n,
            'geometry': [
                box(i * 0.0001, 0, (i + 1) * 0.0001, 0.0001) for i in range(n)
            ],
        },
        index=pd.Index(geo_ids, name='parcel_id'),
        crs='epsg:4326',
    )


def test_cluster_between_old_and_new_group_size_cap_is_recognized():
    parcels = _large_resort_row_gdf(100)
    assert _cluster_condo_parcels(parcels, min_group_size=2, max_group_size=60) is None

    result = _cluster_condo_parcels(parcels, min_group_size=2)
    assert result is not None
    component, _hub_ids = result
    assert set(component.index) == set(parcels.index)


def test_hub_between_old_and_new_area_cap_is_recognized():
    # A 2.5 ha, near-square (aspect ~1.18, well under the 2.5 threshold)
    # common-area parcel -- between the old (2.0 ha) and new (3.0 ha)
    # max_hub_area_ha cap. Real data: Carteret County, NC's 8764JWW6+WV3
    # sits on a 2.63 ha COMMON AREA parcel excluded under the old default.
    # U1/U2 don't touch each other directly (a gap between them) -- the
    # hub is the *only* thing connecting them, so disqualifying it must
    # leave no qualifying cluster at all, not just drop the hub from one.
    parcels = gpd.GeoDataFrame(
        {
            'land_value': [0.0, 0.0, 0.0],
            'improvement_value': [300_000.0, 250_000.0, 0.0],
            'area_ha': [0.012, 0.011, 2.5],
            'geometry': [
                box(0, 0, 0.0001, 0.0001),
                box(0.0003, 0, 0.0004, 0.0001),
                box(0, -0.0013, 0.0011, 0),
            ],
        },
        index=pd.Index(['U1', 'U2', 'H1'], name='parcel_id'),
        crs='epsg:4326',
    )
    assert (
        _cluster_condo_parcels(parcels, min_group_size=2, max_hub_area_ha=2.0) is None
    )

    result = _cluster_condo_parcels(parcels, min_group_size=2)
    assert result is not None
    component, hub_ids = result
    assert set(component.index) == {'U1', 'U2', 'H1'}
    assert hub_ids == {'H1'}
