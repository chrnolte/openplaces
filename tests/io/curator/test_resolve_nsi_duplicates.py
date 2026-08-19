"""Tests for colocated NSI duplicate resolution (flag at link, exclude at merge).

flag_duplicate_points labels low-rank points (source in ignore_sources) that
share a location key with a higher-level record; _link_spatial_point applies
it when the recipe sets thresholds.resolve_duplicates; and
_attribute_point_reference excludes flagged rows from every aggregate, so an
outdated ESRI twin cannot win the value-weighted occupancy pick or inflate
the structure-value sum, even with a higher structure_value.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box

import openplaces.io.harmonizer.attributes as attrs
import openplaces.io.harmonizer.links as links
from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer.links import flag_duplicate_points

CRS = 32617  # UTM 17N (metric)


def _points(rows: list[dict], geoms: list[Point]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(pd.DataFrame(rows), geometry=geoms, crs=CRS)


def test_flag_low_rank_source_colocated_with_higher():
    ref = pd.DataFrame(
        {
            'source': ['Parcel', 'ESRI', 'Parcel'],
            'building_id_ubid': ['B1', 'B1', 'B2'],
        }
    )
    flags = flag_duplicate_points(ref, 'building_id_ubid', ['ESRI'])
    assert flags.tolist()[1] == 'colocated low-rank source'
    assert pd.isna(flags.iloc[0]) and pd.isna(flags.iloc[2])


def test_flag_hazus_colocated_with_parcel():
    ref = pd.DataFrame(
        {
            'source': ['Parcel', 'HAZUS/NSI-2015'],
            'building_id_ubid': ['B1', 'B1'],
        }
    )
    flags = flag_duplicate_points(ref, 'building_id_ubid', ['ESRI', 'HAZUS/NSI-2015'])
    assert flags.tolist()[1] == 'colocated low-rank source'
    assert pd.isna(flags.iloc[0])


def test_group_of_only_ignorable_sources_stays_unflagged():
    # Nothing better to defer to: an ESRI+HAZUS-only group keeps both rows.
    ref = pd.DataFrame(
        {
            'source': ['ESRI', 'HAZUS/NSI-2015'],
            'building_id_ubid': ['B1', 'B1'],
        }
    )
    flags = flag_duplicate_points(ref, 'building_id_ubid', ['ESRI', 'HAZUS/NSI-2015'])
    assert flags.isna().all()


def test_unique_location_never_flagged():
    ref = pd.DataFrame({'source': ['ESRI', 'Parcel'], 'building_id_ubid': ['B1', 'B2']})
    flags = flag_duplicate_points(ref, 'building_id_ubid', ['ESRI'])
    assert flags.isna().all()


def test_null_key_rows_ignored():
    ref = pd.DataFrame({'source': ['ESRI', 'Parcel'], 'building_id_ubid': [None, None]})
    flags = flag_duplicate_points(ref, 'building_id_ubid', ['ESRI'])
    assert flags.isna().all()


def _link_state():
    spine = gpd.GeoDataFrame(
        geometry=[box(0, 0, 100, 100)],
        crs=CRS,
        index=pd.Index(['F1'], name='footprint_id'),
    )
    return HarmonizeState(
        recipe={}, admin_id='US-NC-CE', verbose=False, timer=None, spine=spine
    )


def _run_link(monkeypatch, points, thresholds):
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: points)
    # save_link=False: persistence needs a real spine recipe for the
    # sidecar path and fingerprint; these tests exercise the flag logic.
    state = links._link_spatial_point(
        _link_state(), 'nsi_ref', 'building', None, thresholds, save_link=False
    )
    return state.crosswalks['nsi_ref']


def test_link_flags_by_ubid_and_keeps_rows(monkeypatch):
    points = _points(
        [
            {'source': 'Parcel', 'building_id_ubid': 'B1'},
            {'source': 'ESRI', 'building_id_ubid': 'B1'},
        ],
        [Point(50, 50), Point(50, 50)],
    )
    linked = _run_link(
        monkeypatch,
        points,
        {
            'resolve_duplicates': {
                'key': 'building_id_ubid',
                'ignore_sources': ['ESRI', 'HAZUS/NSI-2015'],
            }
        },
    )
    # Flagged, not dropped: both rows remain on the crosswalk.
    assert len(linked) == 2
    resolution = linked.set_index('source')['duplicate_resolution']
    assert resolution.loc['ESRI'] == 'colocated low-rank source'
    assert pd.isna(resolution.loc['Parcel'])


def test_link_olc_key_catches_different_ubids(monkeypatch):
    # Same ~3 m location cell, different UBIDs (differing footprint extents):
    # the 'olc' key groups them anyway.
    points = _points(
        [
            {'source': 'Parcel', 'building_id_ubid': 'B1-3-3'},
            {'source': 'ESRI', 'building_id_ubid': 'B1-2-2'},
        ],
        [Point(50.0, 50.0), Point(50.0, 51.0)],
    )
    linked = _run_link(
        monkeypatch,
        points,
        {'resolve_duplicates': {'key': 'olc', 'ignore_sources': ['ESRI']}},
    )
    resolution = linked.set_index('source')['duplicate_resolution']
    assert resolution.loc['ESRI'] == 'colocated low-rank source'
    assert pd.isna(resolution.loc['Parcel'])


def test_link_without_config_adds_no_column(monkeypatch):
    points = _points(
        [
            {'source': 'Parcel', 'building_id_ubid': 'B1'},
            {'source': 'ESRI', 'building_id_ubid': 'B1'},
        ],
        [Point(50, 50), Point(50, 50)],
    )
    linked = _run_link(monkeypatch, points, {})
    assert 'duplicate_resolution' not in linked.columns


class _Entity:
    def __init__(self, entity_type):
        self.entity_type = entity_type


def _attr_state(spine):
    return HarmonizeState(
        recipe={'entity': _Entity('footprint')},
        admin_id='US-NC-CE',
        verbose=False,
        timer=None,
        spine=spine,
    )


def test_flagged_esri_excluded_from_merge_despite_higher_value():
    # The acceptance criterion: a colocated ESRI twin with the HIGHER
    # structure_value must not win the occupancy pick, contribute to the
    # value sum, or count as a building.
    spine = pd.DataFrame(index=pd.Index(['F1'], name='footprint_id'))
    crosswalk = pd.DataFrame(
        {
            'footprint_id': ['F1', 'F1'],
            'source': ['Parcel', 'ESRI'],
            'occupancy_type': ['Single Family', 'Retail'],
            'structure_value': [200000.0, 340000.0],
            'duplicate_resolution': [pd.NA, 'colocated low-rank source'],
        }
    )
    state = attrs._attribute_point_reference(
        _attr_state(spine),
        'US_building-nsi-2022',
        'building',
        crosswalk,
        columns=['occupancy_type', 'structure_value'],
    )
    row = state.spine.loc['F1']
    assert row['occupancy_type_building_nsi'] == 'Single Family'
    assert row['structure_value_building_nsi'] == 200000.0
    assert row['n_buildings_nsi'] == 1


def test_unflagged_rows_merge_unchanged():
    # Without any flag the higher-value point still wins (existing behavior).
    spine = pd.DataFrame(index=pd.Index(['F1'], name='footprint_id'))
    crosswalk = pd.DataFrame(
        {
            'footprint_id': ['F1', 'F1'],
            'source': ['Parcel', 'ESRI'],
            'occupancy_type': ['Single Family', 'Retail'],
            'structure_value': [200000.0, 340000.0],
        }
    )
    state = attrs._attribute_point_reference(
        _attr_state(spine),
        'US_building-nsi-2022',
        'building',
        crosswalk,
        columns=['occupancy_type', 'structure_value'],
    )
    row = state.spine.loc['F1']
    assert row['occupancy_type_building_nsi'] == 'Retail'
    assert row['structure_value_building_nsi'] == 540000.0
    assert row['n_buildings_nsi'] == 2
