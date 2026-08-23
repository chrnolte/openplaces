"""Tests for `export_delivery`, the shareable split of a curated entity.

The bundle's contract is that its four data files carry the same index and
that between them they still hold every curated column, so a consumer can
load one part and rejoin the rest. A fifth output, the licence notice, is
text: it travels with the data because the obligations do.
"""

from __future__ import annotations

import os

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from openplaces.core.schema import AdminId, Entity
from openplaces.io import read_parquet, save_parquet
from openplaces.io.delivery import (
    delivery_admin_id,
    delivery_members,
    delivery_paths,
    export_delivery,
)
from openplaces.recipe import get_output_path

CANONICAL = ['admin3_id', 'lat', 'long', 'occupancy_type', 'year_built']
POINT_EXTRA = ['structure_value_per_area']


def _recipe():
    return {
        'recipe_id': 'US_footprint-test-2026',
        'admin_id': AdminId('US'),
        'stage': 'curate',
        'entity': Entity('footprint', 'test', '2026'),
        'process_by': {'admin_level': 3},
        'save_to': {'data_dir': 'share', 'combined': True},
        'share': {
            'columns': CANONICAL,
            'point_columns': POINT_EXTRA,
            'delivery': {'admin_level': 2, 'admin_ids': ['US-NC-AL', 'US-NC-BB']},
        },
    }


def _county(footprint_ids, *, admin3_id, year_built=None, extra=None):
    """Write one county's curated file; return the frame that was written."""
    n = len(footprint_ids)
    frame = gpd.GeoDataFrame(
        {
            'lat': [40.0 + i for i in range(n)],
            'long': [-80.0 - i for i in range(n)],
            'occupancy_type': ['Single-Family'] * n,
            'year_built': year_built if year_built is not None else [1990] * n,
            'structure_value_per_area': [100.0 + i for i in range(n)],
            'occupancy_type_source': ['parcel'] * n,
            'geometry_source': ['obm'] * n,
            'occupancy_type_building_nsi': ['RES1'] * n,
            'n_stories_building_cheer': [2.0] * n,
            **(extra or {}),
        },
        geometry=[box(i, i, i + 1, i + 1) for i in range(n)],
        crs='EPSG:4326',
        index=pd.Index(footprint_ids, name='footprint_id'),
    )
    save_parquet(frame, get_output_path(_recipe(), admin3_id), combined=True)
    return frame


@pytest.fixture
def two_counties(mock_data_root):
    _county(['a', 'b'], admin3_id='US-NC-AL')
    _county(['c', 'd'], admin3_id='US-NC-BB')
    return ['US-NC-AL', 'US-NC-BB']


def _bundle(admin_ids):
    return export_delivery(_recipe(), 'US-NC', admin_ids=admin_ids)


DATA_ROLES = ('canonical', 'point', 'geo', 'evidence')


def test_writes_four_files_sharing_one_index(two_counties):
    paths = _bundle(two_counties)

    # Five outputs, but only four of them are data: the fifth is the
    # licence notice, which is text and shares no index with anything.
    assert set(paths) == {*DATA_ROLES, 'terms'}
    indexes = {}
    for role in DATA_ROLES:
        path = paths[role]
        assert path.exists(), role
        indexes[role] = list(read_parquet(path, geom=role in ('point', 'geo')).index)
    # Identical order, not merely identical membership: a consumer should be
    # able to concatenate two parts column-wise without aligning first.
    assert indexes['canonical'] == ['a', 'b', 'c', 'd']
    assert len(set(map(tuple, indexes.values()))) == 1


def test_canonical_holds_declared_columns_then_sources(two_counties):
    canonical = read_parquet(_bundle(two_counties)['canonical'])

    assert list(canonical.columns) == [
        *CANONICAL,
        'occupancy_type_source',
        'geometry_source',
    ]
    assert canonical.index.name == 'footprint_id'
    assert 'geometry' not in canonical.columns


def test_point_file_swaps_coordinates_for_point_geometry(two_counties):
    point = read_parquet(_bundle(two_counties)['point'], geom=True)

    assert isinstance(point, gpd.GeoDataFrame)
    assert point.geometry.geom_type.eq('Point').all()
    assert 'lat' not in point.columns
    assert 'long' not in point.columns
    assert 'structure_value_per_area' in point.columns
    assert point.loc['a', 'geometry'] == Point(-80.0, 40.0)


def test_geo_file_holds_polygons_only(two_counties):
    geo = read_parquet(_bundle(two_counties)['geo'], geom=True)

    assert list(geo.columns) == ['geometry']
    assert geo.geometry.geom_type.eq('Polygon').all()
    assert geo.crs == 'EPSG:4326'


def test_evidence_holds_the_remainder_plus_sources(two_counties):
    evidence = read_parquet(_bundle(two_counties)['evidence'])

    assert 'occupancy_type_building_nsi' in evidence.columns
    assert 'n_stories_building_cheer' in evidence.columns
    # Sources are duplicated so either file is readable on its own.
    assert 'occupancy_type_source' in evidence.columns
    # Canonical values are not.
    assert 'occupancy_type' not in evidence.columns
    assert 'geometry' not in evidence.columns


def test_bundle_covers_every_curated_column(two_counties):
    paths = _bundle(two_counties)
    delivered = {
        *read_parquet(paths['canonical']).columns,
        *read_parquet(paths['point'], geom=True).columns,
        *read_parquet(paths['evidence']).columns,
        'geometry',
    }
    curated = set(
        read_parquet(get_output_path(_recipe(), 'US-NC-AL'), geom=True).columns
    )
    assert not curated - delivered


def test_shared_id_keeps_the_better_covered_copy(mock_data_root):
    # 'shared' straddles the county line and is curated by both counties;
    # only US-NC-BB has a year_built for it.
    _county(['a', 'shared'], admin3_id='US-NC-AL', year_built=[1990, None])
    _county(['shared', 'd'], admin3_id='US-NC-BB', year_built=[1974, 1990])

    paths = _bundle(['US-NC-AL', 'US-NC-BB'])
    canonical = read_parquet(paths['canonical'])
    evidence = read_parquet(paths['evidence'])

    assert list(canonical.index) == ['a', 'd', 'shared']
    assert canonical.loc['shared', 'year_built'] == 1974
    assert canonical.loc['shared', 'admin3_id'] == 'US-NC-BB'
    # The evidence row must come from the same county as the kept values.
    assert list(evidence.index) == ['a', 'd', 'shared']


def test_county_missing_a_declared_column_still_contributes(mock_data_root):
    _county(['a', 'b'], admin3_id='US-NC-AL')
    frame = _county(['c'], admin3_id='US-NC-BB')
    save_parquet(
        frame.drop(columns='occupancy_type'),
        get_output_path(_recipe(), 'US-NC-BB'),
        combined=True,
    )

    canonical = read_parquet(_bundle(['US-NC-AL', 'US-NC-BB'])['canonical'])

    assert list(canonical.index) == ['a', 'b', 'c']
    assert pd.isna(canonical.loc['c', 'occupancy_type'])
    assert canonical.loc['a', 'occupancy_type'] == 'Single-Family'


def test_recipe_without_share_block_raises(two_counties):
    recipe = {k: v for k, v in _recipe().items() if k != 'share'}

    with pytest.raises(ValueError, match='share'):
        export_delivery(recipe, 'US-NC', admin_ids=two_counties)


def test_no_curated_output_raises(mock_data_root):
    with pytest.raises(FileNotFoundError, match='nothing to deliver'):
        export_delivery(_recipe(), 'US-NC', admin_ids=['US-NC-ZZ'])


def test_recipe_declares_the_whole_delivery(two_counties):
    """A fully declared recipe delivers with no arguments but itself."""
    assert str(delivery_admin_id(_recipe())) == 'US-NC'
    assert delivery_members(_recipe()) == ['US-NC-AL', 'US-NC-BB']

    paths = export_delivery(_recipe())

    assert paths == delivery_paths(_recipe())
    assert list(read_parquet(paths['canonical']).index) == ['a', 'b', 'c', 'd']


def test_declared_members_win_over_the_admin_hierarchy(mock_data_root):
    """A county outside the declared region must not slip into the bundle."""
    _county(['a'], admin3_id='US-NC-AL')
    _county(['b'], admin3_id='US-NC-BB')
    _county(['z'], admin3_id='US-NC-ZZ')

    canonical = read_parquet(export_delivery(_recipe())['canonical'])

    assert list(canonical.index) == ['a', 'b']


def test_shipped_files_are_left_read_only(two_counties):
    paths = export_delivery(_recipe())

    for role, path in paths.items():
        assert not os.access(path, os.W_OK), role


def test_a_second_export_unlocks_its_own_outputs(two_counties):
    """Read-only must guard against other writers, not against reshipping."""
    export_delivery(_recipe())
    paths = export_delivery(_recipe())

    assert list(read_parquet(paths['canonical']).index) == ['a', 'b', 'c', 'd']


def test_ships_a_licence_notice_beside_the_data(two_counties):
    """The obligations travel with the bundle, not in someone's memory."""
    notice = _bundle(two_counties)['terms']

    assert notice.exists()
    text = notice.read_text(encoding='utf-8')
    # Names the recipe it describes and says who decides about sharing.
    assert 'openplaces' in text
    assert 'Sources' in text
