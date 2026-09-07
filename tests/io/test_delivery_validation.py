"""Tests for what `export_delivery` refuses to ship, and how it reads.

A bundle is what leaves this repository, so the failures pinned here are
the ones that used to produce a file anyway: a region missing half its
counties, a bundle filed under the wrong state, a curated output whose
geometry lives in a sidecar, and a share block that cannot be delivered
as declared.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from openplaces.core.schema import AdminId, Entity
from openplaces.io import read_parquet, save_parquet
from openplaces.io.delivery import (
    _deduplicate,
    delivery_admin_id,
    delivery_paths,
    delivery_regions,
    export_delivery,
)
from openplaces.recipe import get_output_path

CANONICAL = ['admin3_id', 'lat', 'long', 'occupancy_type', 'year_built']


def _recipe(*, combined=True, columns=None, point_columns=None, admin_ids=None):
    return {
        'recipe_id': 'US_footprint-test-2026',
        'admin_id': AdminId('US'),
        'stage': 'curate',
        'entity': Entity('footprint', 'test', '2026'),
        'process_by': {'admin_level': 3},
        'save_to': {'data_dir': 'share', 'combined': combined},
        'share': {
            'columns': CANONICAL if columns is None else columns,
            'point_columns': ['structure_value_per_area']
            if point_columns is None
            else point_columns,
            'delivery': {
                'admin_level': 2,
                'admin_ids': admin_ids or ['US-NC-AL', 'US-NC-BB'],
            },
        },
    }


def _county(recipe, footprint_ids, *, admin3_id):
    n = len(footprint_ids)
    frame = gpd.GeoDataFrame(
        {
            'lat': [40.0 + i for i in range(n)],
            'long': [-80.0 - i for i in range(n)],
            'occupancy_type': ['Single-Family'] * n,
            'year_built': [1990] * n,
            'structure_value_per_area': [100.0 + i for i in range(n)],
            'occupancy_type_source': ['parcel'] * n,
            'geometry_source': ['obm'] * n,
        },
        geometry=[box(i, i, i + 1, i + 1) for i in range(n)],
        crs='EPSG:4326',
        index=pd.Index(footprint_ids, name='footprint_id'),
    )
    save_parquet(
        frame,
        get_output_path(recipe, admin3_id),
        combined=bool(recipe['save_to'].get('combined')),
    )
    return frame


def test_a_declared_member_without_curated_output_stops_the_ship(mock_data_root):
    recipe = _recipe()
    _county(recipe, ['a', 'b'], admin3_id='US-NC-AL')

    with pytest.raises(FileNotFoundError, match='US-NC-BB'):
        export_delivery(recipe)


def test_split_layout_curated_output_delivers(mock_data_root):
    """Geometry in a `_geo` sidecar, joined on the key the read must ask for."""
    recipe = _recipe(combined=False)
    _county(recipe, ['a', 'b'], admin3_id='US-NC-AL')
    _county(recipe, ['c', 'd'], admin3_id='US-NC-BB')

    paths = export_delivery(recipe)

    canonical = read_parquet(paths['canonical'])
    geo = read_parquet(paths['geo'], geom=True)
    assert list(canonical.index) == ['a', 'b', 'c', 'd']
    assert list(geo.index) == ['a', 'b', 'c', 'd']
    assert geo.geometry.notna().all()


def test_a_two_state_region_without_its_own_unit_raises():
    recipe = _recipe(admin_ids=['US-NC-AL', 'US-SC-AB'])

    with pytest.raises(ValueError, match='outside US-NC'):
        delivery_admin_id(recipe)


@pytest.mark.parametrize(
    ('columns', 'point_columns', 'message'),
    [
        (CANONICAL, ['year_built'], 'more than once'),
        ([*CANONICAL, 'occupancy_type_source'], [], 'provenance sidecar'),
        (CANONICAL, ['lat'], 'point_columns'),
        (['admin3_id', 'occupancy_type'], [], 'does not declare'),
    ],
)
def test_undeliverable_share_block_raises_before_anything_is_written(
    mock_data_root, columns, point_columns, message
):
    recipe = _recipe(columns=columns, point_columns=point_columns)
    _county(_recipe(), ['a'], admin3_id='US-NC-AL')
    _county(_recipe(), ['b'], admin3_id='US-NC-BB')

    with pytest.raises(ValueError, match=message):
        export_delivery(recipe)
    # Nothing was written: the check runs before the first output file.
    assert not any(path.exists() for path in delivery_paths(recipe).values())


def test_deduplicate_breaks_ties_on_the_lower_admin_id():
    """Equal coverage must not resolve by read order."""
    frame = pd.DataFrame(
        {
            'year_built': [1990, 1974],
            'admin3_id': ['US-NC-BB', 'US-NC-AL'],
        },
        index=pd.Index(['shared', 'shared'], name='footprint_id'),
    )

    kept, n_dropped = _deduplicate(frame, ['year_built'], 'admin3_id')

    assert n_dropped == 1
    assert kept['admin3_id'].tolist() == ['US-NC-AL']
    assert kept['year_built'].tolist() == [1974]


def test_region_registry_is_read_once_per_call(monkeypatch):
    """The CSV was re-parsed for every declared region."""
    import openplaces.io.readers as readers

    calls = []
    real = readers.get_regions

    def counted(region_id=None):
        calls.append(region_id)
        return real(region_id)

    monkeypatch.setattr(readers, 'get_regions', counted)
    regions = delivery_regions('US_footprint-openplaces-2026')

    assert len(regions) > 1
    assert calls == [None]


def test_a_failed_reship_leaves_the_previous_bundle_intact(mock_data_root, monkeypatch):
    """Five files from one curation, or the five from the last one."""
    import openplaces.io.delivery as delivery

    recipe = _recipe()
    _county(recipe, ['a', 'b'], admin3_id='US-NC-AL')
    _county(recipe, ['c', 'd'], admin3_id='US-NC-BB')
    paths = export_delivery(recipe)
    before = read_parquet(paths['canonical'])

    # A second curation with different rows, and a failure after the
    # canonical file has been written.
    _county(recipe, ['e', 'f'], admin3_id='US-NC-AL')

    def _explode(*args, **kwargs):
        raise RuntimeError('interrupted')

    monkeypatch.setattr(delivery, 'bundle_terms', _explode)
    with pytest.raises(RuntimeError, match='interrupted'):
        export_delivery(recipe)

    after = read_parquet(paths['canonical'])
    pd.testing.assert_frame_equal(before, after)
    assert list(after.index) == ['a', 'b', 'c', 'd']


def test_unknown_region_still_raises_the_registry_error():
    recipe = _recipe()
    recipe['share']['delivery'] = {'admin_level': 2, 'regions': ['not-a-region']}

    with pytest.raises(KeyError, match='Unknown region'):
        delivery_regions(recipe)


def test_an_attribute_that_merely_ends_in_source_is_deliverable(mock_data_root):
    """`property_source` is an attribute, not a sidecar of a shared column."""
    recipe = _recipe(columns=[*CANONICAL, 'property_source'])
    _county(recipe, ['a'], admin3_id='US-NC-AL')
    _county(recipe, ['b'], admin3_id='US-NC-BB')

    canonical = read_parquet(export_delivery(recipe)['canonical'])

    assert 'property_source' in canonical.columns
