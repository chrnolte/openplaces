"""Tests for a bundle whose entity has no geometry of its own.

A transaction is an event, not a place: its location belongs to the
parcel it names, and a deed covering several parcels has no single
point at all. Such a recipe declares `share: geometry: false` and ships
the canonical table and its evidence supplement, with no point or
boundary file, rather than a centroid invented to fill one.
"""

from __future__ import annotations

import pandas as pd
import pytest

from openplaces.core.schema import AdminId, Entity
from openplaces.io import read_parquet, save_parquet
from openplaces.io.delivery import (
    _share_spec,
    delivery_paths,
    export_delivery,
    share_is_spatial,
)
from openplaces.recipe import get_output_path, get_recipe_by_id

CANONICAL = ['admin3_id', 'price', 'sale_year', 'parcel_id_local']


def _recipe(**share_overrides):
    share = {
        'columns': CANONICAL,
        'geometry': False,
        'delivery': {'admin_level': 2, 'admin_ids': ['US-NC-AL', 'US-NC-BB']},
    }
    share.update(share_overrides)
    return {
        'recipe_id': 'US_transaction-test-2026',
        'admin_id': AdminId('US'),
        'stage': 'curate',
        'entity': Entity('transaction', 'test', '2026'),
        'process_by': {'admin_level': 3},
        'save_to': {'data_dir': 'share'},
        'share': share,
    }


def _county(sale_ids, *, admin3_id):
    n = len(sale_ids)
    frame = pd.DataFrame(
        {
            'admin3_id': [admin3_id] * n,
            'price': [100_000.0 + i for i in range(n)],
            'sale_year': [2020 + i for i in range(n)],
            'parcel_id_local': [f'p{i}' for i in range(n)],
            'price_source': ['recorder'] * n,
            # An evidence column, to prove the supplement still ships.
            'sale_qualification_code': ['01'] * n,
        },
        index=pd.Index(sale_ids, name='transaction_id'),
    )
    save_parquet(frame, get_output_path(_recipe(), admin3_id))
    return frame


@pytest.fixture
def two_counties(mock_data_root):
    _county(['a', 'b'], admin3_id='US-NC-AL')
    _county(['c', 'd'], admin3_id='US-NC-BB')
    return ['US-NC-AL', 'US-NC-BB']


def test_paths_omit_the_point_and_boundary_files():
    """The path set is the orchestrator's output declaration too.

    Leaving them in would make a delivery job wait on two files nothing
    ever writes.
    """
    paths = delivery_paths(_recipe(), 'US-NC')

    assert set(paths) == {'canonical', 'evidence', 'terms'}


def test_a_non_spatial_share_needs_no_coordinates():
    """No `long`/`lat` demanded, since there is no point file."""
    recipe = _recipe()

    assert share_is_spatial(recipe) is False
    columns, point_columns = _share_spec(recipe)
    assert columns == CANONICAL
    assert point_columns == []


def test_a_spatial_share_still_demands_coordinates():
    """The default is unchanged: every existing recipe stays spatial."""
    recipe = _recipe(geometry=True)

    assert share_is_spatial(recipe) is True
    with pytest.raises(ValueError, match='long, lat'):
        _share_spec(recipe)


def test_point_columns_without_a_point_file_are_refused():
    """Declaring point-only columns with no point file is a recipe error.

    Silently dropping them would ship a bundle missing columns the
    recipe says it carries.
    """
    recipe = _recipe(point_columns=['sale_price_to_assessed_ratio'])

    with pytest.raises(ValueError, match='no point file'):
        _share_spec(recipe)


def test_exports_canonical_and_evidence_only(two_counties):
    """End to end: two files of data, plus the licence notice."""
    paths = export_delivery(_recipe(), 'US-NC', admin_ids=two_counties)

    assert set(paths) == {'canonical', 'evidence', 'terms'}
    canonical = read_parquet(paths['canonical'])
    assert list(canonical.index) == ['a', 'b', 'c', 'd']
    assert [c for c in CANONICAL if c not in canonical.columns] == []
    # The sidecar of a canonical column rides along, as in a spatial
    # bundle.
    assert 'price_source' in canonical.columns
    # Everything else is still delivered, in the supplement.
    evidence = read_parquet(paths['evidence'])
    assert 'sale_qualification_code' in evidence.columns
    assert list(evidence.index) == list(canonical.index)


def test_the_shipped_transaction_recipe_is_non_spatial_and_carries_no_person():
    """The real recipe, not a fixture.

    `export_delivery` refuses a personal column by itself, but the
    declaration should be right before that check has to fire: a
    recorder's table names both parties to every deed.
    """
    recipe = get_recipe_by_id('US_transaction-openplaces-2026')

    assert share_is_spatial(recipe) is False
    columns, point_columns = _share_spec(recipe)
    assert point_columns == []
    assert 'grantor' not in columns
    assert 'grantee' not in columns
    # Both declared regions resolve, so neither bundle ships under the
    # other's name.
    paths = delivery_paths(recipe, region='wisconsin-statewide')
    assert paths['canonical'].name.startswith('US-WI_')
    paths = delivery_paths(recipe, region='florida-statewide')
    assert paths['canonical'].name.startswith('US-FL_')
