"""Tests for the `join_temporal_snapshot` curate step.

Uses the real `US-FL_property-fldor-2026` recipe id (so `get_entities`
resolves a real recipe definition) but writes small, fabricated property
rows to its output path under a temporary data root -- no real property
or transaction data, per AGENTS.md's no-personal-data rule.
"""

from __future__ import annotations

import pandas as pd
import pytest

from openplaces.config import cfg
from openplaces.core.schema import AdminId
from openplaces.io import to_parquet
from openplaces.io.curator import CurateState
from openplaces.io.curator.transactions import join_temporal_snapshot
from openplaces.recipe import get_output_path, get_recipe_by_id

PROPERTY_RECIPE_ID = 'US-FL_property-fldor-2026'
COUNTY = 'US-FL-PO'


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


def _write_property_snapshots(rows):
    recipe = get_recipe_by_id(PROPERTY_RECIPE_ID)
    df = pd.DataFrame(rows)
    path = get_output_path(recipe, AdminId(COUNTY), partition_id='all')
    to_parquet(df, path)


def _state(df: pd.DataFrame) -> CurateState:
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId(COUNTY),
        verbose=False,
        timer=None,
        curated=df,
    )


def test_backward_match_prefers_same_year(data_root):
    _write_property_snapshots(
        [
            {'parcel_id_assessor': 'A', 'tax_year': 2019, 'year_built': 1990},
            {'parcel_id_assessor': 'A', 'tax_year': 2020, 'year_built': 1990},
        ]
    )
    tx = pd.DataFrame({'parcel_id_assessor': ['A'], 'sale_year': [2020]})
    out = join_temporal_snapshot(
        _state(tx),
        recipe_id=PROPERTY_RECIPE_ID,
        join_key='parcel_id_assessor',
        date_column='sale_year',
        vintage_column='tax_year',
        direction='backward',
        match_type_column='property_match_type',
        columns=['tax_year', 'year_built'],
    ).curated
    assert out['tax_year'].iloc[0] == 2020
    assert out['property_match_type'].iloc[0] == 'exact'


def test_backward_match_falls_back_to_prior_year(data_root):
    _write_property_snapshots(
        [{'parcel_id_assessor': 'A', 'tax_year': 2018, 'year_built': 1990}]
    )
    tx = pd.DataFrame({'parcel_id_assessor': ['A'], 'sale_year': [2020]})
    out = join_temporal_snapshot(
        _state(tx),
        recipe_id=PROPERTY_RECIPE_ID,
        join_key='parcel_id_assessor',
        date_column='sale_year',
        vintage_column='tax_year',
        direction='backward',
        match_type_column='property_match_type',
        columns=['tax_year', 'year_built'],
    ).curated
    assert out['tax_year'].iloc[0] == 2018
    assert out['property_match_type'].iloc[0] == 'backward_fallback'


def test_backward_never_matches_a_later_year(data_root):
    _write_property_snapshots(
        [{'parcel_id_assessor': 'A', 'tax_year': 2022, 'year_built': 2021}]
    )
    tx = pd.DataFrame({'parcel_id_assessor': ['A'], 'sale_year': [2020]})
    out = join_temporal_snapshot(
        _state(tx),
        recipe_id=PROPERTY_RECIPE_ID,
        join_key='parcel_id_assessor',
        date_column='sale_year',
        vintage_column='tax_year',
        direction='backward',
        match_type_column='property_match_type',
        columns=['tax_year', 'year_built'],
    ).curated
    assert pd.isna(out['tax_year'].iloc[0])


def test_only_unmatched_and_restrict_to_scope_the_forward_pass(data_root):
    _write_property_snapshots(
        [{'parcel_id_assessor': 'A', 'tax_year': 2022, 'year_built': 2021}]
    )
    tx = pd.DataFrame(
        {
            'parcel_id_assessor': ['A', 'B'],
            'sale_year': [2020, 2020],
            'sale_vacant': ['Improved property', 'Vacant land'],
            'property_match_type': [pd.NA, pd.NA],
        }
    )
    out = join_temporal_snapshot(
        _state(tx),
        recipe_id=PROPERTY_RECIPE_ID,
        join_key='parcel_id_assessor',
        date_column='sale_year',
        vintage_column='tax_year',
        direction='forward',
        only_unmatched=True,
        restrict_to={'column': 'sale_vacant', 'equals': 'Improved property'},
        match_type_column='property_match_type',
        columns=['tax_year', 'year_built'],
    ).curated
    # A: improved and unmatched -> forward-matched to 2022.
    assert out.loc[out['parcel_id_assessor'] == 'A', 'tax_year'].iloc[0] == 2022
    # B: vacant -> restrict_to excludes it even though unmatched.
    assert pd.isna(out.loc[out['parcel_id_assessor'] == 'B', 'tax_year'].iloc[0])


def test_exact_offset_requires_precise_year_and_uses_prefix(data_root):
    _write_property_snapshots(
        [
            {
                'parcel_id_assessor': 'A',
                'tax_year': 2020,
                'use_group_code': 'Vacant Residential',
            },
            {
                'parcel_id_assessor': 'A',
                'tax_year': 2021,
                'use_group_code': 'Single Family',
            },
        ]
    )
    tx = pd.DataFrame({'parcel_id_assessor': ['A'], 'sale_year': [2020]})
    out = join_temporal_snapshot(
        _state(tx),
        recipe_id=PROPERTY_RECIPE_ID,
        join_key='parcel_id_assessor',
        date_column='sale_year',
        vintage_column='tax_year',
        offset_years=1,
        direction='exact',
        prefix='next_year_',
        columns=['use_group_code'],
    ).curated
    assert out['next_year_use_group_code'].iloc[0] == 'Single Family'


def test_exact_offset_requires_offset_years():
    tx = pd.DataFrame({'parcel_id_assessor': ['A'], 'sale_year': [2020]})
    with pytest.raises(ValueError):
        join_temporal_snapshot(
            _state(tx),
            recipe_id=PROPERTY_RECIPE_ID,
            join_key='parcel_id_assessor',
            date_column='sale_year',
            vintage_column='tax_year',
            direction='exact',
            columns=['use_group_code'],
        )


def test_join_key_normalized_across_dash_format_boundary(data_root):
    # Dashed on the property side, bare digits on the transaction side --
    # the same physical parcel, different raw string.
    _write_property_snapshots(
        [
            {
                'parcel_id_assessor': '11-20-26-0300-000-12800',
                'tax_year': 2020,
                'year_built': 1990,
            }
        ]
    )
    tx = pd.DataFrame(
        {'parcel_id_assessor': ['112026030000012800'], 'sale_year': [2020]}
    )
    out = join_temporal_snapshot(
        _state(tx),
        recipe_id=PROPERTY_RECIPE_ID,
        join_key='parcel_id_assessor',
        date_column='sale_year',
        vintage_column='tax_year',
        direction='backward',
        columns=['tax_year', 'year_built'],
    ).curated
    assert out['tax_year'].iloc[0] == 2020
