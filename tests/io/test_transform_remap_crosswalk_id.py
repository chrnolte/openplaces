"""Tests for the remap_file transformation's crosswalk_id variant.

`crosswalk_id` resolves a recipe-relative crosswalk asset (e.g. a
'*-remap.csv' beside the recipe) by recipe id, like the harmonizer's
remap_id — no filesystem path in the recipe.
"""

from __future__ import annotations

import pandas as pd

import openplaces.io.transform as transform_mod
from openplaces.io.transform import apply_transformation


def test_remap_file_accepts_crosswalk_id(monkeypatch):
    crosswalk = pd.Series(
        {'Manufactured Home': 'Manufactured Home', 'Unclassified': None}
    )
    monkeypatch.setattr(transform_mod, 'get_crosswalk', lambda d: crosswalk)

    df = pd.DataFrame({'occupancy_type': ['Manufactured Home', 'Unclassified', 'New']})
    out = apply_transformation(
        df,
        {
            'output': 'group',
            'type': 'remap_file',
            'input': 'occupancy_type',
            'crosswalk_id': 'US_footprint-fema-2023_group-remap',
        },
    )
    assert out['group'].iloc[0] == 'Manufactured Home'
    assert pd.isna(out['group'].iloc[1])  # blank target -> missing (no vote)
    assert pd.isna(out['group'].iloc[2])  # unmapped raw value -> missing


def test_crosswalk_id_resolves_the_partition_admin_unit(monkeypatch):
    """One national recipe can read a per-state crosswalk sidecar.

    A bare county name is not unique nationally -- 17 names occur in
    both North Carolina and Texas -- and the Shovels source carries no
    state column, so one merged table would silently mis-assign them.
    """
    seen = {}

    def _fake(spec):
        seen['recipe_id'] = spec['recipe_id']
        return pd.Series({'CALDWELL': 'US-NC-CD'})

    monkeypatch.setattr(transform_mod, 'get_crosswalk', _fake)
    config = {
        'output': 'admin3_id',
        'type': 'remap_file',
        'input': 'county_name',
        'crosswalk_id': '{admin2_id}_property-shovels-2026_county-name-remap',
    }
    df = pd.DataFrame({'county_name': ['CALDWELL']})

    out = apply_transformation(df, config, admin_id='US-NC-CM')
    assert seen['recipe_id'] == 'US-NC_property-shovels-2026_county-name-remap'
    assert out['admin3_id'].iloc[0] == 'US-NC-CD'

    apply_transformation(df, config, admin_id='US-TX-ARA')
    assert seen['recipe_id'] == 'US-TX_property-shovels-2026_county-name-remap'


def test_an_unresolved_placeholder_is_left_alone(monkeypatch):
    """Failing loudly beats reading whichever table sorts first."""
    seen = {}

    def _fake(spec):
        seen['recipe_id'] = spec['recipe_id']
        return pd.Series(dtype=object)

    monkeypatch.setattr(transform_mod, 'get_crosswalk', _fake)
    apply_transformation(
        pd.DataFrame({'county_name': ['CALDWELL']}),
        {
            'output': 'admin3_id',
            'type': 'remap_file',
            'input': 'county_name',
            'crosswalk_id': '{admin2_id}_property-shovels-2026_county-name-remap',
        },
        admin_id=None,
    )
    assert seen['recipe_id'].startswith('{admin2_id}')


def test_a_plain_crosswalk_id_is_unaffected_by_an_admin_unit(monkeypatch):
    seen = {}

    def _fake(spec):
        seen['recipe_id'] = spec['recipe_id']
        return pd.Series({'x': 'y'})

    monkeypatch.setattr(transform_mod, 'get_crosswalk', _fake)
    apply_transformation(
        pd.DataFrame({'c': ['x']}),
        {
            'output': 'o',
            'type': 'remap_file',
            'input': 'c',
            'crosswalk_id': 'US_footprint-fema-2023_group-remap',
        },
        admin_id='US-NC-CM',
    )
    assert seen['recipe_id'] == 'US_footprint-fema-2023_group-remap'
