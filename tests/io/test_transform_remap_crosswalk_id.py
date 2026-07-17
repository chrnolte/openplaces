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
