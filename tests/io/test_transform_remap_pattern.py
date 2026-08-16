"""Tests for the remap_pattern transformation.

Regression: with no `default`, the all-NaN seed series must be
object-dtype — pandas 2.x types a bare NaN seed float64 and then refuses
to hold the (typically string) pattern values.
"""

from __future__ import annotations

import pandas as pd

from openplaces.io.transform import apply_transformation


def test_remap_pattern_maps_strings_without_a_default():
    # Codes padded with a trailing space, like Robeson's PARDESC1.
    df = pd.DataFrame({'use_subgroup': ['D-10 ', 'C-92 ', 'D-71 ', None]})
    out = apply_transformation(
        df,
        {
            'output': 'use_group',
            'type': 'remap_pattern',
            'input': 'use_subgroup',
            'patterns': [
                {'pattern': r'D-10\s*$', 'value': 'SINGLE FAMILY RESIDENCE'},
                {'pattern': r'C-92\s*$', 'value': 'MOBILE HOME PARK'},
            ],
        },
    )
    assert out['use_group'].iloc[0] == 'SINGLE FAMILY RESIDENCE'
    assert out['use_group'].iloc[1] == 'MOBILE HOME PARK'
    # Unmatched and missing rows stay missing rather than raising.
    assert pd.isna(out['use_group'].iloc[2])
    assert pd.isna(out['use_group'].iloc[3])
