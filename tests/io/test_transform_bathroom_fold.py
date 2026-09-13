"""Tests for folding half baths into n_bathrooms with recipe transformations.

The registry defines n_bathrooms as full + 0.5 x half. Recipes express
that with an expression and a min_count sum, and a source that packs
both counts into one decimal (2.1 for two full and one half) is split
with the round operation. The values below are fabricated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from openplaces.io.transform import apply_transformations

FOLD = [
    {
        'type': 'expression',
        'expression': 'n_half_bathrooms * 0.5',
        'inputs': ['n_half_bathrooms'],
        'output': 'half_bathroom_equivalent',
    },
    {
        'type': 'aggregate',
        'operation': 'sum',
        'inputs': ['n_bathrooms', 'half_bathroom_equivalent'],
        'output': 'n_bathrooms',
    },
]


def test_half_baths_fold_as_one_half_each():
    df = pd.DataFrame(
        {
            'n_bathrooms': [2.0, 1.0, np.nan, 3.0],
            'n_half_bathrooms': [1.0, np.nan, np.nan, 0.0],
        }
    )
    out = apply_transformations(df, {'transformations': FOLD})
    assert out['n_bathrooms'].iloc[0] == 2.5
    # A missing half count reads as no halves when the full count is
    # present, and a row with neither count stays unknown.
    assert out['n_bathrooms'].iloc[1] == 1.0
    assert pd.isna(out['n_bathrooms'].iloc[2])
    assert out['n_bathrooms'].iloc[3] == 3.0


def test_round_splits_a_packed_full_and_half_code():
    df = pd.DataFrame({'BBATH': [2.1, 1.2, 3.0, np.nan]})
    recipe = {
        'transformations': [
            {
                'type': 'expression',
                'expression': 'BBATH * 10',
                'inputs': ['BBATH'],
                'output': 'bath_code',
            },
            {
                'type': 'unary',
                'operation': 'round',
                'input': 'bath_code',
                'output': 'bath_code',
            },
            {
                'type': 'expression',
                'expression': 'bath_code % 10',
                'inputs': ['bath_code'],
                'output': 'n_half_bathrooms',
            },
            {
                'type': 'expression',
                'expression': (
                    '(bath_code - n_half_bathrooms) / 10 + 0.5 * n_half_bathrooms'
                ),
                'inputs': ['bath_code', 'n_half_bathrooms'],
                'output': 'n_bathrooms',
            },
        ]
    }
    out = apply_transformations(df, recipe)
    # 2.1 * 10 is 21.000000000000004 in floating point; without the
    # round the half count would carry that noise.
    assert out['n_half_bathrooms'].tolist()[:3] == [1.0, 2.0, 0.0]
    assert out['n_bathrooms'].tolist()[:3] == [2.5, 2.0, 3.0]
    assert pd.isna(out['n_bathrooms'].iloc[3])
