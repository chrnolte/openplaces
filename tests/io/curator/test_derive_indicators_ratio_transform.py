"""Tests for the transform option of the derive_indicators ratio spec."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.inferers import derive_indicators


def _state(df):
    return CurateState(
        recipe={'entity': {'entity_type': 'transaction'}},
        entity_recipe={},
        admin_id=AdminId('CO'),
        verbose=False,
        timer=None,
        curated=df,
    )


def test_log_ratio_leaves_nonpositive_missing():
    df = pd.DataFrame({'price_real': [100.0, 0.0, 50.0], 'area_ha': [10.0, 5.0, 0.0]})
    out = derive_indicators(
        _state(df),
        indicators=[
            {
                'output': 'price_real_per_ha_log',
                'type': 'ratio',
                'numerator': 'price_real',
                'denominator': 'area_ha',
                'transform': 'log',
            }
        ],
    ).curated
    assert out['price_real_per_ha_log'].iloc[0] == pytest.approx(np.log(10.0))
    assert out['price_real_per_ha_log'].isna().tolist() == [False, True, True]


def test_asinh_and_plain_ratio_agree():
    df = pd.DataFrame({'a': [2.0], 'b': [4.0]})
    out = derive_indicators(
        _state(df),
        indicators=[
            {'output': 'r', 'type': 'ratio', 'numerator': 'a', 'denominator': 'b'},
            {
                'output': 'r_asinh',
                'type': 'ratio',
                'numerator': 'a',
                'denominator': 'b',
                'transform': 'asinh',
            },
        ],
    ).curated
    assert out['r_asinh'].iloc[0] == pytest.approx(np.arcsinh(out['r'].iloc[0]))


def test_unknown_transform_raises():
    df = pd.DataFrame({'a': [1.0], 'b': [1.0]})
    with pytest.raises(ValueError, match='unknown transform'):
        derive_indicators(
            _state(df),
            indicators=[
                {
                    'output': 'r',
                    'type': 'ratio',
                    'numerator': 'a',
                    'denominator': 'b',
                    'transform': 'sqrt',
                }
            ],
        )
