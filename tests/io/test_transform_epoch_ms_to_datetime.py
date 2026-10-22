"""Tests for the `epoch_ms_to_datetime` unary transform.

Esri's f=geojson output writes a date field as epoch milliseconds; the
transform turns those into datetimes, including dates before 1970,
which arrive as negative numbers.
"""

import numpy as np
import pandas as pd

from openplaces.io.transform import apply_transformations

RECIPE = {
    'transformations': [
        {
            'type': 'unary',
            'operation': 'epoch_ms_to_datetime',
            'input': 'x',
            'output': 'x',
        }
    ]
}


def test_milliseconds_become_dates():
    ms = [
        pd.Timestamp('2021-06-15').value // 10**6,
        pd.Timestamp('1927-03-01').value // 10**6,
        np.nan,
    ]
    result = apply_transformations(pd.DataFrame({'x': ms}), RECIPE)
    assert result['x'].iloc[0] == pd.Timestamp('2021-06-15')
    assert result['x'].iloc[1] == pd.Timestamp('1927-03-01')
    assert pd.isna(result['x'].iloc[2])


def test_text_that_is_not_a_number_becomes_missing():
    result = apply_transformations(pd.DataFrame({'x': ['abc']}), RECIPE)
    assert pd.isna(result['x'].iloc[0])
