"""Tests for `index_by`, which normalizes a shared join key that a
multi-table source doesn't expose the same way on every table.
"""

from __future__ import annotations

import pandas as pd
import pytest

from openplaces.io.transform import index_by


def test_index_by_is_noop_when_already_the_index():
    df = pd.DataFrame({'a': [1, 2]}, index=pd.Index(['x', 'y'], name='key'))
    out = index_by(df, 'key')
    assert out.index.name == 'key'
    assert list(out.index) == ['x', 'y']


def test_index_by_promotes_a_plain_column():
    df = pd.DataFrame({'key': ['x', 'y'], 'a': [1, 2]})
    out = index_by(df, 'key')
    assert out.index.name == 'key'
    assert list(out.index) == ['x', 'y']
    assert list(out.columns) == ['a']


def test_index_by_raises_when_name_is_neither_index_nor_column():
    df = pd.DataFrame({'a': [1, 2]})
    with pytest.raises(ValueError, match='key'):
        index_by(df, 'key')
