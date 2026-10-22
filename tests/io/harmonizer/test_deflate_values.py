"""Tests for the harmonize step deflate_values and its deflator series."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openplaces.core.schema import AdminId
from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer.values import deflate_values, deflator_by_year


@pytest.fixture
def index_csv(tmp_path):
    """Two countries; COL inflates 10% a year 2018-2020, 2021 missing."""
    rows = [
        ('COL', 2017, 5.0),
        ('COL', 2018, 10.0),
        ('COL', 2019, 10.0),
        ('COL', 2020, 10.0),
        ('COL', 2021, np.nan),
        ('COL', 2022, 10.0),
        ('USA', 2020, 1.0),
    ]
    path = tmp_path / 'cpi.csv'
    pd.DataFrame(rows, columns=['country', 'year', 'inflation_pct']).to_csv(
        path, index=False
    )
    return path


def _state(df, recipe=None):
    return HarmonizeState(
        recipe=recipe or {'admin_id': 'CO', 'entity': None},
        admin_id=AdminId('CO'),
        verbose=False,
        timer=None,
        spine=df,
    )


def test_deflator_chains_rates_to_base_year(index_csv):
    index = pd.read_csv(index_csv)
    d = deflator_by_year(index, 'COL', 2020)
    assert d.loc[2020] == 1.0
    assert d.loc[2019] == pytest.approx(1.10)
    assert d.loc[2018] == pytest.approx(1.21)
    assert d.loc[2017] == pytest.approx(1.331)
    # A missing rate makes the years beyond it unavailable, both ways.
    assert np.isnan(d.loc[2021])
    assert np.isnan(d.loc[2022])


def test_deflator_divides_for_years_after_base(index_csv):
    index = pd.read_csv(index_csv)
    d = deflator_by_year(index, 'COL', 2018)
    assert d.loc[2020] == pytest.approx(1 / 1.21)


def test_step_writes_real_values_and_provenance(index_csv):
    df = pd.DataFrame(
        {
            'price': [100.0, 100.0, 100.0, 100.0, None],
            'sale_year': [2019, 2020, 2021, None, 2020],
            'price_source': ['reboredo', 'reboredo', None, 'reboredo', 'reboredo'],
        }
    )
    state = deflate_values(
        _state(df),
        columns=['price'],
        year_column='sale_year',
        index=str(index_csv),
        base_year=2020,
        country='COL',
    )
    out = state.spine
    assert out['price_real'].tolist()[:2] == pytest.approx([110.0, 100.0])
    # Year outside the index and missing year: missing, never nominal.
    assert out['price_real'].isna().tolist() == [False, False, True, True, True]
    assert out['price_real_source'].tolist()[:2] == [
        'reboredo+deflated',
        'reboredo+deflated',
    ]
    assert pd.isna(out['price_real_source'].iloc[2])


def test_step_falls_back_to_column_name_as_token(index_csv):
    df = pd.DataFrame({'price': [50.0], 'sale_year': [2019]})
    out = deflate_values(
        _state(df),
        columns=['price', 'absent'],
        year_column='sale_year',
        index=str(index_csv),
        base_year=2020,
        country='COL',
    ).spine
    assert out['price_real_source'].tolist() == ['price+deflated']
    assert 'absent_real' not in out.columns


def test_missing_base_year_raises(index_csv):
    index = pd.read_csv(index_csv)
    with pytest.raises(ValueError, match='base year'):
        deflator_by_year(index, 'USA', 2019)
