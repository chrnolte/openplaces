"""Tests for the registry-driven row helpers in openplaces.table."""

import warnings

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from openplaces.table import aggregate_rows


def _parcels(rows, index=None):
    """Build a small parcel-like frame for the aggregation tests.

    Parameters
    ----------
    rows : dict
        Column name to values, passed straight to the DataFrame.
    index : list, optional
        Index labels; a default RangeIndex when omitted.
    """
    return pd.DataFrame(rows, index=index)


def test_list_columns_produce_a_list_per_group():
    # A pd.NamedAgg inside the dict handed to groupby.agg raises
    # KeyError, because '{col}_list' is not a column of the input.
    df = _parcels(
        {
            'geo_id': ['a', 'a', 'b'],
            'land_value': [10.0, 20.0, 5.0],
            'parcel_id': ['p1', 'p2', 'p3'],
        }
    )

    result = aggregate_rows(df, by='geo_id', list_columns=['parcel_id'])

    assert result.loc['a', 'parcel_id_list'] == ['p1', 'p2']
    assert result.loc['b', 'parcel_id_list'] == ['p3']
    # The scalar aggregation still runs alongside the list column.
    assert result.loc['a', 'land_value'] == 30.0


def test_list_column_of_categorical_dtype():
    df = _parcels(
        {
            'geo_id': ['a', 'a'],
            'land_value': [10.0, 20.0],
            'use_group': pd.Categorical(['residential', 'commercial']),
        }
    )

    result = aggregate_rows(df, by='geo_id', list_columns=['use_group'])

    assert result.loc['a', 'use_group_list'] == ['residential', 'commercial']


def test_area_sort_does_not_multiply_rows_on_a_duplicate_index():
    # get_entities concatenates per-unit frames without resetting the
    # parcel_id index, so a boundary parcel's label repeats.
    # Selecting rows by label then returns every duplicate each time.
    gdf = gpd.GeoDataFrame(
        {
            'geo_id': ['a', 'a', 'b'],
            'land_value': [10.0, 20.0, 5.0],
            'geometry': [box(0, 0, 1, 1), box(0, 0, 2, 2), box(0, 0, 3, 3)],
        },
        index=['p1', 'p1', 'p2'],
        crs='EPSG:3857',
    )

    result = aggregate_rows(gdf, by='geo_id')

    assert result.loc['a', 'land_value'] == 30.0
    assert result.loc['b', 'land_value'] == 5.0


def test_missing_sort_by_column_warns():
    df = _parcels({'geo_id': ['a', 'a'], 'land_value': [10.0, 20.0]})

    with pytest.warns(UserWarning, match='sort_by'):
        aggregate_rows(df, by='geo_id', sort_by='recorded_dt')


def test_present_sort_by_column_does_not_warn():
    df = _parcels(
        {'geo_id': ['a', 'a'], 'land_value': [10.0, 20.0], 'year_built': [1990, 2000]}
    )

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        aggregate_rows(df, by='geo_id', sort_by='year_built')


def test_null_grouping_key_is_reported():
    df = _parcels({'geo_id': ['a', None], 'land_value': [10.0, 20.0]})

    with pytest.warns(UserWarning, match='null'):
        result = aggregate_rows(df, by='geo_id')

    assert result.loc['a', 'land_value'] == 10.0
    assert len(result) == 1


def test_grouping_column_is_not_returned_as_a_column():
    # parcel_id is registered with aggregation='first', so grouping
    # by it returned it as index and column, and reset_index raised.
    df = _parcels({'parcel_id': ['p1', 'p1'], 'land_value': [10.0, 20.0]})

    result = aggregate_rows(df, by='parcel_id')

    assert 'parcel_id' not in result.columns
    assert result.reset_index()['parcel_id'].tolist() == ['p1']


def test_blank_registry_aggregation_is_skipped(monkeypatch):
    # An empty aggregation cell reads back from the CSV as float
    # NaN, not None, and a NaN reaching groupby.agg raises TypeError.
    import openplaces.table as table

    real_get_agg_func = table.get_agg_func

    def fake_get_agg_func(attr):
        if attr == 'postal_code':
            return float('nan')
        return real_get_agg_func(attr)

    monkeypatch.setattr(table, 'get_agg_func', fake_get_agg_func)
    df = _parcels(
        {
            'geo_id': ['a', 'a'],
            'land_value': [10.0, 20.0],
            'postal_code': ['02215', '02215'],
        }
    )

    result = aggregate_rows(df, by='geo_id')

    assert 'postal_code' not in result.columns
    assert result.loc['a', 'land_value'] == 30.0
