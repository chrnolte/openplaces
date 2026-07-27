"""Tests for the area-weighted, registry-driven aggregation function."""

import pandas as pd
import pytest

from openplaces.io.aggregate import aggregate_rows_weighted


def test_weighted_mean_column():
    # year_built is registered with aggregation='mean'.
    df = pd.DataFrame(
        {
            'parcel_id_new': ['a', 'a', 'b'],
            'year_built': [2000, 1980, 1950],
            'area_ha': [1.0, 3.0, 5.0],
        }
    )

    result = aggregate_rows_weighted(df, by='parcel_id_new', wcol='area_ha')

    expected_a = (2000 * 1.0 + 1980 * 3.0) / (1.0 + 3.0)
    assert result.loc['a', 'year_built'] == pytest.approx(expected_a)
    assert result.loc['b', 'year_built'] == pytest.approx(1950)


def test_fraction_weighted_sum_column():
    # land_value is registered with aggregation='sum'.
    df = pd.DataFrame(
        {
            'parcel_id_new': ['a', 'b', 'c'],
            'parcel_id_old': ['x', 'x', 'y'],
            'land_value': [100_000, 100_000, 50_000],
            'fraction_of_old': [0.5, 0.5, 1.0],
        }
    )

    result = aggregate_rows_weighted(df, by='parcel_id_new', wcol='fraction_of_old')

    assert result.loc['a', 'land_value'] == pytest.approx(50_000)
    assert result.loc['b', 'land_value'] == pytest.approx(50_000)
    assert result.loc['c', 'land_value'] == pytest.approx(50_000)
    # Mass conservation: apportioned shares recover the source total.
    assert result.loc[['a', 'b'], 'land_value'].sum() == pytest.approx(100_000)


def test_unweighted_kind_passes_through_unchanged():
    # occupancy_type is registered with aggregation='first'.
    df = pd.DataFrame(
        {
            'parcel_id_new': ['a', 'a'],
            'occupancy_type': ['RES1', 'RES2'],
            'area_ha': [3.0, 1.0],
        }
    )

    result = aggregate_rows_weighted(df, by='parcel_id_new', wcol='area_ha')

    assert result.loc['a', 'occupancy_type'] == 'RES1'


def test_by_as_list_of_columns():
    df = pd.DataFrame(
        {
            'admin_id': ['US-MA', 'US-MA', 'US-MA'],
            'parcel_id_new': ['a', 'a', 'b'],
            'year_built': [2000, 1980, 1950],
            'area_ha': [1.0, 1.0, 1.0],
        }
    )

    result = aggregate_rows_weighted(
        df, by=['admin_id', 'parcel_id_new'], wcol='area_ha'
    )

    assert result.loc[('US-MA', 'a'), 'year_built'] == pytest.approx(1990)
    assert result.loc[('US-MA', 'b'), 'year_built'] == pytest.approx(1950)


def test_no_aggregatable_columns_returns_none():
    df = pd.DataFrame(
        {
            'parcel_id_new': ['a'],
            'not_a_registry_column': [1],
            'area_ha': [1.0],
        }
    )

    assert aggregate_rows_weighted(df, by='parcel_id_new', wcol='area_ha') is None


def test_dict_can_aggregate_non_registry_column():
    # Unlike aggregate_rows, an explicit dict entry aggregates a column even
    # when it has no attribute-registry row at all (e.g. a one-off column
    # from an external dataset not meant to join the shared registry).
    df = pd.DataFrame(
        {
            'parcel_id_new': ['a', 'a'],
            'clim_ppt_summer': [100.0, 200.0],
            'area_ha': [1.0, 3.0],
        }
    )

    result = aggregate_rows_weighted(
        df,
        by='parcel_id_new',
        wcol='area_ha',
        aggregation_function={'clim_ppt_summer': 'mean'},
    )

    expected = (100.0 * 1.0 + 200.0 * 3.0) / (1.0 + 3.0)
    assert result.loc['a', 'clim_ppt_summer'] == pytest.approx(expected)


def test_explicit_aggregation_function_override():
    df = pd.DataFrame(
        {
            'parcel_id_new': ['a', 'a'],
            'year_built': [2000, 1980],
            'area_ha': [1.0, 1.0],
        }
    )

    result = aggregate_rows_weighted(
        df,
        by='parcel_id_new',
        wcol='area_ha',
        aggregation_function={'year_built': 'sum'},
    )

    assert result.loc['a', 'year_built'] == pytest.approx(2000 * 1.0 + 1980 * 1.0)
