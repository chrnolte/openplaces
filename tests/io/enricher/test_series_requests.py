"""Series enrichment: requests, period rules, long output, resume.

Every value here is fabricated: three square parcels on a 50 by 50
grid and random rasters, one per year.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

import openplaces.io.enricher as enricher_mod
from openplaces.core.schema import AdminId
from openplaces.io import read_parquet
from openplaces.io.enricher import Enricher
from openplaces.io.enricher import series as series_mod
from openplaces.io.enricher import zonal as zonal_module
from openplaces.io.enricher.series import (
    SeriesRecipeError,
    request_sources,
    requested_pairs,
    series_layers,
)
from openplaces.io.readers import empty_reason
from openplaces.io.tables import answering_periods, request_years

YEARS = (2001, 2002, 2003, 2004)
UNIT = AdminId('CO-AN-ABC')


def _periods(years, periods, rule):
    return answering_periods(pd.Series(years, dtype='float64'), periods, rule)


def test_before_takes_the_year_before_the_date():
    out = _periods([2016, 2016.0], range(2000, 2025), 'before')
    assert out.tolist() == [2015, 2015]


def test_at_or_before_admits_the_containing_period():
    assert _periods([2016], range(2000, 2025), 'at_or_before').tolist() == [2016]


def test_now_is_the_latest_period_whatever_the_date():
    assert _periods([1950, np.nan], range(2000, 2025), 'now').tolist() == [2024, 2024]


def test_a_date_before_the_first_period_gets_none():
    out = _periods([2000, 1990, np.nan], range(2000, 2025), 'before')
    assert out.isna().all()


def test_later_periods_are_never_taken_and_gaps_fall_back():
    # Periods 2000, 2005, 2010 (a five-yearly product): a 2008 date
    # takes 2005 under both rules, a date after the last the last.
    periods = [2000, 2005, 2010]
    assert _periods([2008, 2030], periods, 'before').tolist() == [2005, 2010]
    assert _periods([2010], periods, 'before').tolist() == [2005]
    assert _periods([2010], periods, 'at_or_before').tolist() == [2010]


def test_request_years_reads_years_and_dates():
    assert request_years(pd.Series([2016.0, np.nan])).tolist()[0] == 2016
    dates = request_years(pd.Series(['2016-03-01', 'not a date']))
    assert dates.iloc[0] == 2016 and np.isnan(dates.iloc[1])


def _layers(rule='before'):
    return series_layers(
        {
            'pipeline': [
                {
                    'step': 'zonal_stats',
                    'years': [2001, 2004],
                    'column': 'forest',
                    'period_rule': rule,
                }
            ]
        }
    )


def test_two_sales_in_one_year_cost_one_pair():
    requests = pd.DataFrame(
        {
            'entity': ['a', 'a', 'b'],
            'year': [2003.0, 2003.0, 2001.0],
            'now': [False, False, False],
        }
    )
    pairs = requested_pairs(requests, _layers())
    # b's sale in 2001 precedes every period ending before it: no row.
    assert sorted(map(tuple, pairs.to_numpy().tolist())) == [('a', 2002)]


def test_a_series_recipe_without_requests_refuses():
    recipe = {
        'stage': 'enrich',
        'series': True,
        'pipeline': [{'step': 'zonal_stats', 'years': [2001, 2002], 'column': 'x'}],
    }
    with pytest.raises(SeriesRecipeError, match='refuses to run'):
        request_sources(recipe)
    with pytest.raises(SeriesRecipeError, match='refuses to run'):
        Enricher(recipe)


def test_a_series_step_keeps_the_year_out_of_its_column():
    with pytest.raises(SeriesRecipeError, match='without'):
        series_layers(
            {'pipeline': [{'step': 'z', 'years': [1, 2], 'column': 'x_{year}'}]}
        )
    with pytest.raises(SeriesRecipeError, match='constant layer'):
        series_layers({'pipeline': [{'step': 'z', 'column': 'x'}]})


def _write(path, data):
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs='EPSG:6933',
        transform=from_origin(0, 50, 1, 1),
        nodata=0,
    ) as dst:
        dst.write(data, 1)


@pytest.fixture
def raster_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(zonal_module, 'resolve_raster_path', lambda p: p)
    rng = np.random.default_rng(7)
    for year in YEARS:
        _write(
            tmp_path / f'layer_{year}.tif',
            rng.integers(0, 3, size=(50, 50)).astype('uint8'),
        )
    return tmp_path


def _spine():
    geoms = gpd.GeoSeries(
        [box(2.2, 3.1, 20.7, 18.4), box(25, 25, 49, 49), box(30.2, 1.5, 44.9, 9.3)],
        crs='EPSG:6933',
    ).to_crs('EPSG:4326')
    return gpd.GeoDataFrame(
        {'sale_year': [2003.0, 2004.0, 1999.0], 'geometry': geoms.values},
        index=pd.Index(['a', 'b', 'c'], name='parcel_id'),
        crs='EPSG:4326',
    )


def _step(raster_dir, column):
    return {
        'step': 'zonal_stats',
        'years': [YEARS[0], YEARS[-1]],
        'raster_key': 'forest_{year}',
        'raster_path': str(raster_dir / 'layer_{year}.tif'),
        'column': column,
        'stat': 'mean',
        'method': 'rasterized',
    }


def _enricher(recipe):
    enricher = Enricher.__new__(Enricher)
    enricher.recipe = recipe
    enricher.entity_recipe = {}
    enricher.verbose = False
    enricher._timer = None
    return enricher


@pytest.fixture
def unit(monkeypatch, tmp_path):
    """The spine served by `get_entities`, outputs under *tmp_path*."""
    spine = _spine()
    out = tmp_path / 'out.parquet'
    monkeypatch.setattr(enricher_mod, 'get_entities', lambda *a, **k: spine)
    monkeypatch.setattr(enricher_mod, 'get_output_path', lambda *a, **k: out)
    monkeypatch.setattr(series_mod, 'get_output_path', lambda *a, **k: out)
    return out


def test_long_output_pivots_to_the_wide_output(raster_dir, unit, monkeypatch):
    saved = {}
    monkeypatch.setattr(
        enricher_mod, 'save_parquet', lambda frame, *a, **k: saved.update(w=frame)
    )
    _enricher({'pipeline': [_step(raster_dir, 'forest_{year}')]})._enrich_one(UNIT)
    wide = saved['w']
    # The long run writes through the series module's own writer.
    spine = _spine()
    recipe = {
        'series': True,
        'requests': 'all',
        'pipeline': [_step(raster_dir, 'forest')],
    }
    _enricher(recipe)._enrich_one(UNIT)
    long = read_parquet(unit)
    assert list(long.index.names) == ['parcel_id', 'period']
    assert len(long) == len(spine) * len(YEARS)
    pivot = long['forest'].unstack('period')
    for year in YEARS:
        pd.testing.assert_series_equal(
            pivot[year].rename(None),
            wide[f'forest_{year}'].rename(None),
            check_names=False,
            check_index_type=False,
        )


def _recipe(raster_dir, requests):
    return {
        'series': True,
        'requests': requests,
        'pipeline': [_step(raster_dir, 'forest')],
    }


def test_resume_computes_only_the_missing_pairs(raster_dir, unit, monkeypatch):
    computed = []
    real = series_mod.extract_pairs

    def recording(enricher, state, missing, layers):
        computed.append(set(map(tuple, missing[['entity', 'period']].to_numpy())))
        return real(enricher, state, missing, layers)

    monkeypatch.setattr(series_mod, 'extract_pairs', recording)
    sales = [{'from': 'spine', 'date': 'sale_year'}]
    _enricher(_recipe(raster_dir, sales))._enrich_one(UNIT)
    first = read_parquet(unit)
    # a sold in 2003 takes 2002, b in 2004 takes 2003, c in 1999 none.
    assert computed[0] == {('a', 2002), ('b', 2003)}

    more = [*sales, {'from': 'spine', 'date': 'now'}]
    _enricher(_recipe(raster_dir, more))._enrich_one(UNIT)
    second = read_parquet(unit)
    assert computed[1] == {('a', 2004), ('b', 2004), ('c', 2004)}
    assert len(second) == 5
    pd.testing.assert_frame_equal(second.loc[first.index], first)

    # Nothing missing: nothing computed, the file left as it was.
    _enricher(_recipe(raster_dir, more))._enrich_one(UNIT)
    assert len(computed) == 2


def test_two_offsets_one_parcel(raster_dir, unit):
    requests = [{'from': 'spine', 'date': 'sale_year', 'offsets_years': [-1, 0]}]
    _enricher(_recipe(raster_dir, requests))._enrich_one(UNIT)
    pairs = set(read_parquet(unit).index)
    assert pairs == {('a', 2001), ('a', 2002), ('b', 2002), ('b', 2003)}


def test_nothing_requested_is_saved_and_says_why(raster_dir, unit, monkeypatch):
    spine = _spine().assign(sale_year=[1990.0, 1991.0, np.nan])
    monkeypatch.setattr(enricher_mod, 'get_entities', lambda *a, **k: spine)
    requests = [{'from': 'spine', 'date': 'sale_year'}]
    _enricher(_recipe(raster_dir, requests))._enrich_one(UNIT)
    assert len(read_parquet(unit)) == 0
    assert empty_reason(unit) == 'nothing_requested'


def test_a_value_does_not_depend_on_what_else_was_requested(
    raster_dir, unit, monkeypatch
):
    # Two overlapping parcels: a burned pixel under both goes to one of
    # them, so sampling a alone would give a different value than the
    # wide mode, which samples the whole spine.
    geoms = gpd.GeoSeries(
        [box(2, 2, 30, 30), box(10, 10, 45, 45)], crs='EPSG:6933'
    ).to_crs('EPSG:4326')
    spine = gpd.GeoDataFrame(
        {'sale_year': [2004.0, np.nan], 'geometry': geoms.values},
        index=pd.Index(['a', 'b'], name='parcel_id'),
        crs='EPSG:4326',
    )
    monkeypatch.setattr(enricher_mod, 'get_entities', lambda *a, **k: spine)
    one = [{'from': 'spine', 'date': 'sale_year'}]
    _enricher(_recipe(raster_dir, one))._enrich_one(UNIT)
    alone = read_parquet(unit).loc[('a', 2003), 'forest']
    _enricher(_recipe(raster_dir, 'all'))._enrich_one(UNIT, reprocess=True)
    together = read_parquet(unit).loc[('a', 2003), 'forest']
    assert alone == together


def test_a_wide_file_is_not_appended_to(raster_dir, unit):
    pd.DataFrame(
        {'forest_2001': [0.5]}, index=pd.Index(['a'], name='parcel_id')
    ).to_parquet(unit)
    requests = [{'from': 'spine', 'date': 'now'}]
    with pytest.raises(SeriesRecipeError, match='no series table'):
        _enricher(_recipe(raster_dir, requests))._enrich_one(UNIT)
