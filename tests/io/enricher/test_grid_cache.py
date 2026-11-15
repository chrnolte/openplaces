"""The per-unit grid cache: shared across a unit's raster steps, then gone.

`zonal_stats` with `method: rasterized` keeps the reprojected spine and
its burned grid in `EnrichState.grid_cache`, so every year of an annual
layer reuses one burn; the Enricher empties the cache when the unit's
pipeline ends, so memory does not grow from unit to unit.
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
from openplaces.io.enricher import Enricher, EnrichState
from openplaces.io.enricher import zonal as zonal_module
from openplaces.io.enricher.zonal import zonal_stats


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
    return path


@pytest.fixture
def rasters(tmp_path, monkeypatch):
    """Three fabricated years of a 0/1/2 uint8 layer, by path."""
    monkeypatch.setattr(zonal_module, 'resolve_raster_path', lambda p: p)
    rng = np.random.default_rng(11)
    return {
        year: str(
            _write(
                tmp_path / f'layer_{year}.tif',
                rng.integers(0, 3, size=(50, 50)).astype('uint8'),
            )
        )
        for year in (2001, 2002, 2003)
    }


def _spine():
    # Stored in a geographic CRS, as a spine is, so every step
    # reprojects it to the raster's CRS.
    geoms = gpd.GeoSeries(
        [box(2.2, 3.1, 20.7, 18.4), box(25, 25, 49, 49), box(30.2, 1.5, 44.9, 9.3)],
        crs='EPSG:6933',
    ).to_crs('EPSG:4326')
    return gpd.GeoDataFrame(
        {'geometry': geoms.values},
        index=pd.Index(['a', 'b', 'c'], name='parcel_id'),
        crs='EPSG:4326',
    )


def _state(spine):
    return EnrichState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('CO-AN-ABC'),
        verbose=False,
        timer=None,
        spine=spine,
        evidence=pd.DataFrame(index=spine.index),
    )


def _run_years(state, rasters, method):
    for year, path in rasters.items():
        zonal_stats(
            state,
            raster_key='forest',
            raster_path=path,
            column=f'forest_{year}',
            stat='mean',
            method=method,
        )


def test_years_share_one_burn_and_match_fresh_state_values(rasters):
    spine = _spine()
    state = _state(spine)
    _run_years(state, rasters, 'rasterized')
    kinds = [key[0] for key in state.grid_cache]
    assert kinds.count('burned_grid') == 1
    assert kinds.count('spine_in_raster_crs') == 1
    # Each year computed alone, on a cold cache, gives the same values.
    for year, path in rasters.items():
        cold = _state(spine)
        _run_years(cold, {year: path}, 'rasterized')
        pd.testing.assert_series_equal(
            state.evidence[f'forest_{year}'], cold.evidence[f'forest_{year}']
        )


def test_other_back_ends_leave_the_cache_alone(rasters):
    state = _state(_spine())
    _run_years(state, rasters, 'exactextract')
    assert state.grid_cache == {}


def _enricher(pipeline):
    enricher = Enricher.__new__(Enricher)
    enricher.recipe = {'pipeline': pipeline}
    enricher.entity_recipe = {}
    enricher.verbose = False
    enricher._timer = None
    return enricher


def test_cache_is_empty_after_the_unit(monkeypatch, tmp_path, rasters):
    spine = _spine()
    seen = []
    monkeypatch.setattr(enricher_mod, 'get_entities', lambda *a, **k: spine)
    monkeypatch.setattr(
        enricher_mod, 'get_output_path', lambda *a, **k: tmp_path / 'out.parquet'
    )
    monkeypatch.setattr(enricher_mod, 'save_parquet', lambda *a, **k: None)

    def recording_step(state, **params):
        seen.append(state)
        return zonal_stats(state, **params)

    monkeypatch.setitem(enricher_mod._STEP_REGISTRY, 'recording_step', recording_step)
    pipeline = [
        {
            'step': 'recording_step',
            'raster_key': 'forest',
            'raster_path': path,
            'column': f'forest_{year}',
            'method': 'rasterized',
        }
        for year, path in rasters.items()
    ]
    _enricher(pipeline)._enrich_one(AdminId('CO-AN-ABC'))
    assert len(seen) == 3
    assert seen[0].grid_cache == {}


def test_cache_is_emptied_when_a_step_fails(monkeypatch, tmp_path):
    spine = _spine()
    seen = []
    monkeypatch.setattr(enricher_mod, 'get_entities', lambda *a, **k: spine)

    def failing_step(state, **params):
        state.grid_cache['x'] = object()
        seen.append(state)
        raise RuntimeError('boom')

    monkeypatch.setitem(enricher_mod._STEP_REGISTRY, 'failing_step', failing_step)
    with pytest.raises(RuntimeError, match='boom'):
        _enricher([{'step': 'failing_step'}])._enrich_one(AdminId('CO-AN-ABC'))
    assert seen[0].grid_cache == {}
