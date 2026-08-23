"""Tests for the save_link crosswalk sidecar (write, reload, invalidate)."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from openplaces.config import cfg
from openplaces.core.schema import AdminId
from openplaces.geo.link import get_entity_link_path
from openplaces.io import cleanup as cl
from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer import links as links_mod
from openplaces.recipe import get_output_path, get_recipe_by_id

SPINE = 'US_footprint-spine-2026'
PARCEL = 'US-NC_parcel-nconemap-2025'
COUNTY = 'US-NC-BR'


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    dirs = dict(cfg.config['directories'])
    dirs['data_root'] = tmp_path
    for name in ('core', 'external', 'raw', 'cache', 'out', 'share'):
        dirs[name] = tmp_path / 'data' / name
    dirs['heap'] = tmp_path / 'data/cache/_heap'
    dirs['logs'] = tmp_path / 'data/cache/_logs'
    monkeypatch.setitem(cfg.config, 'directories', dirs)
    return tmp_path


def _spine_gdf():
    # f1: inside P1 only. f2: straddles P1/P2 evenly (multi-parcel).
    # f3: mostly P1 with a sliver into P2 (trimmed secondary link).
    # f4: outside both parcels (no parcel).
    geoms = {
        'f1': box(1, 1, 3, 3),
        'f2': box(8, 4, 12, 6),
        'f3': box(5, 7, 10.05, 8),
        'f4': box(30, 30, 32, 32),
    }
    gdf = gpd.GeoDataFrame(
        {'geometry': list(geoms.values())},
        index=pd.Index(list(geoms), name='footprint_id'),
        crs='epsg:4326',
    )
    return gdf


def _parcels_gdf():
    return gpd.GeoDataFrame(
        {
            'geo_id': ['P1', 'P2'],
            'geometry': [box(0, 0, 10, 10), box(10, 0, 20, 10)],
        },
        crs='epsg:4326',
    )


@pytest.fixture
def state(data_root, monkeypatch):
    monkeypatch.setattr(links_mod, 'get_entities', lambda *a, **k: _parcels_gdf())
    recipe = get_recipe_by_id(SPINE)
    return HarmonizeState(
        recipe=recipe,
        admin_id=AdminId(COUNTY),
        verbose=False,
        timer=None,
        spine=_spine_gdf(),
    )


def _run_overlay(state, save_link=True, thresholds=None):
    return links_mod._link_spatial_overlay(
        state,
        PARCEL,
        'parcel',
        thresholds
        or {'min_fraction_of_largest': 0.1667, 'area_intersection_m2_min': 10},
        save_link=save_link,
    )


def _fresh_state(state):
    return HarmonizeState(
        recipe=state.recipe,
        admin_id=state.admin_id,
        verbose=False,
        timer=None,
        spine=_spine_gdf(),
    )


def _sidecar_path():
    return get_entity_link_path(SPINE, PARCEL, admin_id=COUNTY)


def test_sidecar_written_with_full_overlay(state):
    _run_overlay(state)
    sidecar = _sidecar_path()
    assert sidecar.exists()
    assert sidecar.parent == get_output_path(SPINE, admin_id=COUNTY).parent

    flat = pd.read_parquet(sidecar)
    # f2 keeps both parcels (n:m retained)
    f2 = flat[flat['footprint_id'] == 'f2']
    assert set(f2['parcel_id']) == {'P1', 'P2'}
    assert (f2['link'] == 'multi-parcel footprint').all()
    # f3's sliver pair into P2 is in the sidecar but not link-labeled
    f3 = flat[flat['footprint_id'] == 'f3']
    assert set(f3['parcel_id'].dropna()) == {'P1', 'P2'}
    sliver = f3[f3['parcel_id'] == 'P2']
    assert sliver['link'].isna().all()
    kept = f3[f3['parcel_id'] == 'P1']
    assert (kept['link'] == 'unique parcel (dropping small neighbor)').all()
    # f4 has no parcel
    f4 = flat[flat['footprint_id'] == 'f4']
    assert f4['parcel_id'].isna().all()
    # No geometry in the sidecar
    assert 'geometry' not in flat.columns


def test_reload_skips_overlay_and_reproduces_crosswalk(state, monkeypatch):
    state = _run_overlay(state)
    crosswalk_fresh = state.crosswalks[PARCEL]

    calls = {'n': 0}
    real_overlay = links_mod.overlay_polygons

    def _counting_overlay(*args, **kwargs):
        calls['n'] += 1
        return real_overlay(*args, **kwargs)

    monkeypatch.setattr(links_mod, 'overlay_polygons', _counting_overlay)

    state2 = _run_overlay(_fresh_state(state))
    assert calls['n'] == 0, 'sidecar reload must skip the overlay'
    crosswalk_reloaded = state2.crosswalks[PARCEL]
    pd.testing.assert_frame_equal(
        crosswalk_fresh.drop(columns='geometry', errors='ignore')
        .reset_index()
        .sort_values(['footprint_id', 'parcel_id'])
        .reset_index(drop=True),
        crosswalk_reloaded.reset_index()
        .sort_values(['footprint_id', 'parcel_id'])
        .reset_index(drop=True),
        check_like=True,
        check_dtype=False,
    )
    # Reloaded overlay is geometry-free
    assert 'geometry' not in state2.overlays[PARCEL].columns


def test_threshold_change_invalidates_sidecar(state, monkeypatch):
    state = _run_overlay(state)
    calls = {'n': 0}
    real_overlay = links_mod.overlay_polygons

    def _counting_overlay(*args, **kwargs):
        calls['n'] += 1
        return real_overlay(*args, **kwargs)

    monkeypatch.setattr(links_mod, 'overlay_polygons', _counting_overlay)
    _run_overlay(
        _fresh_state(state),
        thresholds={'min_fraction_of_largest': 0.3, 'area_intersection_m2_min': 10},
    )
    assert calls['n'] == 1, 'changed thresholds must force a recompute'


def test_reprocess_recomputes_and_rewrites(state, monkeypatch):
    state = _run_overlay(state)
    calls = {'n': 0}
    real_overlay = links_mod.overlay_polygons

    def _counting_overlay(*args, **kwargs):
        calls['n'] += 1
        return real_overlay(*args, **kwargs)

    monkeypatch.setattr(links_mod, 'overlay_polygons', _counting_overlay)
    state2 = _fresh_state(state)
    state2.reprocess = True
    _run_overlay(state2)
    assert calls['n'] == 1, 'reprocess=True must ignore the sidecar'


def test_source_change_invalidates_sidecar(state, monkeypatch):
    # Fingerprint a real source file: the reference parquet
    ref_path = get_output_path(PARCEL, admin_id=COUNTY)
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'geo_id': ['P1']}).to_parquet(ref_path)

    state = _run_overlay(state)

    calls = {'n': 0}
    real_overlay = links_mod.overlay_polygons

    def _counting_overlay(*args, **kwargs):
        calls['n'] += 1
        return real_overlay(*args, **kwargs)

    monkeypatch.setattr(links_mod, 'overlay_polygons', _counting_overlay)

    # Unchanged source: reload
    _run_overlay(_fresh_state(state))
    assert calls['n'] == 0

    # Source rewritten (size/mtime change): recompute
    pd.DataFrame({'geo_id': ['P1', 'P2', 'P3']}).to_parquet(ref_path)
    _run_overlay(_fresh_state(state))
    assert calls['n'] == 1


def test_deleted_source_with_receipt_stays_valid(state, monkeypatch):
    ref_path = get_output_path(PARCEL, admin_id=COUNTY)
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'geo_id': ['P1']}).to_parquet(ref_path)
    stat = None

    state = _run_overlay(state)
    stat = ref_path.stat()

    # Delete the source, leaving a receipt recording its size/mtime
    ref_path.unlink()
    cl.write_receipt(
        ref_path,
        {
            'recipe_id': PARCEL,
            'admin_id': COUNTY,
            'source_size_bytes': stat.st_size,
            'source_mtime': stat.st_mtime,
            'consumers_verified': [{'recipe_id': SPINE, 'path': 'x'}],
        },
    )

    calls = {'n': 0}
    real_overlay = links_mod.overlay_polygons

    def _counting_overlay(*args, **kwargs):
        calls['n'] += 1
        return real_overlay(*args, **kwargs)

    monkeypatch.setattr(links_mod, 'overlay_polygons', _counting_overlay)
    _run_overlay(_fresh_state(state))
    assert calls['n'] == 0, 'receipt must stand in for the deleted source'

    # Without the receipt, the missing source invalidates the sidecar
    cl.discard_receipt(ref_path)
    _run_overlay(_fresh_state(state))
    assert calls['n'] == 1


def test_no_save_link_writes_nothing(state):
    _run_overlay(state, save_link=False)
    assert not _sidecar_path().exists()
