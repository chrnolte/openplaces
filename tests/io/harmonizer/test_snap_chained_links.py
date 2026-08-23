"""Tests for chain-snapping displaced footprint-parcel links.

snap_chained_links collapses a multi-parcel footprint to its dominant parcel
when every minor link's parcel is a different footprint's dominant/unique
parcel and the minor overlap stays below fraction_max; the sidecar records
the adjustment in a link_chain column while the snapped minors lose their
link label (excluded from attribution like trimmed slivers).
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from openplaces.config import cfg
from openplaces.core.schema import AdminId
from openplaces.geo.link import get_entity_link_path
from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer import links as links_mod
from openplaces.io.harmonizer.links import snap_chained_links
from openplaces.recipe import get_recipe_by_id

SPINE = 'US_footprint-spine-2026'
PARCEL = 'US-NC_parcel-nconemap-2025'
COUNTY = 'US-NC-BR'


def _crosswalk(rows: list[tuple]) -> pd.DataFrame:
    """Build a crosswalk frame from (footprint, parcel, link, area) tuples."""
    df = pd.DataFrame(
        rows, columns=['footprint_id', 'parcel_id', 'link', 'area_intersection_m2']
    )
    df['fraction_of_largest'] = df['area_intersection_m2'] / df.groupby('footprint_id')[
        'area_intersection_m2'
    ].transform('max')
    return df.set_index(['footprint_id', 'parcel_id'])


def test_chain_of_three_snaps_all():
    # fA straddles P1 (home) + P2; fB straddles P2 (home) + P3; fC sits on P3.
    crosswalk = _crosswalk(
        [
            ('fA', 'P1', 'multi-parcel footprint', 100.0),
            ('fA', 'P2', 'multi-parcel footprint', 30.0),
            ('fB', 'P2', 'multi-parcel footprint', 100.0),
            ('fB', 'P3', 'multi-parcel footprint', 25.0),
            ('fC', 'P3', 'unique parcel', 100.0),
        ]
    )
    out, snapped = snap_chained_links(crosswalk, 'footprint_id')
    assert set(snapped.index) == {('fA', 'P2'), ('fB', 'P3')}
    assert out.loc[('fA', 'P1'), 'link'] == 'unique parcel (snapped from chain)'
    assert out.loc[('fB', 'P2'), 'link'] == 'unique parcel (snapped from chain)'
    assert out.loc[('fC', 'P3'), 'link'] == 'unique parcel'
    assert ('fA', 'P2') not in out.index and ('fB', 'P3') not in out.index


def test_shared_row_house_footprint_untouched():
    # One genuine shared footprint over three parcels: the minor parcels have
    # no building of their own, so nothing may be snapped.
    crosswalk = _crosswalk(
        [
            ('fR', 'P1', 'multi-parcel footprint', 80.0),
            ('fR', 'P2', 'multi-parcel footprint', 100.0),
            ('fR', 'P3', 'multi-parcel footprint', 80.0),
        ]
    )
    out, snapped = snap_chained_links(crosswalk, 'footprint_id')
    assert snapped.empty
    pd.testing.assert_frame_equal(out, crosswalk)


def test_near_equal_split_untouched():
    # fA's minor overlap is 90% of its dominant one: dominance is ambiguous,
    # so the footprint is left alone even though P2 is fB's home.
    crosswalk = _crosswalk(
        [
            ('fA', 'P1', 'multi-parcel footprint', 100.0),
            ('fA', 'P2', 'multi-parcel footprint', 90.0),
            ('fB', 'P2', 'unique parcel', 100.0),
        ]
    )
    out, snapped = snap_chained_links(crosswalk, 'footprint_id', fraction_max=0.75)
    assert snapped.empty
    assert out.loc[('fA', 'P1'), 'link'] == 'multi-parcel footprint'


def test_mutual_two_cycle_snaps_both():
    # Two neighbors, each footprint spilling onto the other's parcel.
    crosswalk = _crosswalk(
        [
            ('fA', 'P1', 'multi-parcel footprint', 100.0),
            ('fA', 'P2', 'multi-parcel footprint', 20.0),
            ('fB', 'P2', 'multi-parcel footprint', 100.0),
            ('fB', 'P1', 'multi-parcel footprint', 20.0),
        ]
    )
    out, snapped = snap_chained_links(crosswalk, 'footprint_id')
    assert set(snapped.index) == {('fA', 'P2'), ('fB', 'P1')}
    assert out['link'].eq('unique parcel (snapped from chain)').sum() == 2


def test_null_parcel_dominant_not_snapped():
    # fA's largest overlap is the unmatched identity remainder: nothing to
    # snap to, even though its minor parcel is owned elsewhere.
    crosswalk = _crosswalk(
        [
            ('fA', None, 'multi-parcel footprint', 100.0),
            ('fA', 'P2', 'multi-parcel footprint', 20.0),
            ('fB', 'P2', 'unique parcel', 100.0),
        ]
    )
    out, snapped = snap_chained_links(crosswalk, 'footprint_id')
    assert snapped.empty
    assert out.loc[('fA', 'P2'), 'link'] == 'multi-parcel footprint'


def test_no_multi_links_is_noop():
    crosswalk = _crosswalk([('fA', 'P1', 'unique parcel', 100.0)])
    out, snapped = snap_chained_links(crosswalk, 'footprint_id')
    assert snapped.empty
    pd.testing.assert_frame_equal(out, crosswalk)


# Integration through _link_spatial_overlay + the sidecar (fixture pattern
# from test_link_sidecar.py).


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
    # Parcels are 10-wide columns; footprints are displaced +2 in x, so fA
    # and fB each straddle their own parcel and the next one over (a chain),
    # while fC fits inside P3.
    geoms = {
        'fA': box(2, 2, 12, 8),
        'fB': box(12, 2, 22, 8),
        'fC': box(22, 2, 29, 8),
    }
    return gpd.GeoDataFrame(
        {'geometry': list(geoms.values())},
        index=pd.Index(list(geoms), name='footprint_id'),
        crs='epsg:4326',
    )


def _parcels_gdf():
    return gpd.GeoDataFrame(
        {
            'geo_id': ['P1', 'P2', 'P3'],
            'geometry': [box(0, 0, 10, 10), box(10, 0, 20, 10), box(20, 0, 30, 10)],
        },
        crs='epsg:4326',
    )


@pytest.fixture
def state(data_root, monkeypatch):
    monkeypatch.setattr(links_mod, 'get_entities', lambda *a, **k: _parcels_gdf())
    return HarmonizeState(
        recipe=get_recipe_by_id(SPINE),
        admin_id=AdminId(COUNTY),
        verbose=False,
        timer=None,
        spine=_spine_gdf(),
    )


SNAP_THRESHOLDS = {
    'min_fraction_of_largest': 0.1667,
    'area_intersection_m2_min': 10,
    'snap_chains': True,
    'chain_fraction_max': 0.75,
}


def _run_overlay(state, thresholds=None, save_link=True):
    return links_mod._link_spatial_overlay(
        state,
        PARCEL,
        'parcel',
        thresholds or SNAP_THRESHOLDS,
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


def test_overlay_snap_and_sidecar_roundtrip(state):
    state = _run_overlay(state)
    crosswalk = state.crosswalks[PARCEL]
    # Chain resolved to a 1-1 attribution.
    assert crosswalk.loc[('fA', 'P1'), 'link'] == 'unique parcel (snapped from chain)'
    assert crosswalk.loc[('fB', 'P2'), 'link'] == 'unique parcel (snapped from chain)'
    assert ('fA', 'P2') not in crosswalk.index
    assert ('fB', 'P3') not in crosswalk.index
    assert crosswalk.loc[('fC', 'P3'), 'link'] == 'unique parcel'

    flat = pd.read_parquet(get_entity_link_path(SPINE, PARCEL, admin_id=COUNTY))
    flat = flat.set_index(['footprint_id', 'parcel_id'])
    # Snapped minors keep their raw overlap row, lose the link label, and are
    # marked in link_chain; the promoted 1-1 links are marked as dominant.
    for pair in [('fA', 'P2'), ('fB', 'P3')]:
        assert pd.isna(flat.loc[pair, 'link'])
        assert flat.loc[pair, 'link_chain'] == 'snapped minor'
    for pair in [('fA', 'P1'), ('fB', 'P2')]:
        assert flat.loc[pair, 'link_chain'] == 'snapped dominant'
    assert pd.isna(flat.loc[('fC', 'P3'), 'link_chain'])


def test_snap_disabled_by_default(state):
    state = _run_overlay(
        state,
        thresholds={'min_fraction_of_largest': 0.1667, 'area_intersection_m2_min': 10},
    )
    crosswalk = state.crosswalks[PARCEL]
    assert crosswalk.loc[('fA', 'P2'), 'link'] == 'multi-parcel footprint'
    flat = pd.read_parquet(get_entity_link_path(SPINE, PARCEL, admin_id=COUNTY))
    assert 'link_chain' not in flat.columns


def test_reload_reapplies_snap_without_overlay(state, monkeypatch):
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
    # The rewritten sidecar still carries the chain labels.
    flat = pd.read_parquet(get_entity_link_path(SPINE, PARCEL, admin_id=COUNTY))
    assert (flat['link_chain'] == 'snapped minor').sum() == 2


def test_enabling_snap_on_existing_sidecar_needs_no_recompute(state, monkeypatch):
    # Sidecar written without snapping; turning snap_chains on must reuse the
    # stored raw overlay (the snap is crosswalk-level, not overlay-level).
    state = _run_overlay(
        state,
        thresholds={'min_fraction_of_largest': 0.1667, 'area_intersection_m2_min': 10},
    )
    calls = {'n': 0}
    real_overlay = links_mod.overlay_polygons

    def _counting_overlay(*args, **kwargs):
        calls['n'] += 1
        return real_overlay(*args, **kwargs)

    monkeypatch.setattr(links_mod, 'overlay_polygons', _counting_overlay)
    state2 = _run_overlay(_fresh_state(state), thresholds=SNAP_THRESHOLDS)
    assert calls['n'] == 0, 'snap settings must not invalidate the raw overlay'
    crosswalk = state2.crosswalks[PARCEL]
    assert crosswalk.loc[('fA', 'P1'), 'link'] == 'unique parcel (snapped from chain)'
    flat = pd.read_parquet(get_entity_link_path(SPINE, PARCEL, admin_id=COUNTY))
    assert (flat['link_chain'] == 'snapped dominant').sum() == 2
