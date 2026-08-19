"""Tests for the geospine split: fingerprint format 2, point-crosswalk
sidecars, the load_geospine state roundtrip, reader geometry indirection,
and the curate readers' link-owner resolution.

Reuses test_link_sidecar.py's tmp-data_root / monkeypatched get_entities
pattern: real recipe ids anchor the on-disk paths, toy geometries keep the
spatial work trivial.
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from openplaces.config import cfg
from openplaces.core.schema import AdminId
from openplaces.geo.link import get_entity_link_path, get_link_owner_recipe_id
from openplaces.io import save_parquet, to_parquet
from openplaces.io.aggregate import read_file_metadata
from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer import links as links_mod
from openplaces.io.harmonizer import load as load_mod
from openplaces.io.readers import get_entities
from openplaces.recipe import get_output_path, get_recipe_by_id

SPINE = 'US_footprint-spine-2026'
GEOSPINE = 'US_footprint-geospine-2026'
PARCEL = 'US-NC_parcel-nconemap-2025'
COUNTY = 'US-NC-BS'


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
    # f1: inside P1. f2: straddles P1/P2 (multi-parcel). f3: outside both.
    geoms = {
        'f1': box(1, 1, 3, 3),
        'f2': box(8, 4, 12, 6),
        'f3': box(30, 30, 32, 32),
    }
    return gpd.GeoDataFrame(
        {'geometry': list(geoms.values())},
        index=pd.Index(list(geoms), name='footprint_id'),
        crs='epsg:4326',
    )


def _parcels_gdf():
    return gpd.GeoDataFrame(
        {
            'geo_id': ['P1', 'P2'],
            'improvement_value': [100.0, 200.0],
            'geometry': [box(0, 0, 10, 10), box(10, 0, 20, 10)],
        },
        crs='epsg:4326',
    )


def _overlay_thresholds():
    return {'min_fraction_of_largest': 0.1667, 'area_intersection_m2_min': 10}


def _geospine_recipe(pipeline=None):
    """A toy geospine: the real geospine recipe with an explicit pipeline."""
    recipe = dict(get_recipe_by_id(GEOSPINE))
    recipe['pipeline'] = pipeline or [
        {
            'step': 'resolve_spine',
            'thresholds': {'min_area_m2': 10, 'overlap_iou_max': 0.5},
        },
        {
            'step': 'link_to_reference',
            'join': 'spatial_overlay',
            'recipe_id': PARCEL,
            'entity_type': 'parcel',
            'thresholds': _overlay_thresholds(),
            'save_link': True,
        },
    ]
    return recipe


def _run_geospine_overlay(recipe, monkeypatch, reprocess=False):
    """Run the toy geospine's overlay step exactly as its pipeline would."""
    monkeypatch.setattr(links_mod, 'get_entities', lambda *a, **k: _parcels_gdf())
    state = HarmonizeState(
        recipe=recipe,
        admin_id=AdminId(COUNTY),
        verbose=False,
        timer=None,
        spine=_spine_gdf(),
        reprocess=reprocess,
    )
    step_index = next(
        i
        for i, s in enumerate(recipe['pipeline'])
        if s.get('step') == 'link_to_reference'
    )
    state.step_index = step_index
    step = recipe['pipeline'][step_index]
    return links_mod._link_spatial_overlay(
        state,
        PARCEL,
        'parcel',
        step.get('thresholds') or {},
        save_link=True,
    )


def _save_geospine_output(recipe, spine):
    save_parquet(spine, get_output_path(recipe, AdminId(COUNTY)))


class TestFingerprintFormat2:
    def _count_overlays(self, monkeypatch):
        calls = {'n': 0}
        real = links_mod.overlay_polygons

        def _counting(*args, **kwargs):
            calls['n'] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(links_mod, 'overlay_polygons', _counting)
        return calls

    def test_geometry_step_change_invalidates(self, data_root, monkeypatch):
        recipe = _geospine_recipe()
        _run_geospine_overlay(recipe, monkeypatch)

        calls = self._count_overlays(monkeypatch)
        changed = _geospine_recipe()
        changed['pipeline'][0]['thresholds']['min_area_m2'] = 25
        _run_geospine_overlay(changed, monkeypatch)
        assert calls['n'] == 1, 'a changed prior geometry step must invalidate'

    def test_attribute_step_change_does_not_invalidate(self, data_root, monkeypatch):
        recipe = _geospine_recipe()
        # An attribute-phase step before the link must not be fingerprinted.
        recipe['pipeline'].insert(
            1, {'step': 'reconcile_addresses', 'columns': ['address']}
        )
        _run_geospine_overlay(recipe, monkeypatch)

        calls = self._count_overlays(monkeypatch)
        changed = _geospine_recipe()
        changed['pipeline'].insert(
            1, {'step': 'reconcile_addresses', 'columns': ['address', 'city']}
        )
        _run_geospine_overlay(changed, monkeypatch)
        assert calls['n'] == 0, 'attribute-phase config must not be fingerprinted'

    def test_snap_chains_toggle_does_not_invalidate(self, data_root, monkeypatch):
        recipe = _geospine_recipe()
        _run_geospine_overlay(recipe, monkeypatch)

        calls = self._count_overlays(monkeypatch)
        changed = _geospine_recipe()
        changed['pipeline'][1]['thresholds']['snap_chains'] = True
        _run_geospine_overlay(changed, monkeypatch)
        assert calls['n'] == 0, 'snap_chains stays excluded from the fingerprint'

    def test_format_1_sidecar_invalidates_gracefully(self, data_root, monkeypatch):
        import json

        recipe = _geospine_recipe()
        _run_geospine_overlay(recipe, monkeypatch)
        sidecar = get_entity_link_path(GEOSPINE, PARCEL, admin_id=COUNTY)

        # Rewrite the sidecar with a format-1 fingerprint (no
        # prior_geometry_steps), as a pre-upgrade run would have left it.
        stored = json.loads(read_file_metadata(sidecar)[links_mod._LINK_METADATA_KEY])
        stored['format'] = 1
        stored.pop('prior_geometry_steps', None)
        to_parquet(
            pd.read_parquet(sidecar),
            sidecar,
            file_metadata={links_mod._LINK_METADATA_KEY: json.dumps(stored)},
        )

        calls = self._count_overlays(monkeypatch)
        _run_geospine_overlay(_geospine_recipe(), monkeypatch)
        assert calls['n'] == 1, 'a format-1 sidecar must recompute, not crash'


class TestPointLinkSidecar:
    NSI = 'US_building-nsi-2026'

    def _points_gdf(self):
        return gpd.GeoDataFrame(
            {'source': ['Parcel', 'ESRI'], 'occupancy_type': ['RES1', 'RES3']},
            geometry=[Point(2, 2), Point(9, 5)],
            crs='epsg:4326',
        )

    def _run_point(self, recipe, monkeypatch, reprocess=False):
        monkeypatch.setattr(
            links_mod, 'get_entities', lambda *a, **k: self._points_gdf()
        )
        state = HarmonizeState(
            recipe=recipe,
            admin_id=AdminId(COUNTY),
            verbose=False,
            timer=None,
            spine=_spine_gdf(),
            reprocess=reprocess,
        )
        state.step_index = 0
        return links_mod._link_spatial_point(
            state, self.NSI, 'building', None, {}, save_link=True
        )

    def test_write_and_reload(self, data_root, monkeypatch):
        recipe = dict(get_recipe_by_id(GEOSPINE))
        recipe['pipeline'] = [{'step': 'link_to_reference', 'join': 'spatial_point'}]
        state = self._run_point(recipe, monkeypatch)
        fresh = state.crosswalks[self.NSI]
        sidecar = get_entity_link_path(GEOSPINE, self.NSI, admin_id=COUNTY)
        assert sidecar.exists()
        assert 'geometry' not in pd.read_parquet(sidecar).columns

        calls = {'n': 0}
        real_sjoin = links_mod.gpd.sjoin

        def _counting(*args, **kwargs):
            calls['n'] += 1
            return real_sjoin(*args, **kwargs)

        monkeypatch.setattr(links_mod.gpd, 'sjoin', _counting)
        state2 = self._run_point(recipe, monkeypatch)
        assert calls['n'] == 0, 'a valid point sidecar must skip every pass'
        reloaded = state2.crosswalks[self.NSI]
        pd.testing.assert_frame_equal(
            pd.DataFrame(fresh.drop(columns='geometry', errors='ignore')),
            reloaded,
            check_dtype=False,
        )

    def test_reprocess_recomputes(self, data_root, monkeypatch):
        recipe = dict(get_recipe_by_id(GEOSPINE))
        recipe['pipeline'] = [{'step': 'link_to_reference', 'join': 'spatial_point'}]
        self._run_point(recipe, monkeypatch)

        calls = {'n': 0}
        real_sjoin = links_mod.gpd.sjoin

        def _counting(*args, **kwargs):
            calls['n'] += 1
            return real_sjoin(*args, **kwargs)

        monkeypatch.setattr(links_mod.gpd, 'sjoin', _counting)
        self._run_point(recipe, monkeypatch, reprocess=True)
        assert calls['n'] > 0, 'reprocess=True must ignore the point sidecar'


class TestLoadGeospine:
    def _route_get_entities(self, monkeypatch):
        """Route load.py's reads: parcels from the toy frame, others real."""
        real = load_mod.get_entities

        def _router(recipe, *args, **kwargs):
            rid = recipe if isinstance(recipe, str) else None
            if rid == PARCEL:
                return _parcels_gdf()
            if rid is not None and rid != GEOSPINE:
                # Point references etc.: nothing ingested in the tmp root.
                return pd.DataFrame()
            return real(recipe, *args, **kwargs)

        monkeypatch.setattr(load_mod, 'get_entities', _router)

    def test_state_roundtrip(self, data_root, monkeypatch):
        recipe = _geospine_recipe()
        geo_state = _run_geospine_overlay(recipe, monkeypatch)
        _save_geospine_output(recipe, geo_state.spine)

        self._route_get_entities(monkeypatch)
        attr_state = HarmonizeState(
            recipe={'pipeline': [{'step': 'load_geospine'}]},
            admin_id=AdminId(COUNTY),
            verbose=False,
            timer=None,
        )
        attr_state = load_mod.load_geospine(attr_state, entity_recipe_id=recipe)

        # Spine restored with geometry and index intact.
        assert attr_state.spine is not None
        assert attr_state.spine.index.name == 'footprint_id'
        assert set(attr_state.spine.index) == set(geo_state.spine.index)
        assert attr_state.spine.geometry.notna().all()

        # Crosswalk identical to the fresh geometry run's.
        pd.testing.assert_frame_equal(
            geo_state.crosswalks[PARCEL]
            .drop(columns='geometry', errors='ignore')
            .reset_index()
            .sort_values(['footprint_id', 'parcel_id'])
            .reset_index(drop=True),
            attr_state.crosswalks[PARCEL]
            .reset_index()
            .sort_values(['footprint_id', 'parcel_id'])
            .reset_index(drop=True),
            check_like=True,
            check_dtype=False,
        )
        # Overlay restored geometry-free; reference fully re-prepared.
        assert 'geometry' not in attr_state.overlays[PARCEL].columns
        ref = attr_state.references[PARCEL]
        assert 'improvement_value_per_ha' in ref.columns
        assert ref.index.name == 'parcel_id'
        assert attr_state.reference_types[PARCEL] == 'parcel'

    def test_missing_sidecar_raises_with_rerun_instruction(
        self, data_root, monkeypatch
    ):
        recipe = _geospine_recipe()
        geo_state = _run_geospine_overlay(recipe, monkeypatch)
        _save_geospine_output(recipe, geo_state.spine)
        get_entity_link_path(GEOSPINE, PARCEL, admin_id=COUNTY).unlink()

        self._route_get_entities(monkeypatch)
        attr_state = HarmonizeState(
            recipe={'pipeline': [{'step': 'load_geospine'}]},
            admin_id=AdminId(COUNTY),
            verbose=False,
            timer=None,
        )
        with pytest.raises(RuntimeError, match='rerun'):
            load_mod.load_geospine(attr_state, entity_recipe_id=recipe)

    def test_stale_sidecar_raises(self, data_root, monkeypatch):
        recipe = _geospine_recipe()
        geo_state = _run_geospine_overlay(recipe, monkeypatch)
        _save_geospine_output(recipe, geo_state.spine)

        # The geospine's geometry config changed after the sidecar was
        # written: the loader must refuse it, not silently reuse it.
        changed = _geospine_recipe()
        changed['pipeline'][0]['thresholds']['min_area_m2'] = 25

        self._route_get_entities(monkeypatch)
        attr_state = HarmonizeState(
            recipe={'pipeline': [{'step': 'load_geospine'}]},
            admin_id=AdminId(COUNTY),
            verbose=False,
            timer=None,
        )
        with pytest.raises(RuntimeError, match='stale'):
            load_mod.load_geospine(attr_state, entity_recipe_id=changed)


class TestReaderGeometryIndirection:
    def test_geometry_resolved_via_entity_recipe(self, data_root):
        from openplaces.core.schema import Entity

        geo_recipe = dict(get_recipe_by_id(GEOSPINE))
        spine = _spine_gdf()
        save_parquet(spine, get_output_path(geo_recipe, AdminId(COUNTY)))

        attr_recipe = dict(geo_recipe)
        attr_recipe['entity'] = Entity('footprint', 'spine-attrs', '2026')
        attr_recipe['save_to'] = {'data_dir': 'core', 'geometry': False}
        attr_recipe['entity_recipe'] = GEOSPINE
        attrs = pd.DataFrame({'n_stories': [1, 2, 3]}, index=spine.index)
        attr_path = get_output_path(attr_recipe, AdminId(COUNTY))
        save_parquet(attrs, attr_path)

        # A stale _geo sidecar beside the attribute output must be ignored:
        # geometry comes from the declaration chain, not file probing.
        stale = gpd.GeoDataFrame(
            {'_join_id': [1, 2, 3]},
            geometry=[Point(0, 0)] * 3,
            crs='epsg:4326',
        )
        stale.to_parquet(attr_path.with_stem(attr_path.stem + '_geo'))

        out = get_entities(attr_recipe, COUNTY, geom=True)
        assert isinstance(out, gpd.GeoDataFrame)
        assert set(out.columns) >= {'n_stories', 'geometry'}
        joined = out.geometry.sort_index()
        expected = spine.geometry.sort_index()
        assert (joined.geom_equals(expected)).all()

    def test_plain_read_stays_plain(self, data_root):
        from openplaces.core.schema import Entity

        geo_recipe = dict(get_recipe_by_id(GEOSPINE))
        spine = _spine_gdf()
        save_parquet(spine, get_output_path(geo_recipe, AdminId(COUNTY)))

        attr_recipe = dict(geo_recipe)
        attr_recipe['entity'] = Entity('footprint', 'spine-attrs', '2026')
        attr_recipe['save_to'] = {'data_dir': 'core', 'geometry': False}
        attr_recipe['entity_recipe'] = GEOSPINE
        attrs = pd.DataFrame({'n_stories': [1, 2, 3]}, index=spine.index)
        save_parquet(attrs, get_output_path(attr_recipe, AdminId(COUNTY)))

        out = get_entities(attr_recipe, COUNTY, geom=False)
        assert 'geometry' not in out.columns


class TestLinkOwnerResolution:
    def test_link_running_recipe_owns_its_links(self):
        # The geospine's own pipeline runs link_to_reference, so it is its
        # own link owner -- the unsplit-recipe case.
        assert get_link_owner_recipe_id(GEOSPINE) == GEOSPINE

    def test_attribute_recipe_resolves_to_geospine(self):
        # The real registry spine recipe: no link steps of its own, so the
        # chain resolves to the geospine it declares as entity_recipe.
        assert get_link_owner_recipe_id(SPINE) == GEOSPINE
