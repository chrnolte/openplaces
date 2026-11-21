"""A link source without rows in a unit still writes its link sidecar.

The overlay sidecar of a reference that contributes no rows is written
with the columns and dtypes of one with pairs and no rows: the link was
looked for and none found. A missing sidecar keeps meaning the build
did not run or did not save, and the curate reader keeps refusing it.
Every geometry and id below is fabricated.
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from openplaces.config import cfg
from openplaces.core.schema import AdminId
from openplaces.geo.link import get_entity_link_path
from openplaces.io.curator import CurateState
from openplaces.io.curator.evidence import collect_link_ids
from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer import links as links_mod
from openplaces.io.harmonizer import load as load_module
from openplaces.recipe import get_recipe_by_id
from tests.links_patching import patch_links

GEOSPINE = 'US_footprint-geospine-2026'
SPINE = 'US_footprint-spine-2026'
PARCEL = 'US-NC_parcel-nconemap-2025'
WITH_PARCELS = 'US-NC-BR'
WITHOUT_PARCELS = 'US-NC-CUR'
NEVER_BUILT = 'US-NC-WA'
FINGERPRINT = {'format': 2, 'spine_recipe_id': GEOSPINE, 'sources': []}


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


def _spine():
    geoms = {'f1': box(1, 1, 3, 3), 'f2': box(8, 4, 12, 6), 'f3': box(30, 30, 32, 32)}
    return gpd.GeoDataFrame(
        {'geometry': list(geoms.values())},
        index=pd.Index(list(geoms), name='footprint_id'),
        crs='epsg:4326',
    )


def _parcels(admin_id, *args, **kwargs):
    if str(admin_id) != WITH_PARCELS:
        return gpd.GeoDataFrame({'geo_id': []}, geometry=[], crs='epsg:4326')
    return gpd.GeoDataFrame(
        {
            'geo_id': ['P1', 'P2'],
            'geometry': [box(0, 0, 10, 10), box(10, 0, 20, 10)],
        },
        crs='epsg:4326',
    )


@pytest.fixture
def wired(data_root, monkeypatch):
    patch_links(
        monkeypatch, 'get_entities', lambda rid, admin_id, *a, **k: _parcels(admin_id)
    )
    patch_links(monkeypatch, '_link_fingerprint', lambda *a, **k: dict(FINGERPRINT))
    # The reference was saved for both units, empty in one; never for
    # the third.
    patch_links(
        monkeypatch,
        'entity_outputs_exist',
        lambda rid, admin_id: str(admin_id) != NEVER_BUILT,
    )
    return get_recipe_by_id(GEOSPINE)


def _overlay(recipe, admin_id, snap_chains=False):
    state = HarmonizeState(
        recipe=recipe,
        admin_id=AdminId(admin_id),
        verbose=False,
        timer=None,
        spine=_spine(),
    )
    thresholds = {'min_fraction_of_largest': 0.1667, 'area_intersection_m2_min': 10}
    if snap_chains:
        thresholds['snap_chains'] = True
    return links_mod._link_spatial_overlay(
        state, PARCEL, 'parcel', thresholds, save_link=True
    )


def _sidecar(admin_id):
    return get_entity_link_path(GEOSPINE, PARCEL, admin_id=admin_id)


@pytest.mark.parametrize('snap_chains', [False, True], ids=['plain', 'snapped'])
def test_source_without_rows_writes_its_sidecar_like_one_with_rows(wired, snap_chains):
    _overlay(wired, WITH_PARCELS, snap_chains)
    state = _overlay(wired, WITHOUT_PARCELS, snap_chains)

    full = pd.read_parquet(_sidecar(WITH_PARCELS))
    empty = pd.read_parquet(_sidecar(WITHOUT_PARCELS))
    assert len(full) > 0
    assert len(empty) == 0
    assert list(empty.columns) == list(full.columns)
    assert dict(empty.dtypes) == dict(full.dtypes)
    # Nothing to link, so the state carries no crosswalk for it.
    assert PARCEL not in state.crosswalks


def test_unbuilt_reference_writes_no_sidecar(wired):
    state = _overlay(wired, NEVER_BUILT)
    assert not _sidecar(NEVER_BUILT).exists()
    assert PARCEL not in state.crosswalks


def test_entity_outputs_exist_tells_built_empty_from_unbuilt(data_root):
    from openplaces.io import to_parquet
    from openplaces.io.readers import entity_outputs_exist
    from openplaces.recipe import get_output_path, get_save_admin_level

    recipe = get_recipe_by_id(PARCEL)
    assert not entity_outputs_exist(recipe, WITHOUT_PARCELS)
    saved = AdminId(WITHOUT_PARCELS).truncate_to_level(get_save_admin_level(recipe))
    # A placeholder: no rows, no column but the join id.
    to_parquet(
        pd.DataFrame({'_join_id': pd.Series(dtype='str')}),
        get_output_path(recipe, saved),
    )
    assert entity_outputs_exist(recipe, WITHOUT_PARCELS)


def test_empty_sidecar_is_not_a_placeholder(wired):
    from openplaces.io.readers import is_placeholder_output

    _overlay(wired, WITHOUT_PARCELS)
    assert not is_placeholder_output(_sidecar(WITHOUT_PARCELS))


def _curate_state(admin_id):
    curated = pd.DataFrame(
        {'parcel_id': ['P1', 'P2', None]},
        index=pd.Index(['f1', 'f2', 'f3'], name='footprint_id'),
    )
    return CurateState(
        recipe={'recipe_id': 'US_footprint-test-2026'},
        entity_recipe=get_recipe_by_id(SPINE),
        admin_id=AdminId(admin_id),
        verbose=False,
        timer=None,
        curated=curated,
    )


def test_collect_link_ids_reads_the_empty_sidecar_as_no_links(wired):
    _overlay(wired, WITHOUT_PARCELS)
    state = collect_link_ids(_curate_state(WITHOUT_PARCELS), link_recipe_id=PARCEL)
    assert state.curated['parcel_id_all'].isna().all()


def test_collect_link_ids_still_refuses_a_missing_sidecar(wired):
    with pytest.raises(FileNotFoundError, match='Link sidecar not found'):
        collect_link_ids(_curate_state(NEVER_BUILT), link_recipe_id=PARCEL)


def test_the_attribute_recipe_restores_the_empty_sidecar_as_no_link(wired, monkeypatch):
    _overlay(wired, WITHOUT_PARCELS)
    monkeypatch.setattr(
        load_module, '_link_fingerprint', lambda *a, **k: dict(FINGERPRINT)
    )
    monkeypatch.setattr(
        load_module, 'reference_rows', lambda rid, admin, kept_by=None: (None, None)
    )
    state = HarmonizeState(
        recipe=get_recipe_by_id(SPINE),
        admin_id=AdminId(WITHOUT_PARCELS),
        verbose=False,
        timer=None,
        spine=_spine(),
    )
    step_cfg = {
        'step': 'link_to_reference',
        'recipe_id': PARCEL,
        'join': 'spatial_overlay',
        'thresholds': {},
    }
    state = load_module._restore_link(state, wired, 0, step_cfg, 'footprint_id')
    assert PARCEL not in state.crosswalks
    assert PARCEL not in state.overlays
