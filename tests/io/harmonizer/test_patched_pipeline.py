"""Harmonize reads a recipe's pipeline as the unit ran it, patches included.

A county's patch recipe can remove a geospine's link step. The step then
writes no sidecar, so the attribute recipe must not try to restore one
(`load_geospine`) and the skip check must not owe one
(`_missing_link_sidecars`). A harmonize output also records the patches
that shaped it in its footer, as a curate output does. Fabricated
recipes and frames throughout; the patch is a stand-in function.
"""

from __future__ import annotations

import json

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

import openplaces.geo.link as link_mod
import openplaces.io.harmonizer as harmonizer_mod
import openplaces.io.harmonizer.links as links_pkg
from openplaces.core.schema import AdminId, Entity
from openplaces.io.aggregate import read_file_metadata
from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer import load as load_mod
from openplaces.recipe import (
    RECIPE_PATCHES_METADATA_KEY,
    get_output_path,
    get_recipe_by_id,
)

COUNTY = 'US-XX-YY'


def _link(recipe_id):
    return {'step': 'link_to_reference', 'recipe_id': recipe_id, 'join': 'x'}


def _geospine():
    return {
        'recipe_id': 'US_parcel-fabricated-2026',
        'pipeline': [
            {'step': 'resolve_spine'},
            _link('ref-a'),
            _link('ref-b'),
        ],
    }


def _drop_ref_b(recipe, admin_id):
    """A county patch removing the link to ref-b."""
    patched = dict(recipe)
    patched['pipeline'] = [
        s for s in recipe['pipeline'] if s.get('recipe_id') != 'ref-b'
    ]
    return patched, ['US-XX-YY_parcel-fabricated-2026']


def test_load_geospine_restores_only_the_links_the_unit_ran(monkeypatch):
    restored = []
    monkeypatch.setattr(load_mod, 'apply_recipe_patches', _drop_ref_b)
    monkeypatch.setattr(
        load_mod,
        'get_entities',
        lambda *a, **k: gpd.GeoDataFrame(
            index=pd.Index(['p1'], name='parcel_id'),
            geometry=[box(0, 0, 1, 1)],
            crs='EPSG:4326',
        ),
    )
    monkeypatch.setattr(load_mod, '_restore_handoff_metadata', lambda *a: None)
    monkeypatch.setattr(load_mod, 'get_recipe_id', lambda r: r['recipe_id'])

    def _record(state, geospine, step_index, step_cfg, spine_id_col):
        restored.append((step_index, step_cfg['recipe_id']))
        return state

    monkeypatch.setattr(load_mod, '_restore_link', _record)
    state = HarmonizeState(
        recipe={}, admin_id=AdminId(COUNTY), verbose=False, timer=None
    )
    load_mod.load_geospine(state, entity_recipe_id=_geospine())
    # Only ref-a, at the position the patched pipeline gives it.
    assert restored == [(1, 'ref-a')]


def test_a_link_step_a_patch_removed_owes_no_sidecar(monkeypatch, tmp_path):
    built = tmp_path / 'reference.parquet'
    built.write_bytes(b'')
    monkeypatch.setattr(
        links_pkg,
        '_resolve_reference_recipe',
        lambda recipe_id, entity_type, admin_id: (recipe_id, None),
    )
    monkeypatch.setattr(harmonizer_mod, 'get_recipe_by_id', lambda rid: {})
    monkeypatch.setattr(harmonizer_mod, 'get_output_path', lambda *a, **k: built)
    monkeypatch.setattr(
        link_mod,
        'get_entity_link_path',
        lambda a, b, admin_id=None: tmp_path / f'{a}_{b}.parquet',
    )
    unit = AdminId(COUNTY)

    monkeypatch.setattr(harmonizer_mod, 'apply_recipe_patches', lambda r, a: (r, []))
    owed = harmonizer_mod._missing_link_sidecars(_geospine(), unit)
    assert {p.name for p in owed} == {
        'US_parcel-fabricated-2026_ref-a.parquet',
        'US_parcel-fabricated-2026_ref-b.parquet',
    }

    monkeypatch.setattr(harmonizer_mod, 'apply_recipe_patches', _drop_ref_b)
    owed = harmonizer_mod._missing_link_sidecars(_geospine(), unit)
    assert [p.name for p in owed] == ['US_parcel-fabricated-2026_ref-a.parquet']


def test_a_harmonize_output_records_its_patches(mock_data_root, monkeypatch):
    from openplaces.io.harmonizer import _STEP_REGISTRY, Harmonizer

    recipe = dict(get_recipe_by_id('US_footprint-spine-2026'))
    recipe['entity'] = Entity('footprint', 'patchrecord', '2026')
    recipe['save_to'] = {'data_dir': 'core', 'geometry': False}
    recipe['pipeline'] = [{'step': '_seed_patch_record'}]

    def _seed(state):
        state.spine = gpd.GeoDataFrame(
            index=pd.Index(['f1'], name='footprint_id'),
            geometry=[box(0, 0, 1, 1)],
            crs='EPSG:4326',
        )
        return state

    monkeypatch.setitem(_STEP_REGISTRY, '_seed_patch_record', _seed)
    monkeypatch.setattr(
        harmonizer_mod,
        'apply_recipe_patches',
        lambda r, a: (r, ['US-XX-YY_footprint-patchrecord-2026']),
    )
    Harmonizer(recipe, admin_ids=[COUNTY])._harmonize_one(AdminId(COUNTY))

    footer = read_file_metadata(get_output_path(recipe, AdminId(COUNTY)))
    assert json.loads(footer[RECIPE_PATCHES_METADATA_KEY]) == [
        'US-XX-YY_footprint-patchrecord-2026'
    ]


def test_an_unpatched_output_records_none(mock_data_root, monkeypatch):
    from openplaces.io.harmonizer import _STEP_REGISTRY, Harmonizer

    recipe = dict(get_recipe_by_id('US_footprint-spine-2026'))
    recipe['entity'] = Entity('footprint', 'patchrecord', '2026')
    recipe['save_to'] = {'data_dir': 'core', 'geometry': False}
    recipe['pipeline'] = [{'step': '_seed_patch_record'}]

    def _seed(state):
        state.spine = gpd.GeoDataFrame(
            index=pd.Index(['f1'], name='footprint_id'),
            geometry=[box(0, 0, 1, 1)],
            crs='EPSG:4326',
        )
        return state

    monkeypatch.setitem(_STEP_REGISTRY, '_seed_patch_record', _seed)
    monkeypatch.setattr(harmonizer_mod, 'apply_recipe_patches', lambda r, a: (r, []))
    Harmonizer(recipe, admin_ids=[COUNTY])._harmonize_one(AdminId(COUNTY))

    footer = read_file_metadata(get_output_path(recipe, AdminId(COUNTY)))
    assert RECIPE_PATCHES_METADATA_KEY not in footer
