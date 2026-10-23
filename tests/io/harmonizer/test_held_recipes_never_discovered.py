"""A held recipe is never discovered as a source at runtime.

On 2026-10-23 a held, git-ignored permit connector scoped to the whole
country was picked up by the parcel geospine's property join
(`link_by_id(auto_discover: true, entity_type: property)`) and its ZIP
reached parcel values. The graph had never scheduled it: its dependency
scan skips recipes at the consumer's own scope, the runtime join does
not. `core.held_recipes` now keeps held recipes out of every discovery
path, and the run's `exclude_recipe_ids` out of the job's discovery.

Every recipe here is fabricated, in a temporary recipe root: no record
or schema of a real restricted source is read.
"""

import pandas as pd
import pytest
import yaml

import openplaces.diagnostics as diagnostics
import openplaces.io.harmonizer.links as links
import openplaces.path as path_module
import openplaces.recipe as recipe_module
from openplaces.core.held_recipes import EXCLUDE_ENV, is_held
from openplaces.core.schema import AdminId
from openplaces.io.harmonizer import HarmonizeState, discover
from openplaces.path import recipe_path, recipe_root_records, recipe_roots
from openplaces.recipe import discovery as discovery_module
from tests.links_patching import patch_links

OPEN_ID = 'US-NC_property-opentest-2026'
HELD_BY_KEY_ID = 'US-NC_property-heldtest-2027'
# Held through its source id (core.held_recipes.HELD_SOURCE_IDS), with
# a version no real recipe has, scoped like the connector that leaked.
HELD_BY_SOURCE_ID = 'US_property-shovels-2099'
UNIT = 'US-NC-PEN'


def _recipe(admin_id, source_id, version, **extra):
    return {
        'description': 'A fabricated property roll for a discovery test.',
        'admin_id': admin_id,
        'stage': 'ingest',
        'entity': {
            'entity_type': 'property',
            'source': {'source_id': source_id},
            'version': version,
        },
        'save_to': {'data_dir': 'core', 'admin_level': 3},
        **extra,
    }


def _clear_caches():
    recipe_root_records.cache_clear()
    recipe_roots.cache_clear()
    diagnostics._recipe_index.cache_clear()
    recipe_module._scan_ingest_recipe_ids.cache_clear()
    discovery_module._recipe_yaml.cache_clear()


@pytest.fixture
def held_root(tmp_path, monkeypatch):
    root = tmp_path / 'held_recipes'
    fabricated = {
        ('US-NC', 'property-opentest-2026'): _recipe('US-NC', 'opentest', '2026'),
        ('US-NC', 'property-heldtest-2027'): _recipe(
            'US-NC', 'heldtest', '2027', held=True
        ),
        ('US', 'property-shovels-2099'): _recipe(
            'US', 'shovels', '2099', exclude_from_auto_discover=False
        ),
    }
    for (admin_id, entity), recipe in fabricated.items():
        target = recipe_path(admin_id, entity, root=root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(recipe), encoding='utf-8')
    monkeypatch.setattr(
        path_module,
        '_external_recipe_root_records',
        lambda: [
            {
                'root': str(root),
                'provider': 'heldtest',
                'distribution': 'openplaces-heldtest',
                'version': '0.1.0',
            }
        ],
    )
    # Recorded so monkeypatch restores the variable's absence.
    monkeypatch.setenv(EXCLUDE_ENV, '')
    _clear_caches()
    yield root
    _clear_caches()


def _state(admin_id):
    return HarmonizeState(
        recipe={'admin_id': 'US'},
        admin_id=AdminId(admin_id),
        verbose=False,
        timer=None,
    )


def test_find_recipes_leaves_held_recipes_out(held_root):
    found = set(diagnostics.find_recipes('property', stage='ingest')['recipe_id'])
    assert OPEN_ID in found
    assert HELD_BY_KEY_ID not in found
    assert HELD_BY_SOURCE_ID not in found

    everything = diagnostics.find_recipes('property', stage='ingest', include_held=True)
    held = dict(zip(everything['recipe_id'], everything['held'], strict=True))
    assert held[OPEN_ID] is False or not held[OPEN_ID]
    assert held[HELD_BY_KEY_ID] and held[HELD_BY_SOURCE_ID]


def test_the_property_join_never_picks_up_a_held_recipe(held_root):
    """The parcel geospine's property join, on the unit where it leaked."""
    ids = links._find_admin_scoped_recipe_ids(_state(UNIT), 'property')
    assert OPEN_ID in ids
    assert HELD_BY_KEY_ID not in ids
    assert HELD_BY_SOURCE_ID not in ids

    matched = {
        m['recipe_id'] for m in links._discover_link_sources(_state(UNIT), 'property')
    }
    assert OPEN_ID in matched
    assert not matched & {HELD_BY_KEY_ID, HELD_BY_SOURCE_ID}


def test_the_property_join_checks_a_substituted_index(monkeypatch):
    """A frame that still lists held recipes does not bypass the join."""
    rows = pd.DataFrame(
        [
            {
                'admin_id': 'US-NC',
                'source_id': 'opentest',
                'version': '2026',
                'exclude_from_auto_discover': False,
                'recipe_id': OPEN_ID,
                'filename_suffix': '',
            },
            {
                'admin_id': 'US',
                'source_id': 'shovels',
                'version': '2099',
                'exclude_from_auto_discover': False,
                'recipe_id': HELD_BY_SOURCE_ID,
                'filename_suffix': '',
            },
            {
                'admin_id': 'US-NC',
                'source_id': 'heldtest',
                'version': '2027',
                'exclude_from_auto_discover': False,
                'recipe_id': HELD_BY_KEY_ID,
                'filename_suffix': '',
                'held': True,
            },
        ]
    )
    patch_links(monkeypatch, 'find_recipes', lambda *a, **k: rows)
    assert links._find_admin_scoped_recipe_ids(_state(UNIT), 'property') == [OPEN_ID]


def test_the_graph_never_schedules_a_held_recipe(held_root):
    edges = recipe_module.get_recipe_dependencies(
        'US_parcel-geospine-2026', admin_id=UNIT
    )
    upstream = {e.upstream_recipe_id for e in edges}
    assert OPEN_ID in upstream
    assert not upstream & {HELD_BY_KEY_ID, HELD_BY_SOURCE_ID}


def test_the_admin_style_scan_skips_held_recipes(held_root):
    scanned = {s['recipe_id'] for s in discover._scan_entity_ingest_recipes('property')}
    assert OPEN_ID in scanned
    assert not scanned & {HELD_BY_KEY_ID, HELD_BY_SOURCE_ID}


def test_the_best_recipe_resolver_skips_held_recipes(held_root, monkeypatch):
    # find_entity_recipe_id globs one root; point it at the fabricated
    # one. The held recipe has the newer version and would otherwise win.
    monkeypatch.setattr(
        discovery_module,
        'recipe_path',
        lambda *a, **k: recipe_path(*a, root=held_root, **k),
    )
    found = recipe_module.find_entity_recipe_id(
        UNIT, 'property', stage='ingest', silent=True
    )
    assert found == OPEN_ID


def test_a_run_excluded_recipe_is_not_discovered(held_root, monkeypatch):
    monkeypatch.setenv(EXCLUDE_ENV, OPEN_ID)
    assert OPEN_ID not in set(
        diagnostics.find_recipes('property', stage='ingest')['recipe_id']
    )
    assert OPEN_ID not in links._find_admin_scoped_recipe_ids(_state(UNIT), 'property')
    scanned = {s['recipe_id'] for s in discover._scan_entity_ingest_recipes('property')}
    assert OPEN_ID not in scanned
    # Reported, not chosen: the full listing still names it.
    assert OPEN_ID in set(
        diagnostics.find_recipes('property', stage='ingest', include_held=True)[
            'recipe_id'
        ]
    )


def test_a_loaded_recipe_is_recognized_as_held(held_root):
    """Held by source id, read from a loaded recipe's Entity object."""
    loaded = recipe_module.get_recipe_by_id(HELD_BY_SOURCE_ID)
    assert is_held(HELD_BY_SOURCE_ID, loaded)
    assert is_held(HELD_BY_KEY_ID, recipe_module.get_recipe_by_id(HELD_BY_KEY_ID))
    assert not is_held(OPEN_ID, recipe_module.get_recipe_by_id(OPEN_ID))


def test_an_explicit_reference_still_loads_a_held_recipe(held_root):
    """Held keeps a recipe out of discovery, not out of an explicit read.

    The building spine names the permit table by `recipe_id:` for an
    internal check whose columns curate drops before anything ships
    (test_restricted_shovels_columns_are_never_published).
    """
    loaded = recipe_module.get_recipe_by_id(HELD_BY_SOURCE_ID)
    assert loaded['recipe_id'] == HELD_BY_SOURCE_ID
