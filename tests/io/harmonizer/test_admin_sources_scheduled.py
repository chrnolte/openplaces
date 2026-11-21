"""The admin sources discover_sources reads are the ones the graph schedules.

An admin geometry harmonize (admin-openplaces-*) lists its sources by
scanning the recipe tree. The dependency scan now lists them through
the same function, so the plan schedules each source's ingest before
the harmonize, and merge_global raises, rather than saving an empty
table, when a unit's candidates were never ingested.

Recipe ids, units and geometries are fabricated.
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from openplaces.core.schema import AdminId
from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer import discover as discover_module
from openplaces.recipe import discovery as discovery_module
from openplaces.recipe import get_recipe_dependencies

SOURCES = [
    {
        'recipe_id': 'XA-AB_admin-local-2026_admin3',
        'admin_id': 'XA-AB',
        'specificity': 2,
        'version': '2026',
        'restricted': False,
    },
    {
        'recipe_id': 'XA-CD_admin-local-2026_admin3',
        'admin_id': 'XA-CD',
        'specificity': 2,
        'version': '2026',
        'restricted': False,
    },
    {
        'recipe_id': 'XA_admin-national-2026_admin3',
        'admin_id': 'XA',
        'specificity': 1,
        'version': '2026',
        'restricted': False,
    },
    {
        'recipe_id': 'XB_admin-national-2026_admin3',
        'admin_id': 'XB',
        'specificity': 1,
        'version': '2026',
        'restricted': False,
    },
    {
        'recipe_id': 'admin-worldwide-2026_admin3',
        'admin_id': '',
        'specificity': 0,
        'version': '2026',
        'restricted': False,
    },
    {
        'recipe_id': 'admin-licensed-2026_admin3',
        'admin_id': '',
        'specificity': 0,
        'version': '2026',
        'restricted': True,
    },
]

HARMONIZE_RECIPE = {
    'recipe_id': 'admin-merged-2026_admin3',
    'stage': 'harmonize',
    'admin_id': None,
    'admin_level': 3,
    'pipeline': [
        {'step': 'discover_sources'},
        {'step': 'merge_global'},
    ],
}


@pytest.fixture
def scanned(monkeypatch):
    monkeypatch.setattr(
        discovery_module, '_scan_admin_ingest_recipes', lambda level: SOURCES
    )


def _state(admin_id, admin_level=3):
    return HarmonizeState(
        recipe={'admin_id': AdminId(), 'admin_level': admin_level},
        admin_id=AdminId(admin_id) if admin_id else None,
        verbose=False,
        timer=None,
        spine=None,
    )


def _discover(monkeypatch, admin_id, units):
    monkeypatch.setattr(
        discover_module,
        'get_admin_ids',
        lambda level, admin_id=None, allow_empty=False: units,
    )
    return discover_module.discover_sources(_state(admin_id))


def _upstream(admin_id):
    return [
        edge.upstream_recipe_id
        for edge in get_recipe_dependencies(HARMONIZE_RECIPE, admin_id=admin_id)
        if edge.step == 'discover_sources'
    ]


class TestOneListing:
    def test_a_unit_lists_sources_covering_it_or_within_it(self, scanned):
        listed = discovery_module.find_admin_source_recipes(3, 'XA-AB')
        assert [s['recipe_id'] for s in listed] == [
            'XA-AB_admin-local-2026_admin3',
            'XA_admin-national-2026_admin3',
            'admin-worldwide-2026_admin3',
        ]
        within = discovery_module.find_admin_source_recipes(3, 'XA')
        assert 'XA-CD_admin-local-2026_admin3' in {s['recipe_id'] for s in within}

    def test_both_callers_read_the_same_sources(self, scanned, monkeypatch):
        state = _discover(monkeypatch, 'XA-AB', ['XA-AB-001', 'XA-AB-002'])
        read = [s['recipe_id'] for s in state.metadata['discovered_sources']]

        assert read == _upstream('XA-AB')
        assert state.metadata['source_candidates']['XA-AB-001'] == read

    def test_a_restricted_source_is_neither_read_nor_scheduled(self, scanned):
        assert 'admin-licensed-2026_admin3' not in _upstream('XA')
        assert 'admin-licensed-2026_admin3' not in _upstream(None)


class TestDependencyEdges:
    def test_a_discover_sources_step_has_an_edge_per_source(self, scanned):
        edges = [
            edge
            for edge in get_recipe_dependencies(HARMONIZE_RECIPE, admin_id='XB')
            if edge.step == 'discover_sources'
        ]

        assert {edge.upstream_recipe_id for edge in edges} == {
            'XB_admin-national-2026_admin3',
            'admin-worldwide-2026_admin3',
        }
        assert all(edge.resolved and edge.kind == 'auto_discover' for edge in edges)

    def test_a_global_run_depends_on_every_open_source(self, scanned):
        assert set(_upstream(None)) == {
            s['recipe_id'] for s in SOURCES if not s['restricted']
        }

    def test_an_excluded_source_is_pruned(self, scanned):
        edges = get_recipe_dependencies(
            HARMONIZE_RECIPE,
            admin_id='XB',
            exclude_recipe_ids={'admin-worldwide-2026_admin3'},
        )
        assert [e.upstream_recipe_id for e in edges] == [
            'XB_admin-national-2026_admin3'
        ]


def _gdf(rows):
    return gpd.GeoDataFrame(
        {'name': list(rows)},
        geometry=list(rows.values()),
        index=pd.Index(list(rows), name='admin3_id'),
        crs='EPSG:4326',
    )


def _merge_state(candidates):
    state = _state('XA-AB')
    state.metadata['discovered_sources'] = [
        {'recipe_id': rid}
        for rid in dict.fromkeys(r for rids in candidates.values() for r in rids)
    ]
    state.metadata['source_candidates'] = candidates
    return state


def _missing(rid, geom):
    raise FileNotFoundError(f'{rid} was never ingested')


class TestMergeGlobalMissingSources:
    def test_raises_when_no_candidate_was_ingested(self, monkeypatch):
        monkeypatch.setattr(discover_module, 'get_entities', _missing)
        state = _merge_state(
            {
                'XA-AB-001': ['XA-AB_admin-local-2026_admin3', 'XA_admin-x_admin3'],
                'XA-AB-002': ['XA-AB_admin-local-2026_admin3'],
            }
        )

        with (
            pytest.warns(UserWarning, match='Could not load'),
            pytest.raises(discover_module.AdminSourcesNotIngestedError) as raised,
        ):
            discover_module.merge_global(state)

        message = str(raised.value)
        assert 'XA-AB' in message
        assert 'level 3' in message
        assert 'XA-AB_admin-local-2026_admin3' in message
        assert 'XA_admin-x_admin3' in message
        assert 'Ingest them' in message
        assert isinstance(raised.value, RuntimeError)

    def test_one_loaded_source_is_enough(self, monkeypatch):
        def read(rid, geom):
            if rid == 'XA-AB_admin-local-2026_admin3':
                raise FileNotFoundError('never ingested')
            return _gdf({'XA-AB-001': box(0, 0, 1, 1)})

        monkeypatch.setattr(discover_module, 'get_entities', read)
        state = _merge_state(
            {'XA-AB-001': ['XA-AB_admin-local-2026_admin3', 'XA_admin-x_admin3']}
        )

        with pytest.warns(UserWarning, match='Could not load'):
            state = discover_module.merge_global(state)

        assert state.metadata['geometry_source'] == {'XA-AB-001': 'XA_admin-x_admin3'}

    def test_a_unit_with_no_candidates_still_saves_empty(self, monkeypatch):
        monkeypatch.setattr(discover_module, 'get_entities', _missing)
        state = _state('XQ', admin_level=2)
        state.metadata['discovered_sources'] = [
            {'recipe_id': 'admin-worldwide-2026_admin2'}
        ]
        state.metadata['source_candidates'] = {}

        with pytest.warns(UserWarning, match='No admin geometries'):
            state = discover_module.merge_global(state)

        assert state.spine is None
