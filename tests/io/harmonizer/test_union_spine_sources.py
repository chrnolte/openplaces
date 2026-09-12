"""Tests for the harmonizer `union_spine_sources` step.

Non-spatial peer of `resolve_spine`, for entities with no geometry (e.g. a
transaction spine): concatenates rows from every discovered source, with no
IoU/geometry dedup, since sources like transactions are always disjoint by
admin scope.
"""

import pandas as pd
import pytest

import openplaces.io.harmonizer.spine as spine_module
from openplaces.core.schema import AdminId
from openplaces.io.harmonizer import HarmonizeState


def _state(admin_id='US-NC-NH'):
    return HarmonizeState(
        recipe={'admin_id': AdminId('US')},
        admin_id=AdminId(admin_id),
        verbose=False,
        timer=None,
        spine=None,
    )


def _mock_discovery(monkeypatch, rows):
    # `find_recipes` always carries `recipe_id` (the recipe file's stem),
    # which the discovery step reads rather than rebuilding; for these
    # unsuffixed rows the two forms coincide.
    rows = [
        {
            'recipe_id': (
                f'{r["admin_id"]}_{r["entity_type"]}-{r["source_id"]}-{r["version"]}'
            ),
            **r,
        }
        for r in rows
    ]
    monkeypatch.setattr(
        spine_module, 'find_recipes', lambda *a, **k: pd.DataFrame(rows)
    )


def test_concatenates_sources_with_provenance_column(monkeypatch):
    _mock_discovery(
        monkeypatch,
        [
            {
                'admin_id': 'US-NC-NH',
                'source_id': 'nhcgov',
                'version': '2026',
                'entity_type': 'transaction',
                'exclude_from_auto_discover': False,
            }
        ],
    )
    df = pd.DataFrame({'price': [100, 200]})
    monkeypatch.setattr(spine_module, 'get_entities', lambda *a, **k: df)

    state = spine_module.union_spine_sources(
        _state(), sources=[{'auto_discover': True, 'entity_type': 'transaction'}]
    )

    assert len(state.spine) == 2
    assert state.spine['source'].tolist() == ['nhcgov', 'nhcgov']


def test_no_sources_configured_warns_and_leaves_spine_none():
    with pytest.warns(UserWarning, match='no sources configured'):
        state = spine_module.union_spine_sources(_state(), sources=None)
    assert state.spine is None


def test_load_failure_is_skipped_not_fatal(monkeypatch):
    _mock_discovery(
        monkeypatch,
        [
            {
                'admin_id': 'US-NC-NH',
                'source_id': 'nhcgov',
                'version': '2026',
                'entity_type': 'transaction',
                'exclude_from_auto_discover': False,
            }
        ],
    )

    def _raise(*a, **k):
        raise FileNotFoundError('not ingested')

    monkeypatch.setattr(spine_module, 'get_entities', _raise)

    with pytest.warns(UserWarning, match='no rows loaded'):
        state = spine_module.union_spine_sources(
            _state(), sources=[{'auto_discover': True, 'entity_type': 'transaction'}]
        )
    assert state.spine is None


def test_empty_source_is_dropped_but_others_still_load(monkeypatch):
    _mock_discovery(
        monkeypatch,
        [
            {
                'admin_id': 'US-NC-NH',
                'source_id': 'a',
                'version': '1',
                'entity_type': 'transaction',
                'exclude_from_auto_discover': False,
            },
            {
                'admin_id': 'US-NC-NH',
                'source_id': 'b',
                'version': '1',
                'entity_type': 'transaction',
                'exclude_from_auto_discover': False,
            },
        ],
    )
    frames = {
        'US-NC-NH_transaction-a-1': pd.DataFrame(),
        'US-NC-NH_transaction-b-1': pd.DataFrame({'price': [1]}),
    }
    monkeypatch.setattr(
        spine_module, 'get_entities', lambda recipe_id, *a, **k: frames[recipe_id]
    )

    state = spine_module.union_spine_sources(
        _state(), sources=[{'auto_discover': True, 'entity_type': 'transaction'}]
    )

    assert len(state.spine) == 1
    assert state.spine['source'].iloc[0] == 'b'


def test_supplement_recipe_adds_no_rows(monkeypatch):
    # A detail table beside its roll (one row per building component)
    # details the roll's properties. Unioning it counted every component
    # as a property: Victoria TX's spine held the roll's 53,405 rows plus
    # its 46,707 components.
    _mock_discovery(
        monkeypatch,
        [
            {
                'admin_id': 'US-TX-VIC',
                'source_id': 'roll',
                'version': '1',
                'entity_type': 'property',
                'exclude_from_auto_discover': False,
                'supplements': '',
            },
            {
                'admin_id': 'US-TX-VIC',
                'source_id': 'detail',
                'version': '1',
                'entity_type': 'property',
                'exclude_from_auto_discover': False,
                'supplements': 'US-TX-VIC_property-roll-1',
            },
        ],
    )
    frames = {
        'US-TX-VIC_property-roll-1': pd.DataFrame({'parcel_id_local': ['a', 'b']}),
        'US-TX-VIC_property-detail-1': pd.DataFrame(
            {'parcel_id_local': ['a', 'a', 'b']}
        ),
    }
    monkeypatch.setattr(
        spine_module, 'get_entities', lambda recipe_id, *a, **k: frames[recipe_id]
    )

    state = spine_module.union_spine_sources(
        _state('US-TX-VIC'),
        sources=[{'auto_discover': True, 'entity_type': 'property'}],
    )

    assert len(state.spine) == 2
    assert set(state.spine['source']) == {'roll'}
