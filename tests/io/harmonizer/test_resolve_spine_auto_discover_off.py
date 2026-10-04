"""`- auto_discover: false` makes a spine's sources exactly the ones listed.

resolve_spine discovers ingest recipes of the spine's entity type whether
or not a sentinel is listed; the sentinel only places them. The building
geospine is built from the footprint geospine alone, and without a way to
say so the CHEER building inventory was merged in (Currituck, 2026-09-29:
1,637 rows no footprint drew).
"""

from __future__ import annotations

import pandas as pd

from openplaces.core.schema import AdminId, Entity
from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer import spine as spine_module


def _state():
    return HarmonizeState(
        recipe={
            'admin_id': AdminId('US'),
            'entity': Entity('building', 'geospine', '2026'),
        },
        admin_id=AdminId('US-XX-YY'),
        verbose=False,
        timer=None,
    )


def _fake_index():
    return pd.DataFrame(
        [
            {
                'admin_id': 'US-XX',
                'recipe_id': 'US-XX_building-fabricated-2026',
                'source_id': 'fabricated',
                'version': '2026',
                'exclude_from_auto_discover': False,
                'supplements': '',
            }
        ]
    )


def test_discovery_is_on_without_a_sentinel(monkeypatch):
    monkeypatch.setattr(spine_module, 'find_recipes', lambda *a, **k: _fake_index())
    monkeypatch.setattr(
        'openplaces.recipe.find_additional_layer_recipes', lambda *a, **k: []
    )
    sources = [{'recipe_id': 'US_footprint-geospine-2026', 'label': 'footprint'}]
    resolved = spine_module._expand_auto_discover(sources, _state())
    assert [s['recipe_id'] for s in resolved] == [
        'US_footprint-geospine-2026',
        'US-XX_building-fabricated-2026',
    ]


def test_a_false_sentinel_keeps_only_the_listed_sources(monkeypatch):
    monkeypatch.setattr(spine_module, 'find_recipes', lambda *a, **k: _fake_index())
    monkeypatch.setattr(
        'openplaces.recipe.find_additional_layer_recipes', lambda *a, **k: []
    )
    sources = [
        {'recipe_id': 'US_footprint-geospine-2026', 'label': 'footprint'},
        {'auto_discover': False},
    ]
    resolved = spine_module._expand_auto_discover(sources, _state())
    assert resolved == [
        {'recipe_id': 'US_footprint-geospine-2026', 'label': 'footprint'}
    ]
