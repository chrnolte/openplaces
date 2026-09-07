"""Tests for `find_additional_layer_recipes`, the bundled-layer lookup.

A bundled layer never has a recipe file of its own: it is declared inside
a host recipe's `additional_layers` list, so the only way to reach it is
through the host's recipe id. Rebuilding that id from its parts drops any
filename suffix, which is what these tests pin.
"""

import pandas as pd
import pytest

from openplaces import recipe as recipe_module
from openplaces.core.schema import Entity
from openplaces.recipe import find_additional_layer_recipes


def _row(admin_id, source_id, version, filename_suffix=''):
    recipe_id = f'{admin_id}_parcel-{source_id}-{version}'
    if filename_suffix:
        recipe_id += f'_{filename_suffix}'
    return {
        'admin_id': admin_id,
        'source_id': source_id,
        'version': version,
        'exclude_from_auto_discover': False,
        'recipe_id': recipe_id,
        'filename_suffix': filename_suffix,
    }


def _host_recipe(recipe_id):
    """A fabricated host recipe bundling one `property` layer."""
    return {
        'recipe_id': recipe_id,
        'stage': 'ingest',
        'additional_layers': [{'entity': Entity('property', 'igac', '2026')}],
    }


@pytest.fixture
def fake_tree(monkeypatch):
    """Two suffixed sibling recipes sharing one admin unit and source."""
    rows = pd.DataFrame(
        [
            _row('CO', 'igac', '2026', 'rural'),
            _row('CO', 'igac', '2026', 'urban'),
        ]
    )
    import openplaces.diagnostics as diagnostics

    monkeypatch.setattr(diagnostics, 'find_recipes', lambda *a, **k: rows)
    monkeypatch.setattr(recipe_module, 'get_recipe_by_id', _host_recipe)


def test_suffixed_siblings_keep_their_own_recipe_ids(fake_tree):
    matches = find_additional_layer_recipes('property', 'CO-AN')

    # Both files are real tables meant to coexist, and neither is named
    # `CO_parcel-igac-2026`, which is the id a rebuild would produce.
    assert sorted(m['recipe_id'] for m in matches) == [
        'CO_parcel-igac-2026_rural',
        'CO_parcel-igac-2026_urban',
    ]
