"""The terms walk reads a dependency that loads as a table in place.

An admin_id_crosswalk may name the committed admin spine
(admin-spine-2026_admin{N}.csv), which get_recipe_by_id returns as a
DataFrame rather than a recipe. It declares no Source and no input, so
the walk adds no entry for it and walks nothing beneath it, while the
recipes beside it still report their terms. Recipe ids are fabricated.
"""

from types import SimpleNamespace

import pandas as pd

import openplaces.io.delivery.terms as terms_module
from openplaces.recipe import DepEdge

BUNDLE = 'XA_parcel-merged-2026'
SOURCE = 'XA_parcel-local-2026'
TABLE = 'admin-table-2026_admin2'

RECIPES = {
    BUNDLE: {'recipe_id': BUNDLE, 'stage': 'curate'},
    SOURCE: {
        'recipe_id': SOURCE,
        'stage': 'ingest',
        'admin_id': 'XA',
        'entity': {
            'entity_type': 'parcel',
            'source': SimpleNamespace(
                source_id='local',
                license='CC-BY-4.0',
                redistribution_restricted=False,
            ),
        },
    },
    TABLE: pd.DataFrame({'admin2_id': ['XA-AB'], 'name': ['Fabricated']}),
}

EDGES = {
    BUNDLE: [DepEdge(BUNDLE, SOURCE, 'entity_recipe')],
    SOURCE: [DepEdge(SOURCE, TABLE, 'admin_recipe_id')],
}


def _dependencies(recipe, admin_id=None, **kwargs):
    if not isinstance(recipe, dict):
        raise ValueError('Recipe has neither an entity nor a dataset.')
    return EDGES.get(recipe['recipe_id'], [])


def test_a_table_dependency_adds_no_entry_and_is_not_walked(monkeypatch):
    monkeypatch.setattr(terms_module, 'get_recipe_dependencies', _dependencies)
    monkeypatch.setattr(terms_module, 'get_recipe_by_id', lambda rid: RECIPES[rid])

    found = terms_module._source_terms(RECIPES[BUNDLE])

    assert [entry['recipe_id'] for entry in found.values()] == [SOURCE]
