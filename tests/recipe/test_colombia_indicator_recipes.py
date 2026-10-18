"""Consistency tests for the Colombian indicator recipes.

The land value model trains on transactions and predicts on parcels, and
the impact evaluation compares treated polygons with untreated parcels,
so the recipes that produce their columns must stay identical pairwise.
"""

from __future__ import annotations

import pytest

from openplaces.core.attribute_registry import is_personal_attribute, load_registry
from openplaces.recipe import get_recipe_by_id

RECIPE_IDS = [
    'CO_transaction-reboredo-2021',
    'CO_transaction-spine-2026',
    'CO_transaction-openplaces-2026',
    'CO_landcover-arevalo-2025',
    'CO-AN_landcover-forest-hansen-gfc2022',
    'CO-AN_parcel_landcover-forest-annual-2026',
    'CO_transaction_landcover-forest-annual-2026',
    'CO_parcel_land-value-predictors-2026',
    'CO_transaction_land-value-predictors-2026',
]

PAIRS = [
    (
        'CO-AN_parcel_landcover-forest-annual-2026',
        'CO_transaction_landcover-forest-annual-2026',
        {'raster_admin_id'},
    ),
    (
        'CO_parcel_land-value-predictors-2026',
        'CO_transaction_land-value-predictors-2026',
        set(),
    ),
]


@pytest.mark.parametrize('recipe_id', RECIPE_IDS)
def test_recipe_loads(recipe_id):
    recipe = get_recipe_by_id(recipe_id)
    assert recipe is not None, recipe_id


def _steps(recipe_id, ignore):
    steps = get_recipe_by_id(recipe_id)['pipeline']
    return [{k: v for k, v in s.items() if k not in ignore} for s in steps]


@pytest.mark.parametrize('parcel_id, transaction_id, ignore', PAIRS)
def test_paired_recipes_have_identical_steps(parcel_id, transaction_id, ignore):
    assert _steps(parcel_id, ignore) == _steps(transaction_id, ignore)


def _written_columns(recipe_id):
    for step in get_recipe_by_id(recipe_id)['pipeline']:
        column = step.get('column') or step.get('output')
        if column is None:
            continue
        yield column.split('_{year}')[0] if '{year}' in column else column


@pytest.mark.parametrize(
    'recipe_id',
    [
        'CO_parcel_land-value-predictors-2026',
        'CO_transaction_land-value-predictors-2026',
    ],
)
def test_static_predictor_columns_are_registered(recipe_id):
    registry = load_registry()
    missing = [c for c in _written_columns(recipe_id) if c not in registry.index]
    assert not missing, missing


def test_ingest_recipe_maps_no_person_columns():
    """The acquisitions layer carries owner names (`propietari`,
    `propieta_1`) and a free-text `nombre`. None may be read into an
    output column, and all are dropped at ingest."""
    recipe = get_recipe_by_id('CO_transaction-reboredo-2021')
    person_sources = {'propietari', 'propieta_1', 'nombre'}
    outputs = set(recipe['columns']) | {t['output'] for t in recipe['transformations']}
    assert not [c for c in outputs if is_personal_attribute(c)]
    sources = set(recipe['columns'].values())
    for t in recipe['transformations']:
        sources |= set(t.get('inputs', [])) | {t.get('input')}
    assert not sources & person_sources
    assert person_sources <= set(recipe['drop_columns'])
    assert recipe.get('keep_unnamed_columns') is False


def test_derived_layers_reference_known_ops():
    from openplaces.geo.raster_derive import OPS

    for recipe_id in (
        'CO_landcover-arevalo-2025',
        'CO-AN_landcover-forest-hansen-gfc2022',
    ):
        for spec in get_recipe_by_id(recipe_id)['derived_layers']:
            assert spec['op'] in OPS, (recipe_id, spec)
            assert len(spec['years']) == 2
