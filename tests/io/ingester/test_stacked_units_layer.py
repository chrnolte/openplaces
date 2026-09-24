"""A parcel ingest recipe carries an implicit property layer, and the table
ingester writes a parcel table's stacked units into it.

The layer is added when the recipe loads (so discovery, readers and
cleanup see it like a declared one), skipped by the ingester's
source-reading loop, and filled by the split that runs between
preprocessing and saving.
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from openplaces.core.schema import Entity
from openplaces.io.ingester.table_ingester import TableIngester
from openplaces.recipe import (
    STACKED_UNITS_LAYER_KEY,
    _add_stacked_units_layer,
    build_table_recipe,
    get_layers,
)
from openplaces.timing import Timer


def _recipe(**extra):
    recipe = {
        'stage': 'ingest',
        'entity': Entity('parcel', 'countygis', '2026'),
        'save_to': {'data_dir': 'core'},
    }
    recipe.update(extra)
    _add_stacked_units_layer(recipe)
    return recipe


def test_a_parcel_ingest_recipe_gets_an_implicit_property_layer():
    recipe = _recipe()
    assert get_layers(recipe) == ['property']
    spec = recipe['additional_layers'][0]
    assert spec[STACKED_UNITS_LAYER_KEY] is True
    assert str(spec['entity']) == 'property-countygis-2026'
    # A unit names its lot here and keeps its own parcel_id_local.
    assert spec['layer_key'] == 'lot_id_local'
    table = build_table_recipe(recipe, spec)
    assert table['save_to'] == {'data_dir': 'core'}


def test_opting_out_or_declaring_a_property_layer_adds_none():
    assert get_layers(_recipe(stacked_units=False)) == []
    declared = {'entity': Entity('property', 'countygis', '2026'), 'columns': {}}
    recipe = _recipe(additional_layers=[declared])
    assert recipe['additional_layers'] == [declared]


def test_harmonize_and_non_parcel_recipes_get_none():
    assert 'additional_layers' not in _recipe(stage='harmonize')
    assert 'additional_layers' not in _recipe(entity=Entity('property', 's', '1'))


def _make_ingester(recipe, saved):
    ingester = TableIngester.__new__(TableIngester)
    ingester.recipe = recipe
    ingester.processing_chunk = {'admin_id_to_process': 'US-XX-YY'}
    ingester.download_partition = {}
    ingester.verbose = False
    ingester.timer = Timer('test')
    ingester._save_recipe_data = lambda gdf, recipe=None: saved.append(
        (str((recipe or ingester.recipe)['entity']), gdf)
    )
    return ingester


def test_the_split_writes_the_parcel_table_and_the_property_layer():
    recipe = _recipe()
    gdf = gpd.GeoDataFrame(
        {'parcel_id_local': ['u1', 'u2', 'h1'], 'land_value': [1.0, 1.0, 2.0]},
        geometry=[box(0, 0, 1, 1), box(0, 0, 1, 1), box(2, 2, 3, 3)],
        crs='EPSG:4326',
    )
    gdf = TableIngester._preprocess_recipe_data(_make_ingester(recipe, []), gdf)
    saved = []
    ingester = _make_ingester(recipe, saved)
    parcels, properties = ingester._split_stacked_units(gdf)
    ingester._save_recipe_data(parcels)
    ingester._save_recipe_data(properties, recipe=ingester._stacked_units_recipe())

    assert [entity for entity, _ in saved] == [
        'parcel-countygis-2026',
        'property-countygis-2026',
    ]
    assert len(saved[0][1]) == 2 and len(saved[1][1]) == 2
    units = saved[1][1]
    assert units['lot_id_local'].nunique() == 1
    assert sorted(units['parcel_id_local']) == ['u1', 'u2']
    assert units['lot_id_local'].iloc[0] == saved[0][1]['parcel_id_local'].iloc[0]


def test_a_recipe_with_its_own_index_is_left_alone():
    recipe = _recipe(set_index='key')
    gdf = gpd.GeoDataFrame(
        {'key': ['a', 'b']},
        geometry=[box(0, 0, 1, 1), box(0, 0, 1, 1)],
        crs='EPSG:4326',
    )
    gdf = TableIngester._preprocess_recipe_data(_make_ingester(recipe, []), gdf)
    parcels, properties = _make_ingester(recipe, [])._split_stacked_units(gdf)
    assert properties is None and len(parcels) == 2
    assert isinstance(parcels, pd.DataFrame)
