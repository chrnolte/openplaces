"""The core schema is read from the code, and the code stays inside it.

plans/core-schema-and-stage-contracts-review.md, deliverable 1: the
schema page is generated (flow/core_schema.py), the registry's tags are
valid and shrinking in their gaps, the link methods it documents are the
ones the harmonizer writes, and every ingest recipe of a spatial or
keyed entity reaches the schema through a geometry, a key, a layer or a
supplement. These tests need no built data; the observed-columns part of
the page is exercised only where a unit has been built on this machine.
"""

from pathlib import Path

import pytest
import yaml

from openplaces.core.schema import ENTITY_TYPES
from openplaces.flow import core_schema
from openplaces.path import recipe_roots

STAGES = set(core_schema.STAGES)

#: Registry rows with no stage tag on 2026-09-29 and no evidence of a
#: writer in any ingest recipe or built spine. Each is either retired
#: or waits for the recipe that will write it; the list may only shrink.
UNTAGGED_STAGE_ALLOWED = {
    'area_sqft',
    'classified_market_value',
    'n_sales',
    'owner_name_january_1',
    'personal_property_value',
    'purpose_subgroup',
}

#: Entity types whose ingest recipes are expected to reach the schema
#: through a geometry, a local key, an entity id, a layer or a
#: supplement. Admin, tile, image and person recipes are keyed by their
#: own conventions and are not checked here.
KEYED_ENTITY_TYPES = (
    'parcel',
    'footprint',
    'building',
    'dwelling',
    'property',
    'transaction',
)

#: Column names that name a record within its source, any one of which
#: is a way into the schema for a table without geometry.
KEY_ATTRIBUTES = {
    'parcel_id_local',
    'parcel_id_assessor',
    'parcel_id_admin2',
    'parcel_id_admin3',
    'property_id_assessor',
    'property_id_admin2',
    'geo_id',
    'entity_id',
    'address',
    'transaction_id',
    'document_id',
    'sale_document_number',
    'lot_id_local',
}


def _ingest_recipes():
    for root in recipe_roots():
        for path in sorted(root.rglob('*.yaml')):
            try:
                data = yaml.safe_load(path.read_text(encoding='utf8'))
            except yaml.YAMLError:
                continue
            if not isinstance(data, dict) or 'entity' not in data:
                continue
            if data.get('stage', 'ingest') != 'ingest':
                continue
            yield path, data


#: Ingest recipes known not to reach the schema, with the reason; the
#: list may only shrink. A transaction source needs a parcel, property
#: or address key, because a deed number names the document and not
#: what it conveyed.
UNREACHED_ALLOWED = {
    # Indiana's sales disclosure file carries no parcel-identifying column
    # (the recipe's own header says so); its rows cannot reach the parcel
    # spine until a parcel key is found in another table of the file.
    'US-IN_transaction-statsindiana-2026',
}


def _reaches_schema(recipe: dict) -> bool:
    """A geometry, a key column, an entity id, a layer of a host, or a
    supplement of a roll is a way into the schema."""
    if recipe.get('supplements') or recipe.get('entity_id'):
        return True
    if recipe.get('parcel_id_local') or recipe.get('lot_key'):
        return True
    columns = set((recipe.get('columns') or {}).keys())
    for table in recipe.get('additional_layers') or []:
        if isinstance(table, dict):
            columns |= set((table.get('columns') or {}).keys())
    for t in recipe.get('transformations') or []:
        if isinstance(t, dict) and isinstance(t.get('output'), str):
            columns.add(t['output'])
    if recipe['entity'].get('entity_type') == 'transaction':
        # The parcel, property or address the deed conveyed, by any of
        # the names the transaction recipes use for them.
        conveyed = {
            'parcel_id_local',
            'parcel_id_assessor',
            'property_id_assessor',
            'address',
            'street_number',
            'address_number',
            'lot_id_local',
        }
        return bool(columns & conveyed)
    if columns & KEY_ATTRIBUTES or any(
        c.endswith('_id') or c.endswith('_id_local') for c in columns
    ):
        return True
    # A geometry hash (geo_id) is minted from the outline, and the admin
    # overlay places the row: both mean the source is spatial.
    if recipe.get('create_index') or recipe.get('overlay_admin_ids'):
        return True
    # A spatial file supplies geo_id from its geometry: a raster, an
    # ArcGIS layer, a geodatabase, a shapefile or a GeoJSON export.
    source = recipe['entity'].get('source') or {}
    name = ' '.join(
        str(v)
        for v in (
            recipe.get('uncompressed_file_name'),
            source.get('download_url'),
            recipe.get('scraper'),
            recipe.get('reader'),
            recipe.get('layer_name'),
        )
        if v
    ).lower()
    spatial_hints = (
        '.geojson',
        '.gdb',
        '.shp',
        'featureserver',
        'mapserver',
        'arcgis',
        '.gpkg',
        '.fgb',
        'geoparquet',
        '.parquet',
        'kml',
    )
    return any(h in name for h in spatial_hints) or bool(recipe.get('geometry'))


def test_every_entity_type_is_in_the_page():
    text = core_schema.render(admin_id=None)
    for entity_type in ENTITY_TYPES:
        assert f'- {entity_type}\n' in text or entity_type in text


def test_registry_tags_are_valid():
    registry = core_schema.registry_table()
    bad_stage = registry[~registry['stage'].isin(STAGES | {''})]
    assert bad_stage.empty, bad_stage['name'].tolist()
    bad_entity = registry[~registry['entity_type'].isin(set(ENTITY_TYPES) | {''})]
    assert bad_entity.empty, bad_entity['name'].tolist()


def test_untagged_stage_rows_only_shrink():
    registry = core_schema.registry_table()
    missing = set(registry.loc[registry['stage'] == '', 'name'])
    new = missing - UNTAGGED_STAGE_ALLOWED
    assert not new, (
        'registry rows without a stage tag that are not on the allowlist; '
        f'tag them with the stage that first writes them: {sorted(new)}'
    )


def test_documented_link_methods_are_the_ones_written():
    source = (
        Path(core_schema.__file__).parents[1] / 'io/harmonizer/entity_links.py'
    ).read_text(encoding='utf8')
    for method in core_schema.LINK_METHODS:
        assert f"'{method}'" in source, method


def test_every_keyed_ingest_recipe_reaches_the_schema():
    unreached = [
        str(path.stem)
        for path, recipe in _ingest_recipes()
        if recipe['entity'].get('entity_type') in KEYED_ENTITY_TYPES
        and not _reaches_schema(recipe)
    ]
    new = sorted(set(unreached) - UNREACHED_ALLOWED)
    assert not new, (
        'ingest recipes with no geometry, key, entity id, layer or supplement '
        f'that are not on the allowlist: {new}'
    )


@pytest.mark.parametrize('admin_id', ['US-NC-CUR'])
def test_observed_columns_report_where_built(admin_id):
    spines = [
        s
        for row in core_schema.entity_table()['spines']
        if row != '(none)'
        for s in row.split(', ')
    ]
    observed = core_schema.observed_columns(admin_id, spines)
    assert {o['recipe_id'] for o in observed} == set(spines)
    if not any(o['built'] for o in observed):
        pytest.skip(f'{admin_id} is not built in this data root')
    for item in observed:
        if item['built']:
            accounted = item['n_keys'] + item['n_sidecars'] + len(item['unregistered'])
            assert accounted <= item['n_columns']
            assert item['n_rows'] > 0
