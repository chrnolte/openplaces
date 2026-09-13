"""An installed package can contribute a recipe directory.

The `openplaces.recipes` entry-point group is the input-side half of the
hub-and-spoke design: a source-specific or licence-bound recipe lives in
its own package, and the hub reads it as if it were bundled. Bundled
recipes win on a name collision, so a package can add but never shadow.
"""

import pytest
import yaml

import openplaces.diagnostics as diagnostics
import openplaces.path as path_module
import openplaces.recipe as recipe_module
from openplaces.io.harmonizer import discover
from openplaces.path import (
    BUNDLED_RECIPES_DIR,
    recipe_path,
    recipe_root_records,
    recipe_roots,
)

RECIPE_ID = 'US-NC_parcel-spoketest-2026'


@pytest.fixture
def external_root(tmp_path, monkeypatch):
    root = tmp_path / 'spoke_recipes'
    target = recipe_path('US-NC', 'parcel-spoketest-2026', root=root)
    target.parent.mkdir(parents=True)
    target.write_text(
        yaml.safe_dump(
            {
                'description': 'A recipe contributed by an installed package.',
                'admin_id': 'US-NC',
                'stage': 'ingest',
                'entity': {
                    'entity_type': 'parcel',
                    'source': {'source_id': 'spoketest'},
                    'version': '2026',
                },
                'save_to': {'data_dir': 'core', 'admin_level': 3},
            }
        ),
        encoding='utf-8',
    )
    monkeypatch.setattr(
        path_module,
        '_external_recipe_root_records',
        lambda: [
            {
                'root': str(root),
                'provider': 'spoketest',
                'distribution': 'openplaces-spoketest',
                'version': '0.1.0',
            }
        ],
    )
    recipe_root_records.cache_clear()
    recipe_roots.cache_clear()
    diagnostics._recipe_index.cache_clear()
    recipe_module.iter_entity_source_versions.cache_clear()
    recipe_module.iter_entity_sources.cache_clear()
    recipe_module.provenance_suffixes.cache_clear()
    yield root
    recipe_root_records.cache_clear()
    recipe_roots.cache_clear()
    diagnostics._recipe_index.cache_clear()
    recipe_module.iter_entity_source_versions.cache_clear()
    recipe_module.iter_entity_sources.cache_clear()
    recipe_module.provenance_suffixes.cache_clear()


def test_bundled_root_comes_first(external_root):
    assert recipe_roots()[0] == BUNDLED_RECIPES_DIR
    assert external_root in recipe_roots()


def test_a_contributed_recipe_loads_by_id(external_root):
    recipe = recipe_module.get_recipe_by_id(RECIPE_ID)
    assert recipe['recipe_id'] == RECIPE_ID
    assert str(recipe['entity']) == 'parcel-spoketest-2026'


def test_a_contributed_recipe_is_indexed_and_discoverable(external_root):
    found = diagnostics.find_recipes('parcel', stage='ingest')
    assert RECIPE_ID in set(found['recipe_id'])
    scanned = {s['recipe_id'] for s in discover._scan_entity_ingest_recipes('parcel')}
    assert RECIPE_ID in scanned


def test_a_contributed_source_joins_the_suffix_vocabulary(external_root):
    assert ('parcel', 'spoketest') in recipe_module.iter_entity_sources()


def test_a_missing_recipe_still_resolves_under_the_bundled_root(external_root):
    resolved = recipe_path('US-NC', 'parcel-nowhere-2026')
    assert BUNDLED_RECIPES_DIR in resolved.parents


def test_a_broken_entry_point_is_skipped_with_a_warning(monkeypatch, tmp_path):
    class BrokenEntryPoint:
        name = 'broken'
        value = 'nowhere:nothing'

        def load(self):
            raise ImportError('no such module')

    monkeypatch.setattr(
        path_module.importlib.metadata,
        'entry_points',
        lambda group: [BrokenEntryPoint()],
    )
    with pytest.warns(UserWarning, match='could not be loaded'):
        assert path_module._external_recipe_roots() == []


def test_root_records_name_the_provider_and_version(external_root):
    records = recipe_root_records()
    assert records[0]['provider'] == 'bundled'
    assert records[0]['distribution'] == 'openplaces'
    assert records[0]['root'] == str(BUNDLED_RECIPES_DIR)
    assert records[1] == {
        'root': str(external_root),
        'provider': 'spoketest',
        'distribution': 'openplaces-spoketest',
        'version': '0.1.0',
    }


def test_every_attribute_table_records_the_recipe_roots(external_root, tmp_path):
    import json

    import pandas as pd
    import pyarrow.parquet as pq

    from openplaces.io import save_parquet, to_parquet
    from openplaces.path import RECIPE_ROOTS_METADATA_KEY

    plain = tmp_path / 'plain.parquet'
    to_parquet(pd.DataFrame({'a': [1]}), plain)
    meta = pq.read_metadata(plain).metadata
    recorded = json.loads(meta[RECIPE_ROOTS_METADATA_KEY.encode()].decode())
    assert [r['provider'] for r in recorded] == ['bundled', 'spoketest']

    # A caller's own footer entries ride alongside, not instead.
    stamped = tmp_path / 'stamped.parquet'
    to_parquet(pd.DataFrame({'a': [1]}), stamped, file_metadata={'x': 'y'})
    meta = pq.read_metadata(stamped).metadata
    assert meta[b'x'] == b'y' and RECIPE_ROOTS_METADATA_KEY.encode() in meta

    # The split layout's attribute file carries it too.
    import geopandas as gpd
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame({'a': [1]}, geometry=[Point(0, 0)], crs='EPSG:4326')
    split = tmp_path / 'split.parquet'
    save_parquet(gdf, split)
    assert RECIPE_ROOTS_METADATA_KEY.encode() in pq.read_metadata(split).metadata
