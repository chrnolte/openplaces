"""The attribute checkpoint is stamped with every geospine its covered
steps load, so rebuilding the geospine whose links the building spine
re-keys invalidates the building spine's checkpoint.

Found 2026-10-17: only the entity_recipe was stamped, so a footprint
geospine rerun on its own let the building spine restore its old parcel
evidence. Every id below is fabricated.
"""

import openplaces.recipe as recipe_module
from openplaces.io.harmonizer import _checkpoint_chain

PIPELINE = [
    {
        'step': 'load_geospine',
        'links_recipe_id': 'US_footprint-geospine-2026',
        'links_key': 'footprint_id',
    },
    {'step': 'reconcile_attributes', 'checkpoint': True},
]
RECIPE = {'entity_recipe': 'US_building-geospine-2026', 'pipeline': PIPELINE}


def _patch_outputs(monkeypatch, tmp_path):
    paths = {
        'US_building-geospine-2026': tmp_path / 'building.parquet',
        'US_footprint-geospine-2026': tmp_path / 'footprint.parquet',
    }
    for path in paths.values():
        path.write_bytes(b'x')
    monkeypatch.setattr(recipe_module, 'get_recipe_by_id', lambda rid: {'id': rid})
    monkeypatch.setattr(
        recipe_module, 'get_output_path', lambda recipe, admin: paths[recipe['id']]
    )
    return paths


def test_the_links_geospine_is_stamped(monkeypatch, tmp_path):
    _patch_outputs(monkeypatch, tmp_path)
    chain = _checkpoint_chain(RECIPE, PIPELINE, 1, 'US-XX-YY')
    assert chain['format'] == 2
    assert [s['path'] for s in chain['sources']] == [
        'building.parquet',
        'footprint.parquet',
    ]


def test_a_rebuilt_links_geospine_changes_the_chain(monkeypatch, tmp_path):
    paths = _patch_outputs(monkeypatch, tmp_path)
    before = _checkpoint_chain(RECIPE, PIPELINE, 1, 'US-XX-YY')
    paths['US_footprint-geospine-2026'].write_bytes(b'rebuilt')
    after = _checkpoint_chain(RECIPE, PIPELINE, 1, 'US-XX-YY')
    assert before != after


def test_a_recipe_without_links_stamps_its_entity_recipe_only(monkeypatch, tmp_path):
    _patch_outputs(monkeypatch, tmp_path)
    pipeline = [{'step': 'load_geospine'}, {'step': 'x', 'checkpoint': True}]
    chain = _checkpoint_chain(RECIPE, pipeline, 1, 'US-XX-YY')
    assert [s['path'] for s in chain['sources']] == ['building.parquet']
