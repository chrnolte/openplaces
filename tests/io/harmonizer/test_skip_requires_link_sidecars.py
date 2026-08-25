"""A geospine whose link sidecars are missing must not be skipped.

The spine and its sidecars come apart whenever a run wrote the spine
while an upstream ingest was failing. The spine then exists, the next
run skips on that alone, and the attribute recipe fails with "missing or
stale link sidecar" pointing at a geospine that looks finished.
"""

from pathlib import Path

import pytest

from openplaces.core.schema import AdminId
from openplaces.io.harmonizer import _missing_link_sidecars

RECIPE_ID = 'US_footprint-geospine-2026'
REFERENCE_ID = 'US-NC_parcel-nconemap-2025'
ADMIN = AdminId('US', 'NC', 'CM')


@pytest.fixture
def built_references(tmp_path, monkeypatch):
    """Treat every reference as built, unless a test says otherwise."""
    monkeypatch.setattr(
        'openplaces.io.harmonizer.get_recipe_by_id', lambda rid, **k: {'recipe_id': rid}
    )
    monkeypatch.setattr(
        'openplaces.io.harmonizer.get_output_path',
        lambda recipe, admin_id=None: tmp_path / f'built_{recipe["recipe_id"]}',
    )
    return tmp_path


@pytest.fixture
def link_path(tmp_path, monkeypatch, built_references):
    """Route every sidecar lookup into a temporary directory."""
    path = tmp_path / f'{RECIPE_ID}_{REFERENCE_ID}.parquet'

    def fake_path(recipe_id_a, recipe_id_b, admin_id=None):
        other = recipe_id_b if recipe_id_a == RECIPE_ID else recipe_id_a
        return tmp_path / f'{RECIPE_ID}_{other}.parquet'

    monkeypatch.setattr('openplaces.geo.link.get_entity_link_path', fake_path)
    (tmp_path / f'built_{REFERENCE_ID}').write_bytes(b'')
    for name in ('ref-a', 'ref-b'):
        (tmp_path / f'built_{name}').write_bytes(b'')
    return path


def test_unbuilt_reference_owes_nothing(link_path, tmp_path):
    """A reference the linking step will only warn about owes no sidecar.

    Demanding one would re-run the geometry phase on every pass for as
    long as that upstream stays unbuilt.
    """
    (tmp_path / f'built_{REFERENCE_ID}').unlink()
    recipe = _recipe([{'step': 'link_to_reference', 'recipe_id': REFERENCE_ID}])
    assert _missing_link_sidecars(recipe, ADMIN) == []


def _recipe(pipeline):
    return {'recipe_id': RECIPE_ID, 'pipeline': pipeline}


def test_missing_sidecar_is_reported(link_path):
    recipe = _recipe([{'step': 'link_to_reference', 'recipe_id': REFERENCE_ID}])
    assert _missing_link_sidecars(recipe, ADMIN) == [link_path]


def test_present_sidecar_is_not_reported(link_path):
    link_path.write_bytes(b'')
    recipe = _recipe([{'step': 'link_to_reference', 'recipe_id': REFERENCE_ID}])
    assert _missing_link_sidecars(recipe, ADMIN) == []


def test_save_link_false_owes_nothing(link_path):
    recipe = _recipe(
        [{'step': 'link_to_reference', 'recipe_id': REFERENCE_ID, 'save_link': False}]
    )
    assert _missing_link_sidecars(recipe, ADMIN) == []


def test_other_step_owes_nothing_without_save_link(link_path):
    recipe = _recipe([{'step': 'resolve_spine', 'recipe_id': REFERENCE_ID}])
    assert _missing_link_sidecars(recipe, ADMIN) == []


def test_other_step_owes_a_sidecar_when_save_link_is_set(link_path):
    recipe = _recipe(
        [{'step': 'link_by_id', 'recipe_id': REFERENCE_ID, 'save_link': True}]
    )
    assert _missing_link_sidecars(recipe, ADMIN) == [link_path]


def test_empty_and_absent_pipelines_owe_nothing():
    assert _missing_link_sidecars({'recipe_id': RECIPE_ID}, ADMIN) == []
    assert _missing_link_sidecars(_recipe([]), ADMIN) == []
    assert _missing_link_sidecars(_recipe(['not-a-dict']), ADMIN) == []


def test_unresolvable_reference_is_left_to_the_linking_step(
    monkeypatch, built_references
):
    """Resolution failure is the step's error to raise, not the skip check's."""

    def boom(*args, **kwargs):
        raise RuntimeError('no such reference')

    monkeypatch.setattr(
        'openplaces.io.harmonizer.links._resolve_reference_recipe', boom
    )
    recipe = _recipe([{'step': 'link_to_reference', 'entity_type': 'parcel'}])
    assert _missing_link_sidecars(recipe, ADMIN) == []


def test_reference_resolving_to_none_owes_nothing(monkeypatch, built_references):
    monkeypatch.setattr(
        'openplaces.io.harmonizer.links._resolve_reference_recipe',
        lambda *a, **k: (None, None),
    )
    recipe = _recipe([{'step': 'link_to_reference', 'entity_type': 'parcel'}])
    assert _missing_link_sidecars(recipe, ADMIN) == []


def test_every_missing_sidecar_is_listed(tmp_path, monkeypatch, built_references):
    monkeypatch.setattr(
        'openplaces.geo.link.get_entity_link_path',
        lambda a, b, admin_id=None: tmp_path / f'{b}.parquet',
    )
    for name in ('ref-a', 'ref-b'):
        (tmp_path / f'built_{name}').write_bytes(b'')
    recipe = _recipe(
        [
            {'step': 'link_to_reference', 'recipe_id': 'ref-a'},
            {'step': 'link_to_reference', 'recipe_id': 'ref-b'},
        ]
    )
    (tmp_path / 'ref-a.parquet').write_bytes(b'')
    assert _missing_link_sidecars(recipe, ADMIN) == [tmp_path / 'ref-b.parquet']


def test_returns_paths(link_path):
    recipe = _recipe([{'step': 'link_to_reference', 'recipe_id': REFERENCE_ID}])
    assert all(isinstance(p, Path) for p in _missing_link_sidecars(recipe, ADMIN))
