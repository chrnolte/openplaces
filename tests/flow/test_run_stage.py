"""Tests for the run_stage CLI dispatcher."""

import pytest

import openplaces.io.harmonizer as harmonizer_mod
from openplaces.flow import run_stage


@pytest.fixture
def no_orchestrated_env(monkeypatch):
    monkeypatch.delenv('OPENPLACES_ORCHESTRATED', raising=False)


def test_dispatch_and_orchestrated_env(monkeypatch, no_orchestrated_env):
    import os

    calls = {}

    def _fake_harmonize(recipe_id, **kwargs):
        calls['recipe_id'] = recipe_id
        calls['kwargs'] = kwargs
        calls['env'] = os.environ.get('OPENPLACES_ORCHESTRATED')

    monkeypatch.setattr(harmonizer_mod, 'harmonize', _fake_harmonize)
    run_stage.main(['harmonize', 'US_footprint-spine-2026', 'US-NC-BR', '--verbose'])

    assert calls['recipe_id'] == 'US_footprint-spine-2026'
    assert calls['kwargs'] == {
        'admin_ids': ['US-NC-BR'],
        'reprocess': False,
        'verbose': True,
    }
    assert calls['env'] == '1'


def test_no_orchestrated_flag(monkeypatch, no_orchestrated_env):
    import os

    monkeypatch.setattr(harmonizer_mod, 'harmonize', lambda *a, **k: None)
    run_stage.main(
        ['harmonize', 'US_footprint-spine-2026', 'US-NC-BR', '--no-orchestrated']
    )
    assert os.environ.get('OPENPLACES_ORCHESTRATED') is None


def test_enrich_forwards_entity_recipe(monkeypatch, no_orchestrated_env):
    import openplaces.io.enricher as enricher_mod

    calls = {}
    monkeypatch.setattr(
        enricher_mod,
        'enrich',
        lambda recipe_id, **kwargs: calls.update(recipe_id=recipe_id, **kwargs),
    )
    run_stage.main(
        [
            'enrich',
            'US_footprint_built-n-stories-brails-2026',
            'US-NC-BR',
            '--entity-recipe-id',
            'US_footprint-spine-2026',
        ]
    )
    assert calls['entity_recipe_id'] == 'US_footprint-spine-2026'


def test_unknown_stage_rejected():
    with pytest.raises(SystemExit):
        run_stage.main(['transmogrify', 'US_footprint-spine-2026'])


def test_deliver_dispatch(monkeypatch, no_orchestrated_env):
    """deliver takes the region, and forwards it as `region=`.

    Not as the second positional, which is `export_delivery`'s
    `admin_id`: that left `region` unset, and a recipe declaring more
    than one region refuses to guess which bundle is meant, so every
    multi-region delivery failed at job time.
    """
    import os

    import openplaces.io.delivery as delivery_mod

    calls = {}

    def _fake_export(recipe, admin_id=None, **kwargs):
        calls.update(recipe=recipe, admin_id=admin_id, kwargs=kwargs)
        calls['env'] = os.environ.get('OPENPLACES_ORCHESTRATED')

    monkeypatch.setattr(delivery_mod, 'export_delivery', _fake_export)
    run_stage.main(
        ['deliver', 'US_footprint-cheer-2026', 'cheer-eastern-nc', '--verbose']
    )

    assert calls['recipe'] == 'US_footprint-cheer-2026'
    assert calls['admin_id'] is None
    assert calls['kwargs'] == {'region': 'cheer-eastern-nc', 'verbose': True}
    assert calls['env'] == '1'


def test_deliver_without_admin_id_leaves_it_to_the_recipe(
    monkeypatch, no_orchestrated_env
):
    import openplaces.io.delivery as delivery_mod

    calls = {}
    monkeypatch.setattr(
        delivery_mod,
        'export_delivery',
        lambda recipe, admin_id=None, **kw: calls.update(admin_id=admin_id),
    )
    run_stage.main(['deliver', 'US_footprint-cheer-2026'])

    assert calls['admin_id'] is None
