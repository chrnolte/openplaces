"""Tests for the retention vocabulary and its layered resolution."""

import pytest

from openplaces.config import cfg
from openplaces.core.constants import (
    NEVER_DELETE,
    RETENTION_CLASSES,
    STANDARD_DIRS,
)
from openplaces.recipe import get_recipe_by_id, get_recipe_id, get_recipe_retention


def test_standard_dirs_have_valid_retention():
    for name, info in STANDARD_DIRS.items():
        assert info.get('retention') in RETENTION_CLASSES, name


def test_never_delete_buckets_exist_and_are_keep():
    for name in NEVER_DELETE:
        assert name in STANDARD_DIRS
        assert STANDARD_DIRS[name]['retention'] == 'keep'


def test_retention_for_bucket_defaults():
    assert cfg.retention_for('cache') == 'until_consumed'
    assert cfg.retention_for('heap') == 'transient'
    assert cfg.retention_for('core') == 'keep'


def test_retention_for_never_delete_floor():
    # Even explicit recipe-level overrides cannot mark protected buckets
    assert cfg.retention_for('share', recipe_retention='until_consumed') == 'keep'
    assert cfg.retention_for('raw', recipe_retention='transient') == 'keep'


def test_retention_for_recipe_override_order(monkeypatch):
    retention = {
        'cache': 'keep',
        'recipes': {'US_test-recipe-2026': 'until_consumed'},
        'cleanup': {},
    }
    monkeypatch.setitem(cfg.config, 'retention', retention)
    # Bucket override beats the STANDARD_DIRS default
    assert cfg.retention_for('cache') == 'keep'
    # save_to.retention beats the bucket override
    assert cfg.retention_for('cache', recipe_retention='transient') == 'transient'
    # Per-recipe config override beats save_to.retention
    assert (
        cfg.retention_for(
            'cache',
            recipe_id='US_test-recipe-2026',
            recipe_retention='transient',
        )
        == 'until_consumed'
    )


def test_image_recipe_retention_resolves():
    assert get_recipe_retention('image-googlestreetview-2026') == 'until_consumed'


def test_invalid_save_to_retention_raises(tmp_path):
    from openplaces.recipe import get_recipe_dict

    recipe_file = tmp_path / 'US_parcel-test-2026.yaml'
    recipe_file.write_text(
        'entity:\n'
        '  entity_type: parcel\n'
        '  source:\n'
        '    source_id: test\n'
        '  version: "2026"\n'
        'admin_id: US\n'
        'save_to:\n'
        '  data_dir: cache\n'
        '  retention: forever\n',
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='retention'):
        get_recipe_dict(recipe_file, 'US')


def test_get_recipe_id_round_trip():
    recipe = get_recipe_by_id('image-googlestreetview-2026')
    assert get_recipe_id(recipe) == 'image-googlestreetview-2026'
    assert get_recipe_id('US_parcel-massgis-2025') == 'US_parcel-massgis-2025'
