"""The heap and the logs follow the cache unless placed elsewhere.

Moving the cache off a synced drive used to leave the heap and the logs
behind, because their defaults were relative to data_root: the cache
moved, and 180 GB of unzipped downloads kept syncing. They now resolve
against the cache, and a config written under the old rule still lands
in the same place.
"""

import pytest
import yaml

from openplaces.config import OpenPlacesConfig


def _config(monkeypatch, tmp_path, directories):
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.dump({'directories': directories}), encoding='utf-8')
    monkeypatch.setattr(
        OpenPlacesConfig, '_get_user_config_path', lambda self: config_path
    )
    monkeypatch.setattr(OpenPlacesConfig, '_find_project_config', lambda self: None)
    return OpenPlacesConfig(interactive=False)


def test_defaults_sit_inside_the_cache(monkeypatch, tmp_path):
    config = _config(monkeypatch, tmp_path, {'data_root': str(tmp_path / 'store')})

    cache = (tmp_path / 'store' / 'data' / 'cache').resolve()
    assert config.cache_dir == cache
    assert config.heap_dir == cache / '_heap'
    assert config.logs_dir == cache / '_logs'


def test_moving_the_cache_moves_heap_and_logs(monkeypatch, tmp_path):
    elsewhere = tmp_path / 'local' / 'cache'
    config = _config(
        monkeypatch,
        tmp_path,
        {'data_root': str(tmp_path / 'store'), 'cache': str(elsewhere)},
    )

    assert config.heap_dir == elsewhere.resolve() / '_heap'
    assert config.logs_dir == elsewhere.resolve() / '_logs'


@pytest.mark.parametrize('legacy', ['data/cache/_heap', 'data/someone/cache/_heap'])
def test_a_config_written_under_the_old_rule_follows_the_cache(
    monkeypatch, tmp_path, legacy
):
    """Setup wrote every default into the user's file, the heap's too."""
    elsewhere = tmp_path / 'local' / 'cache'
    config = _config(
        monkeypatch,
        tmp_path,
        {
            'data_root': str(tmp_path / 'store'),
            'cache': str(elsewhere),
            'heap': legacy,
        },
    )

    assert config.heap_dir == elsewhere.resolve() / '_heap'


def test_an_absolute_heap_stays_where_it_was_put(monkeypatch, tmp_path):
    scratch = tmp_path / 'scratch'
    config = _config(
        monkeypatch,
        tmp_path,
        {'data_root': str(tmp_path / 'store'), 'heap': str(scratch)},
    )

    assert config.heap_dir == scratch.resolve()


def test_a_custom_relative_heap_resolves_under_the_cache(monkeypatch, tmp_path):
    config = _config(
        monkeypatch,
        tmp_path,
        {'data_root': str(tmp_path / 'store'), 'heap': 'unzipped'},
    )

    assert config.heap_dir == config.cache_dir / 'unzipped'
