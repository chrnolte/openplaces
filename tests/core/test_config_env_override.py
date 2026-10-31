"""OPENPLACES_CONFIG points a process at another user config.

A test run of the workflow needs its own data folder. The user config's
directories override a project openplaces.yaml, so the override has to
replace the user config itself.
"""

import yaml

from openplaces.config import OpenPlacesConfig
from openplaces.config.user_config import CONFIG_ENV, user_config_path


def test_the_variable_names_the_config_file(monkeypatch, tmp_path):
    monkeypatch.setenv(CONFIG_ENV, str(tmp_path / 'test.yaml'))

    assert user_config_path() == tmp_path / 'test.yaml'


def test_without_it_the_platform_file_is_used(monkeypatch):
    monkeypatch.delenv(CONFIG_ENV, raising=False)

    assert user_config_path().name == 'config.yaml'


def test_its_directories_are_the_ones_used(monkeypatch, tmp_path):
    config_path = tmp_path / 'test.yaml'
    store = tmp_path / 'store'
    config_path.write_text(
        yaml.dump({'directories': {'data_root': str(store)}}), encoding='utf-8'
    )
    monkeypatch.setenv(CONFIG_ENV, str(config_path))
    monkeypatch.setattr(OpenPlacesConfig, '_find_project_config', lambda self: None)

    config = OpenPlacesConfig(interactive=False)

    assert config.user_config_path == config_path
    assert config.cache_dir == (store / 'data' / 'cache').resolve()
