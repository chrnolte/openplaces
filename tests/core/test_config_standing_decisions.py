"""Standing decisions accumulate, and a broken config file is not erased.

A person answering [a] at the terms prompt, or resolving a usage-profile
mismatch, records a decision in their own config. Two such answers in one
process used to leave only the second, because the writer rebuilt the
whole mapping from the snapshot taken at import. And a config file that
could not be parsed was treated as an empty one, so the rewrite discarded
everything else it held.
"""

import pytest
import yaml

from openplaces.config import (
    ConfigFileUnreadableError,
    get_terms_consent,
    get_usage_override,
    merge_user_config,
    set_terms_consent,
    set_usage_override,
)


class _StaleCfg:
    """The import-time snapshot, which no later write ever refreshes."""

    def __init__(self, path):
        self.user_config_path = path

    def get(self, key, default=None):
        return default


class _LiveCfg:
    """What reload_config() rebinds `_cfg` to: the file as it stands."""

    def __init__(self, path):
        self.user_config_path = path

    def get(self, key, default=None):
        if not self.user_config_path.exists():
            return default
        content = yaml.safe_load(self.user_config_path.read_text()) or {}
        return content.get(key, default)


@pytest.fixture
def user_config(monkeypatch, tmp_path):
    path = tmp_path / 'config.yaml'
    monkeypatch.setattr('openplaces.config.cfg', _StaleCfg(path))
    monkeypatch.setattr('openplaces.config._cfg', _LiveCfg(path))
    monkeypatch.setattr('openplaces.config.reload_config', lambda *a, **k: None)
    return path


class TestStandingDecisionsAccumulate:
    def test_a_second_accepted_source_does_not_erase_the_first(self, user_config):
        set_terms_consent('portal-a', True)
        set_terms_consent('portal-b', True)

        recorded = yaml.safe_load(user_config.read_text())['consent']['terms']
        assert set(recorded) == {'portal-a', 'portal-b'}
        assert get_terms_consent('portal-a') is True
        assert get_terms_consent('portal-b') is True

    def test_a_second_usage_override_does_not_erase_the_first(self, user_config):
        set_usage_override('source-a', True, reason='judged inapplicable')
        set_usage_override('source-b', True)

        recorded = yaml.safe_load(user_config.read_text())['usage_overrides']
        assert set(recorded) == {'source-a', 'source-b'}
        assert get_usage_override('source-a') is True
        assert get_usage_override('source-b') is True


class TestUnreadableConfigIsNotOverwritten:
    def test_a_parse_error_stops_the_write(self, tmp_path):
        path = tmp_path / 'config.yaml'
        path.write_text('directories:\n  data_root: /data\n\tbad: tab\n')

        with pytest.raises(ConfigFileUnreadableError):
            merge_user_config(path, 'consent', {'terms': {'a': True}})

        assert 'data_root' in path.read_text()

    def test_a_non_mapping_config_stops_the_write(self, tmp_path):
        path = tmp_path / 'config.yaml'
        path.write_text('- just\n- a\n- list\n')

        with pytest.raises(ConfigFileUnreadableError):
            merge_user_config(path, 'consent', {'terms': {'a': True}})

        assert 'just' in path.read_text()

    def test_a_missing_file_is_still_created(self, tmp_path):
        path = tmp_path / 'config.yaml'
        merge_user_config(path, 'consent', {'terms': {'a': True}})
        assert yaml.safe_load(path.read_text())['consent']['terms']['a'] is True
