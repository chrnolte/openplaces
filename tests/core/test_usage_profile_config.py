"""The declared usage profile updates partially and reads completely.

`set_usage_profile` has six independent axes and the cluster case sets
them incrementally across separate job scripts, so a call must never
clobber the axes it does not mention -- the `_UNSET` sentinel, not
None, marks "not passed", because None is itself a meaningful value for
the tri-state `commercial` axis. And because the hierarchical config
loader only deep-merges 'directories' and 'retention', the
`usage_profile` property does its own merge over the defaults so a
partial user config still reports every key.
"""

import copy

import pytest
import yaml

from openplaces.config import (
    OpenPlacesConfig,
    set_usage_profile,
    write_usage_profile,
)


class _StubCfg:
    """Just enough of cfg for set_usage_profile: a path and the merged view."""

    def __init__(self, path):
        self.user_config_path = path

    @property
    def usage_profile(self):
        from openplaces.config import _merge_nested

        stored = {}
        if self.user_config_path.exists():
            content = yaml.safe_load(self.user_config_path.read_text()) or {}
            stored = content.get('usage_profile') or {}
        merged = copy.deepcopy(OpenPlacesConfig.DEFAULTS['usage_profile'])
        return _merge_nested(merged, stored)


@pytest.fixture
def stub_cfg(monkeypatch, tmp_path):
    stub = _StubCfg(tmp_path / 'config.yaml')
    monkeypatch.setattr('openplaces.config.cfg', stub)
    monkeypatch.setattr('openplaces.config.reload_config', lambda *a, **k: None)
    return stub


def test_successive_partial_calls_preserve_untouched_axes(stub_cfg):
    set_usage_profile(commercial=False)
    set_usage_profile(restricted=True)
    profile = set_usage_profile(admin_interests=['US-MA'])

    assert profile['commercial'] is False
    assert profile['environment']['restricted'] is True
    assert profile['environment']['licensed'] is False
    assert profile['admin_interests'] == ['US-MA']


def test_commercial_can_be_returned_to_undeclared(stub_cfg):
    set_usage_profile(commercial=True, offline_only=True)
    profile = set_usage_profile(commercial=None)

    assert profile['commercial'] is None
    assert profile['environment']['offline_only'] is True


def test_admin_interests_replace_wholesale_and_clear_on_empty(stub_cfg):
    set_usage_profile(admin_interests=['US-MA', 'US-NC'])
    assert stub_cfg.usage_profile['admin_interests'] == ['US-MA', 'US-NC']

    profile = set_usage_profile(admin_interests=[])
    assert profile['admin_interests'] == []


def test_property_completes_a_partial_user_config(monkeypatch, tmp_path):
    """A user config declaring one axis still reports every key."""
    config_path = tmp_path / 'config.yaml'
    write_usage_profile(config_path, {'commercial': False})
    monkeypatch.setattr(
        OpenPlacesConfig, '_get_user_config_path', lambda self: config_path
    )

    profile = OpenPlacesConfig(interactive=False).usage_profile

    assert profile['commercial'] is False
    assert set(profile['environment']) == {
        'licensed',
        'restricted',
        'encrypted_at_rest',
        'offline_only',
    }
    assert profile['admin_interests'] == []
