"""Importing openplaces must never block on an interactive prompt.

`config.get_config()` runs at import time. When it defaulted to
``interactive=True``, a machine with no user config -- CI, a container, a
cluster job, a contributor's first `pytest` -- hit `input()` during
`import openplaces` and either hung or died with "reading from stdin while
output is captured".

The docstring always said the default should be False and that only CLI setup
should prompt; the signature disagreed. These tests pin the contract.
"""

import inspect
import sys

from openplaces.config import OpenPlacesConfig, get_config


def test_get_config_does_not_prompt_by_default():
    """The documented contract: imports are non-interactive."""
    default = inspect.signature(get_config).parameters['interactive'].default

    assert default is False


def test_setup_is_skipped_without_a_terminal(monkeypatch, tmp_path):
    """Even an explicit interactive=True must not block a headless run."""
    called = []

    def _fail_if_called(self):
        called.append(True)

    monkeypatch.setattr(OpenPlacesConfig, '_interactive_setup', _fail_if_called)
    monkeypatch.setattr(
        OpenPlacesConfig,
        '_get_user_config_path',
        lambda self: tmp_path / 'absent' / 'config.yaml',
    )

    class _NoTty:
        def isatty(self):
            return False

    monkeypatch.setattr(sys, 'stdin', _NoTty())

    OpenPlacesConfig(interactive=True)

    assert called == [], 'interactive setup ran without a terminal'


def test_config_resolves_directories_without_a_user_config(monkeypatch, tmp_path):
    """Falling back to defaults must still produce usable directories."""
    monkeypatch.setattr(
        OpenPlacesConfig,
        '_get_user_config_path',
        lambda self: tmp_path / 'absent' / 'config.yaml',
    )

    config = OpenPlacesConfig(interactive=False)

    assert config.data_root is not None
    assert config.core_dir.is_absolute()
