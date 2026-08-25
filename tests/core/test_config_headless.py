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

import pytest
import yaml

from openplaces.config import (
    DataRootNotSetError,
    OpenPlacesConfig,
    get_config,
)


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

    # Refusing to guess a data_root is the other half of the contract:
    # the run must neither block on a prompt nor invent a directory.
    with pytest.raises(DataRootNotSetError):
        OpenPlacesConfig(interactive=True)

    assert called == [], 'interactive setup ran without a terminal'


def test_a_missing_data_root_is_refused_not_guessed(monkeypatch, tmp_path):
    """Where the data lives is the user's decision, not a default.

    This used to fall back to the code directory, which does not fail:
    a fresh install quietly built a second data store inside the
    checkout, which on a synced or version-controlled directory is worse
    than an error.
    """
    monkeypatch.setattr(
        OpenPlacesConfig,
        '_get_user_config_path',
        lambda self: tmp_path / 'absent' / 'config.yaml',
    )

    with pytest.raises(DataRootNotSetError, match='will not invent one'):
        OpenPlacesConfig(interactive=False)


def test_a_configured_data_root_resolves_every_directory(monkeypatch, tmp_path):
    """Once set, the rest of the directories derive from it."""
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(
        yaml.dump({'directories': {'data_root': str(tmp_path / 'store')}}),
        encoding='utf-8',
    )
    monkeypatch.setattr(
        OpenPlacesConfig, '_get_user_config_path', lambda self: config_path
    )

    config = OpenPlacesConfig(interactive=False)

    assert config.data_root == (tmp_path / 'store').resolve()
    assert config.core_dir.is_absolute()


class TestCanPrompt:
    """A prompt needs somewhere to read an answer from.

    The predicate used to be `sys.stdin.isatty()`, which conflates
    "has a terminal" with "someone can answer". Those differ exactly
    where it matters: a Jupyter kernel is interactive and is not a tty,
    and it was the one surface the project designates for interactive
    configuration.
    """

    def test_a_notebook_kernel_can_answer(self, monkeypatch):
        from openplaces import config

        class _Shell:
            pass

        _Shell.__name__ = 'ZMQInteractiveShell'

        class _IPython:
            @staticmethod
            def get_ipython():
                return _Shell()

        monkeypatch.delenv('CI', raising=False)
        monkeypatch.delenv('PYTEST_CURRENT_TEST', raising=False)
        monkeypatch.setitem(sys.modules, 'IPython', _IPython)
        assert config.can_prompt()

    def test_an_unattended_runner_never_prompts(self, monkeypatch):
        from openplaces import config

        class _Shell:
            pass

        _Shell.__name__ = 'ZMQInteractiveShell'

        class _IPython:
            @staticmethod
            def get_ipython():
                return _Shell()

        # Even a kernel is refused when the environment says nobody is
        # watching, so a notebook driven by CI cannot hang the job.
        monkeypatch.setitem(sys.modules, 'IPython', _IPython)
        monkeypatch.setenv('CI', '1')
        assert not config.can_prompt()

    def test_a_closed_stdin_is_not_a_prompt(self, monkeypatch):
        from openplaces import config

        class _Closed:
            closed = True

            def isatty(self):
                raise ValueError('I/O operation on closed file')

        monkeypatch.delenv('CI', raising=False)
        monkeypatch.delenv('PYTEST_CURRENT_TEST', raising=False)
        monkeypatch.delitem(sys.modules, 'IPython', raising=False)
        monkeypatch.setattr(sys, 'stdin', _Closed())
        assert not config.can_prompt()
