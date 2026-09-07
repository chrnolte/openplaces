"""Tests for the subprocess helpers in `openplaces.flow.scripts`."""

import subprocess

import pytest

from openplaces.flow import scripts


class _FakeProcess:
    """Minimal stand-in for a finished `subprocess.Popen`."""

    def __init__(self, command, **kwargs):
        self.command = list(command)
        self.stdout = iter(())
        self.stderr = iter(())
        self.returncode = 0

    def wait(self):
        return self.returncode


def test_run_subprocess_does_not_mutate_the_caller_list(monkeypatch):
    """A reused command list must not accumulate arguments across calls."""
    seen = []
    monkeypatch.setattr(
        subprocess,
        'Popen',
        lambda command, **kwargs: seen.append(_FakeProcess(command)) or seen[-1],
    )
    command = ['python', 'run.py']
    scripts.run_subprocess(command, p={'admin_id': 'US-NC-BRU'})
    scripts.run_subprocess(command, p={'admin_id': 'US-NC-CE'})

    assert command == ['python', 'run.py']
    assert seen[0].command == ['python', '-u', 'run.py', '--admin_id', 'US-NC-BRU']
    assert seen[1].command == ['python', '-u', 'run.py', '--admin_id', 'US-NC-CE']


def test_caller_path_error_names_the_offending_path(tmp_path, monkeypatch):
    """The NotImplementedError reports the path, not a literal placeholder."""
    monkeypatch.setattr(scripts, 'get_caller_path', lambda: tmp_path / 'stray.py')
    with pytest.raises(NotImplementedError) as excinfo:
        scripts.get_caller_path_in_code_directory()
    assert 'stray.py' in str(excinfo.value)
    assert '{caller_path}' not in str(excinfo.value)
