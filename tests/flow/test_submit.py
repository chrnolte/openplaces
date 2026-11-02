"""Cluster submission: the dry-run gate and the SCC qsub wrapper."""

import shutil
import subprocess
from pathlib import Path

import pytest

from openplaces.flow import submit

QSUB_SUBMIT = (
    Path(__file__).resolve().parents[2]
    / 'workflow'
    / 'profiles'
    / 'scc'
    / 'qsub_submit.sh'
)


def test_dry_run_uses_the_run_profile(monkeypatch, tmp_path):
    seen = {}

    def fake_run(command, **kwargs):
        seen['command'] = command
        return subprocess.CompletedProcess(command, 0, stdout='', stderr='')

    monkeypatch.setattr(submit.subprocess, 'run', fake_run)
    monkeypatch.setattr(submit.cfg, 'get_dir', lambda name: tmp_path)
    submit.dry_run(profile='workflow/profiles/scc', verbose=False)
    command = seen['command']
    assert '-n' in command
    profile = Path(command[command.index('--profile') + 1])
    assert profile.is_absolute() and profile.parts[-2:] == ('profiles', 'scc')


def _submitted(tmp_path, mem_mb, threads, **env):
    """The qsub arguments the wrapper passes for one job."""
    stub = tmp_path / 'qsub'
    stub.write_text('#!/usr/bin/env bash\necho "$@"\n', encoding='utf-8')
    stub.chmod(0o755)
    environ = {
        'PATH': f'{tmp_path}:/usr/bin:/bin',
        'OPENPLACES_SCC_PROJECT': 'example',
        'OPENPLACES_SGE_LOG_DIR': str(tmp_path / 'sge'),
        **env,
    }
    result = subprocess.run(
        ['bash', str(QSUB_SUBMIT), str(mem_mb), str(threads), 'job.sh'],
        capture_output=True,
        text=True,
        env=environ,
        check=True,
    )
    return result.stdout.split()


@pytest.mark.skipif(
    shutil.which('bash') is None or not Path('/usr/bin').exists(),
    reason='needs a POSIX bash',
)
@pytest.mark.parametrize(
    ('mem_mb', 'threads', 'slots', 'env'),
    [
        (2000, 4, '4', {}),  # threads decide a small job
        (26000, 1, '4', {}),  # 26 GB is four 8 GB slots
        (267000, 1, '16', {}),  # an extrapolated estimate is capped
        (267000, 1, '28', {'OPENPLACES_SCC_MAX_SLOTS': '28'}),
    ],
)
def test_qsub_requests_memory_as_slots(tmp_path, mem_mb, threads, slots, env):
    args = _submitted(tmp_path, mem_mb, threads, **env)
    assert args[args.index('omp') + 1] == slots
    assert 'mem_per_core=8G' in args
    assert args[-1] == 'job.sh'
