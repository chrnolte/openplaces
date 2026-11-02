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


def _fake_snakemake(monkeypatch, tmp_path, returncode=0):
    """Record each snakemake command instead of running it."""
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, returncode, stdout='', stderr='')

    monkeypatch.setattr(submit.subprocess, 'run', fake_run)
    monkeypatch.setattr(submit.cfg, 'get_dir', lambda name: tmp_path)
    return calls


def _main(*argv):
    with pytest.raises(SystemExit) as exit_info:
        submit.main(list(argv))
    return exit_info.value.code


def test_yes_reuses_a_stored_dry_run(monkeypatch, tmp_path):
    calls = _fake_snakemake(monkeypatch, tmp_path)
    config = ['--config', 'recipe=example', 'admin_ids=US-XX-AA']
    assert _main('--dry-run-only', *config) == 0
    assert _main('--yes', *config) == 0
    # The dry run once, then the run itself: no second dry run.
    assert ['-n' in c for c in calls] == [True, False]


def test_yes_builds_a_new_dry_run_for_other_settings(monkeypatch, tmp_path):
    calls = _fake_snakemake(monkeypatch, tmp_path)
    assert _main('--dry-run-only', '--config', 'admin_ids=US-XX-AA') == 0
    assert _main('--yes', '--config', 'admin_ids=US-XX-BB') == 0
    assert ['-n' in c for c in calls] == [True, True, False]


def test_a_failed_dry_run_is_not_reused(monkeypatch, tmp_path):
    _fake_snakemake(monkeypatch, tmp_path, returncode=1)
    config = ['--config', 'admin_ids=US-XX-AA']
    assert _main('--dry-run-only', *config) == 1
    calls = _fake_snakemake(monkeypatch, tmp_path)
    assert _main('--yes', *config) == 0
    assert ['-n' in c for c in calls] == [True, False]


def test_a_dry_run_past_the_window_is_not_reused(monkeypatch, tmp_path):
    calls = _fake_snakemake(monkeypatch, tmp_path)
    config = ['--config', 'admin_ids=US-XX-AA']
    _main('--dry-run-only', *config)
    monkeypatch.setattr(submit, 'DRY_RUN_REUSE_HOURS', 0)
    monkeypatch.setattr(
        submit.time, 'time', lambda: (tmp_path / 'sge').stat().st_mtime + 60
    )
    _main('--yes', *config)
    assert ['-n' in c for c in calls] == [True, True, False]


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
