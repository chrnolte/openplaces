"""
Cluster submission with the inspection gate.

`dry_run` and `deploy` are importable pieces (used by the
deploy_pipeline notebook); `main` composes them into the CLI, which
refuses to submit until a dry run has been produced and shown. The
dry-run output is stored next to the SGE logs for the post-hoc record::

    python -m openplaces.flow.submit --config recipe=US_footprint-cheer-2026 \\
        admin_ids=US-NC-BR,US-NC-AR

With `--yes`, a successful dry run stored in the last
`DRY_RUN_REUSE_HOURS` for the same profile, profile contents, config
and targets stands in for a new one, so a run inspected beforehand
with `--dry-run-only` does not build the job graph twice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from openplaces.config import cfg

# How old a stored dry run may be and still stand in for a new one
# under --yes. Building the graph is the slow part of a large run
# (several minutes for a 267-unit wave), and an unattended submission
# used to build it three times: a dry run to inspect, the gate's, and
# the run's. The window is short because the graph depends on the
# files on disk, which a run started a day later may find changed.
DRY_RUN_REUSE_HOURS = 12


def _snakefile_args(snakefile=None) -> list[str]:
    path = Path(snakefile) if snakefile else cfg.code_root / 'workflow' / 'Snakefile'
    return ['--snakefile', str(path)]


def _config_args(config) -> list[str]:
    """Snakemake --config arguments from a dict or KEY=VALUE strings."""
    if not config:
        return []
    if isinstance(config, dict):
        pairs = [
            f'{key}={",".join(value) if isinstance(value, list) else value}'
            for key, value in config.items()
            if value is not None
        ]
    else:
        pairs = list(config)
    return ['--config', *pairs] if pairs else []


def _run_key(profile, config, targets, snakefile, extra_args) -> str:
    """Short hash of what decides a dry run's job list.

    The profile's own files are hashed, not only its path, because a
    wave's profile is rewritten in place (its forced rules).
    """
    profile_text = ''
    if profile:
        path = Path(_profile_args(profile)[1])
        profile_text = ''.join(
            p.read_text(encoding='utf-8', errors='replace')
            for p in sorted(path.glob('*.yaml'))
        )
    payload = json.dumps(
        [
            profile_text,
            _config_args(config),
            list(targets),
            str(snakefile or ''),
            list(extra_args),
        ]
    )
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]


def stored_dry_run(key: str, max_age_hours: float | None = None) -> Path | None:
    """The newest successful stored dry run for *key*, if recent enough.

    Parameters
    ----------
    key : str
        The run key `dry_run` stored it under.
    max_age_hours : float, optional
        Older dry runs are ignored; `DRY_RUN_REUSE_HOURS` by default.

    Returns
    -------
    Path or None
        The stored output, None when there is none.
    """
    log_dir = Path(cfg.get_dir('logs')) / 'sge'
    if max_age_hours is None:
        max_age_hours = DRY_RUN_REUSE_HOURS
    cutoff = time.time() - max_age_hours * 3600
    for path in sorted(log_dir.glob(f'dryrun_*_{key}_rc0.txt'), reverse=True):
        if path.stat().st_mtime >= cutoff:
            return path
    return None


def _snakemake_command(*args) -> list[str]:
    # sys.executable -m guarantees the notebook/CLI interpreter and its
    # environment (incl. PYTHONPATH) are what snakemake and its jobs see
    return [sys.executable, '-m', 'snakemake', *args]


def _profile_args(profile) -> list[str]:
    path = Path(profile)
    if not path.is_absolute():
        path = cfg.code_root / path
    return ['--profile', str(path)]


def dry_run(
    config=None,
    targets=(),
    snakefile=None,
    extra_args=(),
    verbose=True,
    profile=None,
) -> tuple[int, str, Path]:
    """Produce, print, and store a Snakemake dry run (the inspection gate).

    Parameters
    ----------
    config : dict or list of str, optional
        Workflow config (e.g. ``{'recipe': ..., 'admin_ids': [...]}``
        or ``['recipe=...', 'admin_ids=a,b']``).
    targets : tuple of str
        Optional Snakemake targets.
    snakefile : str or Path, optional
        Defaults to ``<code_root>/workflow/Snakefile``.
    extra_args : tuple of str
        Additional snakemake arguments (e.g. ``('--forcerun', 'rule')``).
    verbose : bool
        Print the dry-run output.
    profile : str or Path, optional
        Snakemake profile directory, as in `deploy`. Pass the profile
        the run will use: its config file and settings (forced reruns,
        default exclusions) decide which jobs the dry run lists.

    Returns
    -------
    (returncode, output, stored_path)
        The dry run's exit code, its combined stdout+stderr, and the path
        it was stored at (under the logs directory, for the post-hoc
        record). The file name carries the run key and the exit code,
        which is how `stored_dry_run` finds it again.
    """
    command = _snakemake_command(
        '-n',
        *(_profile_args(profile) if profile else []),
        *_snakefile_args(snakefile),
        *_config_args(config),
        *extra_args,
        *targets,
    )
    result = subprocess.run(command, capture_output=True, text=True, cwd=cfg.code_root)
    output = result.stdout + result.stderr
    if verbose:
        print(output)

    log_dir = Path(cfg.get_dir('logs')) / 'sge'
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    key = _run_key(profile, config, targets, snakefile, extra_args)
    stored_path = log_dir / f'dryrun_{stamp}_{key}_rc{result.returncode}.txt'
    stored_path.write_text(output, encoding='utf-8')
    if verbose:
        print(f'Dry run stored: {stored_path}')
    return result.returncode, output, stored_path


def deploy(
    profile=None,
    config=None,
    targets=(),
    cores=None,
    snakefile=None,
    extra_args=(),
) -> int:
    """Run the workflow: on the cluster (profile) or locally (cores).

    Streams Snakemake's output to the console and returns its exit code.
    Callers are expected to have produced a dry run first (`dry_run`);
    the CLI (`main`) enforces that.

    Parameters
    ----------
    profile : str or Path, optional
        Snakemake profile directory (e.g. 'workflow/profiles/scc');
        resolved against the code root when relative. Mutually exclusive
        with local execution.
    config, targets, snakefile, extra_args
        As in `dry_run`.
    cores : int, optional
        Local execution with this many cores (default 4) when no profile
        is given.
    """
    if profile:
        mode_args = _profile_args(profile)
    else:
        # The profiles set these; a run without one needs them as
        # flags, or a failed batched ingest deletes every unit's file.
        mode_args = [
            '--cores',
            str(cores or 4),
            '--keep-incomplete',
            '--rerun-incomplete',
        ]
    command = _snakemake_command(
        *mode_args,
        *_snakefile_args(snakefile),
        *_config_args(config),
        *extra_args,
        *targets,
    )
    result = subprocess.run(command, cwd=cfg.code_root)
    return result.returncode


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog='python -m openplaces.flow.submit',
        description='Submit the openplaces workflow to the cluster '
        '(dry-run inspection gate enforced).',
    )
    parser.add_argument(
        '--profile',
        default='workflow/profiles/scc',
        help='Snakemake profile directory',
    )
    parser.add_argument(
        '--config',
        nargs='*',
        metavar='KEY=VALUE',
        help='Workflow config (e.g. recipe=US_footprint-cheer-2026 '
        'admin_ids=US-NC-BR,US-NC-AR)',
    )
    parser.add_argument(
        '--yes',
        action='store_true',
        help='Submit without the interactive confirmation prompt',
    )
    parser.add_argument(
        '--dry-run-only',
        action='store_true',
        help='Produce and store the dry run, submit nothing; a later --yes '
        f'run with the same settings reuses it for {DRY_RUN_REUSE_HOURS} h',
    )
    parser.add_argument('targets', nargs='*', help='Optional Snakemake targets')
    args = parser.parse_args(argv)
    if args.dry_run_only:
        returncode, _, _ = dry_run(
            config=args.config, targets=tuple(args.targets), profile=args.profile
        )
        sys.exit(returncode)

    # Inspection gate: produce and store the dry run before any
    # submission, unless --yes finds a recent one for the same run.
    targets = tuple(args.targets)
    key = _run_key(args.profile, args.config, targets, None, ())
    reused = stored_dry_run(key) if args.yes else None
    if reused is not None:
        print(f'Dry run reused: {reused}')
    else:
        returncode, _, _ = dry_run(
            config=args.config, targets=targets, profile=args.profile
        )
        if returncode != 0:
            sys.exit('Dry run failed; nothing submitted.')

    if not args.yes:
        answer = input('Submit these jobs to the cluster? [y/N] ').strip().lower()
        if answer not in ('y', 'yes'):
            sys.exit('Submission cancelled.')

    sys.exit(deploy(profile=args.profile, config=args.config, targets=targets))


if __name__ == '__main__':
    main()
