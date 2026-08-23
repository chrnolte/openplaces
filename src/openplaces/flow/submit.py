"""
Cluster submission with the inspection gate.

`dry_run` and `deploy` are importable pieces (used by the
deploy_pipeline notebook); `main` composes them into the CLI, which
refuses to submit until a dry run has been produced and shown. The
dry-run output is stored next to the SGE logs for the post-hoc record::

    python -m openplaces.flow.submit --config recipe=US_footprint-cheer-2026 \\
        admin_ids=US-NC-BR,US-NC-AR
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from openplaces.config import cfg


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


def _snakemake_command(*args) -> list[str]:
    # sys.executable -m guarantees the notebook/CLI interpreter and its
    # environment (incl. PYTHONPATH) are what snakemake and its jobs see
    return [sys.executable, '-m', 'snakemake', *args]


def dry_run(
    config=None,
    targets=(),
    snakefile=None,
    extra_args=(),
    verbose=True,
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

    Returns
    -------
    (returncode, output, stored_path)
        The dry run's exit code, its combined stdout+stderr, and the path
        it was stored at (under the logs directory, for the post-hoc
        record).
    """
    command = _snakemake_command(
        '-n',
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
    stored_path = log_dir / f'dryrun_{stamp}.txt'
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
        profile_path = Path(profile)
        if not profile_path.is_absolute():
            profile_path = cfg.code_root / profile_path
        mode_args = ['--profile', str(profile_path)]
    else:
        mode_args = ['--cores', str(cores or 4)]
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
    parser.add_argument('targets', nargs='*', help='Optional Snakemake targets')
    args = parser.parse_args(argv)

    # Inspection gate: produce and store the dry run before any submission
    returncode, _, _ = dry_run(config=args.config, targets=tuple(args.targets))
    if returncode != 0:
        sys.exit('Dry run failed; nothing submitted.')

    if not args.yes:
        answer = input('Submit these jobs to the cluster? [y/N] ').strip().lower()
        if answer not in ('y', 'yes'):
            sys.exit('Submission cancelled.')

    sys.exit(
        deploy(profile=args.profile, config=args.config, targets=tuple(args.targets))
    )


if __name__ == '__main__':
    main()
