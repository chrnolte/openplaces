"""
Cluster submission wrapper with the inspection gate.

Refuses to submit until a dry run (snakemake -n) has been
produced and shown; the dry-run output is stored next to the SGE logs
for the post-hoc record::

    python -m openplaces.flow.submit --config recipe=US_footprint-cheer-2026 \\
        admin_ids=US-NC-BS,US-NC-CE
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from openplaces.config import cfg


def _snakemake_args(args) -> list[str]:
    passthrough = []
    if args.config:
        passthrough += ['--config', *args.config]
    passthrough += args.targets
    return passthrough


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
        'admin_ids=US-NC-BS,US-NC-CE)',
    )
    parser.add_argument(
        '--yes',
        action='store_true',
        help='Submit without the interactive confirmation prompt',
    )
    parser.add_argument('targets', nargs='*', help='Optional Snakemake targets')
    args = parser.parse_args(argv)

    snakefile = ['--snakefile', 'workflow/Snakefile']
    passthrough = _snakemake_args(args)

    # Inspection gate: produce and store the dry run before any submission
    dry_run = subprocess.run(
        ['snakemake', '-n', *snakefile, *passthrough],
        capture_output=True,
        text=True,
    )
    print(dry_run.stdout)
    print(dry_run.stderr, file=sys.stderr)

    log_dir = Path(cfg.get_dir('logs')) / 'sge'
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    dry_run_path = log_dir / f'dryrun_{stamp}.txt'
    dry_run_path.write_text(dry_run.stdout + dry_run.stderr, encoding='utf-8')
    print(f'Dry run stored: {dry_run_path}')

    if dry_run.returncode != 0:
        sys.exit('Dry run failed; nothing submitted.')

    if not args.yes:
        answer = input('Submit these jobs to the cluster? [y/N] ').strip().lower()
        if answer not in ('y', 'yes'):
            sys.exit('Submission cancelled.')

    result = subprocess.run(
        ['snakemake', '--profile', args.profile, *snakefile, *passthrough]
    )
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
