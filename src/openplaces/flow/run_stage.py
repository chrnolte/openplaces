"""
Run one (stage, recipe, admin unit) pipeline job in a fresh process.

The thin CLI behind each Snakemake rule::

    python -m openplaces.flow.run_stage <stage> <recipe_id> [<admin_id>]

Dispatches to the stage entrypoint with reprocess=False by default, so
a re-dispatched already-complete job no-ops in seconds; pass
--reprocess to force a rebuild of exactly this job (the orchestrator
decides which jobs to re-dispatch, so the flag stays per-job). Sets
OPENPLACES_ORCHESTRATED so receipt-based skips are voided and the
physical output file the orchestrator expects is always produced.

'deliver' is the exception. It is not a recipe stage but the terminal
bundling job (openplaces.io.delivery), it takes the region as a single
admin unit rather than a list, and it has no reprocess of its own: the
bundle is always rewritten from whatever the member files currently
hold.

'validate' is the other exception, and takes the region the same way. It
executes, headless, every validation notebook the recipe declares for
that region (`jupyter nbconvert --execute`), passing the recipe and
region through the OPENPLACES_NOTEBOOK_ARGS environment variable that
each notebook's test-arguments cell reads. It then regenerates the docs
validation tables and writes a manifest of what ran into the region's
accuracies/ folder. Executed copies of the notebooks go to the cache's
_logs tree, never beside the notebooks: their outputs can show
row-level reference data.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path

STAGE_MODULES = {
    'ingest': 'openplaces.io.ingester',
    'harmonize': 'openplaces.io.harmonizer',
    'enrich': 'openplaces.io.enricher',
    'curate': 'openplaces.io.curator',
}

# Handled by their own branches in main(), not the STAGE_MODULES
# dispatch.
DELIVER = 'deliver'
VALIDATE = 'validate'

# Read by each validation notebook's test-arguments cell in place of its
# interactive test string.
NOTEBOOK_ARGS_VARIABLE = 'OPENPLACES_NOTEBOOK_ARGS'


def notebook_arguments(recipe_id: str, region: str, verbose: bool = False) -> str:
    """The argument string a validation notebook is executed with.

    Parameters
    ----------
    recipe_id : str
        Curate recipe being validated.
    region : str
        Delivery region being scored.
    verbose : bool, optional
        Append --verbose.

    Returns
    -------
    str
        Space-separated arguments, as the notebooks split them.
    """
    arguments = f'--recipe_id {recipe_id} --region {region}'
    return f'{arguments} --verbose' if verbose else arguments


def run_validation(
    recipe_id: str,
    region: str,
    verbose: bool = False,
    *,
    log_dir=None,
    manifest_path=None,
) -> Path:
    """Execute a region's validation notebooks, then regenerate the docs tables.

    Every notebook runs even when an earlier one fails, so one broken
    reference cannot hide the others' results; the docs generator runs
    after all of them.

    Parameters
    ----------
    recipe_id : str
        Curate recipe being validated.
    region : str
        Delivery region being scored.
    verbose : bool, optional
        Pass --verbose to the notebooks.
    log_dir : str or pathlib.Path, optional
        Where executed notebook copies go. Default: the cache's
        `_logs/validate/{recipe_id}/{region}`.
    manifest_path : str or pathlib.Path, optional
        Where the manifest goes. Default:
        `flow.dag.validation_manifest_path`, the job's declared output.

    Returns
    -------
    pathlib.Path
        The manifest written into the region's accuracies/ folder.

    Raises
    ------
    SystemExit
        After the manifest is written, when a notebook or the generator
        failed, so the orchestrator records the job as failed.
    """
    from openplaces.config import cfg
    from openplaces.flow.dag import validation_manifest_path
    from openplaces.io.curator.validation import validation_notebooks
    from openplaces.recipe import get_recipe_by_id

    started = datetime.now(UTC).isoformat(timespec='seconds')
    repo_root = Path(cfg.code_root)
    if log_dir is None:
        log_dir = Path(cfg.get_dir('cache')) / '_logs' / 'validate' / recipe_id / region
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        NOTEBOOK_ARGS_VARIABLE: notebook_arguments(recipe_id, region, verbose),
    }

    runs = []
    for notebook in validation_notebooks(get_recipe_by_id(recipe_id), region):
        path = repo_root / notebook
        command = [
            sys.executable,
            '-m',
            'jupyter',
            'nbconvert',
            '--to',
            'notebook',
            '--execute',
            str(path),
            '--output-dir',
            str(log_dir),
            '--ExecutePreprocessor.timeout=-1',
        ]
        print(f'validate: executing {notebook}', flush=True)
        result = subprocess.run(command, cwd=path.parent, env=env, check=False)
        runs.append({'notebook': notebook, 'returncode': result.returncode})

    generator = repo_root / 'docs' / '_ext' / 'generate_validation_tables.py'
    generated = None
    if generator.exists():
        generated = subprocess.run(
            [sys.executable, str(generator)], cwd=repo_root, check=False
        ).returncode

    manifest = Path(manifest_path or validation_manifest_path(recipe_id, region))
    manifest.parent.mkdir(parents=True, exist_ok=True)
    record = {
        'recipe_id': recipe_id,
        'region': region,
        'started_at': started,
        'finished_at': datetime.now(UTC).isoformat(timespec='seconds'),
        'notebooks': runs,
        'docs_generator_returncode': generated,
    }
    manifest.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')

    failed = [run['notebook'] for run in runs if run['returncode']]
    if failed or generated:
        raise SystemExit(
            f'validate {recipe_id} {region}: failed notebooks {failed}, '
            f'docs generator exit {generated}; see {log_dir}'
        )
    return manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog='python -m openplaces.flow.run_stage',
        description='Run one (stage, recipe, admin unit) pipeline job.',
    )
    parser.add_argument('stage', choices=sorted([*STAGE_MODULES, DELIVER, VALIDATE]))
    parser.add_argument('recipe_id')
    parser.add_argument(
        'admin_id',
        nargs='?',
        default=None,
        help=(
            'Admin unit to process; for deliver and validate, the region '
            'the bundle covers'
        ),
    )
    parser.add_argument(
        '--entity-recipe-id',
        default=None,
        help='Explicit harmonized entity recipe for enrich jobs',
    )
    parser.add_argument(
        '--reprocess',
        action='store_true',
        help='Re-run this job even if its output already exists',
    )
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument(
        '--no-orchestrated',
        action='store_true',
        help='Do not set OPENPLACES_ORCHESTRATED (honor receipt skips)',
    )
    args = parser.parse_args(argv)

    if not args.no_orchestrated:
        os.environ['OPENPLACES_ORCHESTRATED'] = '1'

    if args.stage == DELIVER:
        from openplaces.io.delivery import export_delivery

        # `region=`, not the second positional. This argument is the
        # region id (see the module docstring and --admin-id's help), and
        # `export_delivery`'s second positional is `admin_id`, so passing
        # it through positionally left `region` unset -- which a recipe
        # declaring more than one region refuses to guess at, failing
        # every multi-region delivery at job time.
        export_delivery(args.recipe_id, region=args.admin_id, verbose=args.verbose)
        return

    if args.stage == VALIDATE:
        run_validation(args.recipe_id, args.admin_id, verbose=args.verbose)
        return

    admin_ids = [args.admin_id] if args.admin_id else None
    module = import_module(STAGE_MODULES[args.stage])
    stage_fn = getattr(module, args.stage)

    kwargs = {
        'admin_ids': admin_ids,
        'reprocess': args.reprocess,
        'verbose': args.verbose,
    }
    if args.stage == 'enrich' and args.entity_recipe_id:
        kwargs['entity_recipe_id'] = args.entity_recipe_id
    stage_fn(args.recipe_id, **kwargs)


if __name__ == '__main__':
    main()
