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
"""

from __future__ import annotations

import argparse
import os
from importlib import import_module

STAGE_MODULES = {
    'ingest': 'openplaces.io.ingester',
    'harmonize': 'openplaces.io.harmonizer',
    'enrich': 'openplaces.io.enricher',
    'curate': 'openplaces.io.curator',
}

# Handled by its own branch in main(), not the STAGE_MODULES dispatch.
DELIVER = 'deliver'


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog='python -m openplaces.flow.run_stage',
        description='Run one (stage, recipe, admin unit) pipeline job.',
    )
    parser.add_argument('stage', choices=sorted([*STAGE_MODULES, DELIVER]))
    parser.add_argument('recipe_id')
    parser.add_argument(
        'admin_id',
        nargs='?',
        default=None,
        help='Admin unit to process; for deliver, the region the bundle covers',
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

        export_delivery(args.recipe_id, args.admin_id, verbose=args.verbose)
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
