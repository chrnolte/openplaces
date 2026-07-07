"""
Run one (stage, recipe, admin unit) pipeline job in a fresh process.

The thin CLI behind each Snakemake rule::

    python -m openplaces.flow.run_stage <stage> <recipe_id> [<admin_id>]

Dispatches to the stage entrypoint with reprocess=False, so a
re-dispatched already-complete job no-ops in seconds. Sets
OPENPLACES_ORCHESTRATED so receipt-based skips are voided and the
physical output file the orchestrator expects is always produced.
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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog='python -m openplaces.flow.run_stage',
        description='Run one (stage, recipe, admin unit) pipeline job.',
    )
    parser.add_argument('stage', choices=sorted(STAGE_MODULES))
    parser.add_argument('recipe_id')
    parser.add_argument('admin_id', nargs='?', default=None)
    parser.add_argument(
        '--entity-recipe-id',
        default=None,
        help='Explicit harmonized entity recipe for enrich jobs',
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

    admin_ids = [args.admin_id] if args.admin_id else None
    module = import_module(STAGE_MODULES[args.stage])
    stage_fn = getattr(module, args.stage)

    kwargs = {'admin_ids': admin_ids, 'reprocess': False, 'verbose': args.verbose}
    if args.stage == 'enrich' and args.entity_recipe_id:
        kwargs['entity_recipe_id'] = args.entity_recipe_id
    stage_fn(args.recipe_id, **kwargs)


if __name__ == '__main__':
    main()
