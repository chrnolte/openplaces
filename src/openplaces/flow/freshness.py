"""Find built outputs that are older than what they were built from.

The check itself (`out_of_date`) lives in `openplaces.io.delivery.freshness`,
below the orchestrator, so that `export_delivery` can refuse a stale
bundle; this module builds the graph it walks and is its command line.
`delivery_graph` is the graph `export_delivery` expects for one region.

The check honors the run's excluded recipes (`exclude_recipe_ids`) and
held ones (`core.held_recipes`): what they wrote, stale or missing,
never marks a consumer stale. `RecipeDAG` does that itself: it gives
an excluded or held recipe no job, and also leaves out an enrich job
whose source is excluded or held, dropping its output from its
consumers' inputs (`flow.dag.adds_left_out_source`). Without that, the
excluded placeslab lane's link tables, older than the parcel spines,
marked Fairfax, Arlington and Alexandria stale in washington-dc
(2026-10-24). A harmonize job that reads an excluded source among
others is kept, since its other inputs still decide whether it is
current.

Usage::

    python -m openplaces.flow.freshness RECIPE_ID ADMIN_ID [ADMIN_ID ...]
        [--exclude-recipe-ids ID,ID]
    python -m openplaces.flow.freshness RECIPE_ID --region REGION_ID

Without ``--exclude-recipe-ids`` the run's exclusions are those of
`default_exclude_recipe_ids`; pass an empty value to exclude nothing.
The exit status is 1 when any job is out of date.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

from openplaces.config import cfg
from openplaces.core.held_recipes import EXCLUDE_ENV, run_excluded_recipe_ids
from openplaces.flow.dag import RecipeDAG
from openplaces.io.delivery.freshness import COLUMNS as COLUMNS
from openplaces.io.delivery.freshness import out_of_date as out_of_date


def profile_defaults_path() -> Path:
    """The local profile's config defaults, which list its exclusions."""
    return Path(cfg.code_root) / 'workflow' / 'profiles' / 'local' / 'defaults.yaml'


def default_exclude_recipe_ids() -> set[str]:
    """The recipe ids a run excludes when the caller names none.

    Inside a job, the ids the workflow passed it (`EXCLUDE_ENV`, set by
    `flow.run_stage`). Elsewhere, those the local profile excludes by
    default (`workflow/profiles/local/defaults.yaml`), which is what a
    run started without `--config exclude_recipe_ids=` builds with.

    Returns
    -------
    set of str
        Empty when neither names any.
    """
    if EXCLUDE_ENV in os.environ:
        return set(run_excluded_recipe_ids())
    try:
        defaults = yaml.safe_load(profile_defaults_path().read_text()) or {}
    except OSError:
        return set()
    return {str(r) for r in defaults.get('exclude_recipe_ids') or [] if r}


def parse_exclude(value) -> set[str] | None:
    """A comma-separated `--exclude-recipe-ids` value, None when absent."""
    if value is None:
        return None
    return {r for r in value.split(',') if r}


def delivery_graph(recipe_id, region=None, exclude_recipe_ids=None) -> RecipeDAG:
    """The dependency graph of the curated files a delivery pools.

    Parameters
    ----------
    recipe_id : str
        The curation recipe being delivered.
    region : str, optional
        The declared region shipped, as for `export_delivery`.
    exclude_recipe_ids : set of str, optional
        Recipes the run pruned, as for `RecipeDAG`. None means
        `default_exclude_recipe_ids()`; an empty set excludes nothing.

    Returns
    -------
    RecipeDAG
        The graph over the region's members, without delivery or
        validation jobs, to pass
        to `export_delivery(graph=...)`.
    """
    from openplaces.io.delivery import delivery_members

    if exclude_recipe_ids is None:
        exclude_recipe_ids = default_exclude_recipe_ids()
    return RecipeDAG(
        recipe_id,
        admin_ids=delivery_members(recipe_id, region=region),
        exclude_recipe_ids=set(exclude_recipe_ids),
        deliver=False,
        validate=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('recipe_id')
    parser.add_argument('admin_ids', nargs='*')
    parser.add_argument('--region', help='check every member of a region')
    parser.add_argument(
        '--exclude-recipe-ids',
        default=None,
        help='comma-separated recipe ids the run excluded (default: the profile)',
    )
    args = parser.parse_args(argv)
    admin_ids = list(args.admin_ids)
    if args.region:
        import openplaces as op

        admin_ids += op.get_region_admin_ids(args.region)
    if not admin_ids:
        parser.error('name admin ids or --region')
    exclude = parse_exclude(args.exclude_recipe_ids)
    if exclude is None:
        exclude = default_exclude_recipe_ids()
    dag = RecipeDAG(
        args.recipe_id,
        admin_ids=admin_ids,
        exclude_recipe_ids=exclude,
        deliver=False,
        validate=False,
    )
    stale = out_of_date(dag)
    checked = stale.attrs['checked']
    if stale.empty:
        print(f'{args.recipe_id}: all {checked} built outputs are current')
        return
    with pd.option_context('display.max_rows', None, 'display.width', 200):
        print(stale.to_string(index=False))
    print(f'{len(stale)} of {checked} built job(s) out of date')
    sys.exit(1)


if __name__ == '__main__':
    main()
