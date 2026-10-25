"""Find built outputs that are older than what they were built from.

The check itself (`out_of_date`) lives in `openplaces.io.delivery.freshness`,
below the orchestrator, so that `export_delivery` can refuse a stale
bundle; this module builds the graph it walks and is its command line.
`delivery_graph` is the graph `export_delivery` expects for one region.

The check honors the run's excluded recipes (`exclude_recipe_ids`) and
held ones (`core.held_recipes`): what they wrote, stale or missing,
never marks a consumer stale. The graph alone does not do that.
`RecipeDAG` gives an excluded or held recipe no job, but an enrich
recipe that adds that recipe's evidence is named by the curation
directly, so it keeps its job with the excluded edge dropped. The
placeslab lane is the case: its ingest is excluded by the local
profile, its enrich (`US_parcel_parcel-placeslab-fmv2026`) stayed, and
its link table, older than the parcel spines, marked Fairfax, Arlington
and Alexandria stale in washington-dc (2026-10-24). `freshness_graph`
therefore also leaves out an enrich job whose source is excluded or
held, and drops its output from its consumers' inputs: an enrich recipe
adds one source's evidence, and without the source it has nothing to
refresh. A harmonize job that reads an excluded source among others
is kept, since its other inputs still decide whether it is current.

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
from openplaces.core.held_recipes import EXCLUDE_ENV, is_held, run_excluded_recipe_ids
from openplaces.core.schema import AdminId
from openplaces.flow.dag import RecipeDAG
from openplaces.io.delivery.freshness import COLUMNS as COLUMNS
from openplaces.io.delivery.freshness import out_of_date as out_of_date
from openplaces.recipe import get_recipe_by_id, get_recipe_dependencies


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


def _reads_left_out_source(dag, node) -> bool:
    """Whether an enrich job adds a source the run excluded or holds."""
    if node.stage != 'enrich':
        return False
    try:
        recipe = get_recipe_by_id(node.recipe_id)
        admin = AdminId(node.admin_id) if node.admin_id else None
        # Unpruned, so that the excluded edge is still there to see
        edges = get_recipe_dependencies(recipe, admin_id=admin)
    except Exception:  # noqa: BLE001 - unresolvable, keep the job
        return False
    excluded = set(getattr(dag, 'exclude_recipe_ids', ()) or ())
    for edge in edges:
        upstream_id = edge.upstream_recipe_id
        if not upstream_id:
            continue
        if upstream_id in excluded:
            return True
        try:
            upstream = get_recipe_by_id(upstream_id)
        except Exception:  # noqa: BLE001 - a held file may be absent
            upstream = None
        if is_held(upstream_id, upstream if isinstance(upstream, dict) else None):
            return True
    return False


class FreshnessGraph:
    """A `RecipeDAG` as the freshness check reads it.

    Leaves out every enrich job whose source the run excluded or holds,
    and drops those jobs' outputs from every other job's inputs (see
    the module docstring). Otherwise it answers as the wrapped graph.

    Parameters
    ----------
    dag : RecipeDAG
        The graph to read, built with the run's `exclude_recipe_ids`.
        Nothing is computed until a method is called.
    """

    def __init__(self, dag):
        self.dag = dag
        self._left_out = None

    def __getattr__(self, name):
        return getattr(self.dag, name)

    def _left_out_jobs(self):
        if self._left_out is None:
            jobs, outputs = set(), set()
            for node in self.dag.nodes():
                if not _reads_left_out_source(self.dag, node):
                    continue
                job = (node.stage, node.recipe_id, node.admin_id, node.region)
                jobs.add(job)
                try:
                    outputs.add(self.dag.output_path(*job))
                except Exception:  # noqa: BLE001 - nothing to drop
                    continue
            self._left_out = (jobs, outputs)
        return self._left_out

    @property
    def left_out(self) -> set[tuple]:
        """The (stage, recipe_id, admin_id, region) jobs left out."""
        return self._left_out_jobs()[0]

    def nodes(self):
        jobs, _ = self._left_out_jobs()
        return [
            node
            for node in self.dag.nodes()
            if (node.stage, node.recipe_id, node.admin_id, node.region) not in jobs
        ]

    def output_path(self, *job, **kwargs):
        return self.dag.output_path(*job, **kwargs)

    def input_paths(self, *job, **kwargs):
        _, outputs = self._left_out_jobs()
        return [p for p in self.dag.input_paths(*job, **kwargs) if p not in outputs]


def freshness_graph(dag) -> FreshnessGraph:
    """Wrap a `RecipeDAG` for the freshness check; see `FreshnessGraph`."""
    return FreshnessGraph(dag)


def parse_exclude(value) -> set[str] | None:
    """A comma-separated `--exclude-recipe-ids` value, None when absent."""
    if value is None:
        return None
    return {r for r in value.split(',') if r}


def delivery_graph(recipe_id, region=None, exclude_recipe_ids=None) -> FreshnessGraph:
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
    FreshnessGraph
        The graph over the region's members, without delivery or
        validation jobs or enrich jobs of a left-out source, to pass
        to `export_delivery(graph=...)`.
    """
    from openplaces.io.delivery import delivery_members

    if exclude_recipe_ids is None:
        exclude_recipe_ids = default_exclude_recipe_ids()
    return freshness_graph(
        RecipeDAG(
            recipe_id,
            admin_ids=delivery_members(recipe_id, region=region),
            exclude_recipe_ids=set(exclude_recipe_ids),
            deliver=False,
            validate=False,
        )
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
    stale = out_of_date(freshness_graph(dag))
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
