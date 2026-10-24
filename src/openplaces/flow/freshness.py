"""Find built outputs that are older than what they were built from.

A delivery pools curated files that are already on disk; it does not
look at whether those files still reflect their inputs. When a spine is
rebuilt in one run and its consumers are not, the consumer's file stays
on disk, older than the spine, and the next export ships it as if it
were current. Transaction curates linked to a property spine that had
since been fixed reached the valuation lane this way (2026-10).

`out_of_date` reports such jobs from file modification times only, the
same trigger Snakemake reruns on. Unlike `RecipeDAG.plan`, it does not
propagate "upstream scheduled" from a missing output: an absent file
can be a held recipe, an opt-in lane or a consumed intermediate, and
treating every one as a pending rebuild marks a whole county's graph
for rerun and buries the real cases. Staleness does propagate through
existing files, because a consumer of a stale output is itself built
from stale data even when its own inputs are older than it.

Usage::

    python -m openplaces.flow.freshness RECIPE_ID ADMIN_ID [ADMIN_ID ...]
    python -m openplaces.flow.freshness RECIPE_ID --region REGION_ID

The exit status is 1 when any job is out of date.
"""

import argparse
import sys

import pandas as pd

from openplaces.flow.dag import RecipeDAG
from openplaces.io.cleanup import _relative_posix

COLUMNS = ['stage', 'recipe_id', 'admin_id', 'region', 'output', 'reason']


def _mtime(path):
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def out_of_date(dag: RecipeDAG) -> pd.DataFrame:
    """List the jobs whose existing output is older than its inputs.

    Parameters
    ----------
    dag : RecipeDAG
        The graph to check. Only the files of its own jobs are statted;
        nothing is read.

    Returns
    -------
    pd.DataFrame
        One row per out-of-date job, with columns stage, recipe_id,
        admin_id, region, output and reason. The reason names the
        newest input newer than the output, or the stale upstream
        output the job was built from. Empty when every existing output
        is current; attrs['checked'] holds the number of built jobs
        compared, so an empty report over nothing built is visible.
    """
    jobs = []
    by_output = {}
    for node in dag.nodes():
        job = (node.stage, node.recipe_id, node.admin_id, node.region)
        try:
            out = dag.output_path(*job)
        except Exception:  # noqa: BLE001 - unresolvable, nothing to check
            continue
        out_mtime = _mtime(out)
        if out_mtime is None:
            continue
        try:
            inputs = dag.input_paths(*job)
        except Exception:  # noqa: BLE001 - same fallback as plan()
            inputs = []
        jobs.append((job, out, out_mtime, inputs))
        by_output[out] = job

    reasons = {}
    for job, out, out_mtime, inputs in jobs:
        newer = [
            (m, p) for p in inputs if (m := _mtime(p)) is not None and m > out_mtime
        ]
        if newer:
            _, newest = max(newer)
            reasons[out] = f'input newer: {_relative_posix(newest)}'

    # Fixpoint over existing files only: a consumer of a stale output
    # is stale too, whatever its own timestamps say.
    changed = True
    while changed:
        changed = False
        for _job, out, _mtime_out, inputs in jobs:
            if out in reasons:
                continue
            stale_input = next((p for p in inputs if p in reasons), None)
            if stale_input is not None:
                reasons[out] = f'built from stale: {_relative_posix(stale_input)}'
                changed = True

    rows = [
        dict(zip(COLUMNS[:4], job), output=_relative_posix(out), reason=reasons[out])
        for job, out, _m, _i in jobs
        if out in reasons
    ]
    report = pd.DataFrame(rows, columns=COLUMNS)
    report.attrs['checked'] = len(jobs)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('recipe_id')
    parser.add_argument('admin_ids', nargs='*')
    parser.add_argument('--region', help='check every member of a region')
    args = parser.parse_args(argv)
    admin_ids = list(args.admin_ids)
    if args.region:
        import openplaces as op

        admin_ids += op.get_region_admin_ids(args.region)
    if not admin_ids:
        parser.error('name admin ids or --region')
    dag = RecipeDAG(args.recipe_id, admin_ids=admin_ids, deliver=False, validate=False)
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
