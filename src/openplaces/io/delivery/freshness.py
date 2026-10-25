"""Find built outputs that are older than what they were built from.

A delivery pools curated files that are already on disk. When a spine is
rebuilt in one run and its consumers are not, the consumer's file stays
on disk, older than the spine, and an export that did not look would
ship it as if it were current. Transaction curates linked to a property
spine that had since been fixed reached the valuation lane this way
(2026-10), which is why `export_delivery` refuses such a bundle
(`refuse_stale`) unless told to allow it.

`out_of_date` reports such jobs from file modification times only, the
same trigger Snakemake reruns on. Unlike `RecipeDAG.plan`, it does not
propagate "upstream scheduled" from a missing output: an absent file
can be a held recipe, an opt-in lane or a consumed intermediate, and
treating every one as a pending rebuild marks a whole county's graph
for rerun and buries the real cases. Staleness does propagate through
existing files, because a consumer of a stale output is itself built
from stale data even when its own inputs are older than it.

The check lives here, below the orchestrator, so that `export_delivery`
can run it. The graph it walks (`openplaces.flow.dag.RecipeDAG`) sits
above this module and is passed in by the caller; anything with the
same three methods (`nodes`, `output_path`, `input_paths`) will do.
`python -m openplaces.flow.freshness` is the command-line entry point.
"""

from pathlib import Path

import pandas as pd

from openplaces.io.cleanup.receipts import _relative_posix, _resolve_relative

COLUMNS = ['stage', 'recipe_id', 'admin_id', 'region', 'output', 'reason']


class StaleOutputError(RuntimeError):
    """A pooled output is older than an input it was built from."""


def _mtime(path):
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _job_files(graph):
    """Yield (job, output, output mtime, inputs) for every built job."""
    for node in graph.nodes():
        job = (node.stage, node.recipe_id, node.admin_id, node.region)
        try:
            out = graph.output_path(*job)
        except Exception:  # noqa: BLE001 - unresolvable, nothing to check
            continue
        out_mtime = _mtime(out)
        if out_mtime is None:
            continue
        try:
            inputs = graph.input_paths(*job)
        except Exception:  # noqa: BLE001 - same fallback as plan()
            inputs = []
        yield job, out, out_mtime, inputs


def out_of_date(graph) -> pd.DataFrame:
    """List the jobs whose existing output is older than its inputs.

    Parameters
    ----------
    graph : RecipeDAG
        The graph to check, or any object with the same `nodes`,
        `output_path` and `input_paths` methods. Only the files of its
        own jobs are statted; nothing is read.

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
    jobs = list(_job_files(graph))

    reasons = {}
    for _job, out, out_mtime, inputs in jobs:
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


def refuse_stale(graph, pooled) -> None:
    """Raise when a file about to be pooled is out of date.

    Parameters
    ----------
    graph : RecipeDAG
        The dependency graph of the pooled files, as for `out_of_date`.
        It has to produce every pooled file: a graph that does not
        would pass a file it never looked at.
    pooled : iterable of pathlib.Path
        The curated files the delivery is about to read.

    Raises
    ------
    ValueError
        When a pooled file is not the output of any job in *graph*.
    StaleOutputError
        When a pooled file is older than an input it was built from,
        directly or through a stale upstream output. The message lists
        every out-of-date output of the graph with the newer input or
        stale upstream file it was found by.
    """
    pooled = {Path(path).resolve() for path in pooled}
    produced = set()
    for node in graph.nodes():
        job = (node.stage, node.recipe_id, node.admin_id, node.region)
        try:
            produced.add(Path(graph.output_path(*job)).resolve())
        except Exception:  # noqa: BLE001 - unresolvable, produces nothing
            continue
    uncovered = sorted(pooled - produced)
    if uncovered:
        raise ValueError(
            f'The dependency graph passed for the freshness check does not '
            f'build {len(uncovered)} of the {len(pooled)} pooled files, e.g. '
            f'{_relative_posix(uncovered[0])}; build it for the units being '
            'delivered.'
        )
    report = out_of_date(graph)
    n_stale = sum(
        _resolve_relative(output).resolve() in pooled for output in report['output']
    )
    if not n_stale:
        return
    lines = [f'  {row.output}: {row.reason}' for row in report.itertuples()]
    raise StaleOutputError(
        f'{n_stale} of {len(pooled)} pooled curated files are older '
        'than an input they were built from. Out-of-date outputs and the '
        'newer input or stale upstream file each was found by:\n'
        + '\n'.join(lines)
        + '\nRebuild them, or pass allow_stale=True (--allow-stale) to '
        'export anyway.'
    )
