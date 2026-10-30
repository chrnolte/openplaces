"""Report built outputs whose code has changed since they were written.

Snakemake cannot see a change to openplaces code. Every rule runs the
same command (`python -m openplaces.flow.run_stage <stage> <recipe>
<unit>`), which no merge changes, and the profiles rerun on mtime only.
A job whose code a merge changed therefore keeps its old output unless
someone forces it, and nothing says so. This module says so; it never
reruns a job.

Each attribute table records in its footer the commit the checkout was
on when the table was written (`path.recipe_roots_footer`). For every
job of a graph that the run will not rebuild anyway (its output is
missing, an input is newer, an upstream job runs, or it is forced),
the report diffs that commit against the working tree, uncommitted
changes included, and keeps the changed files the job depends on:

- the source files its stage imports, followed through every import in
  the package, function-level ones included, plus the step modules a
  stage package loads by name (`io.steps.make_loader`) and any
  openplaces module its recipe names as a string;
- data files beside those modules, such as `core/attribute_registry.csv`;
- its own recipe file, its patch recipes, and the sidecars beside them.

The walk is static and errs toward listing a file: a module counts if
any module the stage reaches imports it, whether or not that code path
runs for this recipe. A listed change may still be one that cannot
affect the output; the report names the files so a person can judge
and force the jobs that need it. An output with no commit in its footer
(a geometry-only file, or one written before the stamp existed) is
counted separately, as is a commit git no longer knows.

Usage::

    python -m openplaces.flow.code_staleness RECIPE_ID [ADMIN_ID ...]
        [--forcerun RULE ...] [--exclude-recipe-ids ID,ID] [--jobs]

The Snakefile prints a summary of the same report (`format_summary`)
while it plans a run, unless `--config code_report=false`. The exit
status is 0 whatever it finds.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from collections import defaultdict
from functools import cache
from pathlib import Path

import openplaces
from openplaces.core.constants import RECIPE_STAGES
from openplaces.flow.dag import STAGES, node_key, rule_name

PACKAGE = 'openplaces'
PACKAGE_DIR = Path(openplaces.__file__).resolve().parent
SOURCE_ROOT = PACKAGE_DIR.parent
RECIPES_DIR = PACKAGE_DIR / 'recipes'

# The modules each stage's job imports first. run_stage dispatches to
# the stage package; deliver and validate have their own branches.
STAGE_ENTRIES = {
    'ingest': ('openplaces.io.ingester',),
    'harmonize': ('openplaces.io.harmonizer',),
    'enrich': ('openplaces.io.enricher',),
    'curate': ('openplaces.io.curator',),
}
# What run_stage itself imports on a recipe stage's path. Not
# run_stage as a whole: its deliver and validate branches import most
# of flow/, which no recipe stage runs.
RUN_STAGE = 'openplaces.core.held_recipes'

# Files beside a module that are documentation, not inputs.
NOT_DATA = {'.py', '.pyc', '.md', '.rst'}

MODULE_NAME = re.compile(r'\bopenplaces(?:\.[A-Za-z_]\w*)+')


def module_file(name: str) -> Path | None:
    """The source file of an openplaces module or package, or None."""
    base = SOURCE_ROOT.joinpath(*name.split('.'))
    for candidate in (base.with_suffix('.py'), base / '__init__.py'):
        if candidate.is_file():
            return candidate
    return None


def _module_name(path: Path) -> str:
    parts = path.relative_to(SOURCE_ROOT).with_suffix('').parts
    return '.'.join(parts[:-1] if parts[-1] == '__init__' else parts)


@cache
def _imports(path: Path) -> frozenset[str]:
    """Every openplaces module *path* imports, at any depth of its body."""
    source = path.read_text(encoding='utf-8')
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return frozenset()
    name = _module_name(path)
    package = name if path.name == '__init__.py' else name.rpartition('.')[0]
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split('.')
                base = base[: len(base) - node.level + 1]
                module = '.'.join([*base, node.module] if node.module else base)
            else:
                module = node.module or ''
            found.add(module)
            # `from package import submodule` imports the submodule.
            found.update(f'{module}.{alias.name}' for alias in node.names)
    # A stage package that loads its step modules by name
    # (io.steps.make_loader) imports every module file beside it.
    if path.name == '__init__.py' and 'make_loader(' in source:
        found.update(
            f'{name}.{p.stem}' for p in path.parent.glob('*.py') if p.stem != '__init__'
        )
    return frozenset(m for m in found if m.split('.')[0] == PACKAGE)


@cache
def import_closure(entries: frozenset[str]) -> frozenset[Path]:
    """Every source file reached from *entries*, package files included.

    Importing `a.b.c` runs `a/__init__.py` and `a/b/__init__.py` first,
    so each module's parent packages are part of the closure.
    """
    seen: set[Path] = set()
    todo = list(entries)
    while todo:
        name = todo.pop()
        parts = name.split('.')
        for depth in range(1, len(parts) + 1):
            path = module_file('.'.join(parts[:depth]))
            if path is None or path in seen:
                continue
            seen.add(path)
            todo.extend(_imports(path))
    return frozenset(seen)


@cache
def _data_files(directory: Path) -> frozenset[Path]:
    """Non-code files directly in a module's directory."""
    if directory == RECIPES_DIR or RECIPES_DIR in directory.parents:
        return frozenset()
    return frozenset(
        p for p in directory.iterdir() if p.is_file() and p.suffix not in NOT_DATA
    )


@cache
def _recipe_files_by_id() -> dict[str, Path]:
    from openplaces.path import BUNDLED_RECIPES_DIR

    return {p.stem: p for p in Path(BUNDLED_RECIPES_DIR).rglob('*.yaml')}


@cache
def recipe_files(recipe_id: str, admin_id) -> frozenset[Path]:
    """A job's recipe file, its patch recipes and the sidecars beside them.

    Only the bundled recipe root is in this repository, so a recipe from
    an installed recipe package contributes nothing.
    """
    from openplaces.recipe import find_recipe_patches

    ids = [recipe_id]
    if admin_id is not None:
        try:
            ids += find_recipe_patches(recipe_id, admin_id)
        except Exception:
            pass
    files = set()
    by_id = _recipe_files_by_id()
    for rid in ids:
        path = by_id.get(rid)
        if path is None:
            continue
        files.add(path)
        # Sidecars (label tables, crosswalks) beside the recipe; the
        # other recipes of the same folder are not this job's.
        files.update(
            p for p in path.parent.iterdir() if p.is_file() and p.suffix != '.yaml'
        )
    return frozenset(files)


def _named_modules(files) -> frozenset[str]:
    """Openplaces modules a recipe names as strings (extensions, hooks)."""
    names = set()
    for path in files:
        if path.suffix != '.yaml':
            continue
        for match in MODULE_NAME.findall(path.read_text(encoding='utf-8')):
            # 'openplaces.io.x:ATTR' or 'openplaces.io.x.function': keep
            # the longest prefix that is a module.
            parts = match.split('.')
            while parts and module_file('.'.join(parts)) is None:
                parts.pop()
            if parts:
                names.add('.'.join(parts))
    return frozenset(names)


def job_dependencies(stage: str, recipe_id: str, admin_id) -> frozenset[str]:
    """Repository-relative paths (posix) a job's output depends on."""
    code, recipes = _dependencies(stage, recipe_id, admin_id)
    return code | recipes


@cache
def _dependencies(stage, recipe_id, admin_id) -> tuple[frozenset[str], frozenset[str]]:
    # Two parts, each cached: the code set is shared by every job whose
    # stage and named modules agree, so a graph of thousands of jobs
    # walks the imports a handful of times.
    recipes = recipe_files(recipe_id, admin_id)
    entries = frozenset((*STAGE_ENTRIES[stage], RUN_STAGE, *_named_modules(recipes)))
    return _code_dependencies(entries), _relative(recipes)


@cache
def _code_dependencies(entries: frozenset[str]) -> frozenset[str]:
    code = import_closure(entries)
    data = {d for path in code for d in _data_files(path.parent)}
    return _relative(frozenset(code | data))


@cache
def _relative(paths: frozenset[Path]) -> frozenset[str]:
    root = repo_root()
    return frozenset(p.resolve().relative_to(root).as_posix() for p in paths)


@cache
def repo_root() -> Path:
    from openplaces.config import cfg

    return Path(cfg.code_root).resolve()


def built_commit(path) -> str | None:
    """The commit recorded in a table's footer, or None.

    Read from the bundled recipe root's record (`path.recipe_roots_footer`),
    which is the checkout's HEAD when the table was written.
    """
    import pyarrow.parquet as pq

    from openplaces.path import RECIPE_ROOTS_METADATA_KEY

    try:
        metadata = pq.read_metadata(path).metadata or {}
        records = json.loads(metadata[RECIPE_ROOTS_METADATA_KEY.encode()])
    except Exception:
        return None
    for record in records:
        if record.get('provider') == 'bundled':
            return record.get('commit')
    return None


@cache
def changed_since(commit: str) -> frozenset[str] | None:
    """Files under src/ that differ between *commit* and the working tree.

    None when git does not know the commit (history rewritten, or a
    shallow clone).
    """
    result = subprocess.run(
        ['git', '-C', str(repo_root()), 'diff', '--name-only', commit, '--', 'src'],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return frozenset(line for line in result.stdout.splitlines() if line)


def _mtime(path) -> float | None:
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return None


def will_run(
    dag, forcerun=(), forceall=False, forcetargets=False, inputs=None
) -> set[tuple]:
    """Node keys of the jobs this run rebuilds anyway.

    A job runs when its output is missing or older than an input (the
    mtime trigger, as `RecipeDAG.plan` mirrors it) or when it is forced,
    and so does every job downstream of one that runs. *forcerun* holds
    rule names or output paths, as Snakemake's `--forcerun` takes them.
    *inputs* maps a node key to its input paths; the Snakefile passes
    the lists it has already built, since deriving them again costs as
    much as building the graph. A node it lacks is derived here.
    """
    forced_names = {str(f) for f in forcerun or ()}
    forced_paths = {Path(f).resolve() for f in forced_names if '/' in f or '\\' in f}
    targets = {Path(p).resolve() for p in dag.target_paths()} if forcetargets else set()
    inputs = inputs or {}
    keys = set()
    for node in dag.nodes():
        key = node_key(node)
        job = (node.stage, node.recipe_id, node.admin_id, node.region)
        output = dag.output_path(*job)
        built = _mtime(output)
        if forceall or built is None or rule_name(node) in forced_names:
            keys.add(key)
            continue
        if (forced_paths or targets) and output.resolve() in forced_paths | targets:
            keys.add(key)
            continue
        paths = inputs.get(key)
        if paths is None:
            paths = dag.input_paths(*job)
        if any((m := _mtime(p)) is not None and m > built for p in paths):
            keys.add(key)
    changed = True
    while changed:
        changed = False
        for upstream, consumer in dag._edges:
            if upstream in keys and consumer not in keys:
                keys.add(consumer)
                changed = True
    return keys


def report(
    dag, forcerun=(), forceall=False, forcetargets=False, inputs=None
) -> list[dict]:
    """One row per job the run keeps although its code changed.

    Jobs the run rebuilds, and delivery and validation jobs (bundles and
    manifests carry no commit), are left out. A row's `status` is
    'code changed', 'no commit recorded' or 'commit unknown to git'.

    Parameters
    ----------
    dag : RecipeDAG
        The run's graph.
    forcerun : iterable of str, optional
        Rule names or output paths the run forces.
    forceall, forcetargets : bool, optional
        Snakemake's `--forceall` and `--force`.
    inputs : dict, optional
        Input paths by node key, as `will_run` takes them.

    Returns
    -------
    list of dict
        Keys: stage, recipe_id, admin_id, rule, commit, status, changed
        (a sorted tuple of repository-relative paths).
    """
    rebuilt = will_run(dag, forcerun, forceall, forcetargets, inputs)
    rows = []
    for node in dag.nodes():
        if node.stage not in RECIPE_STAGES or node_key(node) in rebuilt:
            continue
        output = dag.output_path(node.stage, node.recipe_id, node.admin_id, node.region)
        if not output.exists():
            continue
        commit = built_commit(output)
        row = {
            'stage': node.stage,
            'recipe_id': node.recipe_id,
            'admin_id': node.admin_id,
            'rule': rule_name(node),
            'commit': commit,
            'changed': (),
        }
        if commit is None:
            rows.append({**row, 'status': 'no commit recorded'})
            continue
        changed = changed_since(commit)
        if changed is None:
            rows.append({**row, 'status': 'commit unknown to git'})
            continue
        code, recipes = _dependencies(node.stage, node.recipe_id, node.admin_id)
        relevant = (changed & code) | (changed & recipes)
        if relevant:
            rows.append(
                {**row, 'status': 'code changed', 'changed': tuple(sorted(relevant))}
            )
    return rows


def format_report(rows: list[dict], max_files: int = 4) -> str:
    """The report as text, one line per recipe and commit; '' when empty."""
    if not rows:
        return ''
    order = {stage: rank for rank, stage in enumerate(STAGES)}
    groups = defaultdict(list)
    unknown = defaultdict(int)
    for row in rows:
        if row['status'] == 'code changed':
            groups[(row['stage'], row['recipe_id'], row['commit'])].append(row)
        else:
            unknown[(row['status'], row['stage'], row['recipe_id'])] += 1
    lines = []
    if groups:
        lines.append(
            'Code changed since these outputs were built, and this run keeps '
            'them (Snakemake does not see code changes):'
        )
        for (stage, recipe_id, commit), members in sorted(
            groups.items(), key=lambda kv: (order[kv[0][0]], kv[0][1], kv[0][2])
        ):
            units = len(members)
            changed = sorted({f for m in members for f in m['changed']})
            shown = ', '.join(
                f.removeprefix('src/openplaces/') for f in changed[:max_files]
            )
            if len(changed) > max_files:
                shown += f' and {len(changed) - max_files} more'
            lines.append(
                f'  {stage} {recipe_id}: {units} unit{"s" if units != 1 else ""} '
                f'built at {commit}; changed: {shown}'
            )
        example = min(groups.values(), key=len)[0]['rule']
        lines.append(
            f'Force what needs it with --forcerun <rule>, one rule per unit '
            f'(e.g. {example}); `python -m openplaces.flow.code_staleness` '
            'lists each job.'
        )
    if unknown:
        by_status = defaultdict(list)
        for (status, stage, recipe_id), n in sorted(
            unknown.items(), key=lambda kv: (kv[0][0], order[kv[0][1]], kv[0][2])
        ):
            by_status[status].append(f'{stage} {recipe_id} ({n})')
        for status, entries in by_status.items():
            lines.append(f'Kept outputs with {status}: ' + ', '.join(entries))
    return '\n'.join(lines)


def format_summary(rows: list[dict], recipe_id: str, max_files: int = 5) -> str:
    """A few lines for the start of a run; '' when nothing is kept.

    A graph of hundreds of units yields hundreds of entries in
    `format_report`, which would bury the point in a run's log: this
    gives the counts per stage and the files most jobs share, and names
    the command that lists the rest.
    """
    changed = [r for r in rows if r['status'] == 'code changed']
    unknown = len(rows) - len(changed)
    if not rows:
        return ''
    lines = []
    if changed:
        order = {stage: rank for rank, stage in enumerate(STAGES)}
        per_stage = defaultdict(int)
        per_file = defaultdict(int)
        for row in changed:
            per_stage[row['stage']] += 1
            for path in row['changed']:
                per_file[path] += 1
        stages = ', '.join(
            f'{stage} {n}'
            for stage, n in sorted(per_stage.items(), key=lambda kv: order[kv[0]])
        )
        common = sorted(per_file.items(), key=lambda kv: (-kv[1], kv[0]))[:max_files]
        files = ', '.join(
            f'{p.removeprefix("src/openplaces/")} ({n})' for p, n in common
        )
        lines += [
            f'CODE CHANGED, NOT REBUILT: {len(changed)} jobs this run keeps were built '
            'by older code (Snakemake does not see code changes).',
            f'  by stage: {stages}',
            f'  files changed since, by jobs they reach: {files}',
            f'  list them: python -m openplaces.flow.code_staleness {recipe_id} '
            '[units]; force what needs it with --forcerun <rule>.',
        ]
    if unknown:
        lines.append(
            f'  {unknown} kept outputs record no commit, or one git does not know.'
        )
    return '\n'.join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('recipe_id')
    parser.add_argument('admin_ids', nargs='*')
    parser.add_argument('--forcerun', nargs='*', default=())
    parser.add_argument('--exclude-recipe-ids', default='')
    parser.add_argument(
        '--jobs', action='store_true', help='list every job, one rule per line'
    )
    args = parser.parse_args(argv)

    from openplaces.flow.dag import RecipeDAG

    exclude = {r for r in args.exclude_recipe_ids.split(',') if r}
    dag = RecipeDAG(
        args.recipe_id,
        admin_ids=args.admin_ids,
        exclude_recipe_ids=exclude or None,
        deliver=False,
        validate=False,
    )
    rows = report(dag, forcerun=args.forcerun)
    if args.jobs:
        for row in rows:
            print(f'{row["rule"]}\t{row["status"]}\t{row["commit"] or ""}')
    else:
        text = format_report(rows)
        print(text or 'No kept output was built by code that has changed since.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
