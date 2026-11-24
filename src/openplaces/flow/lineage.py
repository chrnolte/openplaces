"""Which steps read and write a column, across recipes.

A registered step may declare the columns it reads, writes and decides
(`openplaces.io.steps.make_register`). This module resolves those
declarations against every recipe pipeline, without opening any data,
and answers the question a reader of a delivered column asks: which
steps, in which recipes, produced this value, and which steps use it.

Run `python -m openplaces.flow.lineage n_stories` for the chain of one
column, or `python -m openplaces.flow.lineage --coverage` for the share
of recipe steps that declare their columns.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from importlib import import_module
from itertools import combinations

import yaml

from openplaces.core.schema import AdminId
from openplaces.io.steps import resolve_step_columns
from openplaces.path import recipe_roots
from openplaces.recipe import apply_recipe_patches, get_recipe_by_id

#: The stage packages that dispatch a recipe's pipeline by step name.
_STAGE_PACKAGES = {
    'harmonize': 'openplaces.io.harmonizer',
    'enrich': 'openplaces.io.enricher',
    'curate': 'openplaces.io.curator',
}

#: Recipe keys of a pipeline entry that are never step arguments.
_CONTROL_KEYS = frozenset({'step', 'enabled', 'phase_override', 'checkpoint'})


@dataclass(frozen=True)
class StepUse:
    """One step of one recipe pipeline, with its declared columns.

    `declared` is False for a step that declares nothing; its column
    sets are then empty and say nothing about what it touches.
    `entity_type` is the recipe's, None for a recipe of a dataset.
    `drops` names the columns the step removes and `drop_rules` holds
    (rule name, test of a column name) pairs for removals stated as a
    rule (`io.steps.DropRule`); neither counts as a write.
    """

    recipe_id: str
    stage: str
    position: int
    step: str
    declared: bool
    reads: frozenset = frozenset()
    writes: frozenset = frozenset()
    decides: frozenset = frozenset()
    reads_from: tuple = ()
    entity_type: str | None = None
    drops: frozenset = frozenset()
    drop_rules: tuple = field(default=(), compare=False)


def _step_registry(stage: str) -> dict:
    """A stage's name-to-function registry, with every step imported."""
    package = import_module(_STAGE_PACKAGES[stage])
    package._load_steps()
    return package._STEP_REGISTRY


def pipeline_recipes() -> dict[str, dict]:
    """Every recipe that has a pipeline, as parsed YAML keyed by id.

    Patch recipes carry a `pipeline_patch` and no pipeline, so they
    are not listed; `recipe_steps` applies them for an admin unit.
    """
    recipes = {}
    for root in recipe_roots():
        for path in sorted(root.rglob('*.yaml')):
            try:
                data = yaml.safe_load(path.read_text(encoding='utf-8'))
            except yaml.YAMLError:
                continue
            if isinstance(data, dict) and data.get('pipeline'):
                data.setdefault('recipe_id', path.stem)
                recipes[path.stem] = data
    return recipes


def recipe_steps(recipe: dict, admin_id=None) -> list[StepUse]:
    """The steps of one recipe with their declared columns resolved.

    Parameters
    ----------
    recipe : dict
        A recipe with a `pipeline` and a `stage`.
    admin_id : str or AdminId, optional
        Where given, the patch recipes covering this unit are applied
        first, so the result is the pipeline as that unit runs it.

    Raises
    ------
    ValueError
        On a step name its stage does not register, or a declaration
        that does not resolve against the recipe's arguments.
    """
    stage = recipe.get('stage')
    if stage not in _STAGE_PACKAGES:
        return []
    if admin_id is not None:
        recipe, _ = apply_recipe_patches(recipe, admin_id)
    registry = _step_registry(stage)
    recipe_id = recipe.get('recipe_id', '')
    entity_type = _entity_type(recipe)
    uses = []
    for position, entry in enumerate(recipe.get('pipeline') or [], start=1):
        if not isinstance(entry, dict) or entry.get('enabled', True) is False:
            continue
        name = entry.get('step')
        fn = registry.get(name)
        if fn is None:
            raise ValueError(f'{recipe_id} step {position}: unknown step {name!r}')
        params = {k: v for k, v in entry.items() if k not in _CONTROL_KEYS}
        try:
            resolved = resolve_step_columns(fn, params, recipe)
        except ValueError as error:
            where = f'{recipe_id} step {position} {name!r}'
            raise ValueError(f'{where}: {error}') from error
        if resolved is None:
            uses.append(
                StepUse(
                    recipe_id,
                    stage,
                    position,
                    name,
                    declared=False,
                    entity_type=entity_type,
                )
            )
            continue
        uses.append(
            StepUse(
                recipe_id,
                stage,
                position,
                name,
                declared=True,
                reads=resolved.reads,
                writes=resolved.writes,
                decides=resolved.decides,
                reads_from=tuple(sorted(resolved.reads_from.items())),
                entity_type=entity_type,
                drops=resolved.drops,
                drop_rules=resolved.drop_rules,
            )
        )
    return uses


def _entity_type(recipe: dict) -> str | None:
    """A recipe's entity type, parsed YAML or loaded, or None."""
    entity = recipe.get('entity')
    if isinstance(entity, dict):
        entity_type = entity.get('entity_type')
    else:
        entity_type = getattr(entity, 'entity_type', None)
    return str(entity_type) if entity_type is not None else None


def all_steps(admin_id=None) -> list[StepUse]:
    """Every step of every recipe pipeline (see `recipe_steps`).

    With an *admin_id*, each recipe is read through the recipe loader,
    because a patch is checked against the recipe's entity, which
    parsed YAML alone does not carry.
    """
    recipes = pipeline_recipes()
    if admin_id is not None:
        recipes = {recipe_id: get_recipe_by_id(recipe_id) for recipe_id in recipes}
    return [
        use for recipe in recipes.values() for use in recipe_steps(recipe, admin_id)
    ]


def _matches(column: str, names: Iterable[str]) -> bool:
    return any(
        fnmatchcase(column, name) if '*' in name else column == name for name in names
    )


def writers(column: str, steps: Iterable[StepUse]) -> list[StepUse]:
    """The steps that declare writing *column*."""
    return [use for use in steps if _matches(column, use.writes)]


def droppers(column: str, steps: Iterable[StepUse]) -> list[StepUse]:
    """The steps that declare removing *column*: where it leaves the
    table. A removal is not a write, so `writers` does not list them."""
    return [
        use
        for use in steps
        if _matches(column, use.drops)
        or any(test(column) for _, test in use.drop_rules)
    ]


def readers(column: str, steps: Iterable[StepUse]) -> list[StepUse]:
    """The steps that declare reading *column*, from their own table or
    from another recipe's."""
    return [
        use
        for use in steps
        if _matches(column, use.reads)
        or any(_matches(column, columns) for _, columns in use.reads_from)
    ]


def repeated_deciders(steps: Iterable[StepUse]) -> dict[tuple[str, str], list[int]]:
    """Columns that more than one step of a recipe declares deciding.

    Returns
    -------
    dict
        (recipe id, column) mapped to the positions of its deciding
        steps, for every pair with more than one.
    """
    positions: dict[tuple[str, str], list[int]] = {}
    for use in steps:
        for column in use.decides:
            positions.setdefault((use.recipe_id, column), []).append(use.position)
    return {key: value for key, value in positions.items() if len(value) > 1}


def patch_recipes() -> dict[str, dict]:
    """Every patch recipe, as parsed YAML keyed by id.

    A patch recipe names the recipe it amends under `patches` and
    carries a `pipeline_patch` (`openplaces.recipe.apply_recipe_patches`).
    """
    patches = {}
    for root in recipe_roots():
        for path in sorted(root.rglob('*.yaml')):
            try:
                data = yaml.safe_load(path.read_text(encoding='utf-8'))
            except yaml.YAMLError:
                continue
            if isinstance(data, dict) and data.get('pipeline_patch'):
                patches[path.stem] = data
    return patches


def patch_errors() -> dict[str, str]:
    """Patch recipes that do not apply, with the reason.

    Each patch is applied, with the coarser patches before it, to the
    recipe it names, for its own admin unit, and the result resolved
    as `recipe_steps` does. A patch whose anchor names a step the
    pipeline does not hold, or which inserts a step its stage does not
    register, fails here instead of at run time.
    """
    errors = {}
    for patch_id, patch in patch_recipes().items():
        target = patch.get('patches')
        try:
            recipe_steps(get_recipe_by_id(target), patch.get('admin_id'))
        except (ValueError, KeyError, FileNotFoundError) as error:
            errors[patch_id] = str(error)
    return errors


def patched_steps() -> list[StepUse]:
    """The steps of every patched pipeline, one per patch scope.

    Patches that do not apply (`patch_errors`) are left out.
    """
    uses = []
    for patch in patch_recipes().values():
        try:
            uses += recipe_steps(
                get_recipe_by_id(patch.get('patches')), patch.get('admin_id')
            )
        except (ValueError, KeyError, FileNotFoundError):
            continue
    return uses


def entity_deciders(
    steps: Iterable[StepUse], columns: Iterable[str]
) -> dict[tuple[str | None, str], frozenset]:
    """The deciding steps of each column, per entity type, across recipes.

    Parameters
    ----------
    steps : iterable of StepUse
        Steps of any recipes, patched variants included.
    columns : iterable of str
        The columns to report (the attribute registry's names).

    Returns
    -------
    dict
        (entity type, column) mapped to the `recipe_id:step` names that
        declare deciding it, for every pair with two deciders in
        recipes whose admin scopes overlap (`US` and `US-FL` do, `US`
        and `CO` do not: no unit runs both). A step name repeated in
        one recipe counts once; `repeated_deciders` reports that case.
    """
    wanted = frozenset(columns)
    found: dict[tuple[str | None, str], set] = {}
    for use in steps:
        for column in use.decides & wanted:
            key = (use.entity_type, column)
            found.setdefault(key, set()).add(f'{use.recipe_id}:{use.step}')
    return {
        key: frozenset(value)
        for key, value in found.items()
        if any(
            _scopes_overlap(a.split(':')[0], b.split(':')[0])
            for a, b in combinations(sorted(value), 2)
        )
    }


def _scope(recipe_id: str) -> AdminId | None:
    """The admin unit a recipe id starts with, None for a global one."""
    try:
        return AdminId(recipe_id.split('_')[0])
    except ValueError:
        return None


def _scopes_overlap(first: str, second: str) -> bool:
    """Whether some admin unit runs both recipes."""
    a, b = _scope(first), _scope(second)
    if a is None or b is None:
        return True
    return a.is_parent_or_equal_of(b) or b.is_parent_or_equal_of(a)


def writes_after_decider(
    steps: Iterable[StepUse], columns: Iterable[str]
) -> set[tuple[str, str, str, str]]:
    """Steps that write a decided column later in the same recipe.

    *steps* holds one pipeline per recipe id: pass a patched variant
    apart from the unpatched recipe, since their positions differ.

    Returns
    -------
    set of tuple
        (recipe id, column, deciding step, later writer) for each
        column of *columns* a recipe decides. A writer declaring every
        column (`*`) is listed too; it may or may not touch the column.
        A step that only removes the column (`drops`) is not a writer
        and is not listed; `droppers` finds it.
    """
    wanted = frozenset(columns)
    by_recipe: dict[str, list[StepUse]] = {}
    for use in steps:
        by_recipe.setdefault(use.recipe_id, []).append(use)
    found = set()
    for recipe_id, uses in by_recipe.items():
        for decider in uses:
            for column in decider.decides & wanted:
                for later in writers(column, uses):
                    if later.position > decider.position:
                        found.add((recipe_id, column, decider.step, later.step))
    return found


def _print_chain(column: str, steps: list[StepUse]) -> None:
    for title, found in (
        ('written by', writers(column, steps)),
        ('read by', readers(column, steps)),
        ('dropped by', droppers(column, steps)),
    ):
        print(f'{column} {title}:')
        for use in found:
            decides = ' (decides)' if _matches(column, use.decides) else ''
            print(f'  {use.recipe_id} #{use.position} {use.step}{decides}')
        if not found:
            print('  no declared step')
    undeclared = sum(1 for use in steps if not use.declared)
    print(
        f'{undeclared} of {len(steps)} steps declare no columns and may also touch it.'
    )


def _print_coverage(steps: list[StepUse]) -> None:
    for stage in _STAGE_PACKAGES:
        of_stage = [use for use in steps if use.stage == stage]
        declared = sum(1 for use in of_stage if use.declared)
        print(f'{stage}: {declared} of {len(of_stage)} recipe steps declared')


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('column', nargs='?', help='column whose chain to print')
    parser.add_argument('--admin-id', help='apply the patch recipes of this unit')
    parser.add_argument('--coverage', action='store_true')
    args = parser.parse_args(argv)
    steps = all_steps(args.admin_id)
    if args.coverage or not args.column:
        _print_coverage(steps)
    if args.column:
        _print_chain(args.column, steps)


if __name__ == '__main__':
    main()
