"""A stage's named pipeline steps: the registry decorator and the lazy
loader.

The harmonize, enrich and curate packages each dispatch recipe
``pipeline`` entries by name to a function registered under that name.
Until 2026-09-22 each package carried its own copy of the decorator and
of the loader that imports the step submodules; the copies differed only
in that the harmonizer's decorator also records a phase. This module is
the one copy. It sits below the stages in the layer hierarchy and
imports nothing from openplaces at module level (the audit report's
default path reads `openplaces.path` when it is written), so any stage
may use it. It also holds the column declarations and the audit mode
that checks them against what each step did.

Each stage keeps its registry as a module-level dict of its own
(``_STEP_REGISTRY``, and the harmonizer's ``_STEP_PHASES``): tests
monkeypatch those dicts by name, and ``harmonizer.links`` reads the
phases directly, so the dicts stay where they were and only the code
that fills them is shared.
"""

from __future__ import annotations

import inspect
import itertools
import pkgutil
import re
import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from fnmatch import fnmatchcase
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

#: The two phases a harmonize step can declare. Geometry-phase steps
#: mutate spine rows or geometry or run spatial joins, and their configs
#: are part of the link-sidecar fingerprint; attribute-phase steps only
#: read or annotate.
PHASES = ('geometry', 'attributes')


#: The attribute a declared step function carries (a StepColumns).
_COLUMNS_ATTRIBUTE = '__step_columns__'

#: A parameter reference inside a declared column name: `{column}`, or
#: `{mapping:keys}` / `{mapping:values}` for a mapping parameter.
_PARAMETER_REFERENCE = re.compile(r'\{([A-Za-z_][A-Za-z0-9_]*)(?::(keys|values))?\}')

#: The key under which a declaration callable finds the recipe that runs
#: the step, where the caller passes one (`resolve_step_columns`). Not
#: an identifier, so it never shadows a step parameter and no template
#: can name it.
RECIPE_ARGUMENT = '@recipe'


class UndeclaredColumnWarning(UserWarning):
    """A step added a column that its declaration does not list."""


@dataclass(frozen=True)
class DropRule:
    """A named test for the columns a step removes, for a `drops` entry.

    A step that keeps an allow-list (every column the attribute
    registry knows) removes a set that no list of names or patterns
    can state, since patterns cannot negate. The rule states it as a
    test instead: `test(column, arguments)` is True for a column the
    step removes under those arguments. The audit asks it about each
    column a step removed; lineage asks it about the column queried.
    A rule is not callable itself, so a `drops` entry that is a rule
    is never mistaken for a callable returning names.

    Parameters
    ----------
    name : str
        A short label for reports and lineage output.
    test : Callable
        Takes a column name and the step's arguments (as a callable
        entry receives them) and returns True where the step removes
        that column.
    """

    name: str
    test: Callable = field(compare=False)


@dataclass(frozen=True)
class StepColumns:
    """The columns a step declares, as written on its registration.

    Entries are column names, templates naming the step's parameters
    (`'{column}_source'`), or callables taking the step's arguments
    and returning names. A `drops` entry may also be a `DropRule`.
    See :func:`resolve_step_columns`.
    """

    reads: tuple = ()
    writes: tuple = ()
    decides: tuple = ()
    reads_from: Mapping = field(default_factory=dict)
    drops: tuple = ()


@dataclass(frozen=True)
class ResolvedColumns:
    """A step's declared columns for one recipe entry, as plain names.

    `reads_from` maps an upstream recipe id to the columns read from
    its table. A name in `writes` or `drops` may hold `*` as a
    wildcard. `drop_rules` holds each `DropRule` with the arguments
    bound, as (rule name, test of one column name).
    """

    reads: frozenset
    writes: frozenset
    decides: frozenset
    reads_from: Mapping
    drops: frozenset = frozenset()
    drop_rules: tuple = ()

    def drops_column(self, name) -> bool:
        """Whether the declaration says the step removes *name*."""
        if _is_written(name, self.drops):
            return True
        return any(test(str(name)) for _, test in self.drop_rules)


def _identity(fn: Callable) -> tuple[str, str]:
    """A step function's module and qualified name."""
    return (getattr(fn, '__module__', ''), getattr(fn, '__qualname__', repr(fn)))


def _as_entries(value) -> tuple:
    """A declaration argument as a tuple of entries."""
    if value is None:
        return ()
    if isinstance(value, str | DropRule) or callable(value):
        return (value,)
    return tuple(value)


def step_columns(fn: Callable) -> StepColumns | None:
    """The declaration a step function carries, or None without one."""
    return getattr(fn, _COLUMNS_ATTRIBUTE, None)


def _arguments(fn: Callable, params: Mapping) -> dict:
    """The recipe's arguments over the function's own defaults."""
    arguments = {
        name: parameter.default
        for name, parameter in inspect.signature(fn).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    arguments.update(params)
    return arguments


def _names_of(value, part: str | None, reference: str) -> list[str]:
    """The column names one parameter value stands for."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        if part is None:
            raise ValueError(
                f'{reference} is a mapping: write {{name:keys}} or {{name:values}}'
            )
        value = value.keys() if part == 'keys' else value.values()
    # Items that are not names (nested specs) are left to a callable
    # entry, which can read any shape.
    return [item for item in value if isinstance(item, str)]


def _expand(entry, arguments: Mapping) -> list[str]:
    """One declared entry as the column names it stands for."""
    if callable(entry):
        return [name for name in entry(arguments) if isinstance(name, str)]
    references = list(_PARAMETER_REFERENCE.finditer(entry))
    if not references:
        return [entry]
    choices = []
    for match in references:
        name, part = match.group(1), match.group(2)
        if name not in arguments:
            raise ValueError(f'{match.group(0)} names no parameter of the step')
        choices.append(_names_of(arguments[name], part, match.group(0)))
    names = []
    for combination in itertools.product(*choices):
        filled = iter(combination)
        names.append(_PARAMETER_REFERENCE.sub(lambda _: next(filled), entry))
    return names


def _expand_all(entries: Iterable, arguments: Mapping) -> frozenset:
    return frozenset(name for entry in entries for name in _expand(entry, arguments))


def resolve_step_columns(
    fn: Callable, params: Mapping, recipe: Mapping | None = None
) -> ResolvedColumns | None:
    """The columns a step reads and writes under one recipe entry.

    Parameters
    ----------
    fn : Callable
        A registered step function.
    params : Mapping
        The step's arguments as the recipe gives them (control keys
        such as `step` and `enabled` removed).
    recipe : Mapping, optional
        The recipe whose pipeline holds the step. A declaration
        callable finds it under `RECIPE_ARGUMENT`, for a name that
        depends on the recipe rather than on the step's arguments (the
        spine's entity type, a link an earlier step built). Without
        it, such a callable states what it can from the arguments.

    Returns
    -------
    ResolvedColumns or None
        None when the step declares nothing. A template is filled from
        *params*, then from the function's defaults: a list parameter
        stands for each of its items, a mapping parameter for its keys
        or values, a parameter that is None for nothing. A `DropRule`
        in `drops` is kept as a test bound to these arguments.

    Raises
    ------
    ValueError
        On a template that names no parameter of the step, or a mapping
        parameter referenced without `:keys` or `:values`.
    """
    declared = step_columns(fn)
    if declared is None:
        return None
    arguments = _arguments(fn, params)
    if recipe is not None:
        arguments[RECIPE_ARGUMENT] = recipe
    reads_from: dict[str, frozenset] = {}
    for recipe_parameter, entries in declared.reads_from.items():
        columns = _expand_all(_as_entries(entries), arguments)
        for recipe_id in _names_of(
            arguments.get(recipe_parameter), 'keys', recipe_parameter
        ):
            reads_from[recipe_id] = reads_from.get(recipe_id, frozenset()) | columns
    rules = [entry for entry in declared.drops if isinstance(entry, DropRule)]
    named = [entry for entry in declared.drops if not isinstance(entry, DropRule)]
    return ResolvedColumns(
        reads=_expand_all(declared.reads, arguments),
        writes=_expand_all(declared.writes, arguments),
        decides=_expand_all(declared.decides, arguments),
        reads_from=reads_from,
        drops=_expand_all(named, arguments),
        drop_rules=tuple((rule.name, _bound(rule, arguments)) for rule in rules),
    )


def _bound(rule: DropRule, arguments: Mapping) -> Callable[[str], bool]:
    """A rule's test of one column name, under fixed arguments."""
    return lambda column: bool(rule.test(column, arguments))


def undeclared_new_columns(
    fn: Callable, params: Mapping, before: Iterable, after: Iterable
) -> list[str]:
    """Columns a step added that its declaration does not list.

    Compares column names only, so the cost is a set difference. A
    step without a declaration, or one that declares `writes='*'`,
    reports nothing. Changes to existing columns are not seen here.
    """
    resolved = resolve_step_columns(fn, params)
    if resolved is None:
        return []
    added = set(after) - set(before)
    if not added:
        return []
    return sorted(name for name in added if not _is_written(name, resolved.writes))


def _is_written(name, writes: frozenset) -> bool:
    """Whether a column name is among a step's resolved writes."""
    if name in writes:
        return True
    return any(fnmatchcase(str(name), pattern) for pattern in writes if '*' in pattern)


def warn_undeclared_columns(
    step_name: str, fn: Callable, params: Mapping, before: Iterable, after: Iterable
) -> None:
    """Warn when a declared step added columns outside its declaration."""
    undeclared = undeclared_new_columns(fn, params, before, after)
    if undeclared:
        warnings.warn(
            f'step {step_name!r} added columns it does not declare in '
            f'writes: {", ".join(map(str, undeclared))}',
            UndeclaredColumnWarning,
            stacklevel=2,
        )


#: The columns of a step audit report, in order.
AUDIT_COLUMNS = (
    'recipe',
    'admin_id',
    'position',
    'step',
    'table',
    'column',
    'change',
    'n_rows',
    'declared',
    'step_declares',
)


def column_hashes(series: pd.Series) -> np.ndarray:
    """One 64-bit hash per row of a column, of its values alone.

    Geometry is hashed through its WKB. A column holding unhashable
    objects (lists, dicts) is hashed through each value's repr.
    """
    if series.dtype.name == 'geometry':
        import shapely

        wkb = shapely.to_wkb(np.asarray(series.array, dtype=object))
        series = pd.Series(wkb, index=series.index, dtype=object)
    try:
        hashed = pd.util.hash_pandas_object(series, index=False)
    except (TypeError, ValueError):
        hashed = pd.util.hash_pandas_object(series.map(repr), index=False)
    return hashed.to_numpy()


def _snapshot(frame) -> tuple | None:
    """A table's index and, per column, its dtype and row hashes."""
    if frame is None:
        return None
    columns: dict = {}
    for position, name in enumerate(frame.columns):
        hashes = column_hashes(frame.iloc[:, position])
        dtype = str(frame.dtypes.iloc[position])
        if name in columns:
            # A repeated name: fold the second column into the first,
            # so a change to either is still seen.
            held_dtype, held = columns[name]
            hashes = held ^ (hashes * np.uint64(31))
            dtype = f'{held_dtype},{dtype}'
        columns[name] = (dtype, hashes)
    return frame.index, columns


def _compare(before: tuple | None, after: tuple | None) -> list[tuple]:
    """The changes between two snapshots, as (column, change, n_rows).

    Rows are aligned on the index, so a step that reorders rows changes
    nothing and a step that drops rows reports one 'rows' entry, not a
    change in every column. A cell counts as changed when its hash
    differs on a row both tables hold.
    """
    before_index, before_columns = before if before else (pd.Index([]), {})
    after_index, after_columns = after if after else (pd.Index([]), {})
    changes = []
    aligned = before_index.equals(after_index)
    take = None
    if not aligned and before_index.is_unique and after_index.is_unique:
        positions = before_index.get_indexer(after_index)
        kept = positions >= 0
        added, removed = int((~kept).sum()), len(before_index) - int(kept.sum())
        if added or removed:
            changes.append(('', 'rows', added + removed))
        take = (positions[kept], np.flatnonzero(kept))
    for name, (dtype, hashes) in after_columns.items():
        if name not in before_columns:
            changes.append((name, 'added', len(hashes)))
            continue
        held_dtype, held = before_columns[name]
        if aligned:
            n_changed = int((held != hashes).sum())
        elif take is not None:
            n_changed = int((held[take[0]] != hashes[take[1]]).sum())
        else:
            # A repeated index label with a changed row set: no row
            # can be paired, so only whether the column changed is told.
            same = np.array_equal(np.sort(held), np.sort(hashes))
            n_changed = 0 if same else max(len(held), len(hashes))
        if n_changed:
            changes.append((name, 'changed', n_changed))
        elif held_dtype != dtype:
            changes.append((name, 'dtype', 0))
    for name, (_, held) in before_columns.items():
        if name not in after_columns:
            changes.append((name, 'removed', len(held)))
    return changes


class StepAudit:
    """Column hashes before and after every step of one unit's run.

    Audit mode (`audit_steps=True` on a Harmonizer, Enricher or
    Curator) hashes every cell of the stage's tables after each step
    and compares the hashes with those taken after the step before.
    Each column a step added, removed or changed becomes one report
    row, marked by whether the step's declared `writes` name it; a
    removed column also counts as declared where the step's `drops`
    name it or one of its drop rules holds for it. Off,
    no audit object exists and the dispatch loop does nothing more
    than check that it is None.

    The cost is one hash pass over every table per step, and memory
    for one 8-byte hash per cell of the last snapshot, so audit mode
    runs on panel units, never in a wave.

    Parameters
    ----------
    stage : str
        The stage's timer name ('Harmonizer', 'Enricher', 'Curator'),
        used in the report's file name.
    recipe_id : str
        The recipe being run.
    admin_id : str or None
        The unit being run, or a label for a global run.
    tables : Mapping
        The stage's tables before the first step, by name (for
        example {'curated': state.curated}). A table may be None.
    """

    def __init__(self, stage: str, recipe_id: str, admin_id, tables: Mapping):
        self.stage = stage
        self.recipe_id = recipe_id
        self.admin_id = None if admin_id is None else str(admin_id)
        self.rows: list[dict] = []
        self._previous = {name: _snapshot(frame) for name, frame in tables.items()}

    def after_step(
        self,
        position: int,
        step_name: str,
        fn: Callable,
        params: Mapping,
        tables: Mapping,
    ) -> None:
        """Record what one step changed in each table.

        Parameters
        ----------
        position : int
            The step's position in the (unrolled) pipeline.
        step_name : str
            The step's registered name.
        fn : Callable
            The step function, whose declaration decides `declared`.
        params : Mapping
            The step's arguments as the recipe gives them.
        tables : Mapping
            The stage's tables after the step, by the names given to
            the constructor.
        """
        resolved = resolve_step_columns(fn, params)
        writes = resolved.writes if resolved is not None else frozenset()
        for table, frame in tables.items():
            snapshot = _snapshot(frame)
            for column, change, n_rows in _compare(self._previous.get(table), snapshot):
                if change == 'rows':
                    declared = 'n/a'
                elif _is_written(column, writes):
                    declared = 'yes'
                elif change == 'removed' and resolved is not None:
                    declared = 'yes' if resolved.drops_column(column) else 'no'
                else:
                    declared = 'no'
                self.rows.append(
                    {
                        'recipe': self.recipe_id,
                        'admin_id': self.admin_id,
                        'position': position,
                        'step': step_name,
                        'table': table,
                        'column': column,
                        'change': change,
                        'n_rows': n_rows,
                        'declared': declared,
                        'step_declares': resolved is not None,
                    }
                )
            self._previous[table] = snapshot

    def report(self) -> pd.DataFrame:
        """The run's changes, one row per step, table and column."""
        return pd.DataFrame(self.rows, columns=list(AUDIT_COLUMNS))

    def finish(self, path: Path | str | None = None) -> pd.DataFrame:
        """Write the report as CSV and return it.

        Parameters
        ----------
        path : Path or str, optional
            Where to write. By default the unit's log folder, beside the
            stage's timer file: `<Stage>_audit_<timestamp>.csv`.
        """
        report = self.report()
        if path is None:
            from openplaces.path import logs_path

            stamp = datetime.now().strftime('%Y%m%d%H%M%S')
            path = logs_path(self.admin_id, filename=f'{self.stage}_audit_{stamp}.csv')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(path, index=False)
        return report


def make_register(
    registry: dict[str, Callable],
    phases: dict[str, str] | None = None,
    vocabulary: tuple[str, ...] = PHASES,
    default_phase: str | None = 'attributes',
) -> Callable:
    """Return a decorator that registers a step under one or more names.

    Parameters
    ----------
    registry : dict
        The stage's own name-to-function mapping, filled in place.
    phases : dict, optional
        Where given, the returned decorator also accepts ``phase`` and
        records it here per name. Without it the decorator refuses a
        phase, so a stage that has no phases cannot be handed one by
        mistake.
    vocabulary : tuple of str
        The phases this stage's steps may declare. The harmonizer's
        two (geometry, attributes) are the default; the curator passes
        its own five, in pipeline order.
    default_phase : str or None
        The phase a step gets when it declares none. None makes the
        phase mandatory, which is what a stage whose recipes are
        checked for phase order wants: an untagged step there is a
        step nobody has placed.

    Returns
    -------
    Callable
        ``_register(*names, phase=...)``, used as ``@_register('name')``.
        It also takes the step's column declaration: ``reads`` and
        ``writes`` (column names, or templates naming the step's own
        parameters), ``decides`` (the written columns that are the
        final value of a registry column, a subset of ``writes``) and
        ``reads_from`` (a recipe-id parameter mapped to the columns
        read from that recipe's table) and ``drops`` (the columns the
        step removes, as names, templates, callables or a `DropRule`;
        a removal is not a write, so lineage never counts it as one).
        See `resolve_step_columns`.
    """

    def _register(
        *names: str,
        phase: str | None = None,
        reads=None,
        writes=None,
        decides=None,
        reads_from: Mapping | None = None,
        drops=None,
    ) -> Callable:
        declared = None
        given = (reads, writes, decides, reads_from, drops)
        if any(value is not None for value in given):
            declared = StepColumns(
                reads=_as_entries(reads),
                writes=_as_entries(writes),
                decides=_as_entries(decides),
                reads_from=dict(reads_from or {}),
                drops=_as_entries(drops),
            )
            misplaced = [
                entry
                for entry in declared.reads + declared.writes + declared.decides
                if isinstance(entry, DropRule)
            ]
            if misplaced:
                raise TypeError(
                    f'{names[0]!r}: a DropRule belongs in drops only, '
                    f'got {misplaced[0].name!r} elsewhere'
                )
            unwritten = [
                entry for entry in declared.decides if entry not in declared.writes
            ]
            if unwritten and '*' not in declared.writes:
                raise ValueError(
                    f'{names[0]!r}: decides lists entries missing from writes: '
                    f'{unwritten!r}'
                )

        if phases is None:
            if phase is not None:
                raise TypeError('this stage has no step phases')
        else:
            if phase is None:
                if default_phase is None:
                    raise TypeError(
                        f'{names[0]!r}: this stage requires a phase, one of '
                        f'{", ".join(vocabulary)}'
                    )
                phase = default_phase
            if phase not in vocabulary:
                raise ValueError(
                    f'phase must be one of {", ".join(vocabulary)}, got {phase!r}'
                )

        def decorator(fn: Callable) -> Callable:
            for name in names:
                # A second step under a taken name used to replace the
                # first without a word: an admin step registered as
                # 'resolve_overlaps' on 2026-09-30 displaced the
                # footprint step of that name, and the next run used
                # the footprint step on countries. The same function
                # imported again (a module reload) is not a clash.
                held = registry.get(name)
                if held is not None and _identity(held) != _identity(fn):
                    raise ValueError(
                        f'step {name!r} is already registered by '
                        f'{".".join(_identity(held))}'
                    )
                registry[name] = fn
                if phases is not None:
                    phases[name] = phase
            if declared is not None:
                setattr(fn, _COLUMNS_ATTRIBUTE, declared)
            return fn

        return decorator

    return _register


def make_loader(package: str, path) -> Callable[[], None]:
    """Return a function that imports a package's step submodules once.

    The import is deferred until a step is first dispatched, so merely
    importing a stage package (to use its state dataclass in a test, say)
    does not pull every step module. Subpackages are skipped, which keeps
    the enricher's heavy ``detectors`` out of the import.

    Parameters
    ----------
    package : str
        The stage package's ``__name__``.
    path
        The stage package's ``__path__``.
    """
    loaded = False

    def _load_steps() -> None:
        nonlocal loaded
        if loaded:
            return
        for module in pkgutil.iter_modules(path):
            if not module.ispkg:
                import_module(f'{package}.{module.name}')
        loaded = True

    return _load_steps
