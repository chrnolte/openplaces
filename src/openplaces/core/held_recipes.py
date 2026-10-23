"""Recipes kept out of runtime auto-discovery: held and run-excluded.

A *held* recipe is a connector to a source whose terms are under review
(Shovels, while Boston University reviews its terms). Its files may sit
on a machine, git-ignored, and an explicit `recipe_id:` reference may
still read it for an internal check, but it is never picked up as a
source by auto-discovery: not by a spine's union, not by the parcel
geospine's property join, not by any resolver that chooses a recipe for
an admin unit. Before this guard the parcel geospine's property join
read the held permit table, and its float ZIP filled the geospine's
`postal_code`, which the graph had never scheduled as an input:

Measured 2026-10-23 on US-NC-PEN: 8,266 of 55,100 parcels carried it.

A recipe is held when any of these is true:

- its `entity.source.source_id` is in `HELD_SOURCE_IDS`;
- its recipe id is in `HELD_RECIPE_IDS`;
- it declares `held: true` at its top level.

The two lists live here, in tracked code, rather than only in the
recipe, because the held recipe itself is git-ignored: a key inside it
travels with no checkout, and a fresh copy of the file (or a new state
scoped variant, `US-NC_property-shovels-2026`) would be discoverable
again. Holding by source id covers every version and scope of the
source at once. The `held: true` key is for a local connector that
has no entry here yet.

Git-ignore status is deliberately not consulted: an installed package
or an archive copy has no git, `git check-ignore` per file is a
subprocess per recipe in a scan that parses hundreds, and ignoring a
file in git says "do not publish this file", which is narrower than
"do not use this source". The hold is a decision, so it is written
down as one.

A *run-excluded* recipe is one the workflow run pruned from its graph
(`exclude_recipe_ids` in the Snakemake config). The DAG never schedules
it, and `flow.run_stage` passes the same ids to each job through
`EXCLUDE_ENV`, so the job's own discovery skips it too. Without that, a
job read whatever output of an excluded recipe was left on disk by an
earlier run: the graph and the job disagreed about the job's inputs.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping

#: Source ids whose recipes are held, whatever their version or scope.
#: Shovels: proprietary permit data, "internal use only"; its connector
#: is git-ignored pending Boston University's review of the terms
#: (plans/shovels-terms-and-the-withheld-connector.md).
HELD_SOURCE_IDS: frozenset[str] = frozenset({'shovels'})

#: Individual recipe ids that are held, for a source that is otherwise
#: free to discover.
HELD_RECIPE_IDS: frozenset[str] = frozenset()

#: Top-level recipe key that holds a recipe from inside its own file.
HELD_KEY = 'held'

#: Environment variable carrying the run's excluded recipe ids into a
#: job, comma-separated. Set by `flow.run_stage`.
EXCLUDE_ENV = 'OPENPLACES_EXCLUDE_RECIPE_IDS'


def _source_id(recipe: Mapping | None) -> str:
    """The source id a recipe declares, from raw YAML or a loaded recipe."""
    if not recipe:
        return ''
    entity = recipe.get('entity') or recipe.get('dataset')
    if entity is None:
        return ''
    if isinstance(entity, Mapping):
        source = entity.get('source') or {}
        source_id = (
            source.get('source_id')
            if isinstance(source, Mapping)
            else getattr(source, 'source_id', None)
        )
    else:
        source_id = getattr(getattr(entity, 'source', None), 'source_id', None)
    return str(source_id or '')


def is_held(
    recipe_id: str,
    recipe: Mapping | None = None,
    source_id: str | None = None,
) -> bool:
    """Whether a recipe is held from auto-discovery.

    Parameters
    ----------
    recipe_id : str
        The recipe's id (its file stem).
    recipe : Mapping, optional
        The recipe, as parsed YAML or as loaded; read for its `held`
        key and its source id.
    source_id : str, optional
        The recipe's source id, when the caller has it without the
        recipe (a row of the recipe index).

    Returns
    -------
    bool
        True if the recipe must never be discovered as a source.
    """
    if str(recipe_id) in HELD_RECIPE_IDS:
        return True
    if recipe is not None and recipe.get(HELD_KEY) is True:
        return True
    sid = source_id if source_id is not None else _source_id(recipe)
    return str(sid or '') in HELD_SOURCE_IDS


def run_excluded_recipe_ids() -> frozenset[str]:
    """The recipe ids the current run excluded, read from `EXCLUDE_ENV`.

    Read on every call, not cached: a test or a driver may set the
    variable after import.
    """
    raw = os.environ.get(EXCLUDE_ENV, '')
    return frozenset(r.strip() for r in raw.split(',') if r.strip())


def format_exclude_env(recipe_ids: Iterable[str]) -> str:
    """The `EXCLUDE_ENV` value for *recipe_ids*, sorted and comma-joined."""
    return ','.join(sorted({str(r) for r in recipe_ids if r}))


def is_undiscoverable(
    recipe_id: str,
    recipe: Mapping | None = None,
    source_id: str | None = None,
) -> bool:
    """Whether auto-discovery must skip a recipe: held or run-excluded.

    Parameters are those of `is_held`.
    """
    return is_held(recipe_id, recipe, source_id) or (
        str(recipe_id) in run_excluded_recipe_ids()
    )
