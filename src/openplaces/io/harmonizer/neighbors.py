"""Which neighboring units' inputs a spine reads, for one answer everywhere.

A building on a unit line is one building, but each unit is harmonized on
its own. So that both units reconcile it to one envelope, a footprint
geospine declaring `resolve_spine(neighbor_reads: true)` also reads, from
each touching unit, the same footprint ingests `resolve_spine` would read
there (see `spine._cut_at_unit_lines`). Three callers have to agree on
those inputs: the step that reads them, the link-sidecar fingerprint that
must go stale when one changes, and the orchestrator that schedules them.
They all call `neighbor_inputs`, which answers from the admin polygons and
the recipes only, never from which files exist on disk (a DAG that
schedules one thing and a run that reads another is the failure this
avoids).

Neighbors are the units of the same level whose polygons touch or
overlap the unit's, from the admin layer the units are built from. There
is no distance tolerance.

`OPENPLACES_NEIGHBOR_READS=0` in the environment turns the reads off for
a debug run, for the orchestrator and every job it starts alike, since a
job inherits the orchestrator's environment. A geospine built that way
records it in its sidecar fingerprints, so a later full run rebuilds it.
"""

from __future__ import annotations

import os
from functools import cache

from openplaces.core.schema import AdminId
from openplaces.io.readers import get_admin
from openplaces.recipe import (
    get_recipe_by_id,
    get_recipe_dependencies,
    get_recipe_id,
    source_id_from_recipe_id,
)

NEIGHBOR_READS_ENV = 'OPENPLACES_NEIGHBOR_READS'


def neighbor_reads_enabled() -> bool:
    """False when the environment turns neighbor reads off."""
    value = os.environ.get(NEIGHBOR_READS_ENV, '1').strip().lower()
    return value not in ('0', 'false', 'no', 'off')


def recipe_reads_neighbors(recipe: dict) -> bool:
    """Whether *recipe* has a `resolve_spine` step with `neighbor_reads`."""
    return any(
        isinstance(step, dict)
        and step.get('step') == 'resolve_spine'
        and step.get('neighbor_reads')
        for step in recipe.get('pipeline') or []
    )


@cache
def _level_units(country: str, level: int):
    """The country's units at *level*, with geometry, read once."""
    return get_admin([country], level=level, geom=True)


def unit_polygons(admin_id) -> tuple:
    """The unit's own polygon and the touching units' polygons."""
    unit = AdminId(str(admin_id))
    units = _level_units(str(unit.truncate_to_level(1)), unit.get_level())
    own = units.geometry[str(unit)]
    touching = units[units.intersects(own) & (units.index != str(unit))]
    return own, touching


@cache
def touching_units(admin_id: str) -> tuple[str, ...]:
    """Same-level units whose polygons touch or overlap *admin_id*'s."""
    try:
        _, touching = unit_polygons(admin_id)
    except (KeyError, ValueError, FileNotFoundError, OSError):
        return ()
    return tuple(sorted(touching.index))


def _unpatched(recipe: dict) -> dict:
    """The recipe file *recipe* was loaded from, before any patch.

    A recipe that names no recipe file (a test's bare dict) is returned
    as is; nothing can patch it.
    """
    try:
        return get_recipe_by_id(recipe.get('recipe_id') or get_recipe_id(recipe))
    except Exception:  # noqa: BLE001 - an unnamed recipe has no patches
        return recipe


def neighbor_inputs(recipe: dict, admin_id) -> list[tuple[str, str]]:
    """(ingest recipe id, neighbor admin id) pairs the recipe reads.

    Empty unless the recipe declares neighbor reads and the environment
    leaves them on. For each touching unit, the upstream recipes of the
    recipe's `resolve_spine` step as the dependency scan resolves them
    for that unit: the sources the step would read there.

    *recipe* may arrive patched for *admin_id* (the harmonizer passes
    the pipeline the unit runs) or not (the orchestrator passes the
    recipe file). Every unit is scanned from the unpatched recipe, so
    that each takes its own patches and not *admin_id*'s: scanned with
    a county's patch, a neighbor would seem to read a source the patch
    names, which the neighbor neither reads nor schedules.
    """
    if admin_id is None or not neighbor_reads_enabled():
        return []
    if not recipe_reads_neighbors(recipe):
        return []

    # The step's own exclusions (FEMA in the footprint geospine), which
    # the dependency scan does not model: a source the step never reads
    # is not scheduled for a neighbor either.
    excluded = {
        str(source_id)
        for step in recipe.get('pipeline') or []
        if isinstance(step, dict) and step.get('step') == 'resolve_spine'
        for entry in step.get('sources') or []
        if isinstance(entry, dict)
        for source_id in entry.get('exclude_sources') or ()
    }

    unpatched = _unpatched(recipe)

    def spine_sources(unit) -> set[str]:
        return {
            edge.upstream_recipe_id
            for edge in get_recipe_dependencies(unpatched, admin_id=unit)
            if edge.step == 'resolve_spine'
            and edge.upstream_recipe_id
            and source_id_from_recipe_id(edge.upstream_recipe_id) not in excluded
        }

    # Only sources the unit reads itself, so a neighbor's rows join a
    # source the unit ranks; a source only the neighbor has (a state
    # layer across a state line) is not read and not scheduled.
    own = spine_sources(str(admin_id))
    pairs = {
        (recipe_id, neighbor)
        for neighbor in touching_units(str(admin_id))
        for recipe_id in spine_sources(neighbor) & own
    }
    return sorted(pairs)
