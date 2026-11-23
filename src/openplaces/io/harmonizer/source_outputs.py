"""A spine source the unit is owed must have been ingested for it.

An empty ingest table counts as ingested only when it records why it is
empty (`readers.empty_reason`).

A spine step reads each of its sources with a tolerant read, because a
source can legitimately have no rows for a unit. That tolerance also let
a source that was never ingested for the unit pass as one without rows:
a parcel geospine built before one of its county's rolls was ingested
lacked that roll's values, and nothing said so. An ingest that ran for a
unit always leaves a file (rows, or an empty table for a unit the source
does not publish), so a missing file of a source that covers the unit is
an unfinished build, not an observation.
"""

from __future__ import annotations

from openplaces.core.held_recipes import is_undiscoverable
from openplaces.io.ingester.empty_outputs import (  # noqa: F401
    UnexplainedEmptySourceError,
)
from openplaces.io.ingester.scope import unbuilt_outputs


class SourceNotIngestedError(RuntimeError):
    """A spine source covering the unit has no ingest output for it."""


def require_source_ingested(recipe_id: str, admin_id, step: str) -> None:
    """Raise when a source covering *admin_id* has no output file for it.

    A source covers the unit when its admin scope contains the unit and
    its scoped crosswalk sidecar, if it has one, lists the unit: the
    test `RecipeDAG` applies before it schedules the unit's ingest. A
    source outside the unit, a held or run-excluded recipe and a recipe
    that cannot be loaded are not checked (the read that follows skips
    or reports them as before). A saved table without rows that records
    why (`readers.empty_reason`) passes: the unit was ingested and has
    none. One that records no reason raises, and a `marker` (which
    only says where the rows are) counts as missing.

    Parameters
    ----------
    recipe_id : str
        The ingest recipe the step is about to read (its primary table).
    admin_id : str or AdminId
        The unit being built.
    step : str
        The step reading the source, named in the error.

    Raises
    ------
    SourceNotIngestedError
        Naming the unit, the recipe, the step and the first missing file.
    UnexplainedEmptySourceError
        Naming the recipe, the unit and the file, for an output with no
        rows and no recorded reason.
    """
    if admin_id is None or is_undiscoverable(recipe_id):
        return
    # Imported at call time: tests patch `openplaces.recipe.get_recipe_by_id`
    # with fabricated recipes, which a module-level binding would not see.
    from openplaces.recipe import get_recipe_by_id

    try:
        recipe = get_recipe_by_id(recipe_id)
    except (OSError, ValueError, KeyError):
        return
    if (recipe.get('stage') or 'ingest') != 'ingest':
        # A harmonize placeholder records no reason; only ingest
        # outputs are held to one.
        return
    try:
        missing = unbuilt_outputs(recipe, admin_id)
    except UnexplainedEmptySourceError as err:
        raise UnexplainedEmptySourceError(f'{step}: {err}') from err
    if missing:
        raise SourceNotIngestedError(
            f'{step}: {recipe_id} covers {admin_id} but has no ingest output '
            f'for it ({len(missing)} file(s) missing, first: {missing[0]}). '
            'A missing file is not an observed "none": ingest the source for '
            'this unit before building its spine, or hold or exclude the '
            'recipe for the run.'
        )
