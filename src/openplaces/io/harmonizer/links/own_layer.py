"""A lot keeps the values of the parcel layer that draws it.

Where two parcel layers cover a unit (a county's own layer under a
statewide one), `resolve_spine` takes each lot's outline from one of
them and stamps its label in `geometry_source`. The id join of
`link_by_id` then copies every layer's values onto every lot it
matches. Under the coverage rule of `_write_prioritized` alone, the
layer joined later replaced the earlier one's values wherever it
covered half the spine, so a lot drawn by the county could carry the
statewide layer's improvement and the county's total (measured
2026-11-18 on US-NC-CAR: 63,023 lots).

The rule here (maintainer, 2026-11-18): on a lot drawn by a parcel
layer, that layer's values stand and another parcel layer only fills
an empty cell, for every column the join writes. The two layers stay
ordered first non-null; neither is weighted and nothing is voted.

The rule reads the reference's entity type, never a place. It covers
a reference whose rows are parcels: a parcel recipe's own table. A
property layer bundled in a parcel recipe (MassGIS's assessor table,
the stacked-units layer) and a roll (a property recipe) describe
properties, whose values a county states for a lot that a coarser
layer may draw, and they keep the coverage rule (US-TX-HAR: the HCAD
roll over the TxGIO layer).

A county whose own layer is older than the statewide one declares it
in its own ingest recipe, ``values_outdated: true``. On the lots that
layer draws, another parcel layer's stated values then replace its
own, and its own join only fills.
"""

from __future__ import annotations

import pandas as pd

from openplaces.io.harmonizer.links.combine import RowRule
from openplaces.recipe import source_id_from_recipe_id

#: The ingest recipe key a parcel layer declares its values stale by.
VALUES_OUTDATED_KEY = 'values_outdated'


def values_outdated(recipe: dict) -> bool:
    """Whether a parcel ingest recipe declares its values outdated.

    Parameters
    ----------
    recipe : dict
        The loaded ingest recipe.

    Returns
    -------
    bool
        The recipe's ``values_outdated`` (False when absent).

    Raises
    ------
    ValueError
        When the key is set to anything but true or false, or set
        true on a recipe whose rows are not parcels.
    """
    flag = recipe.get(VALUES_OUTDATED_KEY, False)
    name = recipe.get('recipe_id', 'recipe')
    if not isinstance(flag, bool):
        raise ValueError(
            f'{name}: {VALUES_OUTDATED_KEY} must be true or false, got {flag!r}.'
        )
    entity = recipe.get('entity')
    entity_type = getattr(entity, 'entity_type', None)
    if flag and entity_type is not None and str(entity_type) != 'parcel':
        raise ValueError(
            f'{name}: {VALUES_OUTDATED_KEY} applies to a parcel layer; this '
            f'recipe describes {entity_type} rows.'
        )
    return flag


def own_layer_rule(
    spine: pd.DataFrame,
    match: dict,
    matches: list[dict],
    spine_source_ids: set[str],
    entity_type: str,
) -> RowRule | None:
    """The rows where a parcel layer's join overwrites or only fills.

    Parameters
    ----------
    spine : DataFrame
        The parcel spine, carrying `geometry_source`.
    match : dict
        One discovered reference (`_discover_link_sources`).
    matches : list of dict
        Every reference discovered beside it, for the declarations.
    spine_source_ids : set of str
        The recipes `resolve_spine` drew outlines from.
    entity_type : str
        The entity type the discovery looked for.

    Returns
    -------
    RowRule or None
        None when the reference's rows are not parcels or the spine
        has no `geometry_source`; the coverage rule then decides.
        Otherwise the reference overwrites on the lots it draws and
        on lots drawn by a layer declaring its values outdated, and
        only fills on lots drawn by any other parcel layer and on its
        own lots when it declares itself outdated.
    """
    if (match['layer'] or entity_type) != 'parcel':
        return None
    if 'geometry_source' not in spine.columns:
        return None
    label = source_id_from_recipe_id(match['recipe_id'])
    layers = {source_id_from_recipe_id(r) for r in spine_source_ids}
    outdated = {
        source_id_from_recipe_id(m['recipe_id'])
        for m in matches
        if m['layer'] is None and m.get(VALUES_OUTDATED_KEY)
    }
    drawn_by = spine['geometry_source'].astype('string')
    own = drawn_by.eq(label).fillna(False).astype(bool)
    by_layer = drawn_by.isin(layers).fillna(False).astype(bool)
    stale = drawn_by.isin(outdated).fillna(False).astype(bool)
    if label in outdated:
        overwrite = stale & ~own
        fill = (by_layer & ~stale) | own
    else:
        overwrite = own | stale
        fill = by_layer & ~own & ~stale
    return RowRule(overwrite=overwrite, fill=fill)
