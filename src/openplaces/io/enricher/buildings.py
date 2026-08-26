"""
Registered enrich step: attach a reference building entity's attributes onto
footprints by polygon overlap. Works with any building entity recipe that
carries polygon geometry -- not tied to any one source -- so a precomputed
inventory (roof shape, foundation, construction, ...) can be reused instead
of re-deriving the same attributes from imagery.
"""

from __future__ import annotations

import pandas as pd

from openplaces.geo.polygon import overlay_polygons
from openplaces.io.enricher import EnrichState, _register
from openplaces.io.readers import get_entities
from openplaces.recipe import get_recipe_by_id


def _evidence_suffix(reference_recipe: dict) -> str:
    """Provenance suffix for evidence columns from *reference_recipe*.

    Entity level plus source (``_building_cheer``), following the column
    naming convention in AGENTS.md: footprint, building and dwelling are
    easily conflated, so a building-level reference has to say which it is
    as well as who produced it. This deliberately differs from the parcel
    enrichment's `_source_suffix`, which uses the version instead -- there
    the version names the data product, whereas here several sources
    publish an interchangeable ``v0``.
    """
    entity = reference_recipe['entity']
    return f'_{entity.entity_type}_{entity.source.source_id}'


@_register('enrich_footprints_from_reference_buildings')
def enrich_footprints_from_reference_buildings(
    state: EnrichState,
    reference_building_recipe_id: str | None = None,
    columns: list[str] | None = None,
    min_iou: float = 0.1,
    suffix: str | None = None,
    null_with: dict[str, str] | None = None,
) -> EnrichState:
    """Attach a reference building entity's attributes as footprint evidence.

    Each footprint takes its attributes from the single reference building it
    overlaps most, measured by intersection-over-union. IoU rather than plain
    intersection area because the two sides are independent renderings of the
    same buildings: a large reference building clipping the corner of a small
    footprint shares more area with it than the correct small building does,
    and only normalizing by the union rejects that.

    Missing reference coverage for an admin unit is tolerated (returns
    *state* unchanged), matching how the parcel crosswalk step and the
    image-based steps treat an admin unit their input does not cover.

    Parameters
    ----------
    state : EnrichState
        The recipe must set ``spine_geom: true``; matching needs geometry.
    reference_building_recipe_id : str, optional
        Recipe ID of the reference building entity; defaults to the enrich
        recipe's own ``reference_building_recipe_id`` field.
    columns : list of str, optional
        Reference columns to attach. Columns absent from the reference are
        skipped rather than raising, so one recipe can serve reference
        vintages that carry different attributes. Defaults to every
        non-geometry column.
    min_iou : float, default 0.1
        Minimum intersection-over-union for a match. The default is
        deliberately permissive: the two sides digitize the same buildings
        from different imagery, so a correct pair routinely lands well below
        a half (median 0.75 against the CHEER inventory, but a long tail
        below that from roof-vs-wall outlines and building-vs-complex
        splits). Raise it when a source is known to be well aligned.
    suffix : str, optional
        Evidence-column suffix; defaults to the reference recipe's entity
        level and source (e.g. ``_building_cheer``).
    null_with : dict, optional
        Map of column to gating column: the first is nulled wherever the
        second is null. A classifier confidence outlives the class it
        scores when that class is dropped on ingest, and a confidence with
        no class reads as certainty about the wrong thing.

    Returns
    -------
    EnrichState
        With one ``{column}{suffix}`` evidence column per attached column.
    """
    reference_recipe_id = reference_building_recipe_id or state.recipe.get(
        'reference_building_recipe_id'
    )
    if not reference_recipe_id:
        raise ValueError(
            'enrich_footprints_from_reference_buildings requires '
            "'reference_building_recipe_id'."
        )
    reference_recipe = get_recipe_by_id(reference_recipe_id)
    suffix = suffix or _evidence_suffix(reference_recipe)

    def _empty(reason: str) -> EnrichState:
        """Emit the declared columns as all-null rather than nothing.

        The evidence table's schema has to hold whether or not this admin
        unit has reference coverage: the curate stage skips a missing
        evidence file but treats a present one that lacks a declared
        column as a recipe error, so writing a column-less table here
        would poison every later curate run for the unit.
        """
        if state.verbose:
            print(f'  {reason}; writing empty {suffix} evidence.')
        if columns:
            state.evidence = pd.concat(
                [
                    state.evidence,
                    pd.DataFrame(
                        {
                            f'{c}{suffix}': pd.Series(pd.NA, index=state.evidence.index)
                            for c in columns
                        }
                    ),
                ],
                axis=1,
            )
        return state

    try:
        reference = get_entities(reference_recipe, state.admin_id, geom=True)
    except FileNotFoundError:
        return _empty(f'No {reference_recipe_id} for {state.admin_id}')
    if reference is None or reference.empty:
        return _empty(f'{reference_recipe_id} is empty for {state.admin_id}')
    if state.timer:
        state.timer.mark('load reference buildings')

    if 'geometry' not in getattr(state.spine, 'columns', ()):
        raise ValueError(
            'enrich_footprints_from_reference_buildings needs spine geometry; '
            "set 'spine_geom: true' in the enrichment recipe."
        )

    available = [c for c in (reference.columns) if c != 'geometry']
    requested = columns if columns is not None else available
    attach = [c for c in requested if c in available]
    if not attach:
        return _empty(f'{reference_recipe_id} carries none of the declared columns')

    # Overlay on geometry alone, then pull the attributes by index.
    # Carrying them through the overlay would collide wherever the spine
    # already has a column of the same name (n_stories, year_built and
    # area_m2 all recur).
    ref_geo = reference[['geometry']]
    if ref_geo.index.name == state.spine.index.name:
        # A reference of the spine's own entity type (NCDPS footprints
        # enriching the footprint spine) shares the index name, which
        # the overlay refuses as ambiguous. The reference's ids are only
        # used to pull attributes back out below, so a disambiguated
        # name is free.
        ref_geo = ref_geo.rename_axis(f'{ref_geo.index.name}_reference')
    pairs = overlay_polygons(state.spine[['geometry']], ref_geo, iou=True)
    if state.timer:
        state.timer.mark('overlay footprints against reference buildings')
    if pairs.empty:
        return _empty(f'no footprint overlaps any {reference_recipe_id} building')

    pairs = pairs[pairs['iou'] >= min_iou]
    if pairs.empty:
        return _empty(f'no overlap reaches min_iou={min_iou}')

    # One reference building per footprint: the best-overlapping one.
    best = pairs['iou'].groupby(level=0).idxmax()
    footprint_ids = best.index
    reference_ids = [key[1] for key in best.to_numpy()]

    values = reference.loc[reference_ids, attach]
    values.index = footprint_ids

    for column, gate in (null_with or {}).items():
        if column in values.columns and gate in values.columns:
            values.loc[values[gate].isna(), column] = None

    values = values.rename(columns={c: f'{c}{suffix}' for c in attach})

    if state.verbose:
        matched = len(values)
        print(
            f'  Matched {matched:,d} of {len(state.spine):,d} footprints '
            f'to {reference_recipe_id} (min_iou={min_iou})'
        )

    state.evidence = pd.concat(
        [state.evidence, values.reindex(state.evidence.index)], axis=1
    )
    return state
