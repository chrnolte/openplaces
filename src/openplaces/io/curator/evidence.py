"""Curation steps for incorporating enrichment evidence."""

from __future__ import annotations

from openplaces.io import read_parquet
from openplaces.io.curator import CurateState, _register
from openplaces.recipe import get_output_path, get_recipe_by_id


@_register('merge_enrichments')
def merge_enrichments(
    state: CurateState,
    recipes: list[dict],
) -> CurateState:
    """Merge enrichment evidence into canonical columns.

    Each recipe specification must contain ``recipe_id`` and a ``columns``
    mapping from evidence-column names to canonical-column names. Existing
    canonical values take precedence; enrichment fills missing values.
    """
    curated = state.curated

    for recipe_spec in recipes:
        recipe_id = recipe_spec.get('recipe_id')
        columns = recipe_spec.get('columns') or {}
        if not recipe_id:
            raise ValueError("Enrichment specifications require 'recipe_id'.")
        if not columns:
            raise ValueError(
                f"Enrichment specification '{recipe_id}' requires 'columns'."
            )

        enrichment_recipe = get_recipe_by_id(recipe_id)
        if enrichment_recipe.get('stage') != 'enrich':
            raise ValueError(f"Evidence recipe '{recipe_id}' must have stage 'enrich'.")
        evidence_path = get_output_path(
            enrichment_recipe,
            state.admin_id,
            entity_recipe_id=state.entity_recipe,
        )
        evidence = read_parquet(evidence_path)

        for evidence_column, canonical_column in columns.items():
            if evidence_column not in evidence:
                raise ValueError(
                    f"Evidence column '{evidence_column}' is missing from "
                    f'{evidence_path}.'
                )
            values = evidence[evidence_column].reindex(curated.index)
            if canonical_column in curated:
                curated[canonical_column] = curated[canonical_column].combine_first(
                    values
                )
            else:
                curated[canonical_column] = values

    state.curated = curated
    return state
