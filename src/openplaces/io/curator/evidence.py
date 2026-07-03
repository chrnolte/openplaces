"""Curation steps for incorporating enrichment evidence."""

from __future__ import annotations

import pandas as pd

from openplaces.io import read_parquet
from openplaces.io.curator import CurateState, _register
from openplaces.recipe import get_output_path, get_recipe_by_id


def _field(obj, name):
    """Read *name* from *obj*, whether it is a dict or a schema object."""
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


@_register('link_curated_entity')
def link_curated_entity(
    state: CurateState,
    recipe_id: str,
    columns: dict,
    entity_key: str = 'parcel_id',
    ref_key: str = 'parcel_id',
) -> CurateState:
    """Join another curated entity's attributes onto the current one by a shared id.

    Reads the output of a curate-stage recipe and maps each named column onto
    the current entity (1:many ref -> entity) on the id key. This is how, e.g.,
    the footprint lane consumes the parcel curation lane's land-use decisions
    (the refined ``use_group_combined`` and the ``manufactured_home_park``
    flag), so the referenced entity's classification is finalized before the
    current one's is inferred. Existing columns are overwritten — the
    referenced curated value supersedes the raw harmonized one.

    Parameters
    ----------
    recipe_id : str
        A stage ``curate`` recipe whose output supplies the attributes.
    columns : dict
        Mapping of ``{ref_column: entity_column}``. The entity column is the
        name written onto ``state.curated`` (suffix it to mirror a harmonized
        evidence column, e.g. ``_parcel``).
    entity_key : str, optional
        Id column on the current entity (default ``parcel_id``). Unlike other
        cross-attributed columns, an id column keeps its bare name rather than
        the usual ``_parcel``-style suffix (see ``_attributed_name`` in the
        harmonizer) — the entity's own ``parcel_id`` already names the
        referenced row, so a suffix would be redundant. This is the
        referenced entity's globally-unique ``parcel_id`` (geo_id), not
        ``parcel_id_local``: the latter is only locally cross-comparable and
        can collide within a single admin unit, which would silently
        misattribute the joined columns to an unrelated parcel.
    ref_key : str, optional
        Id column on the referenced entity's own curated output (default
        ``parcel_id`` — that entity's own key, not a cross-attributed one).
    """
    ref_recipe = get_recipe_by_id(recipe_id)
    if ref_recipe.get('stage') != 'curate':
        raise ValueError(f"Reference recipe '{recipe_id}' must have stage 'curate'.")

    curated = state.curated
    if entity_key not in curated.columns:
        if state.verbose:
            print(f"  link_curated_entity: entity has no '{entity_key}'; skipping.")
        return state

    ref = read_parquet(get_output_path(ref_recipe, state.admin_id))
    if ref_key not in ref.columns and ref.index.name == ref_key:
        ref = ref.reset_index()
    if ref_key not in ref.columns:
        raise ValueError(f"Curated reference '{recipe_id}' has no '{ref_key}' column.")

    lookup = ref.dropna(subset=[ref_key]).drop_duplicates(ref_key)
    lookup.index = lookup[ref_key].astype('string')
    key = curated[entity_key].astype('string')
    for ref_col, entity_col in columns.items():
        if ref_col not in lookup.columns:
            raise ValueError(
                f"Column '{ref_col}' missing from curated reference '{recipe_id}'."
            )
        curated[entity_col] = key.map(lookup[ref_col])

    if state.verbose:
        matched = int(key.isin(set(lookup.index)).sum())
        print(
            f'  link_curated_entity: {matched:,}/{len(curated):,} rows '
            f'matched {recipe_id} ({len(columns)} columns).'
        )
    state.curated = curated
    return state


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
    from openplaces.io.curator.provenance import record_source

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

        # Provenance token: 'source_id-version' (e.g. 'brails-2026'), an explicit
        # spec 'source_label', or the recipe id as a fallback. The loaded recipe's
        # dataset is a DataSet object (attributes), but tolerate a raw dict too.
        ds = enrichment_recipe.get('dataset')
        source_id = _field(_field(ds, 'source'), 'source_id')
        version = _field(ds, 'version')
        token = recipe_spec.get('source_label')
        if not token:
            token = f'{source_id}-{version}' if source_id and version else source_id
        token = token or recipe_id

        for evidence_column, canonical_column in columns.items():
            if evidence_column not in evidence:
                raise ValueError(
                    f"Evidence column '{evidence_column}' is missing from "
                    f'{evidence_path}.'
                )
            values = evidence[evidence_column].reindex(curated.index)
            if canonical_column in curated:
                was_null = curated[canonical_column].isna()
                curated[canonical_column] = curated[canonical_column].combine_first(
                    values
                )
            else:
                was_null = pd.Series(True, index=curated.index)
                curated[canonical_column] = values
            filled = was_null & curated[canonical_column].notna()
            if filled.any():
                record_source(curated, canonical_column, filled, token)

    state.curated = curated
    return state
