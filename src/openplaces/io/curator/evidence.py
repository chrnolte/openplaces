"""Curation steps for incorporating enrichment evidence."""

from __future__ import annotations

import pandas as pd

from openplaces.io import read_parquet
from openplaces.io.curator import CurateState, _register
from openplaces.recipe import get_output_path, get_recipe_by_id


def _field(obj, name):
    """Read *name* from *obj*, whether it is a dict or a schema object."""
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


@_register('link_curated_parcels')
def link_curated_parcels(
    state: CurateState,
    recipe_id: str,
    columns: dict,
    on: str = 'parcel_id_local',
) -> CurateState:
    """Join curated parcel attributes onto footprints by a shared parcel id.

    Reads the output of a curated parcel recipe and maps each named column onto
    the footprints (1:many parcel -> footprints) on the *on* key. This is how the
    footprint lane consumes the parcel curation lane's land-use decisions
    (e.g. the refined ``use_group_combined`` and the ``manufactured_home_park``
    flag), so the parcel-level classification is finalized before footprint
    occupancy is inferred. Existing footprint columns are overwritten — the
    curated parcel value supersedes the raw harmonized one.

    Parameters
    ----------
    recipe_id : str
        A stage ``curate`` parcel recipe whose output supplies the attributes.
    columns : dict
        Mapping of ``{parcel_column: footprint_column}``. The footprint column is
        the name written onto ``state.curated`` (suffix it with ``_parcel`` where
        it mirrors a harmonized parcel evidence column).
    on : str, optional
        Shared id key on both sides (default ``parcel_id_local``).
    """
    parcel_recipe = get_recipe_by_id(recipe_id)
    if parcel_recipe.get('stage') != 'curate':
        raise ValueError(f"Parcel recipe '{recipe_id}' must have stage 'curate'.")

    curated = state.curated
    if on not in curated.columns:
        if state.verbose:
            print(f"  link_curated_parcels: footprints have no '{on}'; skipping.")
        return state

    parcels = read_parquet(get_output_path(parcel_recipe, state.admin_id))
    if on not in parcels.columns:
        raise ValueError(f"Curated parcels '{recipe_id}' have no '{on}' column.")

    lookup = parcels.dropna(subset=[on]).drop_duplicates(on)
    lookup.index = lookup[on].astype('string')
    key = curated[on].astype('string')
    for parcel_col, fp_col in columns.items():
        if parcel_col not in lookup.columns:
            raise ValueError(
                f"Column '{parcel_col}' missing from curated parcels '{recipe_id}'."
            )
        curated[fp_col] = key.map(lookup[parcel_col])

    if state.verbose:
        matched = int(key.isin(set(lookup.index)).sum())
        print(
            f'  link_curated_parcels: {matched:,}/{len(curated):,} footprints '
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
