"""Curation steps for incorporating enrichment evidence."""

from __future__ import annotations

import pandas as pd

from openplaces.geo.link import get_entity_link_path
from openplaces.io import read_parquet
from openplaces.io.curator import CurateState, _register
from openplaces.io.harmonizer.apportion import (
    APPORTIONED_VALUE_COLUMNS,
    apportion_reference_values,
)
from openplaces.recipe import get_output_path, get_recipe_by_id, get_recipe_id


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


@_register('apportion_curated_values')
def apportion_curated_values(
    state: CurateState,
    recipe_id: str,
    columns: dict,
    entity_type: str = 'parcel',
    link_recipe_id: str | None = None,
    ref_key: str = 'parcel_id',
    entity_key: str = 'parcel_id',
    priority_column: str = 'priority_on_parcel',
    dwelling_column: str = 'n_dwellings_overture',
) -> CurateState:
    """Apportion a curated reference entity's values over the n:m link sidecar.

    The value-column counterpart of :func:`link_curated_entity`: instead of
    mapping each entity's dominant reference's undivided value, this re-uses
    the n:m overlay persisted by the harmonize stage (``link_to_reference``
    with ``save_link: true``) and the shared apportionment implementation
    (:func:`~openplaces.io.harmonizer.apportion.apportion_reference_values`)
    to distribute the *curated* reference's values exactly the way the
    harmonize stage distributes raw reference values: overlap-area shares
    for ``improvement_value``/``n_dwellings``, dwelling-linked suppression,
    ``land_value``/``address`` whole on the dominant reference of principal
    entities only, ``year_built`` as the linked references' mean.

    Synthetic reference-derived fallback geometries (``geometry_source``
    starting with ``'{entity_type}.'``, added by ``infer_spine_additions``
    *after* the overlay was persisted, so never present in the sidecar) are
    appended as full-weight single links via their ``entity_key`` column.

    Parameters
    ----------
    recipe_id : str
        A stage ``curate`` recipe whose output supplies the values.
    columns : dict
        Mapping of ``{ref_column: entity_column}``. Every ref column must be
        one the shared apportionment knows
        (``APPORTIONED_VALUE_COLUMNS``); use :func:`link_curated_entity`
        for plain dominant-reference attributes.
    entity_type : str, optional
        Reference entity type; resolves the link sidecar the same way the
        harmonize overlay did (default ``parcel``). Ignored when
        *link_recipe_id* is given.
    link_recipe_id : str, optional
        Explicit reference recipe id of the link's other side.
    ref_key : str, optional
        Id column on the referenced curated output (default ``parcel_id``).
    entity_key : str, optional
        Id column on the current entity holding its dominant reference id
        (default ``parcel_id``); used only to link synthetic fallback rows,
        which are absent from the sidecar.
    priority_column : str, optional
        Column holding each entity's role on its reference (default
        ``priority_on_parcel``); drives the secondary/principal rules.
    dwelling_column : str, optional
        Column whose positive value marks an entity as dwelling-linked
        (default ``n_dwellings_overture``); drives the suppression rule.

    Raises
    ------
    FileNotFoundError
        When the sidecar is missing: curation cannot recompute the overlay,
        so re-run harmonize with ``save_link: true`` first.
    """
    from openplaces.io.harmonizer.links import _resolve_reference_recipe

    unknown = [c for c in columns if c not in APPORTIONED_VALUE_COLUMNS]
    if unknown:
        raise ValueError(
            f'apportion_curated_values: no apportionment semantics for '
            f'{unknown}; supported: {list(APPORTIONED_VALUE_COLUMNS)}. '
            'Use link_curated_entity for dominant-reference attributes.'
        )

    ref_recipe = get_recipe_by_id(recipe_id)
    if ref_recipe.get('stage') != 'curate':
        raise ValueError(f"Reference recipe '{recipe_id}' must have stage 'curate'.")

    ref = read_parquet(get_output_path(ref_recipe, state.admin_id))
    if ref_key not in ref.columns and ref.index.name == ref_key:
        ref = ref.reset_index()
    if ref_key not in ref.columns:
        raise ValueError(f"Curated reference '{recipe_id}' has no '{ref_key}' column.")
    missing = [c for c in columns if c not in ref.columns]
    if missing:
        raise ValueError(
            f'Columns {missing} missing from curated reference {recipe_id!r}.'
        )
    ref_values = (
        ref.dropna(subset=[ref_key])
        .drop_duplicates(ref_key)
        .set_index(ref_key)[list(columns)]
    )

    resolved_id, _ = _resolve_reference_recipe(
        link_recipe_id, entity_type, state.admin_id
    )
    if resolved_id is None:
        raise ValueError(
            f'apportion_curated_values: no reference recipe found for '
            f'entity_type={entity_type!r} and {state.admin_id}.'
        )
    entity_recipe_id = get_recipe_id(state.entity_recipe)
    sidecar_path = get_entity_link_path(entity_recipe_id, resolved_id, state.admin_id)
    if not sidecar_path.exists():
        raise FileNotFoundError(
            f'Link sidecar not found: {sidecar_path}. Re-run harmonize for '
            f'{entity_recipe_id} with save_link: true on the overlay step '
            'so curation can apportion the curated values.'
        )

    links = pd.read_parquet(sidecar_path)
    curated = state.curated
    id_col = curated.index.name
    if id_col not in links.columns:
        # The sidecar records the harmonizer's working spine id column;
        # fall back to the first (index) column
        id_col = links.columns[0]

    # Only link-labeled pairs (kept by the harmonize crosswalk's sliver
    # thresholds) participate, matching what the harmonize attribution saw.
    if 'link' in links.columns:
        links = links[links['link'].notna()]
    pairs = links[[id_col, ref_key, 'area_intersection_m2']].rename(
        columns={ref_key: 'parcel_id'}
    )
    pairs = pairs[pairs[id_col].isin(curated.index)]

    # Synthetic fallback rows postdate the sidecar; add them as their
    # reference's sole full-weight link.
    if 'geometry_source' in curated.columns and entity_key in curated.columns:
        is_synthetic = (
            curated['geometry_source']
            .astype('string')
            .str.startswith(f'{entity_type}.', na=False)
        )
        synthetic = curated.loc[
            is_synthetic
            & curated[entity_key].notna()
            & ~curated.index.isin(set(pairs[id_col])),
            [entity_key],
        ]
        if len(synthetic):
            pairs = pd.concat(
                [
                    pairs,
                    pd.DataFrame(
                        {
                            id_col: synthetic.index,
                            'parcel_id': synthetic[entity_key].to_numpy(),
                            'area_intersection_m2': 1.0,
                        }
                    ),
                ],
                ignore_index=True,
            )

    result = apportion_reference_values(
        pairs,
        ref_values,
        spine_id_col=id_col,
        priority=curated.get(priority_column),
        dwelling_linked_ids=(
            set(curated.index[curated[dwelling_column] > 0])
            if dwelling_column in curated.columns
            else None
        ),
    )

    for ref_col, entity_col in columns.items():
        attributed = (
            result[ref_col] if ref_col in result.columns else pd.Series(dtype='float64')
        )
        curated[entity_col] = attributed.reindex(curated.index)

    if state.verbose:
        n_linked = curated.index.isin(set(pairs[id_col])).sum()
        print(
            f'  apportion_curated_values: {n_linked:,}/{len(curated):,} rows '
            f'linked to {recipe_id} ({len(columns)} columns).'
        )
    state.curated = curated
    return state


@_register('collect_link_ids')
def collect_link_ids(
    state: CurateState,
    entity_type: str | None = None,
    link_recipe_id: str | None = None,
    column: str = 'parcel_id_all',
    include_below_threshold: bool = False,
) -> CurateState:
    """Materialize all linked reference ids per entity from a link sidecar.

    Reads the n:m link sidecar persisted by the harmonize overlay
    (``link_to_reference`` with ``save_link: true``) and writes a
    pipe-joined id column onto the curated entity, ordered by descending
    intersection area — the dominant reference (e.g. the entity's
    ``parcel_id``) comes first. Harmonized spines stay link-only; this is
    where multi-reference membership becomes a canonical column.

    Parameters
    ----------
    entity_type : str, optional
        Auto-discover the reference recipe of this entity type for the
        current admin unit, the same way the harmonize overlay resolved
        it (so the sidecar filename matches per state/county). Ignored
        when ``link_recipe_id`` is given.
    link_recipe_id : str, optional
        Explicit reference recipe ID of the link's other side.
    column : str, optional
        Output column written onto the curated entity (default
        ``parcel_id_all``). Left missing where the collected value is
        identical to the base id column (e.g. ``parcel_id``).
    include_below_threshold : bool, optional
        When False (default), only link-labeled pairs (those kept by the
        harmonize crosswalk's sliver thresholds) are collected. When
        True, every raw overlap in the sidecar counts.

    Raises
    ------
    FileNotFoundError
        When the sidecar is missing: curation cannot recompute the
        overlay, so re-run harmonize with ``save_link: true`` first.
    """
    from openplaces.io.harmonizer.links import _resolve_reference_recipe

    resolved_id, _ = _resolve_reference_recipe(
        link_recipe_id, entity_type, state.admin_id
    )
    if resolved_id is None:
        if state.verbose:
            print(
                f'  collect_link_ids: no reference recipe for '
                f'entity_type={entity_type!r} and {state.admin_id}; skipping.'
            )
        return state

    entity_recipe_id = get_recipe_id(state.entity_recipe)
    sidecar_path = get_entity_link_path(entity_recipe_id, resolved_id, state.admin_id)
    if not sidecar_path.exists():
        raise FileNotFoundError(
            f'Link sidecar not found: {sidecar_path}. Re-run harmonize for '
            f'{entity_recipe_id} with save_link: true on the overlay step '
            'so curation can collect the n:m link ids.'
        )

    links = pd.read_parquet(sidecar_path)
    curated = state.curated
    id_col = curated.index.name
    if id_col not in links.columns:
        # The sidecar records the harmonizer's working spine id column;
        # fall back to the first (index) column
        id_col = links.columns[0]

    links = links.dropna(subset=['parcel_id'])
    if not include_below_threshold and 'link' in links.columns:
        links = links[links['link'].notna()]
    if 'area_intersection_m2' in links.columns:
        links = links.sort_values('area_intersection_m2', ascending=False)

    collected = links.groupby(id_col)['parcel_id'].agg(
        lambda s: '|'.join(s.dropna().astype(str).unique())
    )
    curated[column] = collected.where(collected != '').reindex(curated.index)

    # Like the harmonizer's `_join_distinct` convention, leave the `_all`
    # column missing where it would just repeat the base id.
    base_column = column.removesuffix('_all')
    if base_column != column and base_column in curated.columns:
        same = (
            curated[column].astype('string') == curated[base_column].astype('string')
        ).fillna(False)
        curated[column] = curated[column].mask(same)

    if state.verbose:
        n_multi = int(curated[column].str.contains(r'\|', na=False).sum())
        print(
            f'  collect_link_ids: {column} set for '
            f'{int(curated[column].notna().sum()):,}/{len(curated):,} rows '
            f'({n_multi:,} multi-reference).'
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
