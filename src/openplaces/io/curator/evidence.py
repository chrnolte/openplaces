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
    (the refined ``use_group_combined`` and the ``manufactured_home_community``
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
    land_use_column: str | None = None,
    non_residential_classes: list[str] | None = None,
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
    land_use_column : str, optional
        Classifier column on the curated reference (e.g. ``land_use_class``)
        used to widen the split for non-residential references -- see
        *non_residential_classes*. Read separately from *columns*: it is a
        classifier input, not one of ``APPORTIONED_VALUE_COLUMNS``, and is
        never itself apportioned. Ignored unless *non_residential_classes* is
        also given; when given but absent from the reference, skipped with a
        verbose message rather than raising (an entity recipe with no
        land-use classification just gets the existing residential-style
        behavior).
    non_residential_classes : list of str, optional
        *land_use_column* values identifying non-residential references,
        passed as ``equal_area_ref_ids`` to
        :func:`~openplaces.io.harmonizer.apportion.apportion_reference_values`
        -- every entity linked to one of these references keeps its full
        overlap-area share of
        :data:`~openplaces.io.harmonizer.apportion.PROPORTIONAL_SPLIT_COLUMNS`
        (``improvement_value``, ``land_value_imputed``,
        ``improvement_value_imputed``), instead of the residential
        dwelling-linked/secondary-structure exclusions: a warehouse plus its
        loading dock, or a retail strip's several units, are all real
        value-bearing structures with no single dwelling to anchor a
        residential-style split to.

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
    # A ref column can be legitimately absent when its source never provided
    # that attribute (e.g. a parcel source with no improvement_value split);
    # skip it -- apportion_reference_values already only produces columns
    # present in ref_values, and the final assignment loop below already
    # defaults an unapportioned column to missing.
    present = [c for c in columns if c in ref.columns]
    missing = [c for c in columns if c not in ref.columns]
    if missing and state.verbose:
        print(
            f'  apportion_curated_values: {missing} missing from curated '
            f"reference '{recipe_id}'; skipping."
        )
    ref_values = (
        ref.dropna(subset=[ref_key])
        .drop_duplicates(ref_key)
        .set_index(ref_key)[present]
    )

    equal_area_ref_ids: set | None = None
    if land_use_column and non_residential_classes:
        if land_use_column in ref.columns:
            land_use_lookup = (
                ref.dropna(subset=[ref_key])
                .drop_duplicates(ref_key)
                .set_index(ref_key)[land_use_column]
            )
            equal_area_ref_ids = set(
                land_use_lookup.index[land_use_lookup.isin(non_residential_classes)]
            )
        elif state.verbose:
            print(
                f'  apportion_curated_values: {land_use_column!r} missing from '
                f"curated reference '{recipe_id}'; skipping equal-area rule."
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
        equal_area_ref_ids=equal_area_ref_ids,
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
    An optional ``record_source`` (default ``False``) writes a ``{column}
    _source`` provenance sidecar for that spec's columns -- reserve this
    for genuinely canonical, possibly-multi-sourced columns (e.g. a value
    later reconciled against a second source); leave it off for a bulk,
    single-source enrichment batch (e.g. dozens of physical/environmental
    predictors from one model), where a sidecar per column is pure
    bookkeeping bloat with no downstream reconciliation to inform.

    A ``recipe_spec`` whose enrichment output doesn't exist yet for this admin
    is skipped rather than raising -- e.g. an imagery-dependent enrichment
    intentionally left unrun this pass (see
    ``notebooks/examples/US_curate_footprints.ipynb``'s ``--no_streetview``/
    ``--no_googlesatellite``, which skip billing-costly image ingestion and
    the enrichment steps that depend on it). A single requested column
    missing from evidence that *does* exist is tolerated the same way and
    skipped on its own (warned when verbose) -- e.g. a per-source predictor
    that simply was never computed for this admin unit's geography (a legacy
    reference dataset's SSURGO-derived soil columns, present for the states
    it originally covered, can be genuinely absent for a newly added one).
    Every *other* column in the same ``recipe_spec`` still merges normally.

    Every column this step touches -- across every ``recipe_spec``, values
    and source sidecars alike -- is collected in memory and written to
    ``curated`` in a single batched ``pd.concat`` at the end, rather than
    one ``curated[col] = ...`` assignment per column: a wide, already
    highly-columned entity table (e.g. the curated parcel table, 170+
    columns) fragments its internal block manager under dozens of
    one-at-a-time inserts, tripping pandas' own
    ``PerformanceWarning: DataFrame is highly fragmented`` -- exactly the
    pattern a large ``merge_enrichments`` call (e.g. ~45 columns from one
    enrichment source) produces.
    """
    from openplaces.io.curator.provenance import _apply_source_mask, source_column

    curated = state.curated
    pending: dict[str, pd.Series] = {}

    for recipe_spec in recipes:
        recipe_id = recipe_spec.get('recipe_id')
        columns = recipe_spec.get('columns') or {}
        want_source = recipe_spec.get('record_source', False)
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
        if not evidence_path.exists():
            if state.verbose:
                print(
                    f"  merge_enrichments: '{recipe_id}' has no evidence for "
                    f'{state.admin_id} (enrichment not run this pass?); skipping.'
                )
            continue
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
                if state.verbose:
                    print(
                        f"  merge_enrichments: '{evidence_column}' missing from "
                        f'{evidence_path.name} (not computed for this admin unit?); '
                        'skipping.'
                    )
                continue
            values = evidence[evidence_column].reindex(curated.index)

            # A prior recipe_spec in this same call may have already
            # queued a value for this column -- read that first so
            # declaration order still acts as a priority chain (a later
            # spec only fills rows still missing), matching the
            # pre-batching combine_first behavior exactly.
            current = pending.get(canonical_column)
            if current is None and canonical_column in curated.columns:
                current = curated[canonical_column]
            if current is not None:
                was_null = current.isna()
                merged = current.combine_first(values)
            else:
                was_null = pd.Series(True, index=curated.index)
                merged = values
            pending[canonical_column] = merged

            if want_source:
                filled = was_null & merged.notna()
                if filled.any():
                    side = source_column(canonical_column)
                    side_existing = pending.get(side)
                    if side_existing is None and side in curated.columns:
                        side_existing = curated[side]
                    pending[side] = _apply_source_mask(
                        side_existing, curated.index, filled, token
                    )

    if pending:
        new_cols = pd.DataFrame(pending, index=curated.index)
        overlap = [c for c in new_cols.columns if c in curated.columns]
        if overlap:
            curated = curated.drop(columns=overlap)
        curated = pd.concat([curated, new_cols], axis=1)

    state.curated = curated
    return state
