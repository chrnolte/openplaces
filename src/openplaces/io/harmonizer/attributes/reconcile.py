"""Attach source columns from established crosswalks as
suffixed evidence columns (add_source_evidence), and
rename_columns.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from functools import cache

import pandas as pd

from openplaces.core.attribute_registry import get_data_type
from openplaces.io.harmonizer import HarmonizeState, _register
from openplaces.io.harmonizer.apportion import APPORTIONED_VALUE_COLUMNS
from openplaces.io.harmonizer.attributes._shared import (
    _ID_COLUMN,
    _POINT_REF_COLS,
    _POLYGON_REF_COLS,
)
from openplaces.io.harmonizer.attributes.point import (
    _attribute_point_reference,
)
from openplaces.io.harmonizer.attributes.polygon import (
    _attribute_polygon_reference,
)
from openplaces.io.steps import RECIPE_ARGUMENT
from openplaces.recipe import (
    get_recipe_by_id,
    resolve_attribute_name,
    source_id_from_recipe_id,
)


def _collect_dwelling_linked(state: HarmonizeState) -> set:
    """Return spine IDs linked to a dwelling point (single_dwelling_point source).

    Used by :func:`_attribute_polygon_reference` to implement Lochhead et al.
    (2026) Table 4: when distributing parcel attributes across multiple
    footprints on the same parcel, restrict attribution to dwelling-linked
    footprints when any footprint in the parcel has dwelling evidence.
    """
    from openplaces.core.schema import SourceGeometryType as _SGT

    if state.spine is None:
        return set()
    spine_id_col = state.spine.index.name
    ids: set = set()
    for rid, sgt in state.source_geometry_types.items():
        if sgt != _SGT.single_dwelling_point:
            continue
        cw = state.crosswalks.get(rid)
        if cw is not None and spine_id_col in cw.columns:
            ids.update(cw[spine_id_col].dropna().unique())
    return ids


def _attribute_absent_point_reference(
    state: HarmonizeState,
    recipe_id: str,
    entity_type: str | None,
    columns: list[str] | None,
    collect_ids: bool = False,
) -> HarmonizeState:
    """Write a zero-coverage point reference's evidence columns as nulls.

    A point link whose reference had no rows for this admin unit leaves no
    crosswalk behind, and the attribution loop used to emit nothing at all:
    a rural county's spine then lacked n_dwellings_overture and its
    siblings while every neighbor carried them. The columns are written
    here instead, null, and the match count 0, matching the enricher's
    contract that a declared column is always present so a missing one
    reads as a recipe error rather than as a coverage gap.

    Only a point reference is handled. A polygon overlay's evidence is
    computed from the reference geometry itself (apportioned values,
    overlap fractions), which cannot be named without loading it, and its
    absence warns instead.

    Parameters
    ----------
    state : HarmonizeState
        Current pipeline state; its spine is written in place.
    recipe_id : str
        Reference recipe the recipe declared and no crosswalk resolved.
    entity_type : str or None
        Entity type the source entry declared, if any.
    columns : list of str or None
        Declared columns; the standard point columns when unset.
    collect_ids : bool, default False
        Forwarded to :func:`_attribute_point_reference`.

    Returns
    -------
    HarmonizeState
        The state, with the reference's evidence columns present.
    """
    sgt = state.source_geometry_types.get(recipe_id)
    ref_entity_type = state.reference_types.get(recipe_id, entity_type)
    if sgt is None or not str(sgt).endswith('point'):
        warnings.warn(
            f'add_source_evidence: crosswalk for {recipe_id!r} not in state; skipping.'
        )
        return state

    spine_id_col = state.spine.index.name
    # Deliberately column-less apart from the join key: every
    # aggregation in _attribute_point_reference then finds nothing to
    # do, and the declared columns are written by its closing null pass.
    empty = pd.DataFrame({spine_id_col: pd.Series(dtype=object)})
    return _attribute_point_reference(
        state,
        recipe_id,
        ref_entity_type,
        empty,
        list(columns) if columns else list(_POINT_REF_COLS),
        collect_ids=collect_ids,
    )


def _entity_type_of(recipe) -> str | None:
    """A recipe's entity type, parsed YAML or loaded, or None."""
    entity = recipe.get('entity') if isinstance(recipe, Mapping) else None
    if isinstance(entity, Mapping):
        entity_type = entity.get('entity_type')
    else:
        entity_type = getattr(entity, 'entity_type', None)
    return str(entity_type) if entity_type is not None else None


def _reference_entity_type(recipe_id: str) -> str:
    """The entity type a link step derives from a reference recipe id."""
    # The rule of links.spatial._resolve_reference_recipe, restated:
    # importing it here would pull the link modules into a declaration.
    return recipe_id.split('_')[-1].split('-')[0]


def _link_joins(pipeline, follow: Callable) -> dict:
    """The join each link step of *pipeline* builds, by reference.

    Keys are a reference recipe id or `('entity_type', <type>)`;
    values are `'spatial_overlay'` or `'spatial_point'`. *follow* is
    called with each recipe id whose links the pipeline restores.
    """
    joins: dict = {}
    for entry in pipeline or ():
        if not isinstance(entry, Mapping) or entry.get('enabled', True) is False:
            continue
        step = entry.get('step')
        if step == 'link_to_reference':
            join = entry.get('join', 'spatial_overlay')
            if entry.get('recipe_id'):
                joins[entry['recipe_id']] = join
            elif entry.get('entity_type'):
                joins[('entity_type', entry['entity_type'])] = join
        elif step == 'split_outlines_at_parcel_lines':
            joins[('entity_type', entry.get('entity_type', 'parcel'))] = (
                'spatial_overlay'
            )
        elif step == 'load_links' and entry.get('links_recipe_id'):
            joins = {**follow(entry['links_recipe_id']), **joins}
    return joins


@cache
def _restored_joins(recipe_id: str) -> dict:
    """The joins a recipe's output carries, its entity recipe's included."""
    try:
        recipe = get_recipe_by_id(recipe_id)
    except (ValueError, KeyError, FileNotFoundError):
        return {}
    return _recipe_joins(recipe)


def _recipe_joins(recipe) -> dict:
    """The joins a recipe builds or restores before its attribute steps."""
    joins = {}
    if recipe.get('entity_recipe'):
        joins.update(_restored_joins(recipe['entity_recipe']))
    joins.update(_link_joins(recipe.get('pipeline'), _restored_joins))
    return joins


def _reconcile_sources(arguments) -> list[dict]:
    """Each source entry resolved to what names its evidence columns.

    A recipe id names its source and entity type; an `entity_type`
    entry is resolved per unit at run time, so its source id is `*`.
    The join is the one a link step of the recipe (or of the recipe
    whose links it restores) builds; without the recipe, or with no
    such step, a point join, which is what the step assumes of a
    reference that has no crosswalk.
    """
    recipe = arguments.get(RECIPE_ARGUMENT)
    joins = _recipe_joins(recipe) if isinstance(recipe, Mapping) else {}
    spine_entity_type = _entity_type_of(recipe)
    resolved = []
    for source in arguments.get('sources') or ():
        recipe_id = source.get('recipe_id')
        entity_type = source.get('entity_type')
        if recipe_id is not None:
            entity_type = entity_type or _reference_entity_type(recipe_id)
            source_id = source_id_from_recipe_id(recipe_id)
            join = joins.get(recipe_id) or joins.get(('entity_type', entity_type))
        elif entity_type is not None:
            source_id = '*'
            join = joins.get(('entity_type', entity_type))
        else:
            continue
        columns = [c for c in source.get('columns') or () if isinstance(c, str)]
        resolved.append(
            {
                'source_id': source_id,
                'entity_type': entity_type,
                'spine_entity_type': spine_entity_type or '*',
                'join': join or 'spatial_point',
                'columns': columns,
                'collect_ids': bool(source.get('collect_ids')),
            }
        )
    return resolved


def _polygon_suffix(source: dict) -> str:
    """`_resolve_suffix` for a declared source (see `_shared`)."""
    entity_type = source['entity_type']
    if entity_type == source['spine_entity_type']:
        return f'_{source["source_id"]}'
    if entity_type == 'parcel':
        return '_parcel'
    return f'_{entity_type}_{source["source_id"]}'


def _polygon_counts(source: dict) -> list[str]:
    """The two relational counts of a polygon overlay."""
    suffix = _polygon_suffix(source)
    label = suffix.lstrip('_')
    entity_type = source['entity_type']
    plural = (
        entity_type + 's' + label[len(entity_type) :]
        if label.startswith(entity_type)
        else f'{label}s'
    )
    spine = source['spine_entity_type']
    return [f'n_{plural}_per_{spine}', f'n_{spine}s_per_{label}']


def _is_numeric_attribute(column: str) -> bool:
    """Whether the registry types *column* as a number."""
    data_type = get_data_type(resolve_attribute_name(column))
    return data_type in ('int', 'float', 'bool', 'Int64', 'int64', 'float64')


def _polygon_writes(source: dict) -> list[str]:
    """The evidence columns `_attribute_polygon_reference` writes."""
    suffix = _polygon_suffix(source)
    label = suffix.lstrip('_')
    columns = source['columns'] or list(_POLYGON_REF_COLS)

    def named(column: str) -> str:
        # An id column is written bare (`_attributed_name`); the step
        # suffixes it only where the spine already holds the name.
        same_entity = source['entity_type'] == source['spine_entity_type']
        if _ID_COLUMN.match(column) and not same_entity:
            return column
        return f'{column}{suffix}'

    names = [f'{label}_id_all', f'overlap_fraction{suffix}']
    if source['entity_type'] == 'parcel':
        names += [named('parcel_id'), f'area_intersection_m2{suffix}']
    names += [f'{c}{suffix}' for c in columns if c in APPORTIONED_VALUE_COLUMNS]
    skipped = {'use_group', 'use_subgroup', 'purpose_group', 'purpose_subgroup'}
    for column in columns:
        if column in skipped or column in APPORTIONED_VALUE_COLUMNS:
            continue
        names.append(named(column))
        # Only a column whose values are not numbers gets the summary
        # of every value; the registry stands in for the data here.
        if not _is_numeric_attribute(column):
            names.append(f'{named(column)}_all')
    return names


def _point_writes(source: dict) -> list[str]:
    """The evidence columns `_attribute_point_reference` writes."""
    entity_type = source['entity_type']
    source_id = source['source_id']
    columns = source['columns'] or list(_POINT_REF_COLS)

    def named(column: str) -> str:
        # `_point_suffix` with its column, and the n_dwellings rename.
        if column == 'n_dwellings':
            return f'n_dwellings_{source_id}'
        if entity_type in column:
            return f'{column}_{source_id}'
        return f'{column}_{entity_type}_{source_id}'

    names = [named(c) for c in columns]
    count = f'n_{entity_type}s_{source_id}'
    if count not in names:
        names.append(count)
    has_value = any('structure_value' in c for c in columns)
    group = next(
        (c for c in columns if 'purpose_subgroup' in c or 'occupancy_type' in c),
        None,
    )
    if has_value:
        names += [f'{named(c)}_all' for c in (group, 'group') if c in columns]
    if source['collect_ids']:
        # The reference's index, an ingest table's `{entity_type}_id`.
        names.append(f'{entity_type}_id')
    return names


def _reconcile_writes(arguments) -> list[str]:
    """Every evidence column `add_source_evidence` writes, but the
    relational counts (`_reconcile_counts`)."""
    names = []
    for source in _reconcile_sources(arguments):
        if source['join'] == 'spatial_overlay':
            names += _polygon_writes(source)
        else:
            names += _point_writes(source)
    return names


def _reconcile_counts(arguments) -> list[str]:
    """The relational counts of each polygon overlay. Each is rebuilt
    on every spine row, so the step decides them."""
    return [
        name
        for source in _reconcile_sources(arguments)
        if source['join'] == 'spatial_overlay'
        for name in _polygon_counts(source)
    ]


# What this step writes depends on its sources, on whether the recipe
# links each one as points or as an overlay, and on the spine's entity
# type: the callables read the last two off the recipe
# (`steps.RECIPE_ARGUMENT`). A name that hangs on a unit's own
# reference (an `entity_type` source) carries `*` for its source id.
@_register(
    'add_source_evidence',
    writes=[_reconcile_writes, _reconcile_counts],
    decides=[_reconcile_counts],
    deprecated=('reconcile_attributes',),
)
def add_source_evidence(
    state: HarmonizeState,
    sources: list[dict] | None = None,
) -> HarmonizeState:
    """Add each linked source's suffixed evidence columns to the spine.

    No existing column is changed. Each value is formed from the source
    rows linked to the spine row: a sum, a mean, the dominant class, a
    parcel's share by overlap or a link count; a value is copied
    unchanged only where one source row links to it.

    For each source in *sources*, looks up the crosswalk in
    ``state.crosswalks`` (resolved via ``recipe_id`` or ``entity_type``) and
    aggregates the requested columns to the spine as source-suffixed evidence
    columns (e.g. ``improvement_value_parcel``, ``occupancy_type_building_nsi``).

    This step only attributes evidence; between-source value selection,
    gap-filling, and occupancy inference now run in the curation stage
    (see ``openplaces.io.curator``).

    Parameters
    ----------
    sources : list of dict
        Each dict describes one reference source and may contain:

        ``recipe_id`` (str, optional)
            Explicit crosswalk key in ``state.crosswalks``.
        ``entity_type`` (str, optional)
            Selects all matching crosswalks via ``state.reference_types``;
            used when ``recipe_id`` is absent.
        ``columns`` (list of str, optional)
            Columns to aggregate.  Defaults to all available columns from
            the corresponding default column list.
        ``remap_id`` (str, optional)
            Recipe id of a two-column value crosswalk (raw -> canonical) applied
            in place to the matching reference column before aggregation (e.g.
            canonicalizing FEMA ``occupancy_type`` via its occupancy-type-remap).
    """
    if state.spine is None or not sources:
        return state

    dwelling_linked_ids = _collect_dwelling_linked(state)

    for src_cfg in sources:
        entity_type = src_cfg.get('entity_type')
        recipe_id = src_cfg.get('recipe_id')
        columns = src_cfg.get('columns')

        if recipe_id is not None:
            crosswalk_keys = [recipe_id] if recipe_id in state.crosswalks else []
            if not crosswalk_keys:
                state = _attribute_absent_point_reference(
                    state,
                    recipe_id,
                    entity_type,
                    columns,
                    collect_ids=src_cfg.get('collect_ids', False),
                )
                continue
        elif entity_type is not None:
            crosswalk_keys = list(state.get_crosswalks_by_type(entity_type).keys())
        else:
            warnings.warn(
                'add_source_evidence: source entry has neither '
                "'recipe_id' nor 'entity_type'; skipping."
            )
            continue

        src_thresholds: dict = src_cfg.get('thresholds') or {}
        remap_id: str | None = src_cfg.get('remap_id')

        for crosswalk_key in crosswalk_keys:
            ref_entity_type = state.reference_types.get(crosswalk_key, entity_type)
            crosswalk = state.crosswalks.get(crosswalk_key)
            if crosswalk is None:
                warnings.warn(
                    f'add_source_evidence: crosswalk for '
                    f'{crosswalk_key!r} not in state; skipping.'
                )
                continue

            if isinstance(crosswalk.index, pd.MultiIndex):
                ref_polys = state.references.get(crosswalk_key)
                state = _attribute_polygon_reference(
                    state,
                    crosswalk_key,
                    ref_entity_type,
                    ref_polys,
                    columns,
                    thresholds=src_thresholds,
                    dwelling_linked_ids=dwelling_linked_ids,
                    remap_id=remap_id,
                )
            else:
                state = _attribute_point_reference(
                    state,
                    crosswalk_key,
                    ref_entity_type,
                    crosswalk,
                    columns,
                    collect_ids=src_cfg.get('collect_ids', False),
                )

    if state.timer:
        state.timer.mark('Attribute')
    return state


# A rename moves values without computing any, so it writes the new
# names and decides none of them.
@_register('rename_columns', reads=['{columns:keys}'], writes=['{columns:values}'])
def rename_columns(state: HarmonizeState, columns: dict[str, str]) -> HarmonizeState:
    """Rename spine columns.

    A small, generic escape hatch for the rare case a later step would
    otherwise silently overwrite a column in place -- e.g. the parcel
    spine renames its raw, as-ingested ``address``/``city`` (attached bare
    by ``link_by_id``'s auto-discovered assessor join, the same convention
    ``resolve_spine``'s own ``keep_columns`` uses for a parcel's other
    native attributes) to ``address_original``/``city_original`` before
    ``reconcile_addresses`` runs, since that step's own output defaults to
    those same bare names.

    Parameters
    ----------
    columns : dict of {old name: new name}
        Missing source columns are skipped, the same "missing evidence is
        tolerated" convention used throughout this codebase.
    """
    if state.spine is None:
        return state
    present = {old: new for old, new in columns.items() if old in state.spine.columns}
    state.spine = state.spine.rename(columns=present)
    return state
