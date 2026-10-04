"""Attach source columns from established crosswalks as
suffixed evidence columns (reconcile_attributes), and
rename_columns.
"""

from __future__ import annotations

import warnings

import pandas as pd

from openplaces.io.harmonizer import HarmonizeState, _register
from openplaces.io.harmonizer.attributes._shared import (
    _POINT_REF_COLS,
)
from openplaces.io.harmonizer.attributes.point import (
    _attribute_point_reference,
)
from openplaces.io.harmonizer.attributes.polygon import (
    _attribute_polygon_reference,
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
            f'reconcile_attributes: crosswalk for {recipe_id!r} not in state; skipping.'
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


@_register('reconcile_attributes')
def reconcile_attributes(
    state: HarmonizeState,
    sources: list[dict] | None = None,
) -> HarmonizeState:
    """Aggregate reference attributes to the spine via established crosswalks.

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
                'reconcile_attributes: source entry has neither '
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
                    f'reconcile_attributes: crosswalk for '
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


@_register('rename_columns')
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
