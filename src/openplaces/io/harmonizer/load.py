"""
Pipeline step that restores a geospine recipe's results into a fresh
HarmonizeState, so an attribute-only recipe can run downstream of the
expensive geometry phase without recomputing any of it.

The geometry/attribute recipe split: a geospine recipe (e.g.
US_footprint-geospine-2026) runs the geometry-phase steps -- spine
resolution, spatial overlays and point joins, overlap resolution -- and
persists its spine plus one link sidecar per join. The attribute recipe
(keeping the established id, e.g. US_footprint-spine-2026) declares the
geospine as its `entity_recipe` and starts its pipeline with
`load_geospine`, which reloads the spine and rebuilds the crosswalks,
overlays, and prepared references from the persisted link tables --
never from a new spatial computation.
"""

from __future__ import annotations

import json

import pandas as pd

from openplaces.core.schema import SourceGeometryType
from openplaces.geo.link import get_entity_link_path
from openplaces.io.aggregate import read_file_metadata
from openplaces.io.harmonizer import (
    HARMONIZE_METADATA_KEY,
    HarmonizeState,
    _register,
)
from openplaces.io.harmonizer.links import (
    _build_crosswalk,
    _link_fingerprint,
    _load_link_sidecar,
    _load_point_link_sidecar,
    _prepare_reference,
    _resolve_reference_recipe,
    snap_chained_links,
)
from openplaces.io.readers import get_entities
from openplaces.recipe import (
    get_output_path,
    get_recipe_by_id,
    get_recipe_id,
    get_save_admin_level,
    raise_if_coverage_complete,
    source_id_from_recipe_id,
)


@_register('load_geospine')
def load_geospine(state: HarmonizeState, entity_recipe_id: str | None = None):
    """Restore the geospine recipe's spine and link products into *state*.

    Reads everything from what the geospine recipe persisted -- its
    entity output (attributes + `_geo` sidecar), its footer-carried
    pipeline metadata, and one link sidecar per `link_to_reference`
    step -- and rebuilds the state the geometry phase would have left
    behind: `spine`, `crosswalks`, `overlays`, prepared `references`
    (overlay links only; nothing downstream reads a point reference),
    `reference_types`, `source_geometry_types`, and the
    `inferred_from_{recipe_id}` metadata frames `reconcile_attributes`
    uses to attribute parcel-inferred spine rows directly.

    Which links to restore, and with what configuration, is read from the
    geospine recipe's own pipeline -- deliberately not re-declared in the
    attribute recipe, so the two YAMLs cannot drift. Each sidecar's footer
    fingerprint is recomputed exactly as the geospine run computed it
    (same recipe, same step position, same config subset); a missing or
    mismatched sidecar raises with instructions to rerun the geospine
    recipe rather than silently recomputing geometry here.

    Parameters
    ----------
    entity_recipe_id : str or dict, optional
        Geospine recipe (id or loaded dict) to load. Defaults to the
        recipe's own `entity_recipe` key.
    """
    geospine = entity_recipe_id or state.recipe.get('entity_recipe')
    if not geospine:
        raise ValueError(
            'load_geospine needs an entity_recipe: declare the geospine '
            "recipe under the attribute recipe's `entity_recipe` key."
        )
    if isinstance(geospine, str):
        geospine = get_recipe_by_id(geospine)
    geospine_id = get_recipe_id(geospine)

    spine = get_entities(geospine, state.admin_id, geom=True, missing='raise')
    if state.verbose:
        print(f'  Load (geospine): {len(spine):,d} {geospine_id} entities')

    _restore_handoff_metadata(state, geospine)

    # Mirror resolve_spine's working-index rename: overlay crosswalks name
    # the reference level 'parcel_id', so a parcel spine's own index must
    # not use that name while the pipeline runs. The harmonizer's save
    # step restores the original from metadata['spine_index_name'].
    if spine.index.name == 'parcel_id':
        state.metadata['spine_index_name'] = spine.index.name
        spine.index = spine.index.rename('spine_id')
    state.spine = spine
    spine_id_col = spine.index.name

    for step_index, step_cfg in enumerate(geospine.get('pipeline') or []):
        if not isinstance(step_cfg, dict):
            continue
        if step_cfg.get('step') != 'link_to_reference':
            continue
        if step_cfg.get('save_link') is False:
            continue
        state = _restore_link(state, geospine, step_index, step_cfg, spine_id_col)

    if state.timer:
        state.timer.mark('Load (geospine)')
    return state


def _restore_handoff_metadata(state: HarmonizeState, geospine: dict) -> None:
    """Restore the footer-carried pipeline metadata of the geospine output.

    The geospine's save step wrote spine_index_name,
    spine_source_recipe_ids, and spine_keep_columns into its attribute
    parquet's footer (in-memory HarmonizeState.metadata does not survive
    the recipe split); `link_by_id(auto_discover=True)` in the attribute
    recipe needs the two sets to protect per-geometry keep_columns from
    being pooled. An output written before the split carries no footer
    entry -- restore nothing and leave the defaults.
    """
    admin_id = state.admin_id
    if admin_id is not None:
        admin_id = admin_id.truncate_to_level(get_save_admin_level(geospine))
    path = get_output_path(geospine, admin_id)
    stored_raw = read_file_metadata(path).get(HARMONIZE_METADATA_KEY)
    if stored_raw is None:
        return
    try:
        handoff = json.loads(stored_raw)
    except json.JSONDecodeError:
        return
    if handoff.get('spine_index_name') is not None:
        state.metadata['spine_index_name'] = handoff['spine_index_name']
    for key in ('spine_source_recipe_ids', 'spine_keep_columns'):
        if handoff.get(key):
            state.metadata[key] = set(handoff[key])


def _restore_link(
    state: HarmonizeState,
    geospine: dict,
    step_index: int,
    step_cfg: dict,
    spine_id_col: str,
) -> HarmonizeState:
    """Rebuild one link_to_reference step's state from its sidecar."""
    resolved_id, resolved_et = _resolve_reference_recipe(
        step_cfg.get('recipe_id'), step_cfg.get('entity_type'), state.admin_id
    )
    if resolved_id is None:
        if state.verbose:
            print(
                '  Load (geospine): no reference recipe for '
                f'entity_type={step_cfg.get("entity_type")!r}; skipping link.'
            )
        return state

    join = step_cfg.get('join', 'spatial_overlay')
    thresholds = step_cfg.get('thresholds') or {}
    sidecar_path = get_entity_link_path(
        get_recipe_id(geospine), resolved_id, state.admin_id
    )

    # Recompute the fingerprint exactly as the geospine run computed it:
    # same recipe, same pipeline position (prior geometry-phase steps),
    # same step-config subset. A stale sidecar then fails closed below.
    shim = HarmonizeState(
        recipe=geospine, admin_id=state.admin_id, verbose=False, timer=None
    )
    shim.step_index = step_index

    if join == 'spatial_overlay':
        min_fraction = thresholds.get('min_fraction_of_largest', 1 / 6)
        area_min_m2 = thresholds.get('area_intersection_m2_min', 10)
        aggregation_function = step_cfg.get('aggregation_function')
        fingerprint = _link_fingerprint(
            shim,
            resolved_id,
            {
                'min_fraction_of_largest': min_fraction,
                'area_intersection_m2_min': area_min_m2,
                'sort_by': step_cfg.get('sort_by'),
                'list_columns': step_cfg.get('list_columns'),
                'aggregation_function': (
                    None if aggregation_function is None else str(aggregation_function)
                ),
            },
        )
        overlay = _load_link_sidecar(
            sidecar_path, fingerprint, spine_id_col, verbose=state.verbose
        )
        ref_raw = get_entities(resolved_id, state.admin_id, geom=True, missing='warn')
        if ref_raw is None or len(ref_raw) == 0:
            # The geospine run skipped this link for the same reason (an
            # expected admin-scoped coverage gap), so there is nothing to
            # restore -- unless a sidecar exists, which means the input
            # disappeared after the link ran. A reference declaring
            # complete coverage escalates instead (see
            # recipe.coverage_is_complete).
            if overlay is None:
                raise_if_coverage_complete(resolved_id, state.admin_id)
                return state
            raise RuntimeError(
                f'Link sidecar {sidecar_path.name} exists but its reference '
                f'{resolved_id} has no data for {state.admin_id}; rerun '
                f'{get_recipe_id(geospine)} for this admin unit.'
            )
        if overlay is None:
            raise RuntimeError(
                f'Missing or stale link sidecar {sidecar_path} for '
                f'{resolved_id}; rerun {get_recipe_id(geospine)} for '
                f'{state.admin_id} to recompute the geometry phase.'
            )
        ref_polys = _prepare_reference(
            ref_raw,
            resolved_id,
            resolved_et,
            state,
            aggregation_function=aggregation_function,
            sort_by=step_cfg.get('sort_by'),
            list_columns=step_cfg.get('list_columns'),
        )
        crosswalk = _build_crosswalk(overlay, spine_id_col, min_fraction, area_min_m2)
        if thresholds.get('snap_chains'):
            crosswalk, _ = snap_chained_links(
                crosswalk,
                spine_id_col,
                fraction_max=float(thresholds.get('chain_fraction_max', 0.75)),
            )
        state.references[resolved_id] = ref_polys
        state.crosswalks[resolved_id] = crosswalk
        state.overlays[resolved_id] = overlay
        _restore_inferred(state, resolved_id, resolved_et)
    elif join == 'spatial_point':
        fingerprint = _link_fingerprint(
            shim,
            resolved_id,
            {
                'join': 'spatial_point',
                'thresholds': thresholds,
                'remap_id': step_cfg.get('remap_id'),
            },
        )
        linked = _load_point_link_sidecar(
            sidecar_path, fingerprint, verbose=state.verbose
        )
        if linked is None:
            probe = get_entities(
                resolved_id, state.admin_id, geom=False, missing='ignore', columns=[]
            )
            if probe is None or len(probe) == 0:
                return state
            raise RuntimeError(
                f'Missing or stale link sidecar {sidecar_path} for '
                f'{resolved_id}; rerun {get_recipe_id(geospine)} for '
                f'{state.admin_id} to recompute the geometry phase.'
            )
        state.crosswalks[resolved_id] = linked
    else:
        raise ValueError(f'Unknown join mode in geospine pipeline: {join!r}')

    if resolved_et:
        state.reference_types[resolved_id] = resolved_et
    sgt = step_cfg.get('source_geometry_type')
    if sgt is not None:
        state.source_geometry_types[resolved_id] = SourceGeometryType(sgt)
    return state


def _restore_inferred(
    state: HarmonizeState, recipe_id: str, entity_type: str | None
) -> None:
    """Rebuild the inferred-footprint metadata frame for one overlay link.

    infer_spine_additions stamps each parcel-inferred spine row with
    `geometry_source = '{entity_type}.{source_id}'` and records the source
    entity's id in `geometry_source_id`; `reconcile_attributes` attributes
    those rows directly (no area weighting) from
    `metadata['inferred_from_{recipe_id}']`, which held the seeding
    reference ids in a 'parcel_id' column. Reconstruct exactly that frame
    from the two persisted spine columns.
    """
    spine = state.spine
    if 'geometry_source' not in spine.columns:
        return
    if 'geometry_source_id' not in spine.columns:
        return
    _et = entity_type or recipe_id.rsplit('_', 1)[-1].split('-', 1)[0]
    token = f'{_et}.{source_id_from_recipe_id(recipe_id)}'
    mask = spine['geometry_source'].astype('string').eq(token).fillna(False)
    if not mask.any():
        return
    inferred = pd.DataFrame(
        {'parcel_id': spine.loc[mask, 'geometry_source_id']}, index=spine.index[mask]
    )
    state.metadata[f'inferred_from_{recipe_id}'] = inferred
