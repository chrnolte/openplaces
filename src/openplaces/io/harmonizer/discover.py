"""
Pipeline steps for admin geometry harmonization:
  - discover_sources: scan ingest admin recipes at the target level
  - merge_global: load and concatenate geometries per discovered source
  - simplify_geometries: apply simplify_coverage for visualization sidecars
"""

from __future__ import annotations

import glob
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml

from openplaces.io.harmonizer import HarmonizeState, _register
from openplaces.io.readers import get_admin_ids, get_entities


@_register('discover_sources', phase='geometry')
def discover_sources(state: HarmonizeState) -> HarmonizeState:
    """Scan all ingest admin recipes at the recipe's ``admin_level``.

    Assigns each admin_id within ``state.admin_id`` to the
    highest-priority ingest recipe that covers it (most-specific admin_id
    wins; newest version wins within the same specificity tier). With no
    ``state.admin_id`` the scope is the whole spine, which is a
    deliberate bulk run rather than the default.

    Stores results in ``state.metadata``:
    - ``'discovered_sources'`` — priority-sorted list of source dicts
    - ``'source_assignment'`` — ``dict[recipe_id, list[admin_id_str]]``
    - ``'unassigned_admin_ids'`` — list of admin_id strings with no coverage
    """
    admin_level = state.recipe.get('admin_level', 1)
    sources = _scan_admin_ingest_recipes(admin_level)

    if not sources:
        warnings.warn(
            f'No ingest admin sources found at level {admin_level}. Skipping.'
        )
        state.metadata['discovered_sources'] = []
        state.metadata['source_assignment'] = {}
        state.metadata['unassigned_admin_ids'] = []
        return state

    if state.verbose:
        scope = state.admin_id or 'the whole spine'
        print(
            f'  Discover: {len(sources)} source(s) at admin level '
            f'{admin_level} within {scope}: '
            + ', '.join(s['recipe_id'] for s in sources)
        )

    # Scoped to the unit being processed rather than the whole world:
    # a reader who wants one town should not pay for a global layer.
    # `state.admin_id` is None only for a deliberately global run.
    all_admin_ids = get_admin_ids(admin_level, state.admin_id)
    assignment: dict[str, list[str]] = {}
    unassigned: list[str] = []
    for aid in all_admin_ids:
        rid = _best_recipe_for(aid, sources)
        if rid is None:
            unassigned.append(aid)
        else:
            assignment.setdefault(rid, []).append(aid)

    if unassigned and state.verbose:
        print(f'  Discover: {len(unassigned)} admin_id(s) have no source coverage.')
    if state.timer:
        state.timer.mark('Discover')

    state.metadata['discovered_sources'] = sources
    state.metadata['source_assignment'] = assignment
    state.metadata['unassigned_admin_ids'] = unassigned
    return state


def _scan_ingest_recipes(entity_type: str, filename_glob: str = '*') -> list[dict]:
    """Scan recipe tree for ingest recipes of entity_type, priority-sorted.

    Searches ``{recipes_root}/**/{entity_type}/*/*/{filename_glob}.yaml``
    (recursive), reads each YAML to verify ``stage == 'ingest'``, and returns
    a list sorted by (specificity, version) descending — most-specific
    admin_id and newest version first.

    Parameters
    ----------
    entity_type : str
        Entity type directory name (e.g. 'admin', 'parcel', 'footprint').
    filename_glob : str, optional
        Glob pattern matched against the recipe filename stem. Defaults to
        '*' (all files). A .yaml extension is appended automatically.
    """
    recipes_root = Path(__file__).parent.parent.parent / 'recipes'
    pattern = str(
        recipes_root / '**' / entity_type / '*' / '*' / f'{filename_glob}.yaml'
    )
    found = sorted(glob.glob(pattern, recursive=True))

    sources: list[dict] = []
    for rp in found:
        try:
            with open(rp, encoding='utf-8') as fh:
                data = yaml.safe_load(fh)
        except Exception:
            continue
        if (data.get('stage') or 'ingest') != 'ingest':
            continue
        # A recipe can withdraw itself from auto-discovery. The field was
        # already set by two recipes and reported by
        # `diagnostics.find_recipes`, but never enforced here -- and this
        # is the path admin-source discovery actually takes, so a
        # superseded layer stayed in the candidate pool and could fill
        # any gap the current vintage did not cover.
        if data.get('exclude_from_auto_discover'):
            continue
        entity = data.get('entity') or {}
        raw_admin_id = data.get('admin_id')
        admin_id_str = (
            str(raw_admin_id)
            if raw_admin_id is not None and str(raw_admin_id) != 'None'
            else ''
        )
        specificity = len(admin_id_str.split('-')) if admin_id_str else 0
        version = str(entity.get('version') or '')
        sources.append(
            {
                'recipe_id': Path(rp).stem,
                'admin_id': admin_id_str,
                'specificity': specificity,
                'version': version,
            }
        )

    sources.sort(key=lambda s: (s['specificity'], s['version']), reverse=True)
    return sources


def _scan_admin_ingest_recipes(admin_level: int) -> list[dict]:
    return _scan_ingest_recipes('admin', filename_glob=f'*_admin{admin_level}')


def _scan_entity_ingest_recipes(entity_type: str) -> list[dict]:
    return _scan_ingest_recipes(entity_type)


def _best_recipe_for(admin_id_str: str, sources: list[dict]) -> str | None:
    """Return recipe_id of the highest-priority source covering admin_id."""
    for src in sources:
        rid = src['admin_id']
        if rid == '' or admin_id_str.startswith(rid):
            return src['recipe_id']
    return None


@_register('merge_global', phase='geometry')
def merge_global(state: HarmonizeState) -> HarmonizeState:
    """Load and concatenate admin geometries from discovered sources.

    Reads ``state.metadata['discovered_sources']`` and
    ``state.metadata['source_assignment']`` produced by ``discover_sources``,
    loads each source's GeoDataFrame, filters to assigned admin_ids, and
    concatenates into ``state.spine``.
    """
    sources = state.metadata.get('discovered_sources', [])
    assignment = state.metadata.get('source_assignment', {})

    if not sources:
        warnings.warn('merge_global: no discovered_sources in metadata.')
        return state

    spine = gpd.GeoDataFrame()
    for src in sources:
        recipe_id = src['recipe_id']
        if recipe_id not in assignment:
            continue
        try:
            gdf = get_entities(recipe_id, geom=True)
        except Exception as exc:
            warnings.warn(f'Could not load {recipe_id}: {exc}. Skipping source.')
            continue

        target_ids = assignment[recipe_id]
        available = gdf[gdf.index.isin(target_ids)]
        if available.empty:
            if state.verbose:
                print(f'  Merge {recipe_id}: 0 / {len(target_ids)} admin_ids found')
            continue

        spine = pd.concat([spine, available]).sort_index()
        if state.verbose:
            print(f'  Merge {recipe_id}: +{len(available)} / {len(target_ids)}')

    if spine.empty:
        admin_level = state.recipe.get('admin_level', '?')
        warnings.warn(f'No admin geometries loaded at level {admin_level}.')
        return state

    if state.timer:
        state.timer.mark('Merge')

    state.spine = spine
    return state


@_register('simplify_geometries', phase='geometry')
def simplify_geometries(
    state: HarmonizeState,
    thresholds: dict | None = None,
) -> HarmonizeState:
    """Simplify spine geometries and store as a sidecar.

    Calls ``GeoSeries.simplify_coverage(tolerance)`` and stores the result
    in ``state.simplified_geometry``.  The sidecar is written by the save
    step in :class:`~openplaces.io.harmonizer.Harmonizer`.

    Parameters
    ----------
    thresholds : dict, optional
        Must contain ``tolerance`` (float). No-op when absent or zero.
    """
    tol = (thresholds or {}).get('tolerance') or (thresholds or {}).get(
        'simplify_tolerance'
    )
    if not tol or state.spine is None:
        return state

    state.simplified_geometry = state.spine.geometry.simplify_coverage(tol)
    if state.verbose:
        print(f'  Simplify: tolerance={tol}')
    if state.timer:
        state.timer.mark('Simplify')
    return state
