"""
Pipeline steps for admin geometry harmonization:
  - discover_sources: rank the ingest admin recipes covering each unit
  - merge_global: take each unit's polygon from the first that has one
  - carve_lumped_territories: hand a territory the area its sovereign's
    polygon lumped in
  - simplify_geometries: apply simplify_coverage for visualization sidecars
"""

from __future__ import annotations

import warnings

import geopandas as gpd
import pandas as pd
import shapely

from openplaces.io.harmonizer import HarmonizeState, _register
from openplaces.io.readers import get_admin_ids, get_entities
from openplaces.recipe.discovery import (
    admin_sources_covering,
    find_admin_source_recipes,
)


class AdminSourcesNotIngestedError(RuntimeError):
    """No candidate source of a unit's admin geometry could be loaded.

    Raised by merge_global when discover_sources assigned at least one
    unit a candidate, no candidate supplied a polygon, and a candidate
    could not be read, which means its ingest has not run. Saving the
    empty table instead would report the unit as built with no
    geometry.
    """


# Ranks the sources into state.metadata and writes no spine column.
@_register('discover_sources', phase='geometry', writes=[])
def discover_sources(
    state: HarmonizeState, include_restricted: bool = False
) -> HarmonizeState:
    """Rank the ingest admin recipes at the recipe's admin_level.

    Lists the sources through `openplaces.recipe.find_admin_source_recipes`,
    the listing the dependency scan also uses, so the graph schedules
    the ingest of each. Ranks, for each admin_id within state.admin_id,
    every ingest recipe
    that covers it: most-specific admin_id first, newest version first
    within the same specificity tier. merge_global takes each unit's
    polygon from the first of them that has one. With no state.admin_id
    the scope is the whole spine, which is a deliberate bulk run rather
    than the default.

    Parameters
    ----------
    state : HarmonizeState
        Harmonize state; its recipe supplies admin_level.
    include_restricted : bool, optional
        Keep sources whose recipe declares redistribution_restricted:
        true. Off by default: the harmonized spine is what readers draw,
        and a restricted source (GADM) may be used locally but must not
        fill the geometry every consumer inherits.

    Notes
    -----
    Stores results in state.metadata:

    - 'discovered_sources': priority-sorted list of source dicts
    - 'source_candidates': dict of admin_id to its covering recipe_ids,
      in priority order
    - 'source_assignment': dict of recipe_id to the admin_ids it is the
      first candidate for
    - 'unassigned_admin_ids': admin_ids no source covers
    """
    admin_level = state.recipe.get('admin_level', 1)
    sources = find_admin_source_recipes(
        admin_level, state.admin_id, include_restricted=include_restricted
    )

    if not sources:
        scope = state.admin_id or 'the whole spine'
        warnings.warn(
            f'No ingest admin sources found at level {admin_level} '
            f'within {scope}. Skipping.'
        )
        state.metadata['discovered_sources'] = []
        state.metadata['source_candidates'] = {}
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
    # A unit with nothing below it at this level (Antarctica at level
    # 2, an Andorran parish at level 3) is an empty assignment, not an
    # error: the harmonizer then writes an empty spine for it, and the
    # rest of the world-level run proceeds.
    all_admin_ids = get_admin_ids(admin_level, state.admin_id, allow_empty=True)
    candidates: dict[str, list[str]] = {}
    assignment: dict[str, list[str]] = {}
    unassigned: list[str] = []
    for aid in all_admin_ids:
        covering = admin_sources_covering(aid, sources)
        if not covering:
            unassigned.append(aid)
        else:
            candidates[aid] = covering
            assignment.setdefault(covering[0], []).append(aid)

    if unassigned and state.verbose:
        print(f'  Discover: {len(unassigned)} admin_id(s) have no source coverage.')
    if state.timer:
        state.timer.mark('Discover')

    state.metadata['discovered_sources'] = sources
    state.metadata['source_candidates'] = candidates
    state.metadata['source_assignment'] = assignment
    state.metadata['unassigned_admin_ids'] = unassigned
    return state


def _best_recipe_for(admin_id_str: str, sources: list[dict]) -> str | None:
    """Return recipe_id of the highest-priority source covering admin_id."""
    covering = admin_sources_covering(admin_id_str, sources)
    return covering[0] if covering else None


@_register('merge_global', phase='geometry')
def merge_global(state: HarmonizeState) -> HarmonizeState:
    """Take each unit's polygon from the first source that has one.

    Reads 'discovered_sources' and 'source_candidates' from
    state.metadata, as discover_sources wrote them, and walks the
    sources in priority order. A unit is taken from the first of its
    candidates whose output holds a non-empty geometry for it; a unit
    the source lacks falls through to the next candidate.

    A unit used to be read from its first candidate only, and a
    national recipe that carries names but no polygons (FR's INSEE,
    JP's Soumu, TR's TUIK, PH's PSA) then left the whole country
    without geometry although geoBoundaries had it. The fall-through
    also fills a unit the first source happens to miss (India's 33 of
    34 geoBoundaries states).

    Writes state.spine, and in state.metadata 'geometry_source' (the
    recipe each unit's polygon came from) and 'unmatched_admin_ids'
    (candidates that no source had a polygon for). A source that cannot
    be read is skipped with a warning while another one supplies a
    polygon. With no unit to fill, the spine stays unset and the
    harmonizer saves an empty table.

    Raises
    ------
    AdminSourcesNotIngestedError
        When at least one unit had a candidate, no polygon was taken,
        and a candidate could not be read: its ingest has not run.
    """
    sources = state.metadata.get('discovered_sources', [])
    candidates = state.metadata.get('source_candidates')
    if candidates is None:
        # A state built before candidates were recorded: one source each.
        assignment = state.metadata.get('source_assignment', {})
        candidates = {aid: [rid] for rid, aids in assignment.items() for aid in aids}

    if not sources:
        warnings.warn('merge_global: no discovered_sources in metadata.')
        return state

    remaining = set(candidates)
    origin: dict[str, str] = {}
    pieces = []
    unreadable: list[str] = []
    for src in sources:
        recipe_id = src['recipe_id']
        wanted = {aid for aid in remaining if recipe_id in candidates[aid]}
        if not wanted:
            continue
        try:
            gdf = get_entities(recipe_id, geom=True)
        except Exception as exc:
            warnings.warn(f'Could not load {recipe_id}: {exc}. Skipping source.')
            unreadable.append(recipe_id)
            continue

        available = gdf[gdf.index.isin(wanted)]
        if 'geometry' in available:
            available = available[
                available.geometry.notna() & ~available.geometry.is_empty
            ]
        available = available[~available.index.duplicated()]
        if available.empty:
            if state.verbose:
                print(f'  Merge {recipe_id}: 0 / {len(wanted)} admin_ids found')
            continue

        pieces.append(available)
        origin.update(dict.fromkeys(available.index, recipe_id))
        remaining -= set(available.index)
        if state.verbose:
            print(f'  Merge {recipe_id}: +{len(available)} / {len(wanted)}')

    state.metadata['geometry_source'] = origin
    state.metadata['unmatched_admin_ids'] = sorted(remaining)
    spine = pd.concat(pieces).sort_index() if pieces else gpd.GeoDataFrame()

    if spine.empty:
        admin_level = state.recipe.get('admin_level', '?')
        # A unit with candidates whose sources could not be read has
        # sources that were never ingested; an empty table saved for it
        # would read as a unit built with no geometry. A unit with no
        # candidates at all (Antarctica at level 2) has nothing to
        # ingest, and its empty table is the right output.
        if candidates and unreadable:
            scope = state.admin_id or 'the whole spine'
            raise AdminSourcesNotIngestedError(
                f'No admin level {admin_level} geometry could be loaded '
                f'for {scope}: candidate source(s) {", ".join(unreadable)} '
                'could not be read. Ingest them, then rerun this harmonize.'
            )
        warnings.warn(f'No admin geometries loaded at level {admin_level}.')
        return state

    if state.timer:
        state.timer.mark('Merge')

    state.spine = spine
    return state


def _parts(geometry):
    """Return a geometry's polygons as a list, empty for an empty one.

    A GeometryCollection's lines and points are dropped: an overlay of
    two polygons that share an edge returns that edge beside the area,
    and only the area is a unit's territory.
    """
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == 'Polygon':
        return [geometry]
    return [
        part
        for g in getattr(geometry, 'geoms', [])
        for part in _parts(g)
        if part.geom_type == 'Polygon'
    ]


def _polygonal(geometry):
    """Return a geometry's polygonal part, or None when it has none."""
    parts = _parts(geometry)
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else shapely.MultiPolygon(parts)


@_register('carve_lumped_territories', phase='geometry')
def carve_lumped_territories(
    state: HarmonizeState,
    thresholds: dict | None = None,
) -> HarmonizeState:
    """Give a unit from a finer source the area a coarser one lumped in.

    Not to be confused with resolve_overlaps (links/overlaps.py), the
    footprint step that drops overlapping rows: step names share one
    registry, and registering this under that name silently replaced
    it, dropping 46 countries on the first run.

    merge_global takes each unit's polygon from the first source that
    has one, and two sources need not split the world the same way.
    geoBoundaries' CGAZ, the level-1 primary, draws France with
    Guadeloupe and the United Kingdom with Akrotiri inside them; Natural
    Earth, which supplies those units because CGAZ has no row for them,
    draws them on their own. Concatenated, the two overlap.

    Where a unit from a lower-priority source overlaps a unit from a
    higher-priority one, each polygon part of the higher unit is
    settled on its own:

    - a part the lower units cover by at least part_share of its area
      is a lumped territory, and is handed over whole, keeping the
      higher source's outline (Guadeloupe keeps CGAZ's coastline rather
      than Natural Earth's coarser one); a part several lower units
      share is split between them, its uncovered rest going to the
      unit covering most of it;
    - a part covered less than that is the higher unit's own land. Its
      overlap with a lower unit is cut out and given to that unit only
      where the overlap is at least cut_share of the lower unit's own
      area, the lower unit lying largely inside it (a base area on an
      island that is otherwise the higher unit's). A smaller overlap is
      the coarser source's border running past the finer one's, and
      the higher unit keeps it: measured 2026-09-30 on admin-naturalearth-5~1~1:
      Natural Earth's West Bank crosses CGAZ's Jordan and Israel by
      about 1 and 2 percent of its area, and French Guiana crosses
      Brazil and Suriname by less than 1.

    A lower unit that received no whole part also keeps its own polygon
    outside every higher unit, which is how an island the primary does
    not draw at all still gets a shape. One that received whole parts
    does not: that remainder is the coarser coastline's disagreement
    with the finer one, not land.

    Parameters
    ----------
    state : HarmonizeState
        State after merge_global, whose metadata 'geometry_source' and
        'discovered_sources' rank each unit's source.
    thresholds : dict, optional
        part_share (float, default 0.5): the covered share at which a
        part is handed over whole. cut_share (float, default 0.25): the
        share of a lower unit's own area its overlap with a higher
        unit's land must reach to be cut out. max_overlap_share (float,
        default 1e-6): the overlap, as a share of the smaller unit's
        area, left after resolution that raises instead of passing.

    Raises
    ------
    ValueError
        When overlap above max_overlap_share remains.
    """
    spine = state.spine
    if spine is None or spine.empty:
        return state
    thresholds = thresholds or {}
    part_share = float(thresholds.get('part_share', 0.5))
    cut_share = float(thresholds.get('cut_share', 0.25))
    max_overlap = float(thresholds.get('max_overlap_share', 1e-6))

    order = [s['recipe_id'] for s in state.metadata.get('discovered_sources', [])]
    rank_of_source = {rid: i for i, rid in enumerate(order)}
    origin = state.metadata.get('geometry_source', {})
    rank = {aid: rank_of_source.get(origin.get(aid), 0) for aid in spine.index}

    original = {
        aid: shapely.make_valid(geom) if geom is not None else geom
        for aid, geom in spine.geometry.items()
    }
    ids = list(original)
    tree = shapely.STRtree([original[aid] for aid in ids])
    lower_of: dict[str, list[str]] = {}
    overlap: dict[tuple[str, str], float] = {}
    for i, j in zip(*tree.query([original[a] for a in ids], 'intersects')):
        high, low = ids[i], ids[j]
        if rank[low] <= rank[high]:
            continue
        area = original[high].intersection(original[low]).area
        if area > 0:
            lower_of.setdefault(high, []).append(low)
            overlap[high, low] = area

    if not lower_of:
        return state

    result = dict(original)
    whole: dict[str, list] = {}
    cut: dict[str, list] = {}
    higher_of: dict[str, list[str]] = {}
    for high in sorted(lower_of, key=lambda a: rank[a]):
        lows = lower_of[high]
        for low in lows:
            higher_of.setdefault(low, []).append(high)
        kept = []
        for part in _parts(original[high]):
            hits = {
                low: part.intersection(original[low])
                for low in lows
                if part.intersects(original[low])
            }
            hits = {low: g for low, g in hits.items() if g.area > 0}
            if not hits:
                kept.append(part)
                continue
            covered = shapely.union_all(list(hits.values()))
            if covered.area >= part_share * part.area:
                if len(hits) == 1:
                    whole.setdefault(next(iter(hits)), []).append(part)
                    continue
                for low, share in hits.items():
                    whole.setdefault(low, []).append(share)
                rest = part.difference(covered)
                if not rest.is_empty:
                    largest = max(hits, key=lambda k: hits[k].area)
                    whole[largest].append(rest)
            else:
                inside = {
                    low: share
                    for low, share in hits.items()
                    if overlap[high, low] >= cut_share * original[low].area
                }
                if inside:
                    part = part.difference(shapely.union_all(list(inside.values())))
                kept.append(part)
                for low, share in inside.items():
                    cut.setdefault(low, []).append(share)
        result[high] = shapely.union_all(kept) if kept else None

    for low, highs in higher_of.items():
        pieces = whole.get(low, []) + cut.get(low, [])
        if not whole.get(low):
            above = shapely.union_all([original[h] for h in highs])
            pieces.append(original[low].difference(above))
        pieces = [p for p in pieces if p is not None and not p.is_empty]
        result[low] = shapely.union_all(pieces) if pieces else None

    geoms = gpd.GeoSeries(
        [_polygonal(result[aid]) for aid in spine.index],
        index=spine.index,
        crs=spine.crs,
    )
    spine = spine.set_geometry(geoms)
    spine = spine[spine.geometry.notna() & ~spine.geometry.is_empty]

    ids = list(spine.index)
    tree = shapely.STRtree(list(spine.geometry))
    left = []
    for i, j in zip(*tree.query(list(spine.geometry), 'intersects')):
        if i >= j:
            continue
        a, b = spine.geometry.iloc[i], spine.geometry.iloc[j]
        overlap = a.intersection(b).area
        if overlap > max_overlap * min(a.area, b.area):
            left.append((ids[i], ids[j], overlap))
    if left:
        raise ValueError(f'carve_lumped_territories: overlap left between {left[:10]}')

    state.metadata['overlaps_resolved'] = {
        high: sorted(lows) for high, lows in lower_of.items()
    }
    if state.verbose:
        print(
            f'  Resolve overlaps: {len(higher_of)} unit(s) carved out of '
            f'{len(lower_of)}'
        )
    state.spine = spine
    return state


# Writes a simplified-geometry sidecar, not a spine column.
@_register('simplify_geometries', phase='geometry', reads=['geometry'], writes=[])
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
