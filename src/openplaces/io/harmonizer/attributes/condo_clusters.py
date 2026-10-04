"""Condominium building cluster detection
(detect_condo_building_clusters).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from openplaces.io.harmonizer import HarmonizeState, _register


@_register('detect_condo_building_clusters', phase='geometry')
def detect_condo_building_clusters(
    state: HarmonizeState,
    land_value_column: str = 'land_value',
    improvement_value_column: str = 'improvement_value',
    area_column: str = 'area_ha',
    use_subgroup_column: str = 'use_subgroup',
    max_unit_area_ha: float = 0.05,
    max_hub_area_ha: float = 3.0,
    max_hub_aspect_ratio: float = 5.0,
    min_group_size: int = 2,
    max_group_size: int = 200,
    group_id_column: str = 'building_cluster_id',
) -> HarmonizeState:
    """Detect stacked-condo-unit parcel clusters sharing one physical building.

    A real, vertically-stacked condo building (unlike the horizontally-
    separated townhome pattern :func:`detect_shared_land_groups` targets) is
    platted as one legal parcel per unit, each a tiny sliver of the
    building's own footprint (tens to low hundreds of m², well below the
    thousands of m² typical of an ordinary house lot), often alongside one
    or more shared "common area" parcels (land held in common -- $0 land
    AND improvement value, e.g. Carteret County, NC's condo-recording
    convention) that the units touch. Confirmed on real data (a 24-99 unit
    oceanfront complex in Carteret County): a single building's real
    footprint can end up split across the parcel/footprint linking pipeline
    as one or two partial real footprints (dominant-linked only to the
    shared common-area parcel, per-unit overlap shares too small to survive
    :func:`~openplaces.io.harmonizer.links.link_to_reference`'s
    ``fraction_of_largest`` trim) plus a scatter of per-unit synthetic
    fallback footprints for whichever units the real footprint(s) don't
    happen to dominate -- this step exists to recognize the underlying
    one-building reality *before* that footprint-side mess, from parcel-side
    evidence alone, so a later footprint-spine step can consolidate it back
    into one coherent shape.

    This reuses :func:`detect_shared_land_groups`'s ``'touches'``-adjacency
    pattern but is a distinct step with a distinct trigger, not a parameter
    variant of it: raising that step's own ``max_group_size`` to cover a
    30-unit condo building would re-admit the county-wide `WATER`/`ROW`
    placeholder-parcel false positives it excludes by design. The
    discriminator here is *unit parcel size* (``max_unit_area_ha``) rather
    than that step's near-zero-*hub*-value-only test, which alone cannot
    tell a condo common area from an HOA park serving ordinary
    single-family lots.

    A unit candidate has area at most *max_unit_area_ha* and positive
    *improvement_value_column*. A hub candidate (optional -- a cluster with
    no separate common-area parcel still forms from mutually-touching units
    alone) has near-zero land and improvement value, area at most
    *max_hub_area_ha*, and oriented aspect ratio at most
    *max_hub_aspect_ratio*. Either candidacy is denied outright to a parcel
    whose *use_subgroup_column* (falling back to ``'use_group_combined'``)
    text matches a small blocklist of known single-family/mobile-home
    labels -- a best-effort, defense-in-depth safety net, since source
    ``use_subgroup`` text is not standardized across recipes (confirmed:
    one county's own single-family label is ``'RESIDENTIAL PRIMARY'``, not
    ``'single family'``), applied only where that column is present and
    non-null; see :func:`_cluster_condo_parcels` for the exact blocklist.

    Clusters are the connected components of the ``'touches'`` adjacency
    graph restricted to unit and hub candidates, with one refinement: a
    shared hub can attach to more than one unit sub-cluster (a common area
    legitimately serving several buildings) but never merges those
    sub-clusters into each other -- unit-unit edges alone determine which
    units belong together (confirmed on real data, Carteret County, NC: a
    0.56 ha shared common-area parcel otherwise chains 3 physically
    separate townhouse-unit groups, 30-130 m apart, into one bogus
    component). A component's *group_id_column* is one of its own unit
    parcel index values (the smallest, for a stable choice), so both a
    hub-anchored and a hub-less cluster resolve the same way. Components
    outside ``[min_group_size, max_group_size]`` are dropped. Because a hub
    can now belong to more than one qualifying cluster, this column can
    only record one of them per hub row (picked deterministically, not
    meaningfully); the full multi-cluster membership is only available via
    :func:`_cluster_condo_parcels`'s own return value.

    Parameters
    ----------
    land_value_column, improvement_value_column : str, optional
        Bare parcel value columns (defaults ``'land_value'``,
        ``'improvement_value'``).
    area_column : str, optional
        Parcel area in hectares (default ``'area_ha'``).
    use_subgroup_column : str, optional
        Land-use subgroup column checked against the single-family/mobile-
        home blocklist (default ``'use_subgroup'``); falls back to
        ``'use_group_combined'`` when absent, and is skipped entirely when
        neither is present.
    max_unit_area_ha : float, optional
        Maximum area for a unit candidate (default 0.05 ha = 500 m² --
        captures ~93% of Carteret County, NC's own condo/townhouse unit
        parcels by area while only 5.8% of ordinary single-family lots
        there fall under it, with *use_subgroup_column* as backstop for
        that residual risk).
    max_hub_area_ha, max_hub_aspect_ratio : float, optional
        Hub candidate filters, same semantics as
        :func:`detect_shared_land_groups` (defaults 3.0 ha, 5.0). The area
        default covers the 95th percentile of Carteret County, NC's own
        zero-value-candidate parcels (2.63 ha) and mainly guards against an
        oversized-but-compact false hub (e.g. a golf course) admitting
        unrelated ordinary lots as "units" -- a risk already reduced by
        *max_unit_area_ha* and the use-subgroup blocklist above, so this
        cap is a secondary backstop. The aspect-ratio default is what
        actually rejects an elongated `ROW`/`WATER` shape regardless of
        area; raised from an earlier, stricter 2.5 because real
        `'COMMON AREA'`-labeled parcels are routinely elongated by their
        own function (beach walkways, drainage strips, canal-front
        access) -- 2.5 excluded roughly a third of them (Carteret County,
        NC: 75th percentile aspect ratio 3.25, 90th 5.15) -- while the
        residual `ROW`/`WATER` false-positive risk this guards against is
        narrower than when this filter was first set, now that
        *max_unit_area_ha* and the use-subgroup blocklist independently
        keep most ordinary lots from ever becoming unit candidates in the
        first place.
    min_group_size, max_group_size : int, optional
        Component size window (default 2-200) -- wide enough for a large
        real resort/motel complex's full unit count (confirmed on real
        data, Carteret County, NC: a legitimate 96-parcel building cluster
        at "1505 Salter Path Rd"), still finite as a backstop against a
        long chain of directly-touching unit candidates with no hub at all
        (e.g. a dense block of many separate small rowhouses) -- the
        *other* runaway-component risk this window guards against, beyond
        the shared-hub risk the two-pass union-find above (see "Clusters
        are..." further up) already handles structurally regardless of
        this cap.
    group_id_column : str, optional
        Output column on every cluster member, hub included (default
        ``'building_cluster_id'``).
    """
    if state.spine is None:
        return state
    spine = state.spine
    result = _cluster_condo_parcels(
        spine,
        land_value_column=land_value_column,
        improvement_value_column=improvement_value_column,
        area_column=area_column,
        use_subgroup_column=use_subgroup_column,
        max_unit_area_ha=max_unit_area_ha,
        max_hub_area_ha=max_hub_area_ha,
        max_hub_aspect_ratio=max_hub_aspect_ratio,
        min_group_size=min_group_size,
        max_group_size=max_group_size,
        verbose=state.verbose,
    )
    if result is None:
        return state
    # A hub parcel is tagged with the same building_cluster_id as its
    # units here regardless -- the geometry-side hub/unit distinction
    # only matters to consolidate_condo_cluster_footprints.
    component, _hub_ids = result
    # A hub attached to multiple qualifying clusters (change 3) has a
    # duplicate index entry in `component`; a scalar column can only hold
    # one, so keep the first (deterministic, not otherwise meaningful).
    n_repeated = int(component.index.duplicated().sum())
    component_for_column = component[~component.index.duplicated(keep='first')]
    if n_repeated and state.verbose:
        print(
            f'  {group_id_column}: {n_repeated:,d} hub parcel(s) belong to more '
            'than one cluster; kept the first cluster id for the column.'
        )

    spine[group_id_column] = spine.index.to_series().map(component_for_column)
    state.spine = spine
    return state


# Case-insensitive substrings of use_subgroup/use_group_combined text that
# indicate a parcel is a known single-family or mobile/manufactured-home
# unit -- never a condo/townhome unit or a common-area hub. Best-effort
# only: source use_subgroup text is not standardized across recipes (one
# county's own single-family label is 'RESIDENTIAL PRIMARY', not 'single
# family'), so this is a defense-in-depth layer on top of the area-based
# candidacy test and the hub-bridging fix below, not a substitute for
# either.
_NON_CONDO_USE_SUBGROUP_TERMS = (
    'single family',
    'single-family',
    'mobile home',
    'manufactured home',
    'residential primary',
)


def _cluster_condo_parcels(
    parcels,
    land_value_column: str = 'land_value',
    improvement_value_column: str = 'improvement_value',
    area_column: str = 'area_ha',
    use_subgroup_column: str = 'use_subgroup',
    max_unit_area_ha: float = 0.05,
    max_hub_area_ha: float = 3.0,
    max_hub_aspect_ratio: float = 5.0,
    min_group_size: int = 2,
    max_group_size: int = 200,
    verbose: bool = False,
) -> tuple[pd.Series, set] | None:
    """Core clustering logic behind :func:`detect_condo_building_clusters`.

    Factored out (state-free, operating on any parcel GeoDataFrame) so
    :func:`~openplaces.io.harmonizer.links.consolidate_condo_cluster_footprints`
    can run the identical detection against the raw parcel reference a
    footprint-spine pipeline already has loaded, not just against an already-
    harmonized parcel spine -- the two recipes' pipelines run in an order
    (footprint spine before parcel spine, since the parcel spine's own
    ``summarize_footprint_morphology`` step reads the harmonized footprint
    spine) that makes a footprint-side step reading the parcel spine's own
    *output* impossible without a circular dependency.

    Parameters are exactly the *state*-independent subset of
    :func:`detect_condo_building_clusters`'s -- see there for the full
    algorithm description.

    Returns
    -------
    tuple of (pandas.Series, set) or None
        ``(component, hub_ids)``: *component* is the cluster id per
        qualifying parcel, indexed like *parcels* (only member rows
        present, not the full index) -- a hub parcel attached to more than
        one qualifying sub-cluster (see the two-pass union-find below)
        appears once per cluster it belongs to, so the index may contain
        duplicate values for such a hub; *hub_ids* is the subset of that
        same index classified as a hub/common-area candidate (real,
        positive ``improvement_value_column`` decides a unit; near-zero
        land *and* improvement value plus a compact shape decides a hub --
        see the ``is_hub_candidate`` test below) -- callers that fall back
        to a cluster's own parcel geometry (e.g.
        :func:`~openplaces.io.harmonizer.links.consolidate_condo_cluster_footprints`)
        must exclude *hub_ids* from that union: a hub's own polygon is the
        surrounding lot/common area, not part of the building, and is
        typically an order of magnitude larger than a real unit parcel.
        ``None`` if no candidates, no touching pairs, or no qualifying
        cluster was found.
    """
    required = {land_value_column, improvement_value_column, area_column}
    if not required.issubset(parcels.columns):
        if verbose:
            print(
                f'  detect_condo_building_clusters: '
                f'{sorted(required - set(parcels.columns))} missing; skipping.'
            )
        return None

    import geopandas as gpd

    from openplaces.geo.polygon import local_metric_crs
    from openplaces.io.harmonizer.spine import get_oriented_dims

    land_value = pd.to_numeric(parcels[land_value_column], errors='coerce').fillna(0)
    improvement_value = pd.to_numeric(
        parcels[improvement_value_column], errors='coerce'
    ).fillna(0)
    area_ha = pd.to_numeric(parcels[area_column], errors='coerce')

    is_unit_candidate = (
        area_ha.notna() & (area_ha <= max_unit_area_ha) & (improvement_value > 0)
    )
    is_hub_candidate = (
        (land_value <= 0)
        & (improvement_value <= 0)
        & area_ha.notna()
        & (area_ha > 0)
        & (area_ha <= max_hub_area_ha)
    )
    if is_hub_candidate.any():
        hub_sub = parcels.loc[is_hub_candidate]
        geom_m = hub_sub.geometry.to_crs(local_metric_crs(hub_sub))
        dims = geom_m.map(get_oriented_dims)
        length = np.array([d[1] for d in dims])
        width = np.clip(np.array([d[2] for d in dims]), 1e-6, None)
        is_hub_candidate.loc[hub_sub.index] = (length / width) <= max_hub_aspect_ratio

    use_subgroup_col = (
        use_subgroup_column
        if use_subgroup_column in parcels.columns
        else 'use_group_combined'
        if 'use_group_combined' in parcels.columns
        else None
    )
    if use_subgroup_col is not None:
        use_subgroup = parcels[use_subgroup_col].astype('string').str.lower()
        is_non_condo_use = use_subgroup.str.contains(
            '|'.join(_NON_CONDO_USE_SUBGROUP_TERMS), na=False, regex=True
        )
        is_unit_candidate &= ~is_non_condo_use
        is_hub_candidate &= ~is_non_condo_use

    is_candidate = is_unit_candidate | is_hub_candidate
    if is_candidate.sum() < min_group_size:
        if verbose:
            print('  detect_condo_building_clusters: no candidates found.')
        return None

    idx_name = parcels.index.name or 'index'
    cand_gdf = parcels.loc[is_candidate, ['geometry']].reset_index()
    a_col, b_col = f'{idx_name}_a', f'{idx_name}_b'
    touching = gpd.sjoin(
        cand_gdf, cand_gdf, predicate='touches', how='inner', lsuffix='_a', rsuffix='_b'
    ).rename(columns={f'{idx_name}__a': a_col, f'{idx_name}__b': b_col})
    if len(touching) == 0:
        if verbose:
            print('  detect_condo_building_clusters: no touching candidates found.')
        return None

    # Connected components over the candidate 'touches' graph (a plain
    # union-find over a small edge list -- one building's worth of parcels at
    # a time, not a large general graph, so no need for a library dependency).
    parent: dict[str, str] = {}

    def _find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def _union(x: str, y: str) -> None:
        rx, ry = _find(x), _find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    hub_ids = set(parcels.index[is_hub_candidate])
    for pid in cand_gdf[idx_name]:
        parent.setdefault(pid, pid)

    # Pass 1: union unit-unit edges only. A hub is deliberately never
    # unioned into this graph -- if it were, a single shared hub could
    # transitively chain multiple, geometrically separate unit groups
    # (e.g. one HOA common-area parcel bordering several distinct
    # townhouse buildings) into a single bogus component. Confirmed on
    # real data (Carteret County, NC): a 0.56 ha COMMON AREA parcel
    # otherwise bridges 3 physically disjoint unit groups, 30-130 m apart,
    # into one. Two adjacent hubs (e.g. neighboring walkway segments)
    # never union with each other either, matching the prior hub-hub
    # guard.
    hub_unit_edges: list[tuple[str, str]] = []
    for a, b in zip(touching[a_col], touching[b_col]):
        a_hub, b_hub = a in hub_ids, b in hub_ids
        if a_hub or b_hub:
            if a_hub != b_hub:
                hub_unit_edges.append((a, b) if a_hub else (b, a))
            continue
        _union(a, b)

    # Pass 2: attach each hub to every unit base-component it directly
    # touches, without merging those base components together -- a hub
    # may legitimately serve multiple separate buildings, but those
    # buildings' unit groups must never become one output row just
    # because they share a hub.
    unit_pids = [pid for pid in cand_gdf[idx_name] if pid not in hub_ids]
    component_index = list(unit_pids)
    component_values = [_find(pid) for pid in unit_pids]
    hub_component_ids: dict[str, set[str]] = {}
    for hub, unit in hub_unit_edges:
        hub_component_ids.setdefault(hub, set()).add(_find(unit))
    for hub, comp_ids in hub_component_ids.items():
        for comp_id in comp_ids:
            component_index.append(hub)
            component_values.append(comp_id)

    component = pd.Series(component_values, index=pd.Index(component_index))
    sizes = component.value_counts()
    qualifying = sizes[(sizes >= min_group_size) & (sizes <= max_group_size)]
    if qualifying.empty:
        if verbose:
            print(
                f'  detect_condo_building_clusters: no cluster with '
                f'{min_group_size}-{max_group_size} parcels found.'
            )
        return None
    component = component[component.isin(qualifying.index)]
    hub_ids = hub_ids & set(component.index)

    if verbose:
        print(
            f'  detect_condo_building_clusters: {len(qualifying):,} building clusters '
            f'covering {component.index.nunique():,} parcels.'
        )
    return component, hub_ids
