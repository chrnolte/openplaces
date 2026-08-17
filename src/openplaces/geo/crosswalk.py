"""
crosswalk.py

Fractional area-weighted linkage between two vintages of the same polygon
entity (e.g. an old and a current parcel layer), for carrying evidence from
one onto the other where boundaries have drifted apart over time.

Two-step linkage: an exact-match fast path on a geometry-hash ``geo_id``
(:func:`openplaces.geo.ids.get_geo_ids`) for unchanged parcels, then a
polygon-overlay fallback for the rest. Kept separate from
:mod:`openplaces.geo.link`'s generic ``create_entity_link``/``overlay_polygons``
because it adds a ``matplotlib``-based QA function, a heavier presentation
dependency that should not leak into that module's lean import surface.
"""

from __future__ import annotations

import math
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from openplaces.geo.ids import _shape_signals, get_geo_ids
from openplaces.geo.overlay import overlay_polygons_with_duckdb
from openplaces.geo.polygon import get_areas, local_metric_crs, overlay_polygons
from openplaces.table import aggregate_rows

ON_DUPLICATE_GEO_ID_OPTIONS = ('raise', 'first', 'groupby')

CROSSWALK_COLUMNS = [
    'parcel_id_old',
    'parcel_id_new',
    'area_ha',
    'match_type',
    'fraction_of_old',
]

# Minimum rows (on the smaller side) required to attempt calibration/direct
# matching -- below this a sample can't say anything meaningful about a
# systematic offset, so fall straight to overlay as before these stages
# existed.
MIN_CALIBRATION_ROWS = 20


def _id_series(gdf, id_col):
    return gdf.index.to_series(index=gdf.index) if id_col is None else gdf[id_col]


def _with_geo_id(gdf, geo_id_col):
    """Return *gdf* with *geo_id_col* present, computing it if absent.

    Mirrors the automatic ``geo_id`` computation `TableIngester` applies to
    every ``entity_type: parcel`` ingest recipe
    (``get_geo_ids(df, handle_duplicates=False)``), so a caller's own
    already-ingested output and a fresh in-memory frame (e.g. a harmonized
    spine, whose ``resolve_spine`` step does not carry ``geo_id`` forward)
    are treated identically.
    """
    if geo_id_col in gdf.columns:
        return gdf
    gdf = gdf.copy()
    gdf[geo_id_col] = get_geo_ids(gdf, handle_duplicates=False)
    return gdf


def _resolve_duplicate_geo_ids(gdf, geo_id_col, id_col, on_duplicate, side_label):
    """Enforce a unique `geo_id_col` on *gdf*, per *on_duplicate*.

    A duplicated `geo_id` can only arise when two rows' geometries are
    quantized identically -- genuinely overlapping or duplicate-record
    parcels (e.g. several condo-unit records sharing one physical footprint).
    That's a structural fact about the input, not matching noise, so it's
    treated as an error unless the caller explicitly opts into a resolution:

    - `'raise'` (default): refuse ambiguous input outright.
    - `'first'`: keep an arbitrary one row per duplicate group.
    - `'groupby'`: aggregate each duplicate group's attribute columns via
      `openplaces.io.aggregate.aggregate_rows` (the existing
      attribute-registry-driven "collapse rows by key" utility -- not new
      aggregation logic), keep the largest-area geometry as representative
      (matching `aggregate_rows`'s own convention for 'first'-aggregated
      columns), and preserve the original ids being merged as an
      `{id_col}_list` column rather than silently dropping them.

    Parameters
    ----------
    gdf : GeoDataFrame
        Must already carry `geo_id_col` and `id_col`.
    geo_id_col : str
    id_col : str
        Column holding this side's caller-facing parcel id (e.g.
        `'parcel_id_new'`/`'parcel_id_old'`).
    on_duplicate : str
        One of `ON_DUPLICATE_GEO_ID_OPTIONS`.
    side_label : str
        `'new'` or `'old'`, for the error message only.

    Returns
    -------
    GeoDataFrame
        *gdf* with a unique `geo_id_col`, unchanged if it already was.
    """
    dup_mask = gdf[geo_id_col].duplicated(keep=False)
    if not dup_mask.any():
        return gdf

    if on_duplicate == 'raise':
        raise ValueError(
            f'{dup_mask.sum():,} {side_label} parcels have a duplicated '
            f'{geo_id_col!r} (identical quantized geometry -- overlapping or '
            'duplicate-record parcels, e.g. several condo-unit records sharing '
            "one footprint). Pass on_duplicate_geo_id='groupby' to aggregate "
            "each group's attributes (via openplaces.io.aggregate.aggregate_rows), "
            "'first' to keep an arbitrary one, or pre-clean/dissolve the input "
            'yourself before calling this function if the duplication reflects '
            'a data problem worth fixing at the source.'
        )
    if on_duplicate not in ON_DUPLICATE_GEO_ID_OPTIONS:
        raise ValueError(
            f'on_duplicate_geo_id must be one of {ON_DUPLICATE_GEO_ID_OPTIONS}, '
            f'got {on_duplicate!r}'
        )

    unique_part = gdf[~dup_mask]
    dup_part = gdf[dup_mask]

    if on_duplicate == 'first':
        resolved = dup_part.drop_duplicates(geo_id_col, keep='first')
        return pd.concat([unique_part, resolved]).reset_index(drop=True)

    # 'groupby': registry-driven aggregation for every column except geometry,
    # geo_id (the grouping key), and id_col (handled explicitly below, since
    # collapsing distinct ids needs to preserve them, not aggregate them away).
    geom_col = dup_part.geometry.name
    agg_cols = [c for c in dup_part.columns if c not in (geo_id_col, id_col, geom_col)]
    aggregated = aggregate_rows(
        pd.DataFrame(dup_part[agg_cols + [geo_id_col]]), by=geo_id_col
    )

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Geometry is in a geographic CRS')
        # fillna(-1): a null/invalid geometry area-quantizes to the same
        # degenerate geo_id as any other null geometry, so a "duplicate"
        # group can end up entirely null-area (no real geometry to prefer).
        # idxmax on an all-NaN group raises; treating null area as -1 still
        # picks a real geometry over a null one when the group has both, and
        # deterministically picks *a* row (arbitrarily, since none is more
        # "representative") when every geometry in the group is null.
        rep_idx = (
            dup_part.geometry.area.fillna(-1).groupby(dup_part[geo_id_col]).idxmax()
        )
    representative = dup_part.loc[rep_idx].set_index(geo_id_col)[[geom_col]]
    ids_list = dup_part.groupby(geo_id_col)[id_col].apply(list).rename(f'{id_col}_list')

    pieces = [representative, ids_list] + (
        [aggregated] if aggregated is not None else []
    )
    resolved = pd.concat(pieces, axis=1).reset_index()
    resolved[id_col] = resolved[f'{id_col}_list'].str[0]
    resolved = gpd.GeoDataFrame(resolved, geometry=geom_col, crs=dup_part.crs)

    return pd.concat([unique_part, resolved]).reset_index(drop=True)


def _shape_gate(geom_a, geom_b, area_tol, compact_tol):
    """Boolean mask: True where *geom_a*/*geom_b* pairs have similar area and
    compactness (`openplaces.geo.ids._shape_signals`), elementwise."""
    area_a, compact_a = _shape_signals(geom_a)
    area_b, compact_b = _shape_signals(geom_b)
    area_ratio = np.where(area_b > 0, area_a / area_b, np.inf)
    compact_ratio = np.where(compact_b > 0, compact_a / compact_b, np.inf)
    return (np.abs(area_ratio - 1) < area_tol) & (
        np.abs(compact_ratio - 1) < compact_tol
    )


def _degrees_per_meter(gdf) -> tuple[float, float]:
    """(lon_deg_per_m, lat_deg_per_m) at *gdf*'s own latitude.

    Longitude degrees shrink toward the poles (`cos(latitude)`); latitude
    degrees don't. Used to convert a meter-based radius to degrees for a
    geographic-CRS input without reprojecting -- a per-call constant, cheap
    enough to not matter next to the `STRtree`/shape-gate cost it replaces
    a reprojection for.
    """
    lat = gdf.total_bounds[[1, 3]].mean()
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
    return 1 / m_per_deg_lon, 1 / m_per_deg_lat


def _estimate_offset_from_sample(
    unmatched_new,
    unmatched_old,
    sample_size=250,
    max_distance_m=5.0,
    area_tol=0.05,
    compact_tol=0.10,
    accept_radius_m=0.15,
    min_frac=0.6,
    min_n=20,
    rng=None,
):
    """Estimate a systematic (dx, dy) offset between two unmatched pools.

    Draws a random sample of *sample_size* rows from *unmatched_old*, pairs
    each sampled centroid to its nearest centroid in *unmatched_new* via a
    `shapely.STRtree` (point-to-point only -- no polygon overlay), and keeps
    pairs whose area and compactness are similar (`_shape_gate`). Detects a
    dominant offset from the gated pairs' own **overall median** (dx, dy) --
    not a grid-quantized bucket -- accepting it only if at least `min_frac`
    of gated pairs sit within `accept_radius_m` of that median: a continuous
    concentration check for whether a consistent offset exists at all,
    rather than a contaminated/scattered sample. (An earlier version of this
    check rounded (dx, dy) onto a `grid_m` grid and required a modal-bucket
    majority; besides being sensitive to where the grid happened to land
    relative to the noise scale -- empirically verified to have a
    data-dependent "dead zone" where one grid size did far worse than both
    finer and coarser ones, for no principled reason -- real per-parcel
    noise is routinely comparable to or larger than a grid fine enough to
    localize the offset well, which made genuine, usable offsets fail
    acceptance almost every time against real data.) The returned estimate
    is the median of just the *concentrated* subset (pairs within
    `accept_radius_m` of the initial overall median), refining the estimate
    the same way the old bucket membership did.

    Parameters
    ----------
    unmatched_new, unmatched_old : GeoDataFrame
        Remaining unmatched pools (any CRS). `unmatched_old` is assumed to be
        the side that may carry a systematic offset; the sample is drawn from
        it.
    max_distance_m : float
        Nearest-neighbor search radius, in meters.
    area_tol, compact_tol : float
        Relative tolerances for the shape gate.
    accept_radius_m : float
        Maximum distance (meters) from the gated pairs' overall median a
        pair may sit and still count toward `min_frac`.
    min_frac : float
        Minimum fraction of gated pairs that must sit within
        `accept_radius_m` of the overall median.
    min_n : int
        Minimum number of gated pairs required to attempt a decision.
    rng : numpy.random.Generator, optional

    Returns
    -------
    tuple(float, float, pyproj.CRS) or None
        `(dx, dy, crs)` -- the offset to *add* to `unmatched_old`'s geometry
        (in `crs`'s units) to align it with `unmatched_new`, or None if no
        sufficiently dominant offset was found. `crs` is `unmatched_old`'s
        own CRS unchanged when it's geographic (see below), or a fresh
        local metric CRS otherwise.
    """
    if len(unmatched_old) == 0 or len(unmatched_new) == 0:
        return None
    rng = rng if rng is not None else np.random.default_rng()

    # A geographic-CRS input (the real-world case -- every caller in this
    # codebase passes EPSG:4326) never needs reprojecting to a local metric
    # CRS at all: the STRtree search and the shape gate's area/compactness
    # *ratios* only ever compare two nearby points/polygons at essentially
    # the same latitude, so the local degrees-to-meters distortion is
    # ~constant across the pair and cancels almost exactly in a ratio (and
    # only weakly biases a raw-degree nearest-neighbor search -- not enough
    # to matter at a several-meter search radius among well-separated
    # parcels). Reprojection was measured to be 95% of this function's cost
    # on real data; skipping it is the single biggest lever on calibration
    # speed, since this runs up to `max_iters` times per group. The one
    # place true anisotropy matters -- converting the meter-based radius
    # parameters to degrees -- is handled by `_degrees_per_meter`. A
    # non-geographic input (no real caller today, but kept working per this
    # function's documented "any CRS" contract) falls back to the previous
    # local-metric-CRS reprojection.
    geographic = unmatched_old.crs is not None and unmatched_old.crs.is_geographic
    if geographic:
        crs = unmatched_old.crs
        old_m, new_m = unmatched_old, unmatched_new
        lon_deg_per_m, lat_deg_per_m = _degrees_per_meter(unmatched_old)
        search_radius = max_distance_m * max(lon_deg_per_m, lat_deg_per_m)
        accept_radius_x = accept_radius_m * lon_deg_per_m
        accept_radius_y = accept_radius_m * lat_deg_per_m
    else:
        crs = local_metric_crs(unmatched_old)
        old_m = unmatched_old.to_crs(crs)
        new_m = unmatched_new.to_crs(crs)
        search_radius = max_distance_m
        accept_radius_x = accept_radius_y = accept_radius_m

    n = min(sample_size, len(old_m))
    sample = old_m.iloc[rng.choice(len(old_m), size=n, replace=False)]

    with warnings.catch_warnings():
        if geographic:
            warnings.filterwarnings('ignore', 'Geometry is in a geographic CRS')
        tree = shapely.STRtree(new_m.geometry.centroid.values)
        idx, dist = tree.query_nearest(
            sample.geometry.centroid.values,
            max_distance=search_radius,
            return_distance=True,
        )
        if len(idx[0]) == 0:
            return None

        geom_old = sample.geometry.values[idx[0]]
        geom_new = new_m.geometry.values[idx[1]]
        gate = _shape_gate(geom_old, geom_new, area_tol, compact_tol)
        if gate.sum() < min_n:
            return None

        c_old = shapely.centroid(geom_old[gate])
        c_new = shapely.centroid(geom_new[gate])

    dx = shapely.get_x(c_new) - shapely.get_x(c_old)
    dy = shapely.get_y(c_new) - shapely.get_y(c_old)

    med_dx, med_dy = np.median(dx), np.median(dy)
    if geographic:
        concentrated = (np.abs(dx - med_dx) <= accept_radius_x) & (
            np.abs(dy - med_dy) <= accept_radius_y
        )
    else:
        concentrated = np.hypot(dx - med_dx, dy - med_dy) <= accept_radius_m
    if concentrated.mean() < min_frac:
        return None

    return float(np.median(dx[concentrated])), float(np.median(dy[concentrated])), crs


def _calibrate_pool(new_pool, old_pool, geo_id_col, max_iters, **kwargs):
    """Iteratively detect and remove a systematic offset from one pool.

    Repeatedly calls `_estimate_offset_from_sample` against the shrinking
    unmatched remainder, shifts, recomputes `geo_id_col`, and re-checks
    against `new_pool`'s geo_ids -- stopping once nothing is left or
    `max_iters` is reached. A round that finds no dominant offset (`None`)
    retries with a fresh random sample rather than giving up: a single
    unlucky draw isn't evidence the pool can't be calibrated, and not
    retrying left calibration silently contributing nothing on most
    real-world runs (see module-level notes). The round count is otherwise
    adaptive: a clean systematic shift may resolve in one round, while
    sub-hash-grid noise left over from an imprecise first estimate can take
    a couple more cheap rounds on the (shrinking) remainder to fully sweep
    up.

    Returns
    -------
    old_calibrated : GeoDataFrame
        `old_pool` with geometry translated and `geo_id_col` recomputed
        wherever calibration made progress (unresolved rows keep their
        original geo_id).
    newly_matched : pd.Series[bool]
        True for rows of `old_calibrated` whose recomputed geo_id now matches
        something in `new_pool`.
    """
    old_calibrated = old_pool.copy()
    new_ids = set(new_pool[geo_id_col])
    still_unmatched = old_calibrated

    for _ in range(max_iters):
        if len(still_unmatched) < MIN_CALIBRATION_ROWS or len(new_pool) == 0:
            break
        result = _estimate_offset_from_sample(new_pool, still_unmatched, **kwargs)
        if result is None:
            continue
        dx, dy, crs = result

        # crs is still_unmatched's own CRS unchanged when
        # _estimate_offset_from_sample took the geographic fast path (the
        # common case) -- shift in place, no reprojection round-trip.
        same_crs = crs == still_unmatched.crs
        shifted = still_unmatched if same_crs else still_unmatched.to_crs(crs)
        shifted = shifted.copy()
        shifted['geometry'] = shifted.translate(xoff=dx, yoff=dy)
        if not same_crs:
            shifted = shifted.to_crs(still_unmatched.crs)
        shifted[geo_id_col] = get_geo_ids(shifted, handle_duplicates=False)

        old_calibrated.loc[shifted.index, 'geometry'] = shifted['geometry']
        old_calibrated.loc[shifted.index, geo_id_col] = shifted[geo_id_col]
        still_unmatched = shifted[~shifted[geo_id_col].isin(new_ids)]

    newly_matched = old_calibrated[geo_id_col].isin(new_ids)
    return old_calibrated, newly_matched


def _calibrate_unmatched(
    new_fallback, old_fallback, geo_id_col, max_iters=3, group_col=None, **kwargs
):
    """Detect and remove a systematic offset from `old_fallback`, per `_calibrate_pool`.

    Parameters
    ----------
    group_col : str, optional
        Column present on both `new_fallback` and `old_fallback` (e.g. a
        town/admin4 id) to calibrate independently per group instead of
        pooling everything together. Which GIS system digitized a given
        parcel -- not the county it happens to sit in -- is what sets its
        systematic offset; pooling several differently-offset groups into
        one estimate blurs each individually-clean shift into noisy,
        harder-to-detect scatter (verified empirically: real MA data's
        per-town offset dispersion is roughly half the county-pooled
        dispersion, with genuinely different medians between towns, not
        just noise around one shared shift). Falls back to a single pooled
        `_calibrate_pool` call when `group_col` is `None` or absent from
        either side. A group below `MIN_CALIBRATION_ROWS` on either side,
        or a row whose group value is missing/unmapped, is left
        uncalibrated -- same fallback (kdtree, then overlay) as today.

    Returns
    -------
    Same as `_calibrate_pool`.
    """
    if (
        group_col is None
        or group_col not in old_fallback.columns
        or group_col not in new_fallback.columns
    ):
        return _calibrate_pool(
            new_fallback, old_fallback, geo_id_col, max_iters, **kwargs
        )

    calibrated_parts = []
    matched_parts = []
    old_groups = dict(tuple(old_fallback.groupby(group_col, dropna=True)))
    new_groups = dict(tuple(new_fallback.groupby(group_col, dropna=True)))
    unmapped_old = old_fallback[old_fallback[group_col].isna()]

    for group, old_group in old_groups.items():
        new_group = new_groups.get(group)
        if (
            new_group is None
            or len(old_group) < MIN_CALIBRATION_ROWS
            or len(new_group) < MIN_CALIBRATION_ROWS
        ):
            calibrated_parts.append(old_group)
            matched_parts.append(pd.Series(False, index=old_group.index))
            continue
        group_calibrated, group_matched = _calibrate_pool(
            new_group, old_group, geo_id_col, max_iters, **kwargs
        )
        calibrated_parts.append(group_calibrated)
        matched_parts.append(group_matched)

    if len(unmapped_old):
        calibrated_parts.append(unmapped_old)
        matched_parts.append(pd.Series(False, index=unmapped_old.index))

    old_calibrated = pd.concat(calibrated_parts).loc[old_fallback.index]
    newly_matched = pd.concat(matched_parts).loc[old_fallback.index]
    return old_calibrated, newly_matched


def _direct_kdtree_match(
    new_fallback, old_fallback, max_distance_m=2.0, area_tol=0.03, compact_tol=0.10
):
    """Directly match the full (non-sampled) remaining unmatched pools.

    A centroid `shapely.STRtree` nearest-neighbor query between *every*
    remaining unmatched row (not a sample), gated on shape similarity
    (`_shape_gate`) and accepted outright as matches -- no polygon overlay.
    Unlike `_estimate_offset_from_sample` (which only detects/corrects a
    systematic offset), this is a resolver in its own right, intended to run
    on whatever's left *after* calibration -- run this first, against an
    uncalibrated offset, and the radius would have to widen enough that in
    dense blocks of small adjacent parcels the nearest centroid could be the
    wrong neighboring parcel.

    `max_distance_m` default (2.0m) was validated against ground truth, not
    guessed: for each of 8 MA counties' real leftover-for-overlay pools
    (after today's calibration + a 0.5m first pass), computed the true best
    match per old parcel via full spatial intersection (requiring >=50%
    area overlap to count as confident ground truth), then checked every
    *new* match a wider radius would add against it. False-positive rate
    was 0.00% through 2.5m (41,938/46,423 new matches recovered across the
    8 counties, respectively, with zero disagreements) and only ticked up to
    0.004% at 3.0m (2 of 49,404) -- 2.0m keeps a margin below where any
    false positive was observed while recovering the large majority of that
    headroom, since per-town calibration now runs first and removes most of
    the systematic-offset risk a tighter radius was originally guarding
    against.

    A candidate pair is only accepted if it is each side's unique nearest
    match passing the gate (mirroring the exact-hash stage's restriction to
    non-duplicated geo_ids) -- an ambiguous pair is left for overlay rather
    than guessed at.

    Returns
    -------
    pd.DataFrame
        Columns `parcel_id_old`, `parcel_id_new` -- one row per matched pair.
    """
    empty = pd.DataFrame(columns=['parcel_id_old', 'parcel_id_new'])
    if len(new_fallback) == 0 or len(old_fallback) == 0:
        return empty

    crs = local_metric_crs(old_fallback)
    old_m = old_fallback.to_crs(crs)
    new_m = new_fallback.to_crs(crs)

    tree = shapely.STRtree(new_m.geometry.centroid.values)
    idx, dist = tree.query_nearest(
        old_m.geometry.centroid.values,
        max_distance=max_distance_m,
        return_distance=True,
    )
    if len(idx[0]) == 0:
        return empty

    gate = _shape_gate(
        old_m.geometry.values[idx[0]],
        new_m.geometry.values[idx[1]],
        area_tol,
        compact_tol,
    )
    if not gate.any():
        return empty

    pairs = pd.DataFrame(
        {
            'parcel_id_old': old_m['parcel_id_old'].to_numpy()[idx[0][gate]],
            'parcel_id_new': new_m['parcel_id_new'].to_numpy()[idx[1][gate]],
        }
    )
    unambiguous = ~pairs['parcel_id_old'].duplicated(keep=False) & ~pairs[
        'parcel_id_new'
    ].duplicated(keep=False)
    return pairs[unambiguous]


def build_id_or_overlay_crosswalk(
    new,
    old,
    new_id_col: str | None = None,
    old_id_col: str | None = None,
    geo_id_col: str = 'geo_id',
    on_duplicate_geo_id: str = 'raise',
    min_overlap_m2: float = 10.0,
    use_duckdb: bool = True,
    calibrate: bool = True,
    calibrate_sample_size: int = 250,
    calibrate_max_iters: int = 3,
    calibrate_group_col: str | None = None,
    kdtree_max_distance_m: float = 2.0,
    verbose: bool = False,
    timer=None,
) -> pd.DataFrame:
    """Build a fractional crosswalk from *old* to *new* parcel geometries.

    Four-stage linkage, cheapest first: an exact-match fast path on
    *geo_id_col* (computed fresh from geometry if not already a column on
    either input); a systematic-offset calibration pass that detects and
    removes a consistent (dx, dy) shift between the two sides and retries the
    exact match (`_calibrate_unmatched`); a full centroid-proximity/shape-gate
    direct match for whatever's left (`_direct_kdtree_match`); and, only for
    what still doesn't resolve, a polygon-overlay fallback
    (`overlay_polygons_with_duckdb` / `overlay_polygons`,
    ``area_intersection=True``).

    The middle two stages exist because two datasets of "the same" parcels
    are rarely reprojected identically -- a systematic datum/reprojection
    offset, or plain vertex-level noise between providers, can leave the
    exact hash failing for a large share of otherwise-identical parcels even
    though full polygon overlay would trivially confirm the match. Both are
    skipped (falling straight to overlay, as before they existed) when either
    side's post-hash-match remainder is smaller than a few dozen rows, since
    a systematic offset can't be estimated from too few points.

    Parameters
    ----------
    new, old : geopandas.GeoDataFrame
        Current and legacy parcel geometries.
    new_id_col, old_id_col : str, optional
        Column holding each side's parcel id; defaults to the frame's index.
    geo_id_col : str, default 'geo_id'
        Column used for the exact-match fast path.
    on_duplicate_geo_id : str, default 'raise'
        A duplicated *geo_id* on either side can only arise from genuinely
        overlapping or duplicate-record parcels (e.g. several condo-unit
        records sharing one physical footprint) -- a structural fact about
        the input, not matching noise. By default this raises `ValueError`
        rather than silently deferring those rows to overlay. Set to
        `'first'` to keep an arbitrary one row per duplicate group, or
        `'groupby'` to aggregate each group's attributes via
        `openplaces.io.aggregate.aggregate_rows` (preserving the original ids
        as a `{id_col}_list` column). See `_resolve_duplicate_geo_ids`.
    min_overlap_m2 : float, default 10.0
        Drop overlay-fallback pairs with less overlap than this (excludes
        sliver/edge-touch artifacts).
    use_duckdb : bool, default True
        Use `overlay_polygons_with_duckdb` (auto-falls back to
        `overlay_polygons` on complexity/OOM) for the fallback step.
    calibrate : bool, default True
        Run the systematic-offset calibration and direct-match stages before
        falling back to overlay. Set False to reproduce the old
        hash-then-overlay-only behavior.
    calibrate_sample_size : int, default 250
        Sample size per calibration round (`_estimate_offset_from_sample`).
    calibrate_max_iters : int, default 3
        Maximum calibration rounds against the shrinking unmatched remainder
        (`_calibrate_unmatched`); stops earlier once nothing is left.
    calibrate_group_col : str, optional
        Column present on both *new* and *old* to calibrate independently
        per group (e.g. a town/admin4 id) instead of pooling everything
        together -- see `_calibrate_unmatched`. `None` (default) pools the
        whole input into a single calibration attempt, as before this
        parameter existed.
    kdtree_max_distance_m : float, default 2.0
        Search radius for the direct centroid match (`_direct_kdtree_match`),
        run after calibration on whatever's still unmatched -- see that
        function's docstring for the ground-truth validation behind this
        default.
    verbose : bool, default False
        Print each stage's match rate before the next, more expensive stage
        runs -- overlay is the expensive step, so this reports how much of
        the work the earlier stages already resolved beforehand.
    timer : openplaces.timing.Timer, optional
        If given, `timer.mark(...)` is called after each of the four stages
        (`'exact geo_id match'`, `'systematic-offset calibration'`,
        `'direct kdtree match'`, `'polygon overlay'`), so a caller's own
        surrounding timer breaks down which stage actually dominates a given
        run instead of lumping the whole function under one mark.

    Returns
    -------
    pd.DataFrame
        Columns: ``parcel_id_old``, ``parcel_id_new``, ``area_ha``,
        ``match_type`` (``'geo_id'``, ``'geo_id_calibrated'``,
        ``'geo_id_kdtree'``, or ``'overlay'``), ``fraction_of_old``
        (share of *old* parcel's area assigned to this *new* parcel; sums to
        1.0 within each ``parcel_id_old`` group among matched rows).
    """
    new = new.copy()
    old = old.copy()
    new['parcel_id_new'] = _id_series(new, new_id_col).to_numpy()
    old['parcel_id_old'] = _id_series(old, old_id_col).to_numpy()
    new = _with_geo_id(new, geo_id_col)
    old = _with_geo_id(old, geo_id_col)
    new = _resolve_duplicate_geo_ids(
        new, geo_id_col, 'parcel_id_new', on_duplicate_geo_id, 'new'
    )
    old = _resolve_duplicate_geo_ids(
        old, geo_id_col, 'parcel_id_old', on_duplicate_geo_id, 'old'
    )

    id_pairs = new[['parcel_id_new', geo_id_col]].merge(
        old[['parcel_id_old', geo_id_col]], on=geo_id_col, how='inner'
    )

    if id_pairs.empty:
        id_matches = pd.DataFrame(columns=CROSSWALK_COLUMNS)
    else:
        area_ha = get_areas(
            old.set_index('parcel_id_old').loc[id_pairs['parcel_id_old'], ['geometry']],
            'ha',
        )
        id_matches = pd.DataFrame(
            {
                'parcel_id_old': id_pairs['parcel_id_old'].to_numpy(),
                'parcel_id_new': id_pairs['parcel_id_new'].to_numpy(),
                'area_ha': area_ha.to_numpy(),
                'match_type': 'geo_id',
                'fraction_of_old': 1.0,
            }
        )

    new_fallback = new[~new['parcel_id_new'].isin(id_matches['parcel_id_new'])]
    old_fallback = old[~old['parcel_id_old'].isin(id_matches['parcel_id_old'])]
    if timer:
        timer.mark('exact geo_id match')

    if verbose:
        n_new_matched = len(new) - len(new_fallback)
        n_old_matched = len(old) - len(old_fallback)
        print(
            f'  geo_id match: {n_new_matched:,}/{len(new):,} new parcels '
            f'({n_new_matched / len(new):.1%}), '
            f'{n_old_matched:,}/{len(old):,} old parcels '
            f'({n_old_matched / len(old):.1%})'
        )

    def _match_pairs_to_crosswalk(pairs, match_type):
        if pairs.empty:
            return pd.DataFrame(columns=CROSSWALK_COLUMNS)
        area_ha = get_areas(
            old.set_index('parcel_id_old').loc[pairs['parcel_id_old'], ['geometry']],
            'ha',
        )
        return pd.DataFrame(
            {
                'parcel_id_old': pairs['parcel_id_old'].to_numpy(),
                'parcel_id_new': pairs['parcel_id_new'].to_numpy(),
                'area_ha': area_ha.to_numpy(),
                'match_type': match_type,
                'fraction_of_old': 1.0,
            }
        )

    calibrated_matches = pd.DataFrame(columns=CROSSWALK_COLUMNS)
    kdtree_matches = pd.DataFrame(columns=CROSSWALK_COLUMNS)
    can_run_cheap_stages = (
        calibrate
        and len(new_fallback) >= MIN_CALIBRATION_ROWS
        and len(old_fallback) >= MIN_CALIBRATION_ROWS
    )

    if can_run_cheap_stages:
        old_calibrated, newly_matched = _calibrate_unmatched(
            new_fallback,
            old_fallback,
            geo_id_col,
            max_iters=calibrate_max_iters,
            group_col=calibrate_group_col,
            sample_size=calibrate_sample_size,
        )
        resolved = old_calibrated[newly_matched]
        new_fallback_unique = new_fallback[
            ~new_fallback[geo_id_col].duplicated(keep=False)
        ]
        resolved_unique = resolved[~resolved[geo_id_col].duplicated(keep=False)]
        calibrated_pairs = resolved_unique[['parcel_id_old', geo_id_col]].merge(
            new_fallback_unique[['parcel_id_new', geo_id_col]],
            on=geo_id_col,
            how='inner',
        )
        calibrated_matches = _match_pairs_to_crosswalk(
            calibrated_pairs, 'geo_id_calibrated'
        )
        if verbose:
            print(
                f'  calibration probe: recovered {len(calibrated_matches):,} '
                'additional matches via systematic-offset alignment'
            )

        new_fallback = new_fallback[
            ~new_fallback['parcel_id_new'].isin(calibrated_matches['parcel_id_new'])
        ]
        old_fallback = old_calibrated[
            ~old_calibrated['parcel_id_old'].isin(calibrated_matches['parcel_id_old'])
        ]
    if timer:
        timer.mark('systematic-offset calibration')

    can_run_cheap_stages = (
        calibrate
        and len(new_fallback) >= MIN_CALIBRATION_ROWS
        and len(old_fallback) >= MIN_CALIBRATION_ROWS
    )

    if can_run_cheap_stages:
        kdtree_pairs = _direct_kdtree_match(
            new_fallback, old_fallback, max_distance_m=kdtree_max_distance_m
        )
        kdtree_matches = _match_pairs_to_crosswalk(kdtree_pairs, 'geo_id_kdtree')
        if verbose:
            print(
                f'  direct match: recovered {len(kdtree_matches):,} additional '
                'matches via centroid/shape-gate proximity'
            )

        new_fallback = new_fallback[
            ~new_fallback['parcel_id_new'].isin(kdtree_matches['parcel_id_new'])
        ]
        old_fallback = old_fallback[
            ~old_fallback['parcel_id_old'].isin(kdtree_matches['parcel_id_old'])
        ]
    if timer:
        timer.mark('direct kdtree match')

    if verbose:
        print(
            f'  running overlay for the remaining '
            f'{len(new_fallback):,}/{len(old_fallback):,}'
        )

    if new_fallback.empty or old_fallback.empty:
        overlay_matches = pd.DataFrame(columns=CROSSWALK_COLUMNS)
    else:
        overlay_fn = overlay_polygons_with_duckdb if use_duckdb else overlay_polygons
        overlay = overlay_fn(
            new_fallback.set_index('parcel_id_new')[['geometry']],
            old_fallback.set_index('parcel_id_old')[['geometry']],
            area_intersection=True,
            how='intersection',
            suffixes=('_new', '_old'),
        ).reset_index()
        overlay = overlay[overlay['area_intersection_m2'] >= min_overlap_m2].copy()
        overlay['area_ha'] = overlay['area_intersection_m2'] / 1e4
        overlay['match_type'] = 'overlay'
        overlay['fraction_of_old'] = overlay['area_intersection_m2'] / overlay.groupby(
            'parcel_id_old'
        )['area_intersection_m2'].transform('sum')
        overlay_matches = overlay[CROSSWALK_COLUMNS]
    if timer:
        timer.mark('polygon overlay')

    crosswalk = pd.concat(
        [id_matches, calibrated_matches, kdtree_matches, overlay_matches],
        ignore_index=True,
    )
    return crosswalk.sort_values(['parcel_id_old', 'parcel_id_new']).reset_index(
        drop=True
    )


def warn_on_geo_id_area_mismatch(
    crosswalk: pd.DataFrame,
    new,
    old,
    new_id_col: str | None = None,
    old_id_col: str | None = None,
    tolerance: float = 0.01,
    n_examples: int = 3,
    silent: bool = False,
) -> pd.DataFrame:
    """Flag geo_id-matched crosswalk pairs whose areas disagree beyond *tolerance*.

    A ``geo_id`` match implies (near-)identical geometry, so
    ``area_new / area_old`` should sit within a fraction of a percent of 1.0;
    larger deviations point at a reprojection issue, a duplicate-suffix
    collision, or a genuinely coincidental hash collision. Restricted to
    ``crosswalk['match_type'] == 'geo_id'`` rows — overlay-fallback rows are
    not checked, since partial-overlap area mismatch there is expected, not a
    QA signal.

    When not *silent*, plots the most-similar-ratio pair, the
    least-similar-ratio pair, and up to ``n_examples - 2`` random flagged
    pairs (old boundary vs. new boundary overlaid, no basemap), then raises a
    `UserWarning` naming the flagged count and rate.

    Parameters
    ----------
    crosswalk : pd.DataFrame
        Output of :func:`build_id_or_overlay_crosswalk`.
    new, old : geopandas.GeoDataFrame
        The same frames the crosswalk was built from.
    new_id_col, old_id_col : str, optional
        Column holding each side's parcel id; defaults to the frame's index.
    tolerance : float, default 0.01
        Maximum allowed ``|area_new / area_old - 1|``. Default 1%, generous
        enough to absorb reprojection/rounding noise while catching genuine
        mismatches.
    n_examples : int, default 3
        Number of example pairs to plot (most-similar and least-similar are
        always included first; the rest are random).
    silent : bool, default False
        Suppress both the plot and the warning.

    Returns
    -------
    pd.DataFrame
        The flagged subset of *crosswalk* (with an added ``ratio`` column),
        regardless of *silent*.
    """
    matched = crosswalk[crosswalk['match_type'] == 'geo_id']
    if matched.empty:
        return matched.assign(ratio=pd.Series(dtype=float))

    new_by_id = new.set_index(_id_series(new, new_id_col))[['geometry']]
    old_by_id = old.set_index(_id_series(old, old_id_col))[['geometry']]

    area_new = get_areas(new_by_id.loc[matched['parcel_id_new']], 'ha').to_numpy()
    area_old = get_areas(old_by_id.loc[matched['parcel_id_old']], 'ha').to_numpy()
    result = matched.assign(ratio=area_new / area_old)
    flagged = result[(result['ratio'] - 1).abs() > tolerance]

    if silent or flagged.empty:
        return flagged

    rate = len(flagged) / len(matched)
    _plot_examples(flagged, new_by_id, old_by_id, n_examples)
    warnings.warn(
        f'{len(flagged)} of {len(matched)} geo_id-matched pairs ({rate:.1%}) have '
        f'area ratios outside ±{tolerance:.1%}. Inspect the plotted examples, '
        'or pass silent=True to suppress this check.',
        UserWarning,
        stacklevel=2,
    )
    return flagged


def _plot_examples(flagged, new_by_id, old_by_id, n_examples):
    """Plot the most-similar, least-similar, and random flagged pairs."""
    import matplotlib.pyplot as plt

    by_dev = flagged.assign(dev=(flagged['ratio'] - 1).abs()).sort_values('dev')
    examples = {'most similar': by_dev.iloc[0]}
    if len(by_dev) > 1:
        examples['least similar'] = by_dev.iloc[-1]
    n_random = max(n_examples - len(examples), 0)
    remaining = by_dev.iloc[1:-1] if len(by_dev) > 2 else by_dev.iloc[0:0]
    if n_random and len(remaining):
        rng = np.random.default_rng()
        pick_size = min(n_random, len(remaining))
        idx = rng.choice(len(remaining), size=pick_size, replace=False)
        picks = remaining.iloc[idx]
        for i, (_, row) in enumerate(picks.iterrows()):
            examples[f'random {i + 1}' if n_random > 1 else 'random'] = row

    fig, axes = plt.subplots(1, len(examples), figsize=(5 * len(examples), 5))
    axes = np.atleast_1d(axes)
    for ax, (label, row) in zip(axes, examples.items(), strict=False):
        old_by_id.loc[[row['parcel_id_old']]].boundary.plot(
            ax=ax, color='tab:blue', label='old'
        )
        new_by_id.loc[[row['parcel_id_new']]].boundary.plot(
            ax=ax, color='tab:orange', label='new'
        )
        ax.set_title(f'{label}\nratio={row["ratio"]:.3f}')
        ax.set_aspect('equal')
        ax.axis('off')
    axes[0].legend()
    fig.tight_layout()
    plt.show()
