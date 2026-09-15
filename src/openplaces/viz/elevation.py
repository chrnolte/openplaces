"""
Terrain-elevation sourcing for the 3D land/building viz.

Samples the ingested USGS 3DEP DEM (see
``recipes/US/_all/land/elevation/usgs/3dep``) to ground `viz.terrain`'s
extruded polygons in real-world elevation: per-vertex draping for land
polygons (`drape_parcel_elevation`), and a single flat zonal-mean elevation
per building footprint (`get_building_elevation`). Results are cached to
disk under `cfg.cache_dir`, keyed by each row's own index (e.g.
``parcel_id`` / ``footprint_id``) -- since that index already encodes a
geometry-shape hash (see `geo.ids.get_geo_ids`), an unchanged shape reuses
its cached value and only new/changed rows touch the raster again. If an
admin unit's DEM hasn't been ingested yet, it's ingested automatically
on first use (see `_ingest_missing_dem`).
"""

import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
import shapely

from openplaces.config import cfg
from openplaces.core.schema import AdminId
from openplaces.geo.raster import sample_raster_at_points, zonal_stats_with_exactextract
from openplaces.io.readers import get_dataset

__all__ = [
    'drape_parcel_elevation',
    'mesh_parcel_elevation',
    'subdivide_polygons',
    'get_building_elevation',
    'get_elevation_datum',
    'resolve_dem_admin_ids',
    'add_z_offset',
    'clamp_z',
    'scale_z',
]


def drape_parcel_elevation(
    gdf: gpd.GeoDataFrame,
    elevation_recipe,
    admin_id_column: str = 'admin3_id',
    cache: bool = True,
    silent: bool = False,
    cache_key: str = '',
):
    """Drape land/parcel polygons onto the ground via per-vertex elevation.

    Extracts every unique vertex position across `gdf` (deduplicating
    shared corners between adjacent parcels), samples the DEM at each
    (vectorized -- see `geo.raster.sample_raster_at_points`), and rebuilds
    each polygon with that elevation as its per-vertex Z coordinate, so the
    resulting geometry follows the real terrain along its whole boundary
    rather than sitting at one flat elevation. Buildings should use
    `get_building_elevation` instead -- a building's base must stay flat.

    Results are cached to disk per admin unit under `cfg.cache_dir`, keyed
    by `gdf`'s own index (e.g. ``parcel_id``); an id already present in the
    cache is reused without resampling the raster (see module docstring).

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Land/parcel polygons to drape.
    elevation_recipe : str or dict
        DEM dataset recipe (e.g. ``'US_land-elevation-usgs-3dep'``), passed
        to `io.readers.get_dataset` per admin unit. Ingested automatically
        (via `io.ingester.ingest`) for any admin unit whose DEM doesn't
        exist on disk yet -- a one-time cost per admin unit.
    admin_id_column : str
        Column on `gdf` naming each row's admin unit at the DEM's own
        ingestion granularity (default ``'admin3_id'``, matching both the
        DEM recipe's and the curated parcel/footprint entities' county
        level) -- used to resolve which DEM tile covers each row and to
        scope the on-disk cache file.
    cache : bool
        If True (default), read/write the on-disk elevation cache. Set
        False to always resample from the raster.
    silent : bool
        If True, suppress the out-of-extent/nodata-vertex warning.
    cache_key : str
        Appended to the cache file name. Required whenever the caller has
        changed the geometry under an unchanged index (a simplified
        polygon keeps its id), or the cache would serve the old shape.

    Returns
    -------
    geometry : geopandas.GeoSeries
        `gdf`'s geometry, unchanged in x/y, with each vertex's Z set to
        its bilinearly sampled elevation.
    mean_elevation : numpy.ndarray
        Each row's own vertex-elevation mean -- a single representative
        ground elevation per row, for ``total_elevation`` bookkeeping
        (`viz.terrain`) even though the geometry itself varies per vertex.
    """

    def compute_miss(miss_gdf, dem_path):
        return _drape_miss(miss_gdf, dem_path, silent=silent)

    combined = _grouped_cache_compute(
        gdf,
        elevation_recipe,
        admin_id_column,
        cache,
        kind=f'parcel_elevation_bilinear{cache_key}',
        geom=True,
        compute_miss=compute_miss,
        silent=silent,
    )
    return (
        gpd.GeoSeries(combined['geometry'], crs=gdf.crs),
        combined['mean_elevation'].to_numpy(dtype=float),
    )


def mesh_parcel_elevation(
    gdf: gpd.GeoDataFrame,
    elevation_recipe,
    admin_id_column: str = 'admin3_id',
    cells: int = 8,
    min_resolution: float = 10.0,
    relief_threshold: float = 0.5,
    profile_tolerance: float = 0.5,
    cache: bool = True,
    silent: bool = False,
    cache_key: str = '',
):
    """Build a terrain-following cap mesh inside each land/parcel polygon.

    `drape_parcel_elevation` gives each boundary vertex its own elevation,
    which is right for the walls of an extruded polygon but wrong for its
    cap: deck.gl triangulates the cap from the ring's x/y alone and lifts
    each triangle corner to its own z, so a large polygon whose road edge
    carries hundreds of vertices is fanned into slivers reaching across the
    whole parcel, each tilted by its three corners. Rendered, that is a
    hatched, partly black fill. No 2D triangulation of a big non-planar
    ring is good, so this function stops asking for one.

    Each polygon is instead subdivided by clipping a square grid to it:
    cells fully inside become two triangles each (a quad with four
    independent elevations is non-planar again), and cells crossing the
    boundary become the cell's intersection with the polygon. Whatever the
    renderer does inside a piece no wider than one cell is off by less than
    one cell of terrain, which is invisible.

    Interior vertices are sampled on the DEM (bilinear). A vertex on the
    polygon's boundary is not: it takes the linear interpolation between
    the two ring vertices it lies between. Extruded, the pieces' outer
    faces are the polygon's walls, and this rule makes a wall top the
    same chord however the grid happened to cut it, so two neighbors
    sharing an edge (identical after `shapely.coverage_simplify`) raise
    identical wall tops and the surface stays closed between them.

    The grid is adaptive so the cost stays bounded. The cell size is
    `max(min_resolution, sqrt(bbox_area) / cells)`, so a polygon gets at
    most about `cells ** 2` cells whatever its size, and a polygon whose
    boundary elevation range is below `relief_threshold` is not subdivided
    at all: it gets one flat piece at its boundary mean, which in gently
    sloped terrain is most polygons. A polygon smaller than one cell is
    likewise one piece, its own shape, at a size where the fan is harmless.

    Results are cached to disk per admin unit under `cfg.cache_dir`, keyed
    by `gdf`'s own index like `drape_parcel_elevation`, under a cache kind
    that encodes the grid parameters.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Land/parcel polygons to mesh.
    elevation_recipe : str or dict
        DEM dataset recipe, as for `drape_parcel_elevation`.
    admin_id_column : str
        Column naming each row's admin unit at the DEM's ingestion
        granularity, as for `drape_parcel_elevation`.
    cells : int
        Target number of grid cells across a polygon's bounding box, per
        axis. Bounds the mesh at roughly `cells ** 2` cells per polygon.
    min_resolution : float
        Smallest grid cell size in meters. Below it a polygon is one piece.
    relief_threshold : float
        Boundary elevation range in meters (raw terrain, before any
        exaggeration) under which a polygon is not subdivided.
    profile_tolerance : float
        Vertical tolerance in meters for thinning the densified rings
        after draping (`thin_profile`): a vertex that `_densify` added
        along a straight edge is dropped where the terrain profile
        between its kept neighbors stays within this of a straight line.
        Corner vertices are never dropped, so neighbors sharing an edge
        keep identical vertices. Default 0.5.
    cache : bool
        If True (default), read/write the on-disk elevation cache.
    silent : bool
        If True, suppress the out-of-extent/nodata-vertex warning.
    cache_key : str
        Appended to the cache file name; see `drape_parcel_elevation`.

    Returns
    -------
    cap : geopandas.GeoSeries
        One 3D (Multi)Polygon per row holding the mesh pieces, in `gdf`'s
        CRS, each vertex's Z set to its sampled elevation.
    walls : geopandas.GeoSeries
        The ring to extrude the walls from: `drape_parcel_elevation`'s
        per-vertex ring for a meshed row, and the ring flattened to the
        boundary mean for a row whose cap is flat, so the wall top and
        the cap meet exactly in both cases.
    mean_elevation : numpy.ndarray
        Each row's boundary-vertex elevation mean, identical to what
        `drape_parcel_elevation` reports for the same rows.
    """

    def compute_miss(miss_gdf, dem_path):
        return _mesh_miss(
            miss_gdf,
            dem_path,
            cells,
            min_resolution,
            relief_threshold,
            profile_tolerance,
            silent,
        )

    kind = (
        f'parcel_mesh_v6_c{cells}_r{min_resolution:g}_t{relief_threshold:g}'
        f'_p{profile_tolerance:g}{cache_key}'
    )
    combined = _grouped_cache_compute(
        gdf,
        elevation_recipe,
        admin_id_column,
        cache,
        kind=kind,
        geom=True,
        compute_miss=compute_miss,
        silent=silent,
    )
    return (
        gpd.GeoSeries(combined['geometry'], crs=gdf.crs),
        gpd.GeoSeries(combined['walls'], crs=gdf.crs),
        combined['mean_elevation'].to_numpy(dtype=float),
    )


def subdivide_polygons(polygons, cells: int = 8, min_resolution: float = 10.0):
    """Clip an adaptive square grid to each polygon; see `mesh_parcel_elevation`.

    Parameters
    ----------
    polygons : array-like of shapely (Multi)Polygon
        2D polygons in a projected CRS with meter units.
    cells : int
        Target grid cells per axis across each polygon's bounding box.
    min_resolution : float
        Smallest cell size in meters.

    Returns
    -------
    pieces : numpy.ndarray of shapely.Polygon
        Triangles (cells fully inside) and clipped cell pieces (cells on
        the boundary). Every piece is a single Polygon. Sorted by `owner`.
    owner : numpy.ndarray of int
        Index into `polygons` of the polygon each piece belongs to.
    """
    polygons = np.asarray(polygons, dtype=object)
    if len(polygons) == 0:
        return np.empty(0, dtype=object), np.empty(0, dtype=np.int64)

    minx, miny, maxx, maxy = shapely.bounds(polygons).T
    width = maxx - minx
    height = maxy - miny
    # Bounding-box area rather than polygon area, so a long diagonal
    # polygon cannot exceed the per-polygon cell budget.
    size = np.maximum(min_resolution, np.sqrt(width * height) / cells)
    n_x = np.maximum(np.ceil(width / size), 1).astype(np.int64)
    n_y = np.maximum(np.ceil(height / size), 1).astype(np.int64)

    n_cells = n_x * n_y
    owner = np.repeat(np.arange(len(polygons)), n_cells)
    offsets = np.repeat(np.cumsum(n_cells) - n_cells, n_cells)
    local = np.arange(int(n_cells.sum())) - offsets
    col = local % n_x[owner]
    row = local // n_x[owner]
    x0 = minx[owner] + col * size[owner]
    y0 = miny[owner] + row * size[owner]
    x1 = x0 + size[owner]
    y1 = y0 + size[owner]
    boxes = shapely.box(x0, y0, x1, y1)

    # `contains`, not `contains_properly`: a cell on the polygon's
    # rim touches its boundary and would otherwise fall through to
    # the clipping path as a full, non-planar quad.
    shapely.prepare(polygons)
    inside = shapely.contains(polygons[owner], boxes)
    crossing = ~inside & shapely.intersects(polygons[owner], boxes)

    # Two triangles per interior cell, split along the same diagonal
    # everywhere so neighboring cells share their edge vertices.
    ix0, iy0, ix1, iy1 = x0[inside], y0[inside], x1[inside], y1[inside]
    lower = np.stack(
        [np.stack([ix0, iy0], -1), np.stack([ix1, iy0], -1), np.stack([ix1, iy1], -1)],
        axis=1,
    )
    upper = np.stack(
        [np.stack([ix0, iy0], -1), np.stack([ix1, iy1], -1), np.stack([ix0, iy1], -1)],
        axis=1,
    )
    triangles = np.concatenate([shapely.polygons(lower), shapely.polygons(upper)])
    triangle_owner = np.concatenate([owner[inside], owner[inside]])

    clipped = shapely.intersection(boxes[crossing], polygons[owner[crossing]])
    parts, part_index = shapely.get_parts(clipped, return_index=True)
    is_polygon = (shapely.get_type_id(parts) == shapely.GeometryType.POLYGON) & ~(
        shapely.is_empty(parts)
    )
    parts = parts[is_polygon]
    part_owner = owner[crossing][part_index[is_polygon]]

    pieces = np.concatenate([triangles, parts])
    piece_owner = np.concatenate([triangle_owner, part_owner])
    order = np.argsort(piece_owner, kind='stable')
    # Clockwise exteriors throughout: deck.gl assumes that winding when
    # lonboard hands it geometry unnormalized, and culls the rest. GEOS
    # emits clipped pieces clockwise; the triangles above are not.
    pieces = shapely.orient_polygons(pieces[order], exterior_cw=True)
    return pieces, piece_owner[order]


def _mesh_miss(
    miss_gdf: gpd.GeoDataFrame,
    dem_path,
    cells: int,
    min_resolution: float,
    relief_threshold: float,
    profile_tolerance: float,
    silent: bool,
):
    """Per-row cap mesh + boundary mean elevation for uncached rows."""
    # Densify the rings first. A wall top is the chord between two ring
    # vertices, and a mesh vertex landing on that edge takes the chord
    # too (so both sides of a shared edge agree), which is only right
    # while the chord is short: a 6-vertex parcel with 160 m edges over
    # 56 m of relief (Lancaster, 2026-08-26) put its rim 39 m under the
    # terrain and folded the cap down to meet it. Capping segments at
    # the cell floor bounds the chord error by the relief within one
    # cell, the same tolerance the cap itself has. `segmentize` splits a
    # segment into equal parts from its endpoints alone, so two
    # neighbors sharing an edge still get identical vertices.
    miss_gdf = miss_gdf.copy()
    miss_gdf.geometry = _densify(miss_gdf.geometry, min_resolution)
    draped = _drape_miss(miss_gdf, dem_path, silent=silent)
    # ... then take back the added vertices the terrain does not need:
    # along a straight run the profile is often close to a straight
    # line too, and every rim vertex costs a triangle plus three walls.
    draped.geometry = thin_profile(draped.geometry, profile_tolerance)
    miss_gdf.geometry = gpd.GeoSeries(
        shapely.force_2d(draped.geometry.to_numpy()),
        index=miss_gdf.index,
        crs=miss_gdf.crs,
    )
    mean_elevation = draped['mean_elevation'].to_numpy(dtype=float)
    xyz, row_index = shapely.get_coordinates(
        draped.geometry.to_numpy(), include_z=True, return_index=True
    )
    z_by_row = pd.Series(xyz[:, 2]).groupby(row_index)
    relief = (
        (z_by_row.max() - z_by_row.min())
        .reindex(range(len(miss_gdf)))
        .fillna(0.0)
        .to_numpy()
    )
    subdivide = relief > relief_threshold

    flat_2d = shapely.force_2d(miss_gdf.geometry.to_numpy())
    mesh = np.empty(len(miss_gdf), dtype=object)
    walls = draped.geometry.to_numpy().copy()
    for i in np.flatnonzero(~subdivide):
        # Flat cap, flat walls: a draped wall under a flat cap would
        # leave a notch of up to `relief_threshold` at the rim.
        mesh[i] = shapely.force_3d(
            shapely.orient_polygons(flat_2d[i], exterior_cw=True), z=mean_elevation[i]
        )
        walls[i] = mesh[i]

    if subdivide.any():
        source = gpd.GeoSeries(flat_2d[subdivide], crs=miss_gdf.crs)
        metric_crs = (
            source.crs if source.crs.is_projected else source.estimate_utm_crs()
        )
        metric = source.to_crs(metric_crs).to_numpy()
        pieces, owner = subdivide_polygons(metric, cells, min_resolution)
        pieces = _insert_ring_vertices(pieces, owner, metric)
        pieces, owner = _triangulate_pieces(pieces, owner)

        xy, piece_index = shapely.get_coordinates(pieces, return_index=True)
        unique_xy, first, inverse = np.unique(
            xy, axis=0, return_index=True, return_inverse=True
        )
        inverse = np.asarray(inverse).reshape(-1)
        with rasterio.open(dem_path) as src:
            dem_crs = src.crs
        sample_x, sample_y = _reproject_points(
            unique_xy[:, 0], unique_xy[:, 1], metric_crs, dem_crs
        )
        unique_z = _fill_unsampled_from_nearest(
            unique_xy,
            sample_raster_at_points(
                dem_path, sample_x, sample_y, interpolation='bilinear'
            ),
            silent=True,
        )
        unique_z = _interpolate_edge_vertices(
            unique_xy,
            unique_z,
            owner[piece_index[first]],
            metric,
            xyz[subdivide[row_index]],
            row_index[subdivide[row_index]],
        )
        out_x, out_y = _reproject_points(
            unique_xy[:, 0], unique_xy[:, 1], metric_crs, miss_gdf.crs
        )
        # A ring vertex must come back with the ring's own xy, not the
        # metric round trip's nanometer-shifted copy, so the outline
        # and ghost layers built from the ring coincide with the mesh.
        ring_xyz = xyz[subdivide[row_index]]
        out_x, out_y, unique_z = _snap_to_ring(
            out_x, out_y, unique_z, ring_xyz, miss_gdf.crs
        )
        coords = np.column_stack([out_x, out_y, unique_z])[inverse]
        pieces = shapely.set_coordinates(shapely.force_3d(pieces), coords)
        mesh[np.flatnonzero(subdivide)] = shapely.multipolygons(pieces, indices=owner)

    return gpd.GeoDataFrame(
        {
            'geometry': mesh,
            'walls': gpd.GeoSeries(walls, index=miss_gdf.index, crs=miss_gdf.crs),
            'mean_elevation': mean_elevation,
        },
        index=miss_gdf.index,
        crs=miss_gdf.crs,
    )


def _densify(geometry: gpd.GeoSeries, max_segment: float) -> gpd.GeoSeries:
    """Insert vertices so no segment exceeds `max_segment` meters.

    Measured in a metric CRS (the geometry's own when projected, else its
    UTM zone) and returned in the input CRS. Original vertices survive
    the round trip to within nanometers; callers that match vertices
    exactly must tolerate that (`_snap_to_ring` does).
    """
    if geometry.crs is None or geometry.crs.is_projected:
        return gpd.GeoSeries(
            shapely.segmentize(geometry.to_numpy(), max_segment),
            index=geometry.index,
            crs=geometry.crs,
        )
    metric_crs = geometry.estimate_utm_crs()
    dense = shapely.segmentize(geometry.to_crs(metric_crs).to_numpy(), max_segment)
    return gpd.GeoSeries(dense, index=geometry.index, crs=metric_crs).to_crs(
        geometry.crs
    )


def thin_profile(geometry: gpd.GeoSeries, tolerance: float) -> gpd.GeoSeries:
    """Drop collinear vertices whose 3D profile is within `tolerance` of straight.

    The z analog of a coverage simplification. Only vertices collinear
    (in the plane) with both neighbors are candidates, i.e. the ones
    `_densify` inserted along a straight edge or a source already had;
    every corner stays. Within each straight run the (distance, z)
    profile is simplified by Douglas-Peucker with `tolerance` meters,
    which depends only on the run's own points and so produces the same
    vertices for both polygons sharing the edge. Lines and polygons
    (holes included) are accepted; 2D input is returned unchanged.

    Parameters
    ----------
    geometry : geopandas.GeoSeries
        3D geometry in any CRS; collinearity is judged in a metric CRS.
    tolerance : float
        Douglas-Peucker tolerance on the vertical profile, in meters.

    Returns
    -------
    geopandas.GeoSeries
        Same index and CRS, with candidate vertices removed.
    """
    geoms = geometry.to_numpy()
    if tolerance <= 0 or len(geoms) == 0 or not shapely.has_z(geoms).any():
        return geometry
    if geometry.crs is None or geometry.crs.is_projected:
        metric = geoms
    else:
        metric = geometry.to_crs(geometry.estimate_utm_crs()).to_numpy()

    def rings_of(geom, geom_metric):
        if geom.geom_type in ('Polygon', 'MultiPolygon'):
            return (
                shapely.get_rings(shapely.get_parts(geom)),
                shapely.get_rings(shapely.get_parts(geom_metric)),
            )
        return shapely.get_parts(geom), shapely.get_parts(geom_metric)

    # Collect every ring's coordinates with its collinear runs, thin the
    # profiles of all runs in one vectorized simplify, then rebuild.
    ring_xyz, keep_masks = [], []
    run_profiles, run_offsets = [], []
    for geom, geom_metric in zip(geoms, metric, strict=True):
        for ring, ring_metric in zip(*rings_of(geom, geom_metric), strict=True):
            xyz = shapely.get_coordinates(ring, include_z=True)
            xy = shapely.get_coordinates(ring_metric)
            keep = np.ones(len(xyz), dtype=bool)
            if len(xyz) > 3:
                prev_v = xy[1:-1] - xy[:-2]
                next_v = xy[2:] - xy[1:-1]
                cross = np.abs(
                    prev_v[:, 0] * next_v[:, 1] - prev_v[:, 1] * next_v[:, 0]
                )
                scale = np.hypot(*prev_v.T) * np.hypot(*next_v.T)
                collinear = np.zeros(len(xyz), dtype=bool)
                collinear[1:-1] = cross <= 1e-6 * scale
                # Each maximal run of collinear vertices, with the corner
                # on either side as anchors, is one profile to simplify.
                d = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
                edges = np.flatnonzero(np.diff(collinear.astype(int)))
                starts = edges[collinear[edges + 1]] + 1
                ends = edges[~collinear[edges + 1]] + 1
                for a, b in zip(starts, ends, strict=True):
                    lo, hi = a - 1, b + 1  # anchors inclusive, [lo, hi)
                    run_profiles.append(np.column_stack([d[lo:hi], xyz[lo:hi, 2]]))
                    run_offsets.append((len(ring_xyz), lo, hi))
            ring_xyz.append(xyz)
            keep_masks.append(keep)
    if run_profiles:
        lengths = np.array([len(p) for p in run_profiles])
        lines = shapely.linestrings(
            np.concatenate(run_profiles),
            indices=np.repeat(np.arange(len(lengths)), lengths),
        )
        thinned = shapely.simplify(lines, tolerance, preserve_topology=False)
        for (ring_i, lo, hi), profile, line in zip(
            run_offsets, run_profiles, thinned, strict=True
        ):
            kept_d = shapely.get_coordinates(line)[:, 0]
            drop = ~np.isin(profile[:, 0], kept_d)
            keep_masks[ring_i][lo:hi][drop] = False

    # Rebuild in input order.
    out = np.empty(len(geoms), dtype=object)
    ring_cursor = 0
    for i, geom in enumerate(geoms):
        if geom.geom_type in ('Polygon', 'MultiPolygon'):
            polygons = []
            for part in shapely.get_parts(geom):
                rings = []
                for _ in shapely.get_rings(part):
                    xyz = ring_xyz[ring_cursor][keep_masks[ring_cursor]]
                    ring_cursor += 1
                    rings.append(xyz)
                polygons.append(shapely.Polygon(rings[0], rings[1:] or None))
            out[i] = (
                polygons[0]
                if geom.geom_type == 'Polygon'
                else shapely.MultiPolygon(polygons)
            )
        else:
            lines = []
            for _ in shapely.get_parts(geom):
                lines.append(
                    shapely.LineString(ring_xyz[ring_cursor][keep_masks[ring_cursor]])
                )
                ring_cursor += 1
            out[i] = lines[0] if len(lines) == 1 else shapely.MultiLineString(lines)
    return gpd.GeoSeries(out, index=geometry.index, crs=geometry.crs)


def _insert_ring_vertices(pieces, owner, polygons):
    """Put every ring vertex back into the boundary pieces that run through it.

    The overlay that clips the grid to a polygon keeps only corners, so
    the collinear vertices `_densify` added along a straight edge (and
    any collinear vertex the source had) vanish from the pieces. The
    rendered walls are the pieces' own outer faces, so without those
    vertices a wall top chords across a whole cell while the outline
    ring beside it follows every densified vertex. Re-inserting them
    makes rim, wall, and outline one vertex set again.

    Done with `shapely.snap`, which inserts a reference vertex into any
    segment passing within the tolerance, in C and over all pieces at
    once; the earlier per-piece Python loop was the mesh's largest cost.

    Parameters
    ----------
    pieces : ndarray of shapely.Polygon
        Output of `subdivide_polygons`, 2D, in meters.
    owner : ndarray of int
        Each piece's polygon index.
    polygons : ndarray of shapely (Multi)Polygon
        The 2D polygons the pieces were cut from, same CRS.

    Returns
    -------
    ndarray of shapely.Polygon
        `pieces`, with ring vertices inserted where they lay on a piece's
        boundary within a millimeter but were not among its vertices.
    """
    ring_xy, ring_owner = shapely.get_coordinates(polygons, return_index=True)
    tree = shapely.STRtree(shapely.points(ring_xy))
    exteriors = shapely.get_exterior_ring(pieces)
    piece_idx, point_idx = tree.query(exteriors, predicate='dwithin', distance=1e-3)
    same_owner = owner[piece_idx] == ring_owner[point_idx]
    piece_idx, point_idx = piece_idx[same_owner], point_idx[same_owner]
    if len(piece_idx) == 0:
        return pieces
    # One reference multipoint per touched piece, holding only the ring
    # vertices near it: snapping against the owner's whole ring costs
    # its length per piece.
    order = np.argsort(piece_idx, kind='stable')
    piece_idx, point_idx = piece_idx[order], point_idx[order]
    touching, rank = np.unique(piece_idx, return_inverse=True)
    references = shapely.multipoints(ring_xy[point_idx], indices=rank)
    snapped = shapely.snap(pieces[touching], references, 1e-3)
    # A sliver whose two sides both pass within a millimeter of a vertex
    # can fold; keep such a piece as it was rather than ship a
    # self-intersection.
    ok = shapely.is_valid(snapped)
    out = pieces.copy()
    out[touching[ok]] = snapped[ok]
    return out


def _triangulate_pieces(pieces, owner):
    """Split every piece into triangles that keep all of its vertices.

    deck.gl triangulates a polygon's cap with earcut, which first drops
    any vertex collinear with its neighbors in x/y. The densified rim
    vertices along a straight parcel edge are exactly that, so the cap
    along such an edge came out as one chord between the corners while
    the walls (built per edge, from every vertex) followed the terrain:
    a slanted face hanging under the rim, blue from inside the parcel,
    and the same on the admin fence. A triangle has no collinear vertex
    to drop, so triangulating here (constrained Delaunay, which keeps
    the vertex set exactly) takes the cap out of earcut's hands.

    Parameters
    ----------
    pieces : ndarray of shapely.Polygon
        2D pieces, in meters.
    owner : ndarray of int
        Each piece's polygon index.

    Returns
    -------
    pieces, owner
        Clockwise triangles and their polygon indices, in piece order.
    """
    # A clipped cell can come out as a zero-area sliver; nothing to
    # draw, and the triangulator refuses it.
    drawable = shapely.area(pieces) > 1e-9
    pieces, owner = pieces[drawable], owner[drawable]
    try:
        triangles = shapely.constrained_delaunay_triangles(pieces)
    except shapely.errors.GEOSException:
        triangles = np.empty(len(pieces), dtype=object)
        for i, piece in enumerate(pieces):
            try:
                triangles[i] = shapely.constrained_delaunay_triangles(piece)
            except shapely.errors.GEOSException:
                triangles[i] = piece
    parts, index = shapely.get_parts(triangles, return_index=True)
    parts = shapely.orient_polygons(parts, exterior_cw=True)
    return parts, owner[index]


def _interpolate_edge_vertices(xy, z, owner, polygons, ring_xyz, ring_row):
    """Replace z of vertices on a polygon's boundary by ring interpolation.

    `xy`/`z`/`owner` describe unique mesh vertices in the metric CRS with
    the index of the polygon (into `polygons`, metric, 2D) each belongs to.
    `ring_xyz`/`ring_row` are the draped ring coordinates of the same
    polygons in the original CRS, in `shapely.get_coordinates` order,
    which `shapely.boundary` preserves: exterior then holes, polygon by
    polygon. A vertex within a micrometer of its polygon's boundary takes
    the z linearly interpolated along the ring between the ring vertices
    it lies between.
    """
    from scipy.spatial import cKDTree  # noqa: F401  (kept import-light)

    boundaries = shapely.boundary(polygons)
    points = shapely.points(xy)
    distance = shapely.distance(points, boundaries[owner])
    on_edge = distance < 1e-6
    if not on_edge.any():
        return z

    # Per-polygon profile of (cumulative length along the boundary, z),
    # cumulative within each ring part and continuing across parts,
    # which is how GEOS's line_locate_point measures a MultiLineString.
    metric_xy, part_index = shapely.get_coordinates(
        shapely.get_parts(boundaries), return_index=True
    )
    part_owner = np.repeat(
        np.arange(len(boundaries)), shapely.get_num_geometries(boundaries)
    )
    vertex_owner = part_owner[part_index]
    step = np.hypot(*np.diff(metric_xy, axis=0).T)
    step = np.concatenate([[0.0], step])
    step[np.concatenate([[True], np.diff(part_index) != 0])] = 0.0
    cumulative = np.cumsum(step)
    # Restart the count at each polygon by subtracting the running total
    # at its first vertex (vertices are grouped by polygon, in order).
    _, first_vertex = np.unique(vertex_owner, return_index=True)
    cumulative = cumulative - cumulative[first_vertex][vertex_owner]
    assert len(metric_xy) == len(ring_xyz), 'ring/boundary vertex order mismatch'

    # One np.interp over all polygons at once: offset each polygon's
    # profile by a stride larger than any boundary length, so profiles
    # never overlap on the x axis.
    stride = cumulative.max() * 2 + 1.0
    profile_x = cumulative + vertex_owner * stride
    profile_z = ring_xyz[:, 2]
    located = shapely.line_locate_point(boundaries[owner[on_edge]], points[on_edge])
    z = z.copy()
    z[on_edge] = np.interp(located + owner[on_edge] * stride, profile_x, profile_z)
    return z


def _snap_to_ring(x, y, z, ring_xyz, crs):
    """Replace mesh vertices within a hair of a ring vertex by that vertex."""
    from scipy.spatial import cKDTree

    if len(ring_xyz) == 0:
        return x, y, z
    ring_xy, first = np.unique(ring_xyz[:, :2], axis=0, return_index=True)
    ring_z = ring_xyz[first, 2]
    # A centimeter in either kind of CRS; the round-trip error is a
    # million times smaller, and no real vertex pair sits that close.
    tolerance = 1e-7 if pyproj.CRS(crs).is_geographic else 1e-2
    distance, index = cKDTree(ring_xy).query(
        np.column_stack([x, y]), distance_upper_bound=tolerance
    )
    on_ring = np.isfinite(distance)
    x, y, z = x.copy(), y.copy(), z.copy()
    x[on_ring] = ring_xy[index[on_ring], 0]
    y[on_ring] = ring_xy[index[on_ring], 1]
    z[on_ring] = ring_z[index[on_ring]]
    return x, y, z


def get_building_elevation(
    gdf: gpd.GeoDataFrame,
    elevation_recipe,
    admin_id_column: str = 'admin3_id',
    cache: bool = True,
    silent: bool = False,
):
    """Average ground elevation under each building footprint.

    Buildings render as flat extrusions (a real building's base is flat),
    so unlike `drape_parcel_elevation` this returns one scalar per row: the
    DEM's zonal mean over each footprint polygon
    (`geo.raster.zonal_stats_with_exactextract`), independent of whichever
    parcel the footprint is later stacked on via `viz.terrain`'s `stack_on`.

    Caching and parameters otherwise match `drape_parcel_elevation`.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Building/footprint polygons.
    elevation_recipe : str or dict
        DEM dataset recipe, passed to `io.readers.get_dataset` per admin
        unit.
    admin_id_column : str
        Column naming each row's admin unit (default ``'admin3_id'``).
    cache : bool
        If True (default), read/write the on-disk elevation cache.
    silent : bool
        Unused directly (zonal stats raise no out-of-extent warning of
        their own); kept for signature parity with `drape_parcel_elevation`.

    Returns
    -------
    numpy.ndarray
        One elevation value (meters) per row, row-aligned to `gdf`. NaN for
        any row `zonal_stats_with_exactextract` drops as an invalid
        geometry (mirrors `viz.terrain`'s own missing-value handling).
    """

    def compute_miss(miss_gdf, dem_path):
        stats = zonal_stats_with_exactextract(miss_gdf, dem_path, stats='mean')
        elevation = stats['mean'].reindex(miss_gdf.index)
        return pd.DataFrame(
            {'elevation': elevation.to_numpy(dtype=float)}, index=elevation.index
        )

    combined = _grouped_cache_compute(
        gdf,
        elevation_recipe,
        admin_id_column,
        cache,
        kind='building_elevation',
        geom=False,
        compute_miss=compute_miss,
        silent=silent,
    )
    return combined['elevation'].to_numpy(dtype=float)


def resolve_dem_admin_ids(
    gdf: gpd.GeoDataFrame,
    elevation_recipe,
    admin_id_column: str | None = None,
) -> pd.Series:
    """Each row's admin id truncated to the DEM recipe's own save level.

    A DEM is tiled per admin unit, and `io.readers.get_dataset` refuses an
    id at any other level, so a frame keyed by a finer unit has to be mapped
    up: a level-4 town is covered by its level-3 county's DEM. The level is
    read off the recipe rather than assumed, so this keeps working if the
    DEM is ever re-tiled at a different granularity.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Frame whose rows carry admin ids, in `admin_id_column` or the index.
    elevation_recipe : str or dict
        DEM dataset recipe.
    admin_id_column : str, optional
        Column holding each row's admin id. When None (default), the ids are
        read from `gdf`'s index.

    Returns
    -------
    pandas.Series of str
        Admin ids at the DEM's own level, indexed like `gdf`.
    """
    from openplaces.recipe import get_recipe_by_id, get_save_admin_level

    dem_recipe = (
        get_recipe_by_id(elevation_recipe)
        if isinstance(elevation_recipe, str)
        else elevation_recipe
    )
    dem_level = get_save_admin_level(dem_recipe)

    raw = gdf[admin_id_column] if admin_id_column is not None else gdf.index
    try:
        parsed = [AdminId(*str(value).split('-')) for value in raw]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            'Cannot resolve a DEM for these rows: each needs an admin id to '
            'find the raster covering it, and this frame is not indexed by '
            'admin id. Pass a frame from `get_admin`, or name the column '
            'holding the ids.',
        ) from exc

    resolved = []
    for admin, original in zip(parsed, raw, strict=True):
        parent = admin.truncate_to_level(dem_level)
        if parent is None:
            raise ValueError(
                f'Admin unit {original!r} is coarser than the level-{dem_level} '
                'tiling the DEM recipe is saved at, so no single DEM covers '
                'it. Use a finer admin level.',
            )
        resolved.append(str(parent))
    return pd.Series(resolved, index=gdf.index)


def get_elevation_datum(
    gdf: gpd.GeoDataFrame,
    elevation_recipe,
    quantile: float = 0.001,
    admin_id_column: str | None = None,
):
    """Ground elevation to treat as z=0 for a scene, in meters.

    The 3D viz extrudes from sea level by default, which puts the whole
    scene as far above the flat basemap as the land happens to be above the
    ocean — a few hundred meters here, more once `terrain_exaggeration`
    multiplies it. That offset is not harmless: with the camera tilted to
    pitch `p`, anything `h` meters up appears shifted from its own
    basemap position by `h * tan(p)`, so at pitch 75 a scene 345 m up
    (Lancaster at 3x) reads about 1.3 km away from the streets it sits on.

    Referencing the scene to the ground beneath it removes that whole term.
    What is left is only the relief *within* the extent, which is what the
    terrain is actually meant to show.

    Defaults to a low quantile (0.1%) rather than the mean or the
    minimum. A mean would put half the terrain below z=0, underneath a
    flat basemap, where it is hidden. The minimum is hostage to a few bad
    pixels: Boston's 3DEP tile holds some 600 returns below -30 m in the
    harbor, and referencing to them lifted a city at sea level 77 m off
    its basemap. The layers clamp at z=0 (`viz.terrain`), so the handful
    of vertices below a low quantile sit on the ground plane rather than
    under it; pass `quantile=0` for the strict minimum.

    Compute this **once per scene** and pass the same value to every layer
    — parcels, buildings, boundaries, the draped basemap. A datum that
    differs between layers slides them vertically relative to each other,
    the same failure mode as a mismatched `terrain_exaggeration`.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Geometries defining the scene's extent. Only their bounds and admin
        ids are used, so any layer of the scene gives the same answer as
        long as they cover the same extent.
    elevation_recipe : str or dict
        DEM dataset recipe, passed to `io.readers.get_dataset`.
    quantile : float
        Quantile of the sampled elevations to use, in [0, 1]. Default
        0.001. Raise it to reference a scene whose extent dips into a
        valley or offshore that would otherwise drag the datum down; 0 is
        the strict minimum.
    admin_id_column : str, optional
        Column naming each row's admin unit at the DEM's own tiling. When
        None (default), the ids are read from `gdf`'s index.

    Returns
    -------
    float
        The reference elevation in meters. 0.0 if the DEM has no data over
        the extent, which keeps the sea-level behavior rather than raising.
    """
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f'quantile must be in [0, 1], got {quantile!r}.')

    ids = resolve_dem_admin_ids(gdf, elevation_recipe, admin_id_column).unique()

    values = []
    for admin_id in ids:
        dem_path = Path(get_dataset(elevation_recipe, admin_id=admin_id))
        if not dem_path.exists():
            _ingest_missing_dem(elevation_recipe, admin_id, dem_path, silent=True)
        with rasterio.open(dem_path) as src:
            bounds = gdf.to_crs(src.crs).total_bounds
            window = rasterio.windows.from_bounds(*bounds, transform=src.transform)
            band = src.read(1, window=window, masked=True, boundless=True)
        finite = np.asarray(band.compressed(), dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            values.append(np.quantile(finite, quantile))

    return float(min(values)) if values else 0.0


def add_z_offset(geometry, offset_per_row):
    """Add a per-row scalar Z offset to every vertex of each geometry.

    Unlike `shapely.force_3d` (which sets an absolute Z on 2D input but
    leaves an already-3D geometry's existing Z untouched), this *adds*
    `offset_per_row[i]` to whatever Z geometry `i` already has -- 0 for
    still-2D geometries, or each vertex's own already-draped elevation
    otherwise. Used in `viz.terrain` to layer a `stack_on`/`elevation_column`
    scalar offset on top of `drape_parcel_elevation`'s per-vertex terrain
    without flattening it back to a single Z per row.

    Parameters
    ----------
    geometry : array-like of shapely geometries
        2D or 3D polygon geometries.
    offset_per_row : array-like of float
        One Z offset per geometry, added to every one of that geometry's
        own vertices.

    Returns
    -------
    numpy.ndarray of shapely geometries
    """
    geom3d = shapely.force_3d(np.asarray(geometry), z=0.0)
    xyz, row_index = shapely.get_coordinates(geom3d, include_z=True, return_index=True)
    xyz[:, 2] = xyz[:, 2] + np.asarray(offset_per_row, dtype=float)[row_index]
    return shapely.set_coordinates(geom3d, xyz)


def clamp_z(geometry, lower: float = 0.0):
    """Raise every vertex Z below `lower` up to it, leaving x/y alone.

    Used by `viz.terrain`'s `elevation_datum` to guarantee that referenced
    ground never sinks below the basemap plane, where a flat basemap would
    simply hide it. Clamping (rather than shifting the whole scene down to
    fit) keeps the datum meaning what it says for the rest of the extent;
    only the part that would have gone under is flattened onto the plane.

    Parameters
    ----------
    geometry : array-like of shapely geometries
        2D or 3D geometries. A still-2D geometry has an implicit Z of 0 and
        is unaffected by the default `lower`.
    lower : float
        Floor applied to every vertex's Z. Defaults to 0.

    Returns
    -------
    numpy.ndarray of shapely geometries
    """
    geom3d = shapely.force_3d(np.asarray(geometry), z=0.0)
    xyz = np.asarray(shapely.get_coordinates(geom3d, include_z=True))
    xyz[:, 2] = np.maximum(xyz[:, 2], lower)
    return shapely.set_coordinates(geom3d, xyz)


def scale_z(geometry, factor: float):
    """Multiply every vertex's own Z by `factor`.

    Used by `viz.terrain`'s `terrain_exaggeration` to visually exaggerate
    real-world terrain relief -- e.g. `drape_parcel_elevation`'s per-vertex
    Z, which on a typical parcel spans only a few meters and can otherwise
    be nearly imperceptible next to the value-based extrusion height. x/y
    are untouched; a still-2D input geometry is unaffected (its implicit
    Z of 0 stays 0 regardless of `factor`).

    Parameters
    ----------
    geometry : array-like of shapely geometries
        2D or 3D polygon geometries.
    factor : float
        Multiplier applied to each vertex's own Z value.

    Returns
    -------
    numpy.ndarray of shapely geometries
    """
    geom3d = shapely.force_3d(np.asarray(geometry), z=0.0)
    xyz = np.asarray(shapely.get_coordinates(geom3d, include_z=True))
    xyz[:, 2] = xyz[:, 2] * factor
    return shapely.set_coordinates(geom3d, xyz)


def _drape_miss(miss_gdf: gpd.GeoDataFrame, dem_path, silent: bool):
    """Per-vertex draped geometry + per-row mean elevation for uncached rows."""
    geom3d = shapely.force_3d(miss_gdf.geometry.to_numpy(), z=0.0)
    xyz, row_index = shapely.get_coordinates(geom3d, include_z=True, return_index=True)

    unique_xy, inverse = np.unique(xyz[:, :2], axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)

    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
    sample_x, sample_y = _reproject_points(
        unique_xy[:, 0], unique_xy[:, 1], miss_gdf.crs, dem_crs
    )
    unique_z = sample_raster_at_points(
        dem_path, sample_x, sample_y, interpolation='bilinear'
    )

    unique_z = _fill_unsampled_from_nearest(unique_xy, unique_z, silent=silent)

    xyz[:, 2] = unique_z[inverse]
    draped = shapely.set_coordinates(geom3d, xyz)

    mean_elevation = (
        pd.Series(xyz[:, 2])
        .groupby(row_index)
        .mean()
        .reindex(range(len(miss_gdf)))
        .to_numpy(dtype=float)
    )
    return gpd.GeoDataFrame(
        {'geometry': draped, 'mean_elevation': mean_elevation},
        index=miss_gdf.index,
        crs=miss_gdf.crs,
    )


def _fill_unsampled_from_nearest(xy, z, silent: bool):
    """Give every unsampled vertex the elevation of its nearest sampled one.

    A vertex outside the DEM's extent or on a nodata pixel (the DEM is
    clipped to the admin unit, so a parcel straddling the boundary has
    them routinely) used to get 0 m. That is not a missing value but a
    wrong one: the mesh interpolates edge vertices along the ring and
    chords wall tops between vertices, so one sea-level vertex pulls a
    whole rim down toward the ground plane. The nearest sampled vertex
    is the closest available reading of the same terrain.

    Parameters
    ----------
    xy : ndarray of shape (n, 2)
        Vertex coordinates, in any single planar or geographic CRS
        (only relative distance matters).
    z : ndarray of shape (n,)
        Sampled elevations, NaN where sampling failed.
    silent : bool
        If True, suppress the warning.

    Returns
    -------
    ndarray of shape (n,)
        `z` with every NaN replaced. All-NaN input is returned as zeros,
        because there is no terrain reading to borrow.
    """
    z = np.asarray(z, dtype=float)
    missing = np.isnan(z)
    n_missing = int(missing.sum())
    if not n_missing:
        return z
    if n_missing == len(z):
        if not silent:
            warnings.warn(
                f'None of {len(z)} vertices sampled a DEM value; setting their '
                'elevation to 0.',
                stacklevel=3,
            )
        return np.zeros_like(z)
    if not silent:
        warnings.warn(
            f'{n_missing} of {len(z)} sampled vertex(es) fell outside the DEM '
            'extent or on a nodata pixel; taking the elevation of the nearest '
            'sampled vertex.',
            stacklevel=3,
        )
    valid = np.flatnonzero(~missing)
    tree = shapely.STRtree(shapely.points(xy[valid]))
    nearest = tree.nearest(shapely.points(xy[missing]))
    filled = z.copy()
    filled[missing] = z[valid[nearest]]
    return filled


def _reproject_points(x, y, src_crs, dst_crs):
    """Reproject point coordinate arrays; no-op when CRSs already match."""
    if src_crs is None or dst_crs is None or pyproj.CRS(src_crs) == pyproj.CRS(dst_crs):
        return x, y
    transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return transformer.transform(x, y)


def _cache_path(admin_id, kind: str) -> Path:
    """Cache file path for one admin unit's derived elevation values.

    Not a recipe-defined output (this is regenerable derived data, not a
    canonical entity/dataset) -- lives directly under `cfg.cache_dir`,
    mirroring the admin-id directory structure `path.py` uses elsewhere.
    """
    admin_id = admin_id if isinstance(admin_id, AdminId) else AdminId(admin_id)
    return (
        cfg.cache_dir
        / admin_id.to_path()
        / 'viz_elevation'
        / f'{admin_id}_{kind}.parquet'
    )


def _load_cache(path: Path, geom: bool):
    if not path.exists():
        return None
    return gpd.read_parquet(path) if geom else pd.read_parquet(path)


def _save_cache(existing, fresh, path: Path, geom: bool) -> None:
    combined = (
        fresh
        if existing is None
        else pd.concat([existing[~existing.index.isin(fresh.index)], fresh])
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if geom:
        gpd.GeoDataFrame(combined).to_parquet(path)
    else:
        combined.to_parquet(path)


def _grouped_cache_compute(
    gdf: gpd.GeoDataFrame,
    elevation_recipe,
    admin_id_column: str,
    cache: bool,
    kind: str,
    geom: bool,
    compute_miss,
    silent: bool = False,
):
    """Shared cache + per-admin-unit-DEM orchestration.

    Groups `gdf` by `admin_id_column` (matching the DEM's own per-admin-unit
    tiling), reuses cached rows, and calls `compute_miss(miss_gdf, dem_path)`
    only for rows not already cached for that admin unit. Returns a frame
    row-aligned to `gdf` (its original order/index), combining cache hits
    and freshly computed rows.
    """
    if gdf.empty:
        return (
            gpd.GeoDataFrame(index=gdf.index) if geom else pd.DataFrame(index=gdf.index)
        )

    pieces = []
    for admin_id, group in gdf.groupby(admin_id_column, sort=False):
        cache_file = _cache_path(admin_id, kind)
        cached = _load_cache(cache_file, geom) if cache else None

        is_hit = (
            group.index.isin(cached.index)
            if cached is not None
            else np.zeros(len(group), dtype=bool)
        )
        if is_hit.any():
            pieces.append(cached.loc[group.index[is_hit]])

        miss = group[~is_hit]
        if len(miss):
            dem_path = get_dataset(elevation_recipe, admin_id=admin_id)
            if not Path(dem_path).exists():
                _ingest_missing_dem(elevation_recipe, admin_id, dem_path, silent)
            fresh = compute_miss(miss, dem_path)
            pieces.append(fresh)
            if cache:
                _save_cache(cached, fresh, cache_file, geom)

    combined = pd.concat(pieces) if pieces else gdf.iloc[:0]
    return combined.reindex(gdf.index)


def _ingest_missing_dem(
    elevation_recipe, admin_id, dem_path: Path, silent: bool
) -> None:
    """Ingest `elevation_recipe` for `admin_id` when its DEM hasn't been fetched yet.

    Imported lazily (not at module level) since `io.ingester` pulls in the
    full ingestion dependency stack (~1.4s), which every other caller of
    this module -- i.e. every `viz.terrain` call, even ones never touching
    `elevation_recipe` -- would otherwise pay for unconditionally.
    """
    from openplaces.io.ingester import ingest

    if not silent:
        warnings.warn(
            f'No DEM ingested yet for {admin_id!s} -- ingesting {elevation_recipe!r} '
            'now (one-time; cached to disk afterward).',
            stacklevel=4,
        )
    ingest(elevation_recipe, admin_ids=str(admin_id), verbose=not silent)

    if not dem_path.exists():
        raise FileNotFoundError(
            f'Ingesting {elevation_recipe!r} for {admin_id!s} did not produce the '
            f'expected DEM at {dem_path} -- this admin unit may have no 3DEP coverage.'
        )
