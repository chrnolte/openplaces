"""
Registered enrich step: link parcels to lakes and rivers.

For each parcel it writes evidence on four separate channels, because the
hedonic literature prices them separately and a model that omits one
biases the others (Nicholls and Crompton 2018, citing Muller 2009):

- frontage: shoreline length the parcel holds on a lake or river;
- proximity: distance from the parcel's primary building, and from the
  parcel edge, to the nearest lake or river;
- the lake it fronts and the lake nearest it, with their areas, so that
  size enters a model as its own variable rather than through an index.

View and public access are later phases (see
plans/lake-frontage-and-distance-enrichment.md).

Frontage is allocated, not buffered. The predecessor model
(places/core/b7_pc_nhd.py) intersected parcels with a 25 m ring outside
each lake and divided the overlap area by 25 m. That credits a
second-row parcel behind a thin first-row lot, a shore road or an
association strip, counts the same shoreline once for every parcel in
the ring, and misstates length wherever the shore curves. Here each
shoreline is cut into short segments, and each segment goes to the one
parcel nearest it, provided that parcel lies within a tolerance. A
shoreline meter therefore belongs to at most one parcel, a second-row
parcel gets nothing because the first-row parcel is nearer, a parcel
drawn into the water is at distance zero and wins, and a misalignment
between the cadastre and the NHD shoreline up to the tolerance is
absorbed. One property follows from nearest-parcel allocation and is
bounded by the tolerance: where no parcel holds a stretch of shore (a
park or a road drawn without a parcel), the lots at its ends take up
to one tolerance of it each.

Distances are measured from a parcel's land, not its whole polygon.
This keeps what the predecessor's ring did implicitly (the ring lay
outside the water, so only land could overlap it) and covers the
cadastres the ring was written for: a lake drawn as its own parcel,
which has no land and so claims no shore; a lake bed subdivided among
the lots around it, each claiming the shore in front of its dry part; a
lake parcel that also holds a house on a strip of shore, which claims
the shore in front of that strip. A lake parcel drawn slightly larger
than the NHD shoreline leaves a thin ring of "land" around the lake at
distance zero from every segment; for parcels that are mostly water,
land strips narrower than twice *sliver_m* are removed (an opening,
applied to that one parcel's land), so real dry land survives and the
misalignment ring does not.

The opening removes a rim only up to twice *sliver_m* wide, and around
large lakes the rim can be wider. A parcel that is mostly one
waterbody and holds most of it (a lake drawn as its own parcel)
therefore also yields each segment to the nearest other parcel within
the tolerance, except near its own building. It keeps the segments no
other parcel reaches, which it does hold. The two tests are those of an
earlier rule that excluded such parcels outright; used as a local
preference instead, they leave a lake parcel with a house its shore.

This measures a relation between parcels and a waterbody. It groups no
parcels into a community, unions no enlarged boundaries of a group
(the opening shrinks then regrows one parcel's own land), finds no
border-intersection points and imputes no address, so it is at least
four steps different from US10248731B1 claim 1 (AGENTS.md, patent
shape 1).
"""

from __future__ import annotations

from functools import cache

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from openplaces.geo.polygon import subdivide_polygons
from openplaces.io import save_parquet
from openplaces.io.enricher import EnrichState, _register
from openplaces.io.readers import get_admin, get_dataset, get_entities
from openplaces.path import recipe_path
from openplaces.recipe import get_output_path, get_recipe_by_id

WATERBODY_RECIPES = (
    'US_water-waterbody-usgs-nhdhr2023',
    'US_water-waterbody-usgs-nhdhr2023_area',
)
# The spine whose rows say which structure is a parcel's primary one
# (`parcel_id`, `priority_on_parcel`): the building spine since b3 step 7,
# when the footprint spine stopped carrying those attribute columns.
FOOTPRINT_RECIPE = 'US_building-spine-2026'
ID = 'waterbody_id_nhd'


@cache
def _feature_classes(recipe_id: str) -> pd.DataFrame:
    """Load the FCode classes sidecar beside a waterbody recipe.

    One row per NHD FCode: its feature class (lake_pond, reservoir, pool,
    river, swamp_marsh, ...) and one boolean column per lake definition
    (is_lake_lagos, is_lake_lakesense, is_river_places).
    """
    recipe = get_recipe_by_id(recipe_id)
    stem = recipe_id.split('_')[0] + '_' + recipe_id.split('_')[1]
    folder = recipe_path(recipe['admin_id'], dataset=recipe['dataset'], as_dir=True)
    classes = pd.read_csv(folder / f'{stem}_fcode-classes.csv')
    return classes.set_index('feature_code_nhd')


@cache
def _great_lake_names(recipe_id: str) -> frozenset[str]:
    """GNIS names of the Great Lakes, from the sidecar beside the recipe.

    Matched by name, not id: NHD splits each Great Lake into several
    polygons (some unnamed pieces are missed, which the sidecar notes),
    and their ids changed between versions.
    """
    recipe = get_recipe_by_id(recipe_id)
    folder = recipe_path(recipe['admin_id'], dataset=recipe['dataset'], as_dir=True)
    path = folder / f'{recipe_id}_great-lakes.csv'
    if not path.exists():
        return frozenset()
    return frozenset(pd.read_csv(path)['gnis_name'])


def _load_waterbodies(region_wgs84, recipe_ids, crs) -> gpd.GeoDataFrame:
    """Waterbodies of every state touching *region_wgs84*, deduplicated.

    A state's NHD file carries the whole HU basins touching the state, so
    a lake near a border appears in two files under one id; the first
    copy is kept. Rows are clipped to the region by bounding box only.
    """
    states = get_admin('US', level=2, geom=True).to_crs('EPSG:4326')
    touching = states.index[states.intersects(region_wgs84)]
    frames = []
    for recipe_id in recipe_ids:
        for admin_id in touching:
            try:
                gdf = get_dataset(recipe_id, admin_id=admin_id, geom=True)
            except FileNotFoundError:
                continue
            if gdf is None or gdf.empty:
                continue
            gdf = gdf.to_crs('EPSG:4326')
            minx, miny, maxx, maxy = region_wgs84.bounds
            gdf = gdf.cx[minx:maxx, miny:maxy]
            frames.append(gdf)
    if not frames:
        return gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=crs)
    water = pd.concat(frames)
    water = water[~water.index.duplicated(keep='first')]
    water.index.name = ID
    return gpd.GeoDataFrame(water, geometry='geometry').to_crs(crs)


def _shoreline_segments(water: gpd.GeoSeries, segment_m: float, clip):
    """Cut every waterbody boundary into segments of at most *segment_m*.

    Islands count: a polygon's boundary includes its interior rings, so
    an island's shore is shoreline too. Returns segment midpoints, their
    lengths and the waterbody id of each segment.
    """
    lines = gpd.GeoSeries(water.boundary, crs=water.crs).explode(index_parts=False)
    lines = lines[~lines.is_empty]
    if clip is not None:
        lines = lines.clip(clip)
        lines = lines[~lines.is_empty].explode(index_parts=False)
    lines = lines[lines.geom_type == 'LineString']
    if lines.empty:
        return np.empty((0, 2)), np.empty(0), np.empty(0, dtype=object)
    dense = shapely.segmentize(lines.values, segment_m)
    coords, part = shapely.get_coordinates(dense, return_index=True)
    same = part[1:] == part[:-1]
    start, end = coords[:-1][same], coords[1:][same]
    owner = lines.index.to_numpy()[part[:-1][same]]
    midpoints = (start + end) / 2
    lengths = np.hypot(*(end - start).T)
    return midpoints, lengths, owner


def _waterbody_holders(
    parcels, water, holder_share: float, pieces=None
) -> pd.MultiIndex:
    """(parcel, waterbody) pairs where the parcel is mostly that waterbody.

    A holder has at least *holder_share* of its area under the
    waterbody and holds at least *holder_share* of the waterbody's area:
    a lake drawn as its own parcel, not a lot drawn into a lake. Holding
    that share needs that share of the area, so pairs are filtered by
    area before any intersection is computed. With *pieces* (the
    waterbodies subdivided, indexed by waterbody id), the shared area is
    summed over pieces, so no overlay touches a whole large lake.
    """
    empty = pd.MultiIndex.from_arrays([[], []], names=['parcel', ID])
    if parcels.empty or water.empty:
        return empty
    if pieces is None:
        pieces = water.geometry
    pieces = pieces[pieces.index.isin(water.index)]
    geoms = parcels.geometry.values
    i_p, i_k = shapely.STRtree(pieces.values).query(geoms, predicate='intersects')
    wid = pieces.index.to_numpy()[i_k]
    p_area = shapely.area(geoms[i_p])
    w_area = water.geometry.area.reindex(wid).to_numpy()
    big = p_area >= holder_share * w_area
    i_p, i_k, wid = i_p[big], i_k[big], wid[big]
    if len(i_p) == 0:
        return empty
    pairs = (
        pd.DataFrame(
            {
                'p': i_p,
                'w': wid,
                'shared': shapely.area(
                    shapely.intersection(geoms[i_p], pieces.values[i_k])
                ),
            }
        )
        .groupby(['p', 'w'], sort=False)['shared']
        .sum()
    )
    p_idx = pairs.index.get_level_values('p').to_numpy()
    w_idx = pairs.index.get_level_values('w').to_numpy()
    p_area = shapely.area(geoms[p_idx])
    w_area = water.geometry.area.reindex(w_idx).to_numpy()
    shared = pairs.to_numpy()
    held = (shared >= holder_share * p_area) & (shared >= holder_share * w_area)
    return pd.MultiIndex.from_arrays(
        [parcels.index.to_numpy()[p_idx[held]], w_idx[held]], names=['parcel', ID]
    )


def _min_distances(geoms: gpd.GeoSeries, pieces: gpd.GeoSeries, within_m: float):
    """Distance from each geometry to each waterbody within *within_m*.

    *pieces* are waterbody pieces indexed by waterbody id; the distance
    to a waterbody is the smallest distance to any of its pieces.
    Returns a Series indexed by (geometry label, waterbody id).
    """
    names = [geoms.index.name or 'parcel', ID]
    if geoms.empty or pieces.empty:
        return pd.Series(
            dtype=float, index=pd.MultiIndex.from_arrays([[], []], names=names)
        )
    g = geoms.values
    i_g, i_k = shapely.STRtree(pieces.values).query(
        g, predicate='dwithin', distance=within_m
    )
    d = pd.DataFrame(
        {
            names[0]: geoms.index.to_numpy()[i_g],
            ID: pieces.index.to_numpy()[i_k],
            'd': shapely.distance(g[i_g], pieces.values[i_k]),
        }
    )
    return d.groupby(names, sort=False)['d'].min()


def _allocate_frontage(
    water,
    parcels,
    tolerance_m,
    segment_m,
    clip,
    holders=None,
    buildings=None,
    holder_house_m: float = 20.0,
    return_segments: bool = False,
):
    """Frontage per (parcel, waterbody) pair, by nearest-parcel allocation.

    Each segment goes to the nearest parcel within *tolerance_m*. A
    parcel that holds the segment's waterbody (a pair in *holders*)
    yields the segment to the nearest other parcel within the
    tolerance, unless its building in *buildings* lies within
    *holder_house_m* of the segment; with no other parcel in reach, it
    keeps the segment.

    Returns a Series of meters indexed by (parcel id, waterbody id).
    With *return_segments*, also returns the segment table (one row per
    shoreline segment: x, y, length, waterbody id, and the parcel it
    went to, null where no parcel lies within the tolerance), which the
    access measures start from.
    """
    midpoints, lengths, owner = _shoreline_segments(water.geometry, segment_m, clip)
    if len(midpoints) == 0 or parcels.empty:
        empty = pd.Series(dtype=float)
        if return_segments:
            return empty, pd.DataFrame(
                columns=['x', 'y', 'length', ID, 'parcel', 'distance']
            )
        return empty
    geoms = parcels.geometry.values
    points = shapely.points(midpoints)
    i_seg, i_par = shapely.STRtree(geoms).query(
        points, predicate='dwithin', distance=tolerance_m
    )
    pid = parcels.index.to_numpy()[i_par]
    wid = owner[i_seg]
    cand = pd.DataFrame(
        {
            'seg': i_seg,
            'parcel': pid,
            'distance': shapely.distance(points[i_seg], geoms[i_par]),
            'yields': False,
        }
    )
    if holders is not None and len(holders):
        yields = np.asarray(pd.MultiIndex.from_arrays([pid, wid]).isin(holders))
        if yields.any() and buildings is not None and len(buildings):
            house = buildings.reindex(pid[yields]).to_numpy()
            has = ~pd.isna(house)
            near = np.zeros(int(yields.sum()), dtype=bool)
            near[has] = (
                shapely.distance(points[i_seg[yields]][has], house[has])
                <= holder_house_m
            )
            yields[np.flatnonzero(yields)[near]] = False
        cand['yields'] = yields
    # One parcel per segment: non-yielding before yielding, then nearest.
    # A tie (two parcels touching a segment) goes to one of them, which
    # keeps the sum of frontage at or below the shoreline length.
    cand = cand.sort_values(['seg', 'yields', 'distance'], kind='stable')
    cand = cand.drop_duplicates('seg')
    pairs = pd.DataFrame(
        {
            'parcel': cand['parcel'].to_numpy(),
            ID: owner[cand['seg'].to_numpy()],
            'frontage_m': lengths[cand['seg'].to_numpy()],
        }
    )
    frontage = pairs.groupby(['parcel', ID])['frontage_m'].sum()
    if not return_segments:
        return frontage
    segments = pd.DataFrame(
        {
            'x': midpoints[:, 0],
            'y': midpoints[:, 1],
            'length': lengths,
            ID: owner,
        }
    )
    taken = cand.set_index('seg')
    segments['parcel'] = taken['parcel'].reindex(segments.index)
    segments['distance'] = taken['distance'].reindex(segments.index)
    return frontage, segments


def _access_through_shore_parcels(segments, parcels, max_distance_m: float, roads=None):
    """Lots that reach the water through the parcel holding the shore.

    For each shoreline segment held by a parcel S, the segment's access
    goes to the nearest other parcel L within *max_distance_m* that
    shares a boundary with S (strict contiguity: the two polygons
    intersect, with no distance tolerance) and holds no frontage on the
    same waterbody itself. This is the lot behind a common-area strip,
    a marsh parcel or a beach parcel. With *roads*, a lot whose path to
    the segment crosses a road centerline is skipped, so a road drawn as
    a parcel, or a road running across S, passes no access on.
    Which kinds of S grant access is decided by the caller from S's
    attributes; this function only finds the lot behind.

    Returns one row per segment with a lot behind: seg, via (S), parcel
    (L), distance (from the segment to L).
    """
    cols = ['seg', 'via', 'parcel', 'distance']
    held = segments[segments['parcel'].notna()]
    if held.empty or parcels.empty:
        return pd.DataFrame(columns=cols)
    geoms = parcels.geometry.values
    ids = parcels.index.to_numpy()
    pos = pd.Series(np.arange(len(ids)), index=ids)
    # Adjacency of each shore-holding parcel, by strict contiguity
    shore = pd.unique(held['parcel'])
    s_pos = pos.reindex(shore).dropna().astype(int).to_numpy()
    i_s, i_l = shapely.STRtree(geoms).query(geoms[s_pos], predicate='intersects')
    adj = pd.MultiIndex.from_arrays([ids[s_pos[i_s]], ids[i_l]])
    adj = adj[adj.get_level_values(0) != adj.get_level_values(1)]
    # Candidate lots near each held segment
    pts = shapely.points(held['x'].to_numpy(), held['y'].to_numpy())
    i_p, i_c = shapely.STRtree(geoms).query(
        pts, predicate='dwithin', distance=max_distance_m
    )
    cand = pd.DataFrame(
        {
            'seg': held.index.to_numpy()[i_p],
            'via': held['parcel'].to_numpy()[i_p],
            'parcel': ids[i_c],
            ID: held[ID].to_numpy()[i_p],
        }
    )
    cand = cand[
        np.asarray(pd.MultiIndex.from_arrays([cand['via'], cand['parcel']]).isin(adj))
    ]
    fronting = pd.MultiIndex.from_frame(held[['parcel', ID]].drop_duplicates())
    cand = cand[
        ~np.asarray(
            pd.MultiIndex.from_arrays([cand['parcel'], cand[ID]]).isin(fronting)
        )
    ]
    if cand.empty:
        return pd.DataFrame(columns=cols)
    seg_pts = shapely.points(segments.loc[cand['seg'], ['x', 'y']].to_numpy())
    lot_geoms = geoms[pos.reindex(cand['parcel']).to_numpy()]
    cand['distance'] = shapely.distance(seg_pts, lot_geoms)
    if roads is not None and len(roads):
        cand = cand[~_crosses_road(shapely.shortest_line(seg_pts, lot_geoms), roads)]
    cand = cand.sort_values(['seg', 'distance'], kind='stable').drop_duplicates('seg')
    return cand[cols].reset_index(drop=True)


def _crosses_road(lines, roads) -> np.ndarray:
    """Whether each path line crosses a road centerline in *roads*."""
    out = np.zeros(len(lines), dtype=bool)
    if roads is None or len(roads) == 0 or len(lines) == 0:
        return out
    i_l, _ = shapely.STRtree(roads.values).query(lines, predicate='intersects')
    out[np.unique(i_l)] = True
    return out


def _access_across_open_land(
    segments, parcels, max_distance_m: float, n_probe=5, roads=None
):
    """Lots that reach unheld shore across land no parcel covers.

    A segment no parcel holds within the frontage tolerance (a beach, or
    a shore drawn without parcels) goes to the nearest parcel within
    *max_distance_m* whose straight path to the segment crosses no other
    parcel: *n_probe* points along the path, ends excluded, must lie
    outside every other parcel. With *roads* (centerlines), a path that
    crosses a road is not access either: a lot across a shore road does
    not reach the water (maintainer's decision, 2026-10-01). A park or
    beach without a parcel is open land; the caller decides which gaps
    count, from the gap width and the water class.

    Returns one row per segment reached: seg, parcel, distance.
    """
    cols = ['seg', 'parcel', 'distance']
    free = segments[segments['parcel'].isna()]
    if free.empty or parcels.empty:
        return pd.DataFrame(columns=cols)
    geoms = parcels.geometry.values
    ids = parcels.index.to_numpy()
    pts = shapely.points(free['x'].to_numpy(), free['y'].to_numpy())
    tree = shapely.STRtree(geoms)
    i_p, i_c = tree.query_nearest(pts, max_distance=max_distance_m, all_matches=False)
    if len(i_p) == 0:
        return pd.DataFrame(columns=cols)
    lines = shapely.shortest_line(pts[i_p], geoms[i_c])
    clear = ~_crosses_road(lines, roads)
    for f in np.linspace(0, 1, n_probe + 2)[1:-1]:
        probe = shapely.line_interpolate_point(lines, f, normalized=True)
        j_p, j_c = tree.query(probe, predicate='within')
        blocked = j_c != i_c[j_p]
        clear[np.unique(j_p[blocked])] = False
    out = pd.DataFrame(
        {
            'seg': free.index.to_numpy()[i_p],
            'parcel': ids[i_c],
            'distance': shapely.length(lines),
        }
    )
    return out[clear].reset_index(drop=True)


def _shore_land(
    parcels, water, sliver_m: float, sliver_water_share: float, buildings=None
):
    """Each parcel's land part, and its water share.

    Returns ``(land, share)``: *land* is a GeoSeries of the parcel minus
    the water polygons it overlaps (unchanged for a parcel touching no
    water; absent where nothing is left), *share* the fraction of the
    parcel's area under water. For parcels at least *sliver_water_share*
    water, strips of land narrower than ``2 * sliver_m`` are removed,
    except within ``2 * sliver_m`` of the parcel's building in
    *buildings* (a GeoSeries indexed by parcel id): a house on a small
    dry corner keeps that corner.
    """
    share = pd.Series(0.0, index=parcels.index)
    land = parcels.geometry.copy()
    geoms = parcels.geometry.values
    tree = shapely.STRtree(water.geometry.values)
    i_parcel, i_water = tree.query(geoms, predicate='intersects')
    if len(i_parcel) == 0:
        return land, share
    wet_geom = water.geometry.values
    counts = np.bincount(i_parcel, minlength=len(geoms))
    # Most wet parcels touch one water polygon: one vectorized difference.
    single = counts[i_parcel] == 1
    p_one = i_parcel[single]
    dry = np.empty(len(geoms), dtype=object)
    dry[p_one] = shapely.difference(geoms[p_one], wet_geom[i_water[single]])
    # The rest subtract the union of their water polygons.
    multi = pd.Series(i_water[~single]).groupby(i_parcel[~single])
    for p, group in multi:
        dry[p] = shapely.difference(
            geoms[p], shapely.union_all(wet_geom[group.to_numpy()])
        )
    wet_p = np.unique(i_parcel)
    area = shapely.area(geoms[wet_p])
    with np.errstate(divide='ignore', invalid='ignore'):
        s = np.clip(1 - shapely.area(dry[wet_p]) / area, 0.0, 1.0)
    s = np.nan_to_num(s)
    # Opening on the land of mostly-water parcels: drop strips narrower
    # than 2 * sliver_m, except around the parcel's building.
    opened = wet_p[(s >= sliver_water_share) & ~shapely.is_empty(dry[wet_p])]
    if len(opened):
        keep = shapely.buffer(shapely.buffer(dry[opened], -sliver_m), sliver_m)
        if buildings is not None and len(buildings):
            house = buildings.reindex(parcels.index[opened]).values
            has = ~pd.isna(house)
            keep[has] = shapely.union(
                keep[has], shapely.buffer(house[has], 2 * sliver_m)
            )
        dry[opened] = shapely.intersection(dry[opened], keep)
    share.iloc[wet_p] = s
    land.iloc[wet_p] = dry[wet_p]
    land = land[~land.is_empty & land.notna()]
    return land, share


def _primary_buildings(admin_id, parcels, footprint_recipe, crs) -> gpd.GeoSeries:
    """One footprint per parcel: its largest primary footprint.

    Falls back to the parcel's largest footprint where none is marked
    primary. Parcels without a footprint are absent.
    """
    try:
        fp = get_entities(
            footprint_recipe,
            admin_id,
            geom=True,
            columns=['parcel_id', 'priority_on_parcel'],
        )
    except FileNotFoundError:
        return gpd.GeoSeries(crs=crs)
    fp = fp[fp['parcel_id'].isin(parcels.index)].to_crs(crs)
    if fp.empty:
        return gpd.GeoSeries(crs=crs)
    fp = fp.assign(
        area=fp.geometry.area,
        primary=fp['priority_on_parcel'].eq('primary'),
    )
    fp = fp.sort_values(['primary', 'area'], ascending=False)
    fp = fp.drop_duplicates('parcel_id')
    return gpd.GeoSeries(fp.geometry.values, index=fp['parcel_id'].values, crs=crs)


def _nearest(geoms: gpd.GeoSeries, water: gpd.GeoDataFrame, max_distance_m):
    """Nearest waterbody and its distance for each geometry, within a cap.

    Rows with nothing within *max_distance_m* get null, not the cap: a
    fill value turns "farther than" into a mass point at one distance.
    """
    out = pd.DataFrame(index=geoms.index, columns=['distance', ID], dtype=object)
    if geoms.empty or water.empty:
        out['distance'] = np.nan
        return out
    left = gpd.GeoDataFrame(geometry=geoms)
    right = gpd.GeoDataFrame(geometry=water.geometry)
    right.index.name = ID
    joined = gpd.sjoin_nearest(
        left, right, how='inner', max_distance=max_distance_m, distance_col='distance'
    )
    joined = joined[~joined.index.duplicated(keep='first')]
    out['distance'] = joined['distance'].reindex(out.index).astype(float)
    out[ID] = joined[f'{ID}'].reindex(out.index)
    return out


@_register('link_parcels_to_waterbodies')
def link_parcels_to_waterbodies(
    state: EnrichState,
    waterbody_recipe_id: str = WATERBODY_RECIPES[0],
    area_recipe_id: str | None = WATERBODY_RECIPES[1],
    footprint_recipe_id: str = FOOTPRINT_RECIPE,
    lake_definition: str = 'is_lake_lagos',
    river_definition: str = 'is_river_places',
    tolerance_m: float = 20.0,
    segment_m: float = 2.0,
    frontage_min_area_ha: float = 1.0,
    distance_min_area_ha: float = 4.0,
    max_distance_m: float = 10000.0,
    link_distance_m: float = 500.0,
    sliver_m: float = 5.0,
    sliver_water_share: float = 0.5,
    holder_share: float = 0.5,
    holder_house_m: float = 20.0,
    max_vertices: int = 256,
    save_links: bool = True,
) -> EnrichState:
    """Write lake and river frontage and distance evidence for parcels.

    Parameters
    ----------
    state : EnrichState
        The recipe must set ``spine_geom: true``.
    waterbody_recipe_id : str
        NHD waterbody ingest recipe. Its sidecar ``*_fcode-classes.csv``
        classifies FCodes. Named with a ``recipe_id`` key, as is every
        input here, so `RecipeDAG` sees it as a dependency.
    area_recipe_id : str, optional
        NHD area-feature ingest recipe (wide rivers); None skips rivers
        drawn as polygons.
    footprint_recipe_id : str
        Footprint spine giving each parcel its primary building.
    lake_definition : str
        Boolean sidecar column naming which FCodes count as lakes:
        ``is_lake_lagos`` (default; LAGOS-US LOCUS, which excludes
        decorative and swimming pools) or ``is_lake_lakesense``.
    river_definition : str
        Boolean sidecar column naming which FCodes count as rivers.
    tolerance_m : float
        Largest gap between a parcel and a shoreline segment for the
        segment to count as that parcel's frontage. Measured 2026-09-29
        on US-TX-ARA: against the roll's waterfront land types, 20 m
        gives F1 0.798 (precision 0.807, recall 0.788) for frontage on
        water >= 1 ha; F1 plateaus beyond 20 m while view-only lots
        flagged as frontage rise from 7% to 14% at 30 m.
    segment_m : float
        Shoreline segment length. 2 m resolves a lot line to about 1 m.
    frontage_min_area_ha : float
        Smallest lake counted in the per-parcel frontage summary. 1 ha is
        the LAGOS-US census floor; NHD HR is unreliable below it.
    distance_min_area_ha : float
        Smallest lake counted as "nearest lake". 4 ha follows LAGOS-NE,
        Mamun et al. (2023) and Swedberg et al. (2024).
    max_distance_m : float
        Distances beyond this are left null.
    link_distance_m : float
        Parcel-waterbody pairs closer than this enter the link table,
        with every pair that has frontage.
    sliver_m, sliver_water_share : float
        Frontage is allocated to a parcel's land, never its water. For a
        parcel at least *sliver_water_share* water, land strips narrower
        than ``2 * sliver_m`` are removed first, so a lake parcel drawn a
        few meters past the NHD shoreline cannot take the shore from the
        lots around it. No parcel is excluded outright: a lake parcel
        that also holds a house on a strip of shore keeps that strip.
    holder_share, holder_house_m : float
        A parcel with at least *holder_share* of its area under one
        waterbody, holding at least *holder_share* of that waterbody's
        area (a lake drawn as its own parcel), yields each shoreline
        segment to the nearest other parcel within the tolerance, except
        within *holder_house_m* of its own building. This covers a lake
        parcel whose rim of land is wider than the opening removes.
        Measured 2026-09-30 on US-FL-LA: with the yield, parcels
        fronting open water >= 1 ha rise from 16,195 to 17,881, and the
        lake parcels that held 9 to 13 km of shore each leave the
        largest holders.
    max_vertices : int
        Waterbodies are cut into pieces of at most this many coordinates
        (`openplaces.geo.polygon.subdivide_polygons`) before any overlay
        or distance, so that no operation scans a whole large lake or
        sea polygon once per parcel.
    save_links : bool
        Write the parcel x waterbody link table beside the evidence.

    Returns
    -------
    EnrichState
        With lake_* and river_* evidence columns suffixed with the NHD
        recipe's version (``_nhdhr2023``).
    """
    recipe_ids = [r for r in (waterbody_recipe_id, area_recipe_id) if r]
    suffix = '_' + get_recipe_by_id(recipe_ids[0])['dataset'].version
    columns = [
        'lake_frontage',
        'lake_id',
        'lake_area_ha',
        'lake_distance',
        'lake_distance_parcel',
        'lake_id_nearest',
        'lake_area_ha_nearest',
        'river_frontage',
        'river_distance',
        'river_distance_parcel',
        'waterbody_area_share',
    ]
    evidence = pd.DataFrame(index=state.spine.index, columns=columns, dtype=object)

    spine = state.spine
    if 'geometry' not in getattr(spine, 'columns', ()):
        raise ValueError(
            "link_parcels_to_waterbodies needs spine geometry; set 'spine_geom: true'."
        )
    parcels = gpd.GeoDataFrame(geometry=spine.geometry, crs=spine.crs)
    parcels = parcels[~parcels.geometry.is_empty & parcels.geometry.notna()]
    crs = parcels.estimate_utm_crs()
    parcels = parcels.to_crs(crs)

    region = shapely.box(*parcels.total_bounds).buffer(max_distance_m)
    region_wgs84 = gpd.GeoSeries([region], crs=crs).to_crs('EPSG:4326').iloc[0]
    water = _load_waterbodies(region_wgs84, recipe_ids, crs)
    if state.timer:
        state.timer.mark('load waterbodies')

    def _finish(links=None):
        for c in columns:
            state.evidence[f'{c}{suffix}'] = evidence[c].reindex(state.evidence.index)
        if save_links and links is not None:
            out_path = get_output_path(
                state.recipe, state.admin_id, entity_recipe_id=state.entity_recipe
            )
            save_parquet(links, out_path.with_stem(out_path.stem + '_links'))
        return state

    if water.empty:
        if state.verbose:
            print('  no waterbodies within reach; writing empty evidence.')
        return _finish()

    classes = _feature_classes(recipe_ids[0])
    code = water['feature_code_nhd'].astype(int)
    water = water.assign(
        feature_class=code.map(classes['feature_class']).fillna('other').values,
        is_lake=code.map(classes[lake_definition]).fillna(False).astype(bool).values,
        is_river=code.map(classes[river_definition]).fillna(False).astype(bool).values,
        area_ha=water['area_km2'].astype(float).values * 100,
        is_great_lake=water['gnis_name'].isin(_great_lake_names(recipe_ids[0])).values,
    )
    linkable = water[water['is_lake'] | water['is_river']]
    # Every overlay and distance below runs against pieces of at most
    # max_vertices coordinates, never a whole large lake or sea polygon.
    # Shoreline segments still come from the whole polygons, so the cut
    # lines between pieces never count as shore.
    pieces = subdivide_polygons(water.geometry, max_vertices=max_vertices)
    if state.timer:
        state.timer.mark('subdivide waterbodies')

    # Frontage: allocate each shoreline segment to the nearest parcel
    # land (the parcel minus water).
    buildings = _primary_buildings(state.admin_id, parcels, footprint_recipe_id, crs)
    wet = water.index[water['feature_class'] != 'other']
    land, share = _shore_land(
        parcels,
        pieces[pieces.index.isin(wet)],
        sliver_m,
        sliver_water_share,
        buildings=buildings,
    )
    evidence['waterbody_area_share'] = share
    recipients = gpd.GeoDataFrame(geometry=land, crs=crs)
    clip = shapely.box(*parcels.total_bounds).buffer(tolerance_m + segment_m)
    holders = _waterbody_holders(parcels, linkable, holder_share, pieces=pieces)
    frontage = _allocate_frontage(
        linkable,
        recipients,
        tolerance_m,
        segment_m,
        clip,
        holders=holders,
        buildings=buildings,
        holder_house_m=holder_house_m,
    )
    if state.timer:
        state.timer.mark('allocate shoreline')

    # Link table: every pair with frontage, plus pairs within
    # link_distance_m, with parcel-edge and building distances.
    link_pieces = pieces[pieces.index.isin(linkable.index)]
    d_parcel = _min_distances(parcels.geometry, link_pieces, link_distance_m)
    d_parcel.index = d_parcel.index.set_names(['parcel', ID])
    near = d_parcel.index
    pairs = near.union(frontage.index) if len(frontage) else near
    links = pd.DataFrame(index=pairs)
    links['frontage_m'] = frontage.reindex(pairs).fillna(0.0).to_numpy()
    links['distance_parcel_m'] = d_parcel.reindex(pairs).to_numpy()
    # A building can stand farther from the water than its parcel's
    # edge; distances beyond four link ranges are left null.
    d_building = _min_distances(buildings, link_pieces, 4 * link_distance_m)
    d_building.index = d_building.index.set_names(['parcel', ID])
    links['distance_building_m'] = d_building.reindex(pairs).to_numpy()
    attrs = linkable[
        ['feature_class', 'is_lake', 'is_river', 'is_great_lake', 'area_ha']
    ]
    links = links.join(attrs, on=ID)
    links.index = links.index.set_names([spine.index.name or 'parcel_id', ID])
    if state.timer:
        state.timer.mark('link table')

    # Per-parcel frontage summaries
    parcel_level = links.index.names[0]
    lakes_f = links[links['is_lake'] & links['area_ha'].ge(frontage_min_area_ha)]
    fronted = lakes_f[lakes_f['frontage_m'] > 0]
    evidence['lake_frontage'] = fronted.groupby(level=parcel_level)['frontage_m'].sum()
    largest = fronted.reset_index().sort_values('area_ha', ascending=False)
    largest = largest.drop_duplicates(parcel_level).set_index(parcel_level)
    evidence['lake_id'] = largest[ID]
    evidence['lake_area_ha'] = largest['area_ha']
    rivers = links[links['is_river'] & links['frontage_m'].gt(0)]
    evidence['river_frontage'] = rivers.groupby(level=parcel_level)['frontage_m'].sum()

    # Nearest-lake and nearest-river distances, beyond the link range
    big = linkable['area_ha'].ge(distance_min_area_ha)
    lakes_d = linkable[linkable['is_lake'] & big]
    rivers_d = linkable[linkable['is_river']]
    for name, targets in (('lake', lakes_d), ('river', rivers_d)):
        target_pieces = gpd.GeoDataFrame(
            geometry=pieces[pieces.index.isin(targets.index)]
        )
        by_parcel = _nearest(parcels.geometry, target_pieces, max_distance_m)
        by_building = _nearest(buildings, target_pieces, max_distance_m)
        evidence[f'{name}_distance_parcel'] = by_parcel['distance']
        evidence[f'{name}_distance'] = by_building['distance']
        if name == 'lake':
            nearest_id = by_building[ID].fillna(by_parcel[ID])
            evidence['lake_id_nearest'] = nearest_id
            evidence['lake_area_ha_nearest'] = nearest_id.map(targets['area_ha'])
    if state.timer:
        state.timer.mark('nearest distances')

    for c in columns:
        if c not in ('lake_id', 'lake_id_nearest'):
            evidence[c] = pd.to_numeric(evidence[c], errors='coerce')
    # A parcel with no frontage holds 0 m, not an unknown
    for c in ('lake_frontage', 'river_frontage'):
        evidence[c] = evidence[c].fillna(0.0)

    if state.verbose:
        n = int(evidence['lake_frontage'].gt(0).sum())
        print(
            f'  {n:,d} of {len(evidence):,d} parcels front a lake >= '
            f'{frontage_min_area_ha} ha; {len(links):,d} parcel-waterbody links.'
        )
    return _finish(links.reset_index())
