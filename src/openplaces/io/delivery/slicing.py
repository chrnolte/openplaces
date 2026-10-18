"""Cut an outline that two units both delivered at the line between them.

Each admin unit is harmonized and curated on its own, so an outline that
crosses a unit line can be curated by both units, and each unit's parcel
puts its own assessed value on its own copy. Pooled into one region, the
building is drawn twice. Measured 2026-10-17 on boston-inner-core: 150
outline pairs in two of 11 neighboring Massachusetts towns, 52 of them
the same outline and 98 two outlines of one building (MassGIS in one
town, OBM in the other).

The rule is the maintainer's (2026-10-17): choose one outline, cut it
along the unit line, and give each piece the values that unit's own
assessor layer put on its copy. Nothing is summed across units, and the
pieces are never merged again.

- **Which outline.** The copy whose representative point lies in its own
  unit, else the larger.
- **Pieces.** The chosen outline intersected with each unit's polygon
  (the admin geometry the units are built from). A piece takes the row
  of its unit's copy, with the piece's geometry, area and centroid.
- **Slivers.** Where the smaller piece is under `MIN_PIECE_M2` or under
  `MIN_PIECE_SHARE` of the outline, nothing is cut: the copy of the unit
  holding most of the building stays as it is, and the other copy is
  left out, its value staying on that unit's parcel. Measured
  2026-10-17 on boston-inner-core: 62 of 150 pairs, among them 34 whose
  smaller piece is under 1 m2 (a building wholly in one town that the
  other drew too).
- **Ids.** A piece's id is the Open Location Code of its centroid, the
  way every footprint id is made; a code that collides with another id
  is replaced by a longer one (a finer cell), so a collision is broken
  geometrically, never by a sequence number. `sliced_from` keeps the
  chosen outline's id.

An outline only one unit delivered is left whole, even where it crosses
the line: the other unit's parcel value never reached a footprint, and
this step has no parcels to apportion (maintainer, 2026-10-17).
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from openplaces.geo.ids import get_openlocationcodes

#: Outlines overlapping by at least this share of the smaller one are
#: one building drawn twice (the pair test of resolve_spine and
#: spine_resolution).
PAIR_SHARE_MIN = 0.5
MIN_PIECE_M2 = 10.0
MIN_PIECE_SHARE = 0.05
METRIC_CRS = 'epsg:6933'
SLICED_FROM = 'sliced_from'


def _piece_ids(pieces: gpd.GeoSeries, taken: set) -> pd.Series:
    """Centroid Open Location Codes, lengthened where one is taken."""
    ids = get_openlocationcodes(
        gpd.GeoDataFrame(geometry=pieces), handle_duplicates=False
    ).astype(object)
    for length in (13, 15):
        clash = ids.isin(taken) | ids.duplicated(keep=False)
        if not clash.any():
            break
        ids[clash] = get_openlocationcodes(
            gpd.GeoDataFrame(geometry=pieces[clash]),
            codelength=length,
            handle_duplicates=False,
        ).astype(object)
    if ids.isin(taken).any() or ids.duplicated().any():
        raise ValueError('slice_at_unit_lines: a piece id stays ambiguous.')
    return ids


def slice_at_unit_lines(
    pooled: gpd.GeoDataFrame,
    unit_column: str,
    units: gpd.GeoDataFrame,
    *,
    min_piece_m2: float = MIN_PIECE_M2,
    min_piece_share: float = MIN_PIECE_SHARE,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Replace outlines two units delivered by one outline cut at the line.

    Parameters
    ----------
    pooled : GeoDataFrame
        The region's pooled rows, indexed by entity id (an id may repeat
        across units), with *unit_column* naming each row's unit.
    unit_column : str
        Column holding each row's admin unit id.
    units : GeoDataFrame
        The units' polygons, indexed by admin id.
    min_piece_m2, min_piece_share : float
        Below either, the smaller piece is not cut (see the module).

    Returns
    -------
    tuple of (GeoDataFrame, DataFrame)
        The rows, with each sliced pair replaced by its pieces and each
        sliver pair by its majority copy; and the pieces' provenance,
        columns `piece_id`, `source_id` (the curated row it came from)
        and the unit column, for the evidence pass.
    """
    empty = pd.DataFrame(columns=['piece_id', 'source_id', unit_column])
    if pooled.empty or units.empty:
        return pooled, empty
    crs = pooled.crs
    frame = pooled.reset_index()
    id_column = frame.columns[0]
    metric = frame.geometry.to_crs(METRIC_CRS)
    units_m = units.to_crs(METRIC_CRS)
    left = gpd.GeoDataFrame(
        {'u': frame[unit_column]}, geometry=metric.to_numpy(), crs=METRIC_CRS
    )
    j = gpd.sjoin(left, left, predicate='intersects')
    j = j[(j.index < j['index_right']) & (j['u_left'] != j['u_right'])]
    if j.empty:
        return pooled, empty
    ga = metric.to_numpy()[j.index]
    gb = metric.to_numpy()[j['index_right'].to_numpy()]
    inter = gpd.GeoSeries(ga).intersection(gpd.GeoSeries(gb)).area.to_numpy()
    small = np.minimum(
        gpd.GeoSeries(ga).area.to_numpy(), gpd.GeoSeries(gb).area.to_numpy()
    )
    j = j[inter >= PAIR_SHARE_MIN * small]

    drop: set[int] = set()
    pieces: list[tuple[int, object]] = []  # (row position, piece geometry)
    sliced_from: dict[int, str] = {}
    used: set[int] = set()
    for a, b in zip(j.index, j['index_right'], strict=True):
        if a in used or b in used:
            continue
        used.update((a, b))
        unit_a, unit_b = frame.at[a, unit_column], frame.at[b, unit_column]
        if unit_a not in units_m.index or unit_b not in units_m.index:
            continue
        poly_a, poly_b = units_m.geometry[unit_a], units_m.geometry[unit_b]
        geom_a, geom_b = metric.iloc[a], metric.iloc[b]
        own_a = poly_a.contains(geom_a.representative_point())
        own_b = poly_b.contains(geom_b.representative_point())
        if own_a != own_b:
            kept, kept_pos = (geom_a, a) if own_a else (geom_b, b)
        else:
            kept, kept_pos = (geom_a, a) if geom_a.area >= geom_b.area else (geom_b, b)
        piece_a, piece_b = kept.intersection(poly_a), kept.intersection(poly_b)
        minor = min(piece_a.area, piece_b.area)
        if minor < min_piece_m2 or minor < min_piece_share * kept.area:
            drop.add(b if piece_a.area >= piece_b.area else a)
            continue
        source_id = str(frame.at[kept_pos, id_column])
        for position, piece in ((a, piece_a), (b, piece_b)):
            pieces.append((position, piece))
            sliced_from[position] = source_id

    if not drop and not pieces:
        return pooled, empty
    positions = [p for p, _ in pieces]
    piece_geoms = gpd.GeoSeries([g for _, g in pieces], crs=METRIC_CRS).to_crs(crs)
    keep_mask = ~frame.index.isin(list(drop) + positions)
    taken = set(frame.loc[keep_mask, id_column].astype(str))
    new_ids = _piece_ids(piece_geoms, taken)

    sliced = frame.loc[positions].copy()
    provenance = pd.DataFrame(
        {
            'piece_id': new_ids.to_numpy(),
            'source_id': sliced[id_column].astype(str).to_numpy(),
            unit_column: sliced[unit_column].to_numpy(),
        }
    )
    sliced[id_column] = new_ids.to_numpy()
    sliced[SLICED_FROM] = [sliced_from[p] for p in positions]
    sliced = sliced.set_geometry(piece_geoms.to_numpy())
    area = piece_geoms.to_crs(METRIC_CRS).area.to_numpy()
    if 'area_m2' in sliced.columns:
        sliced['area_m2'] = area
    if 'area_ha' in sliced.columns:
        sliced['area_ha'] = area / 1e4
    if {'long', 'lat'} <= set(sliced.columns):
        cents = piece_geoms.to_crs('epsg:4326').centroid
        sliced['long'] = cents.x.to_numpy()
        sliced['lat'] = cents.y.to_numpy()

    kept_rows = frame.loc[keep_mask]
    out = pd.concat([kept_rows, sliced], ignore_index=True)
    out = gpd.GeoDataFrame(out.set_index(id_column), geometry='geometry', crs=crs)
    return out, provenance
