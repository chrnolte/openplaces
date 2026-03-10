"""
ids.py

Functions for computing geographic identifiers.

- `geo_id`: to link identical polygons securely through time (within a
  small spatial tolerance, to minimize corrections).
- `openlocationcode`: for point locations
- unique building IDs (UBID) for building footprints
"""

import hashlib
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely
from openlocationcode import openlocationcode as olc

from openplaces.io.transform import add_unique_suffix


def get_geo_ids(
    gdf,
    grid_degrees=0.000001,  # ~11cm at equator, ~8cm at 45°N
    hash_length=24,
    handle_duplicates=True,
    verbose=False,
):
    """Generate stable, unique parcel IDs from polygon geometry.

    Uses fixed degree grid in EPSG:4326 for full Earth coverage.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame with parcel geometries
    grid_degrees : float
        Grid size in degrees (default 0.00003)
    hash_length : int
        Number of hex characters in output (default 18 = 72 bits)
    handle_duplicates : bool
        If True, adds numeric suffix to duplicate GIDs (default True)
    verbose: bool
        If True, prints information on duplicates

    Returns
    -------
    pd.Series
        Series of geo_ids with same index as input GeoDataFrame

    Notes
    -----
    Why degrees instead of projected CRS:
    - No projection covers entire Earth without distortion/singularities
    - Degree grid is globally consistent (same grid cell = same x/y)
    - Simple, fast (no reprojection needed)
    - Works everywhere including poles

    Trade-off:
    - Grid "size" in meters varies by latitude (larger at equator)
    - But parcels at same location always use same grid
    - This guarantees non-overlapping parcels get different IDs
    """

    # Ensure EPSG:4326
    if gdf.crs != 'epsg:4326':
        print('Reprojecting vector data to `epsg:4326` to compute `geo_ids`.')
        gdf = gdf.to_crs('epsg:4326')

    # Get bounds for each parcel
    # (using fillna(0) to avoid checking for empty geometries)
    bounds = gdf.bounds.fillna(0)

    # Quantize bbox corners (consistent grid for all parcels)
    minx_q = (bounds['minx'] / grid_degrees).round().astype(int)
    miny_q = (bounds['miny'] / grid_degrees).round().astype(int)
    maxx_q = (bounds['maxx'] / grid_degrees).round().astype(int)
    maxy_q = (bounds['maxy'] / grid_degrees).round().astype(int)

    # Area in square degrees (log scale)
    # Note: Area in degrees² varies with latitude, but that's okay
    # because we're comparing relative sizes at similar locations
    warnings.filterwarnings('ignore', 'Geometry is in a geographic CRS')
    area_deg2 = gdf.area.fillna(0)
    warnings.filterwarnings('default', 'Geometry is in a geographic CRS')
    area_q = (
        (np.log10(area_deg2 * 1e10 + 1) * 100).round().fillna(0).astype(int)
    )  # Scale up for precision

    # Compactness: perimeter²/area (dimensionless, so units don't matter)
    warnings.filterwarnings('ignore', 'Geometry is in a geographic CRS')
    compactness = (gdf.length**2) / (area_deg2 + 1e-10)
    warnings.filterwarnings('default', 'Geometry is in a geographic CRS')
    compact_q = (compactness * 10).round().fillna(0).astype(int)

    # Create hash inputs
    hash_inputs = (
        minx_q.astype(str)
        + ','
        + miny_q.astype(str)
        + ','
        + maxx_q.astype(str)
        + ','
        + maxy_q.astype(str)
        + ','
        + area_q.astype(str)
        + ','
        + compact_q.astype(str)
    )

    # Generate hash
    geo_ids = hash_inputs.apply(
        lambda s: hashlib.sha256(s.encode()).hexdigest()[:hash_length]
    )

    # Check for duplicates
    duplicates = geo_ids.duplicated(keep=False)

    geo_ids.loc[gdf['geometry'].is_empty] = 'no-geometry'

    if duplicates.any():
        n_dupl = duplicates.sum()
        if verbose:
            print(
                f'Warning: {n_dupl} polygons with duplicate GIDs '
                f'({n_dupl / len(geo_ids) * 100:.2g}%)'
            )

            # Show some examples
            print('\nExample duplicates:')
            dup_examples = (
                geo_ids[duplicates].sort_values().head(min(10, duplicates.sum())).index
            )
            print(
                pd.concat(
                    [
                        geo_ids[dup_examples].rename('geo_id'),
                        hash_inputs[dup_examples].rename('hash_inputs'),
                        gdf.loc[dup_examples],
                    ],
                    axis=1,
                )
            )

        if handle_duplicates:
            # Handle collisions with suffix
            if verbose:
                print('Adding suffixes...')

            geo_ids.loc[duplicates] = add_unique_suffix(geo_ids[duplicates])

    return geo_ids.rename('geo_id')


def add_geo_id_index(
    gdf,
    name='geo_id',
    handle_duplicates=True,
    verbose=False,
):
    """Return the GeoDataFrame using `geo_id` as the index

    Parameters
    ----------
    gdf : GeoDataFrame
        Polygon data
    name : str
        Name of index column
    handle_duplicates : bool
        If True, adds numeric suffix to duplicate GIDs (default True)
    verbose: bool
        If True, prints information on duplicates
    """

    gdf = gdf.copy()
    gdf.index = pd.Index(
        get_geo_ids(gdf, handle_duplicates=handle_duplicates, verbose=verbose),
        name=name,
    )
    if gdf.index.duplicated().any():
        raise ValueError(
            'Unhandled duplicates found in `geo_id` index. '
            'Set `handle_duplicates=True` or pick a different index.'
        )
    return gdf


def get_openlocationcodes(
    gdf: gpd.GeoDataFrame,
    name='openlocationcode',
    codelength=11,
    handle_duplicates=True,
):
    """Assign a location index based on centroid (Open Location Code).

    Parameters
    ----------
    gdf: GeoDataFrame
        Vector data with polygon geometries in any CRS.
    name : str
        Name of index
    codelength : int
        `openlocationcode` code length
    handle_duplicates : bool
        If True, adds numeric suffix to duplicate OLCs (default True)
    """
    geom_arr = gdf.geometry.values
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        geom_arr = shapely.transform(
            geom_arr,
            pyproj.Transformer.from_crs(gdf.crs, 'epsg:4326', always_xy=True).transform,
        )

    cents = shapely.centroid(geom_arr)
    lats = shapely.get_y(cents)
    lngs = shapely.get_x(cents)

    if codelength == 11:
        # Fast implementation (default, vectorized)
        raw_ids = _olc_encode_11(lats, lngs)
    else:
        raw_ids = [
            olc.encode(la, lo, codeLength=codelength) for la, lo in zip(lats, lngs)
        ]

    if not handle_duplicates:
        return pd.Series(raw_ids, index=gdf.index, name=name)

    counts: dict[str, int] = {}
    for code in raw_ids:
        counts[code] = counts.get(code, 0) + 1

    seen: dict[str, int] = {}
    footprint_ids = []
    for code in raw_ids:
        if counts[code] == 1:
            footprint_ids.append(code)
        else:
            seen[code] = seen.get(code, 0) + 1
            footprint_ids.append(f'{code}-{seen[code]}')

    return pd.Series(footprint_ids, index=gdf.index, name=name)


def add_openlocationcode_index(
    gdf,
    name='openlocationcode',
    codelength=11,
    handle_duplicates=True,
):
    """Return the GeoDataFrame using `geo_id` as the index

    Parameters
    ----------
    gdf : GeoDataFrame
        Polygon data
    name : str
        Name of index column
    codelength : int
        `openlocationcode` code length
    handle_duplicates : bool
        If True, adds numeric suffix to duplicate GIDs (default True)
    """

    gdf = gdf.copy()
    gdf.index = pd.Index(
        get_openlocationcodes(
            gdf, name=name, codelength=codelength, handle_duplicates=handle_duplicates
        ),
        name=name,
    )
    if gdf.index.duplicated().any():
        raise ValueError(
            'Unhandled duplicates found in `openlocationcode` index. '
            'Set `handle_duplicates=True` or pick a different index.'
        )
    return gdf


# Constants and encode logic for the fast computation of unique building
# IDs (UBID). Vectorized computations of Level-11 Open Location Codes
# (OLC) make this function much faster for large polygon datasets than
# the original from Pacific Northwest National Laboratory (PNNL).
# Tested against original for 74K Microsoft footprints in US-NC-BU (100%
# match). Original sources:
# https://github.com/pnnl/buildingid
# https://github.com/google/open-location-code

OLC_FINAL_LAT_PREC = 25_000_000
OLC_FINAL_LONG_PREC = 8_192_000
OLC_CELL_H_UNITS = 625  # GRID_LAT_FIRST_PLACE_VALUE_
OLC_CELL_W_UNITS = 256  # GRID_LNG_FIRST_PLACE_VALUE_
OLC_CELL_H = OLC_CELL_H_UNITS / OLC_FINAL_LAT_PREC  # 2.5e-5 degrees
OLC_CELL_W = OLC_CELL_W_UNITS / OLC_FINAL_LONG_PREC  # 3.125e-5 degrees
OLC_GRID_ROWS = 5
OLC_GRID_COLS = 4
OLC_ENC_BASE = 20
OLC_ALPHABET = np.frombuffer(b'23456789CFGHJMPQRVWX', dtype=np.uint8)
OLC_ALPHABET_INV = {chr(c): i for i, c in enumerate(OLC_ALPHABET)}


def _olc_snap_11(lats: np.ndarray, lngs: np.ndarray) -> tuple:
    """Return OLC cell (lat_lo, lat_hi, lng_lo, lng_hi) for codelength=11."""
    lat_val = np.round((lats + 90) * OLC_FINAL_LAT_PREC, 6).astype(np.int64)
    lng_val = np.round((lngs + 180) * OLC_FINAL_LONG_PREC, 6).astype(np.int64)
    lat_lo = (lat_val // OLC_CELL_H_UNITS) * OLC_CELL_H_UNITS / OLC_FINAL_LAT_PREC - 90
    lng_lo = (
        lng_val // OLC_CELL_W_UNITS
    ) * OLC_CELL_W_UNITS / OLC_FINAL_LONG_PREC - 180
    return lat_lo, lat_lo + OLC_CELL_H, lng_lo, lng_lo + OLC_CELL_W


def _olc_encode_11(lats: np.ndarray, lngs: np.ndarray) -> list[str]:
    """Vectorized OLC encode for codelength=11."""
    lats = np.clip(lats, -90, 90).copy()
    lngs = ((lngs + 180) % 360) - 180
    lats[lats == 90.0] -= 0.000125
    lat_val = np.round((lats + 90) * OLC_FINAL_LAT_PREC, 6).astype(np.int64)
    lng_val = np.round((lngs + 180) * OLC_FINAL_LONG_PREC, 6).astype(np.int64)
    n = len(lats)
    lv = lat_val.copy()
    lv_lng = lng_val.copy()

    grid_chars = np.zeros((n, 5), dtype=np.uint8)
    for i in range(5):
        ld = lv % OLC_GRID_ROWS
        lc = lv_lng % OLC_GRID_COLS
        grid_chars[:, 4 - i] = OLC_ALPHABET[ld * OLC_GRID_COLS + lc]
        lv //= OLC_GRID_ROWS
        lv_lng //= OLC_GRID_COLS

    pair_chars = np.zeros((n, 10), dtype=np.uint8)
    for i in range(5):
        pair_chars[:, 9 - 2 * i] = OLC_ALPHABET[lv_lng % OLC_ENC_BASE]
        pair_chars[:, 9 - 2 * i - 1] = OLC_ALPHABET[lv % OLC_ENC_BASE]
        lv //= OLC_ENC_BASE
        lv_lng //= OLC_ENC_BASE

    codes = np.empty((n, 12), dtype=np.uint8)
    codes[:, 0:8] = pair_chars[:, 0:8]
    codes[:, 8] = ord('+')
    codes[:, 9:11] = pair_chars[:, 8:10]
    codes[:, 11] = grid_chars[:, 0]
    return [row.tobytes().decode('ascii') for row in codes]


def get_ubids(gdf, duplicates: str = 'raise'):
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    geom_arr = gdf.geometry.values
    bounds = shapely.bounds(geom_arr)
    lng_lo = bounds[:, 0]
    lat_lo = bounds[:, 1]
    lng_hi = bounds[:, 2]
    lat_hi = bounds[:, 3]

    cents = shapely.centroid(geom_arr)
    lat_c = shapely.get_y(cents)
    lng_c = shapely.get_x(cents)

    center_codes = _olc_encode_11(lat_c, lng_c)
    c_lat_lo, c_lat_hi, c_lng_lo, c_lng_hi = _olc_snap_11(lat_c, lng_c)
    ne_lat_lo, ne_lat_hi, ne_lng_lo, ne_lng_hi = _olc_snap_11(lat_hi, lng_hi)
    sw_lat_lo, sw_lat_hi, sw_lng_lo, sw_lng_hi = _olc_snap_11(lat_lo, lng_lo)
    cell_h = OLC_CELL_H
    cell_w = OLC_CELL_W

    north = np.round((ne_lat_hi - c_lat_hi) / cell_h).astype(int)
    east = np.round((ne_lng_hi - c_lng_hi) / cell_w).astype(int)
    south = np.round((c_lat_lo - sw_lat_lo) / cell_h).astype(int)
    west = np.round((c_lng_lo - sw_lng_lo) / cell_w).astype(int)

    if (north < 0).any() or (east < 0).any() or (south < 0).any() or (west < 0).any():
        raise ValueError('Negative UBID extent: bounding box corners may be inverted')

    ubids = pd.Series(
        [
            f'{c}-{n}-{e}-{s}-{w}'
            for c, n, e, s, w in zip(center_codes, north, east, south, west)
        ],
        index=gdf.index,
    )

    if duplicates == 'raise' and ubids.duplicated().any():
        raise ValueError(f'{ubids.duplicated().sum()} duplicate UBIDs found.')
    elif duplicates == 'drop':
        ubids = ubids[~ubids.duplicated()]

    return ubids


def add_ubid_index(
    gdf,
    name='ubid',
    duplicates='raise',
):
    """Return the GeoDataFrame using `geo_id` as the index

    Parameters
    ----------
    gdf : GeoDataFrame
        Polygon data
    name : str
        Name of index column
    duplicates : str
        'raise' or 'drop'. Duplicate UBID indices are not permitted.
    """

    gdf = gdf.copy()
    gdf.index = pd.Index(get_ubids(gdf, duplicates=duplicates), name=name)
    if gdf.index.duplicated().any():
        raise ValueError(
            'Unhandled duplicates found in `ubid` index. '
            "Set `duplicates='drop'` or pick a different index."
        )
    return gdf


def _olc_decode_11(codes: list[str]) -> tuple:
    """Decode OLC codes (codelength=11) to cell bounds

    Returns: (lat_low, lat_high, long_low, lng_high)"""
    n = len(codes)
    pair_chars = np.zeros((n, 10), dtype=np.int64)
    grid_chars = np.zeros((n, 1), dtype=np.int64)

    for i, code in enumerate(codes):
        # Strip '+', giving 10 pair chars + 1 grid char
        stripped = code[:8] + code[9:12]
        for j in range(10):
            pair_chars[i, j] = OLC_ALPHABET_INV[stripped[j]]
        grid_chars[i, 0] = OLC_ALPHABET_INV[stripped[10]]

    # Decode pair characters (5 lat/lng pairs, high to low)
    lat_val = np.zeros(n, dtype=np.int64)
    lng_val = np.zeros(n, dtype=np.int64)
    for i in range(5):
        lat_val = lat_val * OLC_ENC_BASE + pair_chars[:, 2 * i]
        lng_val = lng_val * OLC_ENC_BASE + pair_chars[:, 2 * i + 1]

    # Decode single grid character
    grid_lat = grid_chars[:, 0] // OLC_GRID_COLS
    grid_lng = grid_chars[:, 0] % OLC_GRID_COLS

    lat_val = lat_val * OLC_GRID_ROWS + grid_lat
    lng_val = lng_val * OLC_GRID_COLS + grid_lng

    lat_lo = lat_val * OLC_CELL_H_UNITS / OLC_FINAL_LAT_PREC - 90
    lng_lo = lng_val * OLC_CELL_W_UNITS / OLC_FINAL_LONG_PREC - 180

    return lat_lo, lat_lo + OLC_CELL_H, lng_lo, lng_lo + OLC_CELL_W


def decode_openlocationcodes(
    codes: list[str] | pd.Series,
) -> gpd.GeoDataFrame:
    """Decode level-11 Open Location Codes to a GeoDataFrame of points.

    Returns the center of each OLC cell as a Point geometry.

    Parameters
    ----------
    codes : list[str] or pd.Series
        Sequence of OLC strings with codelength=11 (e.g. '85G8Q23G+CFM').
        If a Series, its index is preserved in the output.
    name : str
        Name for the geometry column (default 'geometry').

    Returns
    -------
    GeoDataFrame
        Points at OLC cell centers, CRS EPSG:4326.
    """
    if isinstance(codes, pd.Series):
        index = codes.index
    elif isinstance(codes, pd.Index):
        index = codes
    elif isinstance(codes, list):
        index = pd.Index(codes, name='openlocationcode')
    code_list = list(codes)

    lat_lo, lat_hi, lng_lo, lng_hi = _olc_decode_11(code_list)

    lats = (lat_lo + lat_hi) / 2
    lngs = (lng_lo + lng_hi) / 2

    geometry = gpd.points_from_xy(lngs, lats)

    return gpd.GeoDataFrame(
        index=index,
        geometry=geometry,
        crs='epsg:4326',
    )


def decode_ubids(ubids: pd.Series, outer: bool = True) -> gpd.GeoDataFrame:
    """Decode a Series of UBIDs to a GeoDataFrame of bounding boxes.

    Parameters
    ----------
    ubids : pd.Series
        Series of UBID strings in the format 'OLC-N-E-S-W'.
    outer : bool
        If True (default), return the outermost plausible bounding box.
        If False, return the innermost plausible bounding box (shrinks
        each side by 0.5 OLC cells to account for rounding).
    """
    parts = ubids.str.split('-', n=4, expand=True)
    # Re-join the OLC code (contains one internal '-' after position 8: 'XXXXXXXX+XX X')
    # UBID format is: <8chars>+<2chars><1grid>-N-E-S-W  (the + is in the OLC)
    # str.split on '-' with n=4 splits: [olc, N, E, S, W] correctly since
    # OLC uses '+' not '-' as separator.
    center_codes = parts[0].tolist()
    north = parts[1].astype(int).to_numpy()
    east = parts[2].astype(int).to_numpy()
    south = parts[3].astype(int).to_numpy()
    west = parts[4].astype(int).to_numpy()

    c_lat_lo, c_lat_hi, c_lng_lo, c_lng_hi = _olc_decode_11(center_codes)

    shrink = 0.5 if not outer else 0.0

    lat_south = c_lat_lo - (south - shrink) * OLC_CELL_H
    lat_north = c_lat_hi + (north - shrink) * OLC_CELL_H
    lng_west = c_lng_lo - (west - shrink) * OLC_CELL_W
    lng_east = c_lng_hi + (east - shrink) * OLC_CELL_W

    geometries = shapely.box(lng_west, lat_south, lng_east, lat_north)

    return gpd.GeoDataFrame(
        geometry=geometries,
        index=ubids.index,
        crs='EPSG:4326',
    )
