"""
ids.py

Functions for computing geographic identifiers.

- `geo_id`: to link identical polygons securely through time (within a
  small spatial tolerance, to minimize corrections).
- `openlocationcode`: for point locations
- unique building IDs (UBID) for building footprints
"""

import hashlib
import re
import warnings
from functools import cache
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely
from openlocationcode import openlocationcode as olc

from openplaces.geo.polygon import reproject
from openplaces.io.transform import add_unique_suffix

# Default area/compactness quantization, expressed as the historical hardcoded
# multipliers (area_q = round(log10(area*1e10+1) * 100), compact_q =
# round(compact * 10)). AREA_TOLERANCE_DEFAULT/COMPACTNESS_STEP_DEFAULT below are
# derived from these so get_geo_ids' default output is unchanged.
_AREA_PRECISION_DEFAULT = 100
_COMPACT_PRECISION_DEFAULT = 10
AREA_TOLERANCE_DEFAULT = 10 ** (1 / _AREA_PRECISION_DEFAULT) - 1
COMPACTNESS_STEP_DEFAULT = 1 / _COMPACT_PRECISION_DEFAULT


def _shape_signals(geom_arr):
    """Return (area, compactness) for an array of geometries.

    Compactness is perimeter^2 / area (dimensionless). Shared by `get_geo_ids`
    and the crosswalk's shape-similarity gate (`crosswalk.py`), so both use the
    exact same formula.
    """
    area = np.nan_to_num(shapely.area(geom_arr))
    length = np.nan_to_num(shapely.length(geom_arr))
    compact = length**2 / (area + 1e-10)
    return area, compact


def get_geo_ids(
    gdf,
    grid_degrees=0.000001,  # ~11cm at equator, ~8cm at 45°N
    area_tolerance=AREA_TOLERANCE_DEFAULT,
    compactness_step=COMPACTNESS_STEP_DEFAULT,
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
        Grid size in degrees
    area_tolerance : float
        Relative area tolerance (e.g. 0.02 = 2%): two areas whose ratio is
        within this bound quantize to the same bucket. Default reproduces the
        historical hardcoded area quantization exactly.
    compactness_step : float
        Absolute step size for compactness (perimeter^2/area) quantization.
        Default reproduces the historical hardcoded compactness quantization
        exactly. Kept linear (not ratio-based, unlike `area_tolerance`) because
        compactness is not log-scaled internally -- ratio-scaling it would
        silently change existing `geo_id` values for parcels away from the
        tolerance's reference point.
    hash_length : int
        Number of hex characters in output
    handle_duplicates : bool
        Add unique numeric suffix to duplicate geo_ids (default True)
    verbose: bool
        Print information on duplicates (default False)

    Returns
    -------
    pd.Series
        Series of geo_ids with same index as input GeoDataFrame
    """

    # Ensure GeoDataFrame is in EPSG:4326 projection
    if gdf.crs != 'epsg:4326':
        print('Reprojecting vector data to `epsg:4326` to compute `geo_ids`.')
        gdf = reproject(gdf, 'epsg:4326')

    # Quantize bbox corners (consistent grid for all parcels; nan_to_num
    # avoids checking for empty geometries)
    geom = gdf.geometry.values
    bounds_q = np.round(np.nan_to_num(shapely.bounds(geom)) / grid_degrees).astype(
        np.int64
    )

    # Area in square degrees (log scale, scaled up for precision)
    # Note: Area in degrees² varies with latitude, but that's okay
    # because we're comparing relative sizes at similar locations
    area_deg2, compact = _shape_signals(geom)
    area_precision = 1 / np.log10(1 + area_tolerance)
    area_q = np.round(np.log10(area_deg2 * 1e10 + 1) * area_precision).astype(np.int64)

    # Compactness: perimeter²/area (dimensionless, so units don't matter)
    compact_precision = 1 / compactness_step
    compact_q = np.round(compact * compact_precision).astype(np.int64)

    # Create hash inputs
    cols = [
        c.tolist()
        for c in (
            bounds_q[:, 0],
            bounds_q[:, 1],
            bounds_q[:, 2],
            bounds_q[:, 3],
            area_q,
            compact_q,
        )
    ]
    hash_inputs = [f'{a},{b},{c},{d},{e},{f}' for a, b, c, d, e, f in zip(*cols)]

    # Generate hash
    sha = hashlib.sha256
    geo_ids = pd.Series(
        [sha(s.encode()).hexdigest()[:hash_length] for s in hash_inputs],
        index=gdf.index,
    )

    # Give empty geometries a 'no-geometry' string ID
    geo_ids.loc[gdf.geometry.is_empty] = 'no-geometry'

    # Check for duplicates
    duplicates = geo_ids.duplicated(keep=False)

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
            hash_input_series = pd.Series(hash_inputs, index=gdf.index)
            print(
                pd.concat(
                    [
                        geo_ids[dup_examples].rename('geo_id'),
                        hash_input_series[dup_examples].rename('hash_inputs'),
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
        # interleaved=False: pyproj's Transformer.transform(xx, yy) takes x/y as
        # separate arguments, matching this calling convention -- the default
        # interleaved=True instead passes one combined (N, 2) array, which
        # Transformer.transform doesn't accept.
        geom_arr = shapely.transform(
            geom_arr,
            pyproj.Transformer.from_crs(gdf.crs, 'epsg:4326', always_xy=True).transform,
            interleaved=False,
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
    """Return the GeoDataFrame using Open Location Code (OLC) as the index.

    Parameters
    ----------
    gdf : GeoDataFrame
        Polygon data
    name : str
        Name of index column
    codelength : int
        `openlocationcode` code length
    handle_duplicates : bool
        If True, adds numeric suffix to duplicate OLCs (default True)
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
    """Return the GeoDataFrame using Unique Building ID (UBID) as the index.

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


# Parcel identifier standardization
#
# Convert a raw source parcel id (`parcel_id_assessor`) into a standardized,
# locally cross-comparable matching key (`parcel_id_local`) so parcel, tax, and
# transaction datasets in the same locality join without re-deriving ids.
# Ported from the places APN-matching methodology; only the conversion
# operations that appear in the auto-selected best solutions are implemented.
# The per-admin-unit conversion table (`parcel_id_links.csv`) and the pattern
# library (`parcel_id_patterns.csv`) live beside this module.

_PARCEL_ID_DIR = Path(__file__).parent


@cache
def _parcel_id_patterns() -> pd.DataFrame:
    """Active parcel-id extraction patterns (regex), indexed by pattern name."""
    patterns = pd.read_csv(_PARCEL_ID_DIR / 'parcel_id_patterns.csv')
    return patterns[patterns['active'] == 1].set_index('pattern')


@cache
def _parcel_id_links() -> pd.DataFrame:
    """Default per-admin-unit conversions (parcel + tax kinds), by admin_id."""
    return pd.read_csv(_PARCEL_ID_DIR / 'parcel_id_links.csv', dtype=str).set_index(
        'admin_id'
    )


def _pattern_regex(pattern) -> str:
    """Resolve a pattern name (or raw regex) to an extraction regex."""
    if pattern is None or (isinstance(pattern, float) and pd.isna(pattern)):
        return r'^(.*)$'
    patterns = _parcel_id_patterns()
    if pattern in patterns.index:
        return patterns.at[pattern, 'regex']
    if pattern in ('Unrecognized', 'Useless', 'Ignored'):
        return r'^(.*)$'
    if pattern == 'Empty':
        return r'^()$'
    if isinstance(pattern, str) and pattern.startswith('^') and pattern.endswith('$'):
        return pattern
    return r'^(.*)$'


def _conv_dict(conv_code: str) -> dict[str, str]:
    """Parse a conversion code ('op: value & op: value') into an ordered dict."""
    if not isinstance(conv_code, str) or not conv_code.strip():
        return {}
    out: dict[str, str] = {}
    for part in conv_code.split(' & '):
        if ': ' in part:
            key, value = part.split(': ', 1)
            out[key.strip()] = value.strip()
        elif part.strip():
            out[part.strip()] = ''
    return out


def _split_groups(cols: pd.DataFrame, spec: str) -> pd.DataFrame:
    """Split selected capture-group columns in place (by char or position)."""
    for split in spec.split(' '):
        col, splitchar = split.split('|')
        if col not in cols.columns:
            continue
        ic = cols.columns.get_loc(col)
        if re.fullmatch(r'-?\d+', splitchar):
            pos = int(splitchar)
            left = cols[col].str.slice(0, pos).rename(0)
            right = cols[col].str.slice(pos).rename(1)
            piece = pd.concat([left, right], axis=1)
        else:
            piece = cols[col].str.split(splitchar, n=1, expand=True)
        piece = piece.fillna('')
        if len(piece.columns) == 1:
            piece[1] = ''
        piece = piece.rename(columns={i: f'{col}_{i}' for i in (0, 1)})
        cols = cols.join(piece)
        cols = cols[
            list(cols.columns[:ic])
            + [f'{col}_0', f'{col}_1']
            + list(cols.columns[ic + 1 : -2])
        ]
    return cols


def _switch_groups(cols: pd.DataFrame, spec: str) -> pd.DataFrame:
    """Switch the positions of pairs of capture-group columns."""
    for sw in spec.split(' '):
        col1, col2 = sw.split('|')
        if {col1, col2} <= set(cols.columns):
            order = list(cols.columns)
            i1, i2 = order.index(col1), order.index(col2)
            order[i1], order[i2] = order[i2], order[i1]
            cols = cols[order]
    return cols


def _merge_after(cols: pd.DataFrame, spec: str) -> pd.DataFrame:
    """Concatenate each named column with its right neighbour (no separator)."""
    for col in spec.split(' '):
        if col in cols.columns:
            ic = cols.columns.get_loc(col)
            if ic + 1 < len(cols.columns):
                neighbour = cols.columns[ic + 1]
                cols[col] = cols[col] + cols[neighbour]
                cols = cols.drop(columns=neighbour)
    return cols


def convert_parcel_id(series: pd.Series, pattern=None, conv_code: str = 'simple'):
    """Standardize raw parcel ids into a matching key via a conversion code.

    Implements only the operations seen in the auto-selected best solutions:
    ``simple`` (keep alphanumerics), ``pipe`` (keep alphanumerics, but
    replace each run of separator characters with ``|`` instead of deleting
    it), ``no_conv``, ``string_lengths``, ``split_groups``, ``drop_cols``,
    ``keep_length``, ``fill_zeros``, ``switch``, ``merge_after``,
    ``max_length``, ``join_char``, ``skip_empty``.

    Parameters
    ----------
    series : pd.Series
        Raw parcel identifiers (``parcel_id_assessor``).
    pattern : str, optional
        Pattern name (in ``parcel_id_patterns.csv``) or a raw ``^...$`` regex
        with capture groups. Ignored when ``string_lengths`` is in the code,
        or when ``conv_code`` is ``'simple'`` or ``'pipe'`` (both act on the
        whole string without a pattern).
    conv_code : str
        Conversion code, e.g. ``'string_lengths: 2 2 3 & skip_empty: 1'`` or
        the bare ``'simple'`` / ``'pipe'`` / ``'no_conv'``.

    Returns
    -------
    pd.Series
        Standardized matching key (``parcel_id_local``); NA where extraction
        failed or the result is empty.
    """
    s = series.astype('string').str.strip().str.upper()
    if conv_code == 'simple':
        out = s.str.replace(r'[^0-9A-Z]', '', regex=True)
        return out.where(out.ne(''), pd.NA)
    if conv_code == 'pipe':
        # Less lossy than 'simple': collapsing every separator run to '|'
        # (rather than deleting it) keeps segment boundaries and leading
        # zeros significant, so two raw ids that only match after 'simple'
        # strips their punctuation (e.g. '1-23' and '12-3', both 'simple'
        # -> '123') stay distinct here ('1|23' vs '12|3').
        out = s.str.replace(r'[^0-9A-Z]+', '|', regex=True).str.strip('|')
        return out.where(out.ne(''), pd.NA)

    p = _conv_dict(conv_code)
    if 'no_conv' in p:
        return s.where(s.ne(''), pd.NA)

    if 'string_lengths' in p:
        regex = ''.join(
            '([0-9A-Z ]' + (('{' + x + '}') if x not in ('+', '*', '?') else x) + ')'
            for x in p['string_lengths'].split(' ')
        )
    else:
        regex = _pattern_regex(pattern)

    cols = s.str.extract(regex, expand=True)
    i_null = cols.isnull().sum(axis=1).eq(len(cols.columns))
    cols.columns = [str(c) for c in cols.columns]

    if 'split_groups' in p:
        cols = _split_groups(cols, p['split_groups'])
    if 'drop_cols' in p:
        cols = cols.drop(
            columns=[c for c in p['drop_cols'].split(' ') if c in cols.columns]
        )

    keep = set(p['keep_length'].split(' ')) if 'keep_length' in p else set()
    strip_cols = [c for c in cols.columns if c not in keep]
    if strip_cols:
        cols[strip_cols] = cols[strip_cols].apply(
            lambda x: x.str.lstrip(' ').str.lstrip('0')
        )

    if 'fill_zeros' in p:
        fz = [c for c in p['fill_zeros'].split(' ') if c in cols.columns]
        if fz:
            cols[fz] = cols[fz].replace('', '0').fillna('0')
    if 'switch' in p:
        cols = _switch_groups(cols, p['switch'])

    cols = cols.fillna('')

    if 'merge_after' in p:
        cols = _merge_after(cols, p['merge_after'])
    if 'max_length' in p:
        for maxl in p['max_length'].split(' '):
            col, length = maxl.split('|')[0], int(maxl.split('|')[1])
            if col in cols.columns:
                cols[col] = cols[col].str.slice(0, length)

    join_char = p.get('join_char', '|')
    skip_empty = 'skip_empty' in p

    def _join(row):
        values = [v for v in row if v != ''] if skip_empty else list(row)
        return join_char.join(values)

    out = cols.apply(_join, axis=1).astype('string')
    out[i_null.to_numpy() | out.eq('')] = pd.NA
    return out


def dominant_parcel_id_pattern(series: pd.Series, min_match_ratio: float = 0.5) -> str:
    """Return the dominant extraction pattern of raw parcel ids.

    Offline helper used to (re)generate the per-admin-unit conversion table;
    not used on the ingest path. Picks the active pattern with the highest
    match ratio, breaking ties by lower complexity.
    """
    s = series.astype('string').str.strip().str.upper()
    s = s[s.notnull() & s.ne('')]
    if len(s) == 0:
        return 'Empty'
    patterns = _parcel_id_patterns()
    best_pattern, best_ratio, best_complexity = None, 0.0, 10**9
    for name, row in patterns.iterrows():
        ratio = s.str.match(row['regex']).mean()
        if ratio > best_ratio or (
            ratio == best_ratio and ratio > 0 and row['complexity'] < best_complexity
        ):
            best_pattern, best_ratio, best_complexity = name, ratio, row['complexity']
    if best_ratio > min_match_ratio:
        return best_pattern
    if s.str.len().ge(3).mean() > 0.5 and s.duplicated(False).mean() < 0.75:
        return 'Unrecognized'
    return 'Useless'


def simplest_parcel_id_pattern(
    series: pd.Series,
    min_match_ratio: float = 0.5,
    conv_code: str = 'skip_empty: 1',
    tolerance: float = 0.005,
) -> str:
    """Return the simplest pattern that matches well without adding duplicates.

    Offline helper used to (re)generate the per-admin-unit conversion table;
    not used on the ingest path. Complements :func:`dominant_parcel_id_pattern`
    (which optimizes for match ratio alone, breaking ties by lower
    complexity): this instead walks the active pattern library from lowest to
    highest ``complexity`` and returns the first pattern that both clears
    ``min_match_ratio`` and, once converted with ``conv_code``, does not
    collapse distinct raw ids beyond ``tolerance`` (see ``_adds_duplicates``,
    the same guard :func:`compute_parcel_id_local` applies at ingest time). A
    simpler pattern that already clears both bars generalizes better to id
    variants absent from the sample than the most specific pattern that
    happens to match it -- useful for a family like
    ``Sx-[S.]x(-Sx)(-Sx)(-Sx)`` where several ladder members (``Sx-Sx``,
    ``Sx-Sx(-Sx)``, ...) may fit a given county equally well.

    Falls back to :func:`dominant_parcel_id_pattern`'s result if no candidate
    clears both gates.

    Caveat: ranking trusts the bundled ``complexity`` column, which is
    hand-curated from the original ZTRAX port and occasionally under-scores
    old, overly permissive patterns (e.g. ones tolerating optional
    whitespace around every separator, which match more incidentally than a
    strict fixed-delimiter pattern despite a low stored complexity). Treat
    the return value as a starting point to eyeball, not a final answer --
    especially when it comes back looking fussier (more optional groups,
    looser separators) than a stricter same- or lower-complexity candidate
    you'd expect to win instead.
    """
    s = series.astype('string').str.strip().str.upper()
    s = s[s.notnull() & s.ne('')]
    if len(s) == 0:
        return 'Empty'

    patterns = _parcel_id_patterns().sort_values('complexity', kind='stable')
    for name, row in patterns.iterrows():
        if s.str.match(row['regex']).mean() < min_match_ratio:
            continue
        candidate = convert_parcel_id(s, name, conv_code)
        if not _adds_duplicates(s, candidate, tolerance):
            return name

    return dominant_parcel_id_pattern(series, min_match_ratio)


def _adds_duplicates(raw: pd.Series, candidate: pd.Series, tolerance: float) -> bool:
    """True if *candidate* collapses distinct raw ids beyond *tolerance*.

    Compares duplicate counts over rows where both are non-null; the converted
    key must not introduce duplicates over those already present in the raw
    ``parcel_id_assessor`` input.
    """
    mask = raw.notna() & candidate.notna()
    n = int(mask.sum())
    if n == 0:
        return False
    extra = int(candidate[mask].duplicated().sum() - raw[mask].duplicated().sum())
    return extra > tolerance * n


def _resolve_instruction(admin_unit_id, instruction, kind):
    """Resolve (pattern, conv_code, tolerance) for an admin unit and source kind.

    Order: explicit recipe ``instruction`` for the most-specific admin id, then
    the bundled default table, then ``(None, 'simple', None)``. ``tolerance``
    (the duplicate-guard override, see ``compute_parcel_id_local``) is only
    ever read from ``instruction`` entries -- the bundled default table has no
    such column, and always resolves it as ``None`` (caller's default).
    """
    if instruction:
        aid = str(admin_unit_id) if admin_unit_id is not None else None
        while aid:
            if aid in instruction:
                entry = instruction[aid]
                return (
                    entry.get('pattern'),
                    entry.get('conv', 'simple'),
                    entry.get('tolerance'),
                )
            aid = aid.rsplit('-', 1)[0] if '-' in aid else None
    links = _parcel_id_links()
    aid = str(admin_unit_id) if admin_unit_id is not None else None
    while aid:
        if aid in links.index:
            row = links.loc[aid]
            return row.get(f'pattern_{kind}'), row.get(f'conv_{kind}') or 'simple', None
        aid = aid.rsplit('-', 1)[0] if '-' in aid else None
    return None, 'simple', None


def compute_parcel_id_local(
    series: pd.Series,
    admin_unit_id=None,
    instruction: dict | None = None,
    kind: str = 'parcel',
    tolerance: float = 0.005,
) -> pd.Series:
    """Compute the standardized ``parcel_id_local`` key for a parcel id column.

    Resolves the admin-unit-specific conversion (recipe ``instruction`` then the
    bundled default table, by source ``kind`` ``'parcel'`` or ``'tax'``), and
    applies a hardened duplicate guard: if the conversion would collapse
    distinct ``parcel_id_assessor`` values beyond *tolerance*, it falls back to
    ``pipe`` and then to the raw (uppercased, alphanumeric) id, never adding
    new duplicates over the source. ``pipe`` (not ``simple``) is the fallback
    because it keeps separators between segments (``'1-23'`` -> ``'1|23'``)
    instead of deleting them (``'1-23'`` -> ``'123'``, now indistinguishable
    from ``'12-3'``) -- a fallback's whole job is to avoid manufacturing new
    collisions, so it must never be more lossy than necessary. An
    ``instruction`` entry may set its own
    ``tolerance`` (e.g. for a county where standardizing away an inconsistently
    zero-padded segment is expected to legitimately collapse repeat-sale
    filings of the same parcel beyond the default 0.5%) -- it overrides the
    *tolerance* parameter when present.
    """
    raw = series.astype('string').str.strip().str.upper()
    pattern, conv_code, resolved_tolerance = _resolve_instruction(
        admin_unit_id, instruction, kind
    )
    if resolved_tolerance is not None:
        tolerance = float(resolved_tolerance)

    candidate = convert_parcel_id(raw, pattern, conv_code)
    if not _adds_duplicates(raw, candidate, tolerance):
        return candidate

    warnings.warn(
        f'parcel_id_local conversion for admin {admin_unit_id} (kind={kind}, '
        f'conv={conv_code!r}) added duplicates; falling back to pipe.',
        stacklevel=2,
    )
    candidate = convert_parcel_id(raw, None, 'pipe')
    if not _adds_duplicates(raw, candidate, tolerance):
        return candidate

    return raw.where(raw.ne(''), pd.NA)
