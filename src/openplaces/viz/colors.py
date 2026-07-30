"""Default category-color mappings for openplaces standard columns.

Labels are strings as stored in parquet files after ingestion (i.e.
post-labels-CSV remapping, not raw source codes). The mapping itself lives
in ``category_colors.csv`` (same directory) rather than as Python dicts, so
it can be shared with non-Python consumers (e.g. a future QGIS map-styling
export) instead of being locked into this module.
"""

import hashlib
from functools import cache
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

_CATEGORY_COLORS_CSV = Path(__file__).parent / 'category_colors.csv'


@cache
def load_category_colors_table() -> pd.DataFrame:
    """Return ``category_colors.csv`` as a DataFrame.

    Columns: ``column`` (the categorical column a row applies to, e.g.
    ``'occupancy_type'``), ``label``, ``color`` (hex ``#rrggbb``), and
    ``parent`` (optional -- for a subgroup label, the `label` within the
    same `column` it's a variant of, e.g. ``'Low-Rise Multi-Family'``'s
    parent is ``'Multi-Family'``; blank for top-level/unspecified labels).
    """
    return pd.read_csv(_CATEGORY_COLORS_CSV, keep_default_na=False)


def _build_category_colors() -> dict[str, dict[str, str]]:
    table = load_category_colors_table()
    return {
        column: dict(zip(rows['label'], rows['color'], strict=True))
        for column, rows in table.groupby('column')
    }


def _build_category_parents() -> dict[str, dict[str, str]]:
    table = load_category_colors_table()
    with_parent = table[table['parent'] != '']
    return {
        column: dict(zip(rows['label'], rows['parent'], strict=True))
        for column, rows in with_parent.groupby('column')
    }


# building.group -- standardized building use groups, as defined in the
# purpose-subgroup-remap CSV. Covers NSI HAZUS occupancy classes; additional
# categories from other datasets (FEMA, OBM, etc.) may be added to
# category_colors.csv with consistent color shading.
#
# building.purpose_group -- NSI: RES / COM / IND / PUB, post-label remapping.
#
# footprint.occupancy_type -- curate-stage canonical classes, colored
# consistently with _GROUP's matching entries where the concepts overlap
# (e.g. 'Single-Family'/'Single Family'). Height-banded Multi-Family
# subclasses ('Low-Rise'/'Mid-Rise'/'High-Rise') are colored as related-but-
# distinguishable shades of the unspecified 'Multi-Family' color, recorded
# via each row's `parent` in category_colors.csv, rather than aliased to an
# identical color.
#
# building.source -- NSI, post-label remapping from source-labels CSV. Green
# for highest-quality (Parcel), grading to grey for legacy/fallback, matching
# the ordered categorical priority set in the source-labels CSV.
_CATEGORY_COLORS_BY_COLUMN = _build_category_colors()
_CATEGORY_PARENTS = _build_category_parents()
_GROUP = _CATEGORY_COLORS_BY_COLUMN['group']
_PURPOSE_GROUP = _CATEGORY_COLORS_BY_COLUMN['purpose_group']
_OCCUPANCY_TYPE = _CATEGORY_COLORS_BY_COLUMN['occupancy_type']
_SOURCE = _CATEGORY_COLORS_BY_COLUMN['geometry_source']

# Category values with no entry in category_colors.csv (e.g. a footprint's
# occupancy_type filled in from a raw, non-residential evidence label that
# CHEER's occupancy class-map doesn't recognize) get a color deterministically
# auto-assigned by resolve_category_colors below, rather than all sharing one
# flat, ambiguous fallback. Missing/NaN values are the one exception: they
# always get the fixed RESERVED_NEUTRAL_COLOR below, not an auto-assigned one.
MISSING_LABEL = '(missing)'
RESERVED_NEUTRAL_COLOR = '#999999'
_AUTO_ASSIGN_BINS = 64

# Continuous "value"/"price" colormaps: many stops, full saturation
# throughout (no washed-out midpoint the way a typical diverging colormap
# fades to white/pale in the middle), for value/price choropleths and
# terrains. Each entry is a list of (position, hex) control points spanning
# [0, 1], suitable for `matplotlib.colors.LinearSegmentedColormap.from_list`
# (or pass it straight to `get_diverging_colormap`).
#
# Two entities rendered together (e.g. buildings + parcels, `viz.terrain`)
# read as visually separable when given colormaps from *different* families
# below rather than the same one twice -- e.g. an 'ibm'-family (blue/purple/
# magenta/orange) colormap for one and a green-anchored 'price' family
# colormap for the other. Using the same colormap for both remains a valid,
# supported choice (the original single-palette look); nothing here requires
# pairing two different ones.
DIVERGING_COLORMAPS = {
    # -- green-anchored "price" family: green (cheap) -> yellow -> orange ->
    # red -> darker (expensive). All variants below share this backbone with
    # different endpoints/stop placement.
    # Dark green -> green -> yellow -> orange -> red -> black.
    'price': [
        (0.0, '#003a0a'),
        (0.125, '#005100'),
        (0.25, '#009c00'),
        (0.325, '#96c400'),
        (0.5, '#fffb00'),
        (0.625, '#ff950b'),
        (0.75, '#ee0000'),
        (0.85, '#c40003'),
        (1.0, '#490001'),  #'black'),
    ],
    # -- blue-anchored "price_blue" family: like "price" but starts at dark blue
    # -> blue instead of green.
    'price_blue': [
        (0.0, '#000a3a'),
        (0.1, '#001f7f'),
        (0.2, '#0047d4'),
        (0.29, '#708ffa'),
        (0.5, '#fffb00'),
        (0.625, '#ff950b'),
        (0.75, '#ee0000'),
        (0.875, '#c40003'),
        (1.0, 'black'),
    ],
    # 'price', bookended by black on both ends instead of starting green --
    # reads as more clearly "closed"/symmetric at a glance.
    'price300': [
        (0.0, 'black'),
        (0.05, '#033203'),
        (0.18, '#005100'),
        (0.35, '#009c00'),
        (0.40, '#96c400'),
        (0.54, '#fffb00'),
        (0.68, '#ff950b'),
        (0.8, '#ee0000'),
        (0.9, '#c40003'),
        (1.0, 'black'),
    ],
    # Near-black -> green -> yellow -> red -> dark red -> pink -> white: a
    # wider, more saturated spread than 'price', ending light rather than dark.
    'price3d': [
        (0.0, '#2b1800'),
        (0.17, '#0b6107'),
        (0.27, '#1ec90d'),
        (0.45, '#fff02a'),
        (0.55, '#fbf485'),
        (0.65, '#ff9c00'),
        (0.73, '#ff0000'),
        (0.85, '#6f0000'),
        (0.95, '#ff62f2'),
        (1.0, 'white'),
    ],
    # -- blue-anchored family: blue/cyan (cheap) -> green -> yellow -> red ->
    # magenta (expensive). Visually distinct from the green-anchored family
    # above at a glance -- pairs well with one of those for two-layer scenes.
    'price3db': [
        (0.0, '#1453c8'),
        (0.1, '#0adfff'),
        (0.25, '#1de418'),
        (0.40, '#fffb00'),
        (0.75, '#ff1c1c'),
        (0.85, '#e52ee3'),
        (1.0, 'white'),
    ],
    # 'price3db' with a white (rather than blue) low end.
    'price3dw': [
        (0.0, 'white'),
        (0.1, '#0adfff'),
        (0.25, '#1de418'),
        (0.40, '#fffb00'),
        (0.75, '#ff1c1c'),
        (0.85, '#e52ee3'),
        (1.0, 'white'),
    ],
    # Blue -> yellow -> red, staying saturated (no pale middle).
    'BuYlRd': [
        (0.0, '#2f28b5'),
        (0.1, '#648fff'),
        (0.5, '#fffb00'),
        (0.625, '#ff950b'),
        (0.75, '#ee0000'),
        (0.875, '#c40003'),
        (1.0, '#771010'),
    ],
    'land_value': [
        (0.00, '#203826'),
        (0.18, '#4c6b2f'),
        (0.38, '#8fa34a'),
        (0.58, '#d8c96a'),
        (0.74, '#d88a3d'),
        (0.88, '#b84b2d'),
        (1.00, '#7d1818'),
    ],
    'building_value': [
        (0.00, '#3a3a3a'),
        (0.12, '#5c6f89'),
        (0.28, '#6d8b8b'),
        (0.46, '#8e9a63'),
        (0.64, '#b89a50'),
        (0.82, '#b56f52'),
        (1.00, '#8d4c72'),
    ],
    # -- simple 3-stop diverging colormaps --
    # Magenta -> yellow -> green.
    'MgYlGn': [(0.0, '#da00e8'), (0.5, '#fff000'), (1.0, '#0db90d')],
    # Green -> yellow -> warm red/salmon.
    'GnYlWm': [(0.0, '#0db90d'), (0.5, '#fff000'), (1.0, '#ec694a')],
    # -- IBM Design Library colorblind-safe palette, as a continuous ramp --
    # Blue -> purple -> magenta -> orange -> gold.
    'ibm': [
        (0.0, '#648fff'),
        (0.25, '#785ef0'),
        (0.5, '#dc267f'),
        (0.75, '#fe6100'),
        (1.0, '#ffb000'),
    ],
    # Blue -> gold -> magenta (diverging 3-stop subset of 'ibm').
    'ibm_div': [(0.0, '#648fff'), (0.5, '#ffb000'), (1.0, '#dc267f')],
    # nipy_spectral ending in purple instead of light grey, and starting in dark blue.
    'nipy_spectral_purple': [
        (0.0, '#000055'),
        (0.11, '#0000dd'),
        (0.22, '#0098dd'),
        (0.33, '#00aa88'),
        (0.44, '#00bc00'),
        (0.55, '#00ff00'),
        (0.66, '#efed00'),
        (0.77, '#ff9900'),
        (0.88, '#dc0000'),
        (1.0, '#870098'),
    ],
}


def get_diverging_colormap(name: str) -> mpl.colors.Colormap:
    """Build a matplotlib colormap from `DIVERGING_COLORMAPS[name]`, falling back
    to standard matplotlib colormaps if name is not in the custom registry.

    Parameters
    ----------
    name : str
        Key into `DIVERGING_COLORMAPS` or a standard matplotlib colormap name.

    Returns
    -------
    matplotlib.colors.Colormap
    """
    if name in DIVERGING_COLORMAPS:
        return mpl.colors.LinearSegmentedColormap.from_list(
            name, DIVERGING_COLORMAPS[name]
        )
    return mpl.colormaps[name]


# Public registry  {column_name: {label: color}}
# 'use_group'/'purpose_group' precede 'group' so the substring match in
# match_palette prefers the occupancy palette for use_group*/purpose_group*
# columns. use_group (parcel "what it is used for") shares the occupancy palette;
# match_palette falls back when its land-use labels do not match.
CATEGORY_COLORS = {
    'use_group': _PURPOSE_GROUP,
    'purpose_group': _PURPOSE_GROUP,
    'group': _GROUP,
    'occupancy_type': _OCCUPANCY_TYPE,
    'geometry_source': _SOURCE,
}


def match_palette(values, col_name=None, weights=None, threshold=0.5):
    """Return the best-matching color palette for a set of category values.

    Tries matches in order:

    1. Exact column-name key lookup.
    2. Any palette key is a substring of the column name.
    3. Frequency-weighted coverage: fraction of total weight (row count)
       whose category label has a defined color, using ``weights`` if
       provided, otherwise falling back to unweighted unique-label coverage.

    Parameters
    ----------
    values : iterable
        The category labels present in the plot (e.g. ``widths.columns``).
    col_name : str, optional
        Column name hint; checked first for an exact or substring key match.
    weights : array-like, optional
        Total weight (e.g. row count) for each value in ``values``, in the
        same order. When provided, coverage is computed as the fraction of
        total weight that falls in palette-covered categories, making the
        match robust to many rare/uncovered categories.
    threshold : float
        Minimum coverage required for a value-based match.

    Returns
    -------
    dict or None
        Matching ``{label: color}`` palette, or ``None`` if no good match.
    """
    str_values = [str(v) for v in values]

    def _coverage(palette):
        palette_keys = set(palette)
        if weights is not None:
            total = sum(weights)
            return (
                sum(w for v, w in zip(str_values, weights) if v in palette_keys) / total
            )
        return sum(1 for v in str_values if v in palette_keys) / len(str_values)

    # 1. Exact column name match
    if col_name in CATEGORY_COLORS:
        return CATEGORY_COLORS[col_name]
    # 2. Any palette key is a substring of the column name — only if values match
    if col_name:
        for key, palette in CATEGORY_COLORS.items():
            if key in col_name and _coverage(palette) >= threshold:
                return palette
    # 3. Frequency-weighted (or unweighted) value coverage
    best, best_score = None, 0.0
    for palette in CATEGORY_COLORS.values():
        score = _coverage(palette)
        if score > best_score:
            best_score, best = score, palette
    if best_score >= threshold:
        return best
    return None


@cache
def _auto_assign_bin_colors() -> list[str]:
    """`_AUTO_ASSIGN_BINS` hex colors, evenly spaced across `'full_hue_vivid'`.

    Sampled at evenly spaced positions (not adjacent stops of the
    underlying 256-line source file, which form a smooth ramp and would
    look near-identical) so that two different auto-assigned bins read as
    visually distinct. Cached since it's rebuilt from a 256-line file
    otherwise.
    """
    cmap = get_diverging_colormap('full_hue_vivid')
    return [
        mpl.colors.to_hex(cmap(i / _AUTO_ASSIGN_BINS)) for i in range(_AUTO_ASSIGN_BINS)
    ]


def _auto_assign_color(label: str) -> str:
    """Deterministic per-label color for a label with no curated entry.

    Uses a stable hash (`hashlib.md5`, not the builtin `hash()`, which is
    salted per-process) so the same label always gets the same color across
    calls, admin-unit scopes, and rendering tracks (Lonboard vs Datashader).
    """
    digest = hashlib.md5(label.encode('utf-8')).hexdigest()
    bin_index = int(digest[:8], 16) % _AUTO_ASSIGN_BINS
    return _auto_assign_bin_colors()[bin_index]


def resolve_category_colors(values, col_name: str | None = None) -> dict[str, str]:
    """Resolve one color per distinct label actually present in `values`.

    Curated colors (via `match_palette`) are used where available; every
    other distinct label -- including ones with no palette entry at all,
    e.g. a footprint's `occupancy_type` filled in from a raw evidence label
    CHEER's occupancy class-map doesn't recognize -- gets a deterministic
    per-label color from `_auto_assign_color` instead of a single shared
    fallback. This guarantees the same label always maps to the same color
    regardless of admin-unit scope or which rendering track (Lonboard vs
    Datashader) is asking, which a flat "unmatched" fallback color cannot.

    Not a collision-free guarantee for arbitrarily many uncurated labels,
    but legends only ever display a handful of labels at once
    (`openplaces.viz.legend.MAX_CATEGORICAL_ENTRIES`), so it only needs to
    keep whatever actually competes for those slots visually distinct in
    practice.

    Parameters
    ----------
    values : pandas.Series
        Category values actually present in the plotted/rasterized data.
        `NaN` is treated as the literal label `MISSING_LABEL`.
    col_name : str, optional
        Column name hint, forwarded to `match_palette`.

    Returns
    -------
    dict
        ``{label: hex_color}``, covering every distinct label in `values`
        (as `str`, with `NaN` folded to `MISSING_LABEL`). Never `None`.
    """
    # dropna=True here (rather than folding NaN into the coverage count)
    # matches match_palette's pre-existing calling convention across every
    # other call site (interactive.py/raster.py/tabulation.py all compute
    # weights from a dropna=True value_counts()) -- NaN's color is decided
    # separately below, not by the curated-palette match itself.
    counts = values.value_counts(dropna=True)
    labels = [str(label) for label in counts.index]
    curated = match_palette(labels, col_name=col_name, weights=counts.to_numpy()) or {}

    resolved = {
        label: curated.get(label, _auto_assign_color(label)) for label in labels
    }
    if values.isna().any():
        resolved[MISSING_LABEL] = RESERVED_NEUTRAL_COLOR
    return resolved


def to_rgba_array(
    values, palette: dict, default: str = '#999999', alpha: int = 200
) -> np.ndarray:
    """Map category values to an RGBA uint8 array via a ``{label: hex}`` palette.

    Values absent from `palette` (including missing/NaN) receive `default`.
    Written for GPU renderers (e.g. Lonboard/deck.gl) that expect a fixed-width
    uint8 color per row rather than deferring color resolution themselves —
    unlike `match_palette`'s output, which some CPU renderers (e.g. Datashader's
    `color_key`) can consume directly as a label:hex dict.

    Parameters
    ----------
    values : iterable
        Category labels, one per row.
    palette : dict
        ``{label: '#rrggbb'}`` color mapping, e.g. from `match_palette`.
    default : str
        Hex color assigned to values not found in `palette`.
    alpha : int
        Alpha channel (0-255) applied to every row.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(len(values), 4)``, dtype ``uint8``.
    """
    hex_colors = pd.Series(values, dtype='object').map(palette).fillna(default)
    rgb = np.array(
        [[int(h[i : i + 2], 16) for i in (1, 3, 5)] for h in hex_colors],
        dtype=np.uint8,
    )
    return np.column_stack([rgb, np.full(len(rgb), alpha, dtype=np.uint8)])


def _hex_to_rgb(hex_color: str) -> np.ndarray:
    return np.array([int(hex_color[i : i + 2], 16) for i in (1, 3, 5)])


def continuous_to_rgba(
    values,
    low: str = '#e5e7db',
    high: str = '#5a2828',
    cmap: str | mpl.colors.Colormap | None = None,
    alpha: int = 200,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """Map continuous values to an RGBA uint8 array via a min-max color ramp.

    Scales `values` linearly over their finite range into [0, 1], then colors
    them either with a simple two-endpoint ramp (`low` -> `high`) or, if
    `cmap` is given, a full matplotlib colormap (e.g. one of
    `DIVERGING_COLORMAPS`) — `low`/`high` are ignored when `cmap` is given.
    The scaling itself is always linear in whatever units `values` are
    already in — apply a transform (e.g. `np.log1p`) to `values` yourself
    beforehand for a log-color scale; doing the transform before calling
    this function (rather than adding a `log` flag here) keeps the
    quantization as fine as the ramp's own continuous resolution in the
    transformed space.

    Parameters
    ----------
    values : iterable
        Continuous values, one per row. NaN/non-finite entries are colored
        at the low end of the ramp.
    low, high : str
        Hex colors ('#rrggbb') for the minimum and maximum ends of the ramp.
        Ignored when `cmap` is given.
    cmap : str or matplotlib.colors.Colormap, optional
        A colormap name in `DIVERGING_COLORMAPS`, or a `Colormap` instance
        (e.g. from `get_diverging_colormap` or matplotlib's own registry).
    alpha : int
        Alpha channel (0-255) applied to every row.
    vmin, vmax : float, optional
        Explicit bounds for the linear scaling step, in the same
        (already-transformed, if the caller applied one) units as `values`.
        Default (`None`) is each call's own observed
        `values.min()`/`values.max()` — pass both explicitly to pin the
        color scale (e.g. across repeated calls, or to match another call's
        range) instead of auto-ranging independently each time.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(len(values), 4)``, dtype ``uint8``.
    """
    colormap = None
    if cmap is not None:
        colormap = get_diverging_colormap(cmap) if isinstance(cmap, str) else cmap

    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        if colormap is not None:
            fallback = np.array(colormap(0.0)[:3]) * 255
        else:
            fallback = _hex_to_rgb(low)
        return np.tile(np.append(fallback, alpha), (len(values), 1)).astype(np.uint8)

    scale_min = finite.min() if vmin is None else vmin
    scale_max = finite.max() if vmax is None else vmax
    span = scale_max - scale_min or 1.0
    filled = np.where(np.isfinite(values), values, scale_min)
    scaled = np.clip((filled - scale_min) / span, 0, 1)

    if colormap is not None:
        rgb = (np.asarray(colormap(scaled))[:, :3] * 255).astype(np.uint8)
    else:
        low_rgb = _hex_to_rgb(low)
        high_rgb = _hex_to_rgb(high)
        rgb = (low_rgb + scaled[:, None] * (high_rgb - low_rgb)).astype(np.uint8)
    return np.column_stack([rgb, np.full(len(values), alpha, dtype=np.uint8)])


def adjust_brightness(rgba: np.ndarray, factor: float) -> np.ndarray:
    """Scale an RGBA uint8 array's brightness (HSV value channel), alpha kept.

    Uses HSV rather than a naive `rgb * factor` so hue/saturation stay
    unchanged and results don't clip asymmetrically across channels — a
    darkened or brightened color is still recognizably "the same color at a
    different shade", useful both for distinguishing two otherwise-similar
    layers (e.g. one rendered brighter than the other) and for deriving a
    per-row outline color as a darker shade of that row's own fill color.

    Parameters
    ----------
    rgba : numpy.ndarray
        Array of shape ``(n, 4)``, dtype ``uint8`` (e.g. from
        `continuous_to_rgba`/`to_rgba_array`).
    factor : float
        Multiplier on the HSV value channel. `1.0` leaves colors unchanged;
        `<1.0` darkens; `>1.0` brightens (clipped at full brightness).

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n, 4)``, dtype ``uint8``.
    """
    rgb_float = rgba[:, :3].astype(float) / 255.0
    hsv = mpl.colors.rgb_to_hsv(rgb_float)
    hsv[:, 2] = np.clip(hsv[:, 2] * factor, 0, 1)
    rgb = np.round(mpl.colors.hsv_to_rgb(hsv) * 255).astype(np.uint8)
    return np.column_stack([rgb, rgba[:, 3]])


# Load external colormaps from package files and register third-party colormaps
def _load_and_register_external_colormaps():
    import os

    import matplotlib as mpl

    # Helper to load hex codes
    def _load_hex_colormap(filename: str):
        path = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(path):
            try:
                with open(path, encoding='utf-8') as f:
                    colors = []
                    for line in f:
                        stripped = line.strip()
                        if (
                            stripped
                            and not stripped.startswith('# ')
                            and stripped.startswith('#')
                        ):
                            colors.append(stripped)
                if colors:
                    return colors
            except Exception:
                pass
        return None

    def _colors_to_stops(colors):
        n = len(colors)
        return [(i / (n - 1), color) for i, color in enumerate(colors)]

    # 1. full_hue_vivid
    vivid = _load_hex_colormap('full_hue_vivid_hex.txt')
    if vivid:
        stops = _colors_to_stops(vivid)
        DIVERGING_COLORMAPS['full_hue_vivid'] = stops
        DIVERGING_COLORMAPS['full_hue_vivid_hex'] = stops

    # 2. full_hue_natural
    natural = _load_hex_colormap('full_hue_natural_hex.txt')
    if natural:
        stops = _colors_to_stops(natural)
        DIVERGING_COLORMAPS['full_hue_natural'] = stops
        DIVERGING_COLORMAPS['full_hue_natural_hex'] = stops

    # 3. CMasher pride & Colorcet R1 / R4
    try:
        import cmasher as cmr  # noqa: F401

        if 'cmr.pride' in mpl.colormaps and 'pride' not in mpl.colormaps:
            mpl.colormaps.register(mpl.colormaps['cmr.pride'], name='pride')
    except ImportError:
        pass

    try:
        import colorcet as cc

        if 'CET_R1' not in mpl.colormaps:
            mpl.colormaps.register(cc.cm.CET_R1, name='CET_R1')
        if 'CET_R4' not in mpl.colormaps:
            mpl.colormaps.register(cc.cm.CET_R4, name='CET_R4')
    except ImportError:
        pass


_load_and_register_external_colormaps()
