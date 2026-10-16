"""Static web page of the 3D land and building value scene.

Writes the scene of `show_value_terrain_layer` (parcels extruded by land
value per area, buildings stacked on them by structure value per area)
as static files any plain web host can serve: a page, its script, a
manifest, and gzip-compressed binary files per admin unit. The host runs
no code and holds no database.

Three properties of the page are maintainer's decisions, pinned by
tests:

- It draws only the scene's own layers and the unit boundaries, on a
  plain background: no basemap or tile layer, like every map in
  `openplaces.viz`.
- It has no search, no filter and no control choosing which units to
  show. Every unit of the scene loads at start; a hovered feature shows
  the fields shipped with its own geometry, and nothing is looked up.
- It reads delivery bundles (`export_delivery`), never curated per-unit
  files, so whatever a bundle withholds never reaches the page, and the
  tooltip fields are an explicit allow-list that refuses personal
  columns.

The binary layout, one file per (layer, unit), little-endian:

- `OPVS`, then a uint32 header length and a UTF-8 JSON header naming
  each array (`name`, `dtype`, `offset`, `count`), padded to 8 bytes;
- `xy` (float32): vertex longitude and latitude as offsets from the
  header's `origin`. Offsets keep float32 precise to millimeters within
  a town, and deck.gl draws them directly (`LNGLAT_OFFSETS`), which
  halves the size of float64 coordinates;
- `z` (float32, only in a layer draped on terrain): each vertex's
  height in meters, from which the page extrudes instead of `base`;
- `ring_offsets`, `part_ring_offsets` (uint32): vertex index of each
  ring's start, ring index of each polygon part's start (both with a
  closing entry), and `part_feature` (uint32), the feature of each part,
  so a multipolygon's parts share one feature's values;
- `triangles` (uint32): the parts' top faces as vertex index triples
  into `xy`, computed at build time so the page does no triangulation
  (rings are written outer counterclockwise, holes clockwise);
- per feature: `base` and `height` (float32, meters) and `value`
  (float32), the value per area unit the colors encode, NaN where
  there is none. The page colors from it with the palette the visitor
  picks (`PALETTES`), so switching palettes needs no new files.

The tooltip values go into a second gzip file per (layer, unit), as
columnar JSON. A self-describing JSON-plus-arrays file was chosen over
GeoParquet or GeoArrow because the page then needs one library
(deck.gl's standalone bundle) instead of four that must agree on
versions (parquet-wasm, Apache Arrow, the GeoArrow layers, deck.gl).
"""

import gzip
import json
import shutil
import struct
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import numpy as np
import pandas as pd
import shapely

from openplaces.core.attribute_registry import is_personal_attribute
from openplaces.core.provenance import is_imputed
from openplaces.io.delivery import PersonalColumnError, delivery_paths
from openplaces.io.readers import get_admin
from openplaces.viz.colors import get_diverging_colormap

MAGIC = b'OPVS'
# 2: adds `triangles`, which the page now requires.
FORMAT_VERSION = 2
SOURCE_SUFFIX = '_source'
STATIC_DIR = files('openplaces.viz.web_scene') / 'static'
# deck.gl's standalone bundle, pinned; the page loads nothing else.
DECK_GL_URL = 'https://cdn.jsdelivr.net/npm/deck.gl@9.1.14/dist.min.js'
TOOLTIP_KINDS = ('text', 'money', 'money_per_area', 'year', 'count')
# The page's only heading (maintainer, 2026-10-15).
DEFAULT_TITLE = 'How land and building value is distributed'


@dataclass(frozen=True)
class TooltipField:
    """One field shown when a feature is hovered.

    Parameters
    ----------
    column : str
        Column of the rendered table. A personal column is refused.
    label : str
        Label shown on the page.
    kind : str
        How the page formats the value: one of `TOOLTIP_KINDS`.
    imputed_from : str, optional
        Provenance column whose `imputed` marker flags the value as
        filled by openplaces; the page then says so beside it.
    """

    column: str
    label: str
    kind: str = 'text'
    imputed_from: str | None = None

    def __post_init__(self):
        if self.kind not in TOOLTIP_KINDS:
            raise ValueError(
                f'TooltipField kind must be one of {TOOLTIP_KINDS}, got {self.kind!r}.'
            )


@dataclass(frozen=True)
class SceneLayer:
    """One extruded layer of the scene.

    Parameters
    ----------
    name : str
        Short id, used in file names (`parcels`, `buildings`).
    label : str
        Name shown in the legend.
    recipe : str
        Curation recipe whose delivery bundle supplies the entities.
    value_column : str
        Value divided by area for height and color.
    area_unit : str
        Display unit of the color scale (`ac`, `sqft`, `ha`, `m2`).
    vmin, vmax : float
        Color-scale bounds, in dollars per `area_unit`.
    cmap : str
        Colormap name (`viz.colors.get_diverging_colormap`; cmasher's
        `cmr.*` names included).
    cmap_range : tuple of float
        The part of the colormap used, as fractions (0-1). Cuts an end
        that would vanish into the page's dark background or wash out
        to white.
    tooltip : tuple of TooltipField
        Fields shown on hover; nothing else of the bundle is written.
    stack_on : str, optional
        Name of an earlier layer this one stands on.
    render : dict
        Further keyword arguments for `show_value_terrain_layer`.
    simplify_m : float, optional
        Simplify each polygon by this many meters before writing it
        (topology-preserving, polygon by polygon). For layers that are
        not a coverage, such as building outlines; a coverage (parcels)
        is simplified inside `show_value_terrain_layer` instead, so
        neighbors keep their shared edges.
    """

    name: str
    label: str
    recipe: str
    value_column: str
    area_unit: str
    vmin: float
    vmax: float
    cmap: str = 'turbo'
    cmap_range: tuple[float, float] = (0.0, 1.0)
    tooltip: tuple[TooltipField, ...] = ()
    stack_on: str | None = None
    render: dict = field(default_factory=dict)
    simplify_m: float | None = None


# The notebook's scene (notebooks/09_show/03_show_land_and_building_
# values_3d), with buildings without a value drawn flat at their
# parcel's top rather than as floating outlines.
#
# Colors (maintainer's review, 2026-10-15): the page's job is to show
# where the money is, on a dark background. Land is a cool ground that
# recedes (indigo to teal to seafoam) and buildings are heat that ends
# in gold, so the two never share a hue and the costliest buildings
# are the brightest things on the page. Both ramps rise steadily in
# lightness, so their order also reads without color vision. The
# notebook's rainbows (turbo, nipy_spectral) shared every hue between
# land and buildings and ended in white. Land bounds: measured
# 2026-10-15 on boston-inner-core, land value per acre had median
# $3.0M and p99 $51M, so $50K-$50M puts the median mid-ramp; the
# notebook's $250-$10M left nine parcels in ten in its top fifth.
ZERO_VALUE_RGBA = (34, 40, 54, 255)
VALUE_SCENE_LAYERS = (
    SceneLayer(
        name='parcels',
        label='Land value',
        recipe='US_parcel-openplaces-2026',
        value_column='land_value_imputed',
        area_unit='ac',
        vmin=50_000,
        vmax=50_000_000,
        cmap='cmr.ocean',
        cmap_range=(0.05, 0.85),
        tooltip=(
            TooltipField('address', 'Address'),
            TooltipField('use_group', 'Land use'),
            TooltipField(
                'land_value_imputed',
                'Land value',
                'money',
                imputed_from='land_value_imputed_source',
            ),
            TooltipField(
                'land_value_imputed_per_ac',
                'Land value per acre',
                'money_per_area',
                imputed_from='land_value_imputed_source',
            ),
        ),
        render={
            'drop_corridors': True,
            'height_clip_percentile': 99.9999,
            'simplify_tolerance': 0.5,
        },
    ),
    SceneLayer(
        name='buildings',
        label='Building value',
        recipe='US_footprint-openplaces-2026',
        value_column='structure_value',
        area_unit='sqft',
        # Measured 2026-10-15 on boston-inner-core: structure value per
        # sq ft had median $453, p90 $1,251, p99 $3,407; $50-$5,000
        # puts the median near mid-ramp and leaves the brightest gold
        # to the costliest buildings.
        vmin=50,
        vmax=5_000,
        cmap='cmr.amber',
        cmap_range=(0.2, 0.95),
        stack_on='parcels',
        tooltip=(
            TooltipField('address', 'Address'),
            TooltipField('occupancy_type', 'Occupancy'),
            TooltipField('year_built', 'Year built', 'year'),
            TooltipField(
                'structure_value',
                'Building value',
                'money',
                imputed_from='structure_value_source',
            ),
            TooltipField(
                'structure_value_per_sqft',
                'Building value per sq ft',
                'money_per_area',
                imputed_from='structure_value_source',
            ),
        ),
        # A building whose height was capped keeps its color: the
        # notebook's translucent magenta marker read as a pink wash
        # over exactly the tallest towers.
        render={
            'height_clip_percentile': 99.999,
            'alpha': 0.9,
            'clipped_fill_rgba': None,
        },
        # Outlines from building structure layers trace every jog of a
        # wall. Measured 2026-10-15 on boston-inner-core: 2.45M vertices
        # for 151k building parts (16.2 per part) against 7.4 per
        # parcel, which a phone could not hold.
        simplify_m=0.5,
    ),
)


def refuse_personal_fields(layers) -> None:
    """Raise if any layer's tooltip names a personal column.

    The same rule as a delivery's `share` block: a personal column
    never leaves the installation, and a scene that asks for one is
    refused rather than written without it.

    Parameters
    ----------
    layers : sequence of SceneLayer

    Raises
    ------
    PersonalColumnError
        If a tooltip field, or the provenance column it reads, is
        personal.
    """
    for layer in layers:
        for tip in layer.tooltip:
            for column in (tip.column, tip.imputed_from):
                if column is None:
                    continue
                attribute = column.removesuffix(SOURCE_SUFFIX)
                if is_personal_attribute(attribute):
                    raise PersonalColumnError(
                        f'Scene layer {layer.name!r} would show the personal '
                        f'column {column!r}. A web scene never carries one.'
                    )


def read_bundle(recipe, region, admin_ids=None) -> gpd.GeoDataFrame:
    """Load a delivery bundle's attributes joined to its polygons.

    Parameters
    ----------
    recipe : str or dict
        Curation recipe that declares `region` under `share: delivery:`.
    region : str
        Region id of the bundle.
    admin_ids : sequence of str, optional
        Keep only these units (`admin3_id`).

    Returns
    -------
    geopandas.GeoDataFrame
        Indexed like the bundle, with its canonical columns and
        `_source` sidecars.
    """
    paths = delivery_paths(recipe, region=region)
    for role in ('canonical', 'geo'):
        if not paths[role].exists():
            raise FileNotFoundError(
                f'The {role} file of the {region!r} bundle of {recipe!r} does '
                f'not exist ({paths[role]}). Run export_delivery first.'
            )
    attributes = pd.read_parquet(paths['canonical'])
    if admin_ids is not None:
        attributes = attributes[attributes['admin3_id'].isin(list(admin_ids))]
    geometry = gpd.read_parquet(paths['geo'], columns=['geometry'])
    gdf = gpd.GeoDataFrame(
        attributes.join(geometry, how='inner'), geometry='geometry', crs=geometry.crs
    )
    if len(gdf) < len(attributes):
        raise ValueError(
            f'{len(attributes) - len(gdf):,} rows of the {region!r} bundle of '
            f'{recipe!r} have no polygon in its _geo file; the bundle is '
            'inconsistent and should be re-exported.'
        )
    return gdf


def _region_members(recipe, region) -> list[str]:
    """Units a recipe's delivery region pools."""
    from openplaces.io.delivery import delivery_members

    return [str(a) for a in delivery_members(recipe, region=region)]


def bundle_notice(recipe, region) -> str:
    """Return the terms notice shipped with a bundle, for the page."""
    return delivery_paths(recipe, region=region)['terms'].read_text(encoding='utf-8')


def _ragged_parts(geometry: np.ndarray):
    """Return coordinates and ring/part offsets of (multi)polygons.

    Polygons and multipolygons are written alike: a polygon is a
    multipolygon of one part, so the page needs one code path.
    """
    include_z = bool(shapely.has_z(geometry).any())
    geom_type, coords, offsets = shapely.to_ragged_array(
        geometry if include_z else shapely.force_2d(geometry), include_z=include_z
    )
    if geom_type == shapely.GeometryType.POLYGON:
        ring_offsets, part_ring_offsets = offsets
        part_feature = np.arange(len(geometry), dtype=np.uint32)
    elif geom_type == shapely.GeometryType.MULTIPOLYGON:
        ring_offsets, part_ring_offsets, feature_part_offsets = offsets
        part_feature = np.repeat(
            np.arange(len(geometry), dtype=np.uint32), np.diff(feature_part_offsets)
        )
    else:
        raise ValueError(f'Expected polygons, got {geom_type!r}.')
    return coords, ring_offsets, part_ring_offsets, part_feature


def _triangulate_parts(coords, ring_offsets, part_ring_offsets) -> np.ndarray:
    """Triangles of every polygon part, as indices into `coords`.

    The page would otherwise triangulate every polygon itself (deck.gl
    runs earcut per polygon) and keep one JavaScript object per part;
    on a phone that ran out of memory and stalled the load. Done here,
    the page hands the arrays to the GPU as they are.

    GEOS's constrained Delaunay triangulation uses only the polygon's
    own vertices, so each triangle corner is found again among the
    part's vertices by its coordinates. A part GEOS cannot triangulate
    (an invalid ring) gets no triangles: its walls still show, its top
    does not.

    Returns
    -------
    numpy.ndarray of uint32
        Flat `[a0, b0, c0, a1, b1, c1, ...]`, global vertex indices.
    """
    ring_offsets = np.asarray(ring_offsets, dtype=np.int64)
    part_ring_offsets = np.asarray(part_ring_offsets, dtype=np.int64)
    n_parts = len(part_ring_offsets) - 1
    if n_parts == 0:
        return np.zeros(0, dtype=np.uint32)
    polygons = shapely.from_ragged_array(
        shapely.GeometryType.POLYGON,
        np.ascontiguousarray(coords[:, :2], dtype=float),
        (ring_offsets, part_ring_offsets),
    )
    try:
        triangulated = shapely.constrained_delaunay_triangles(polygons)
    except shapely.errors.GEOSException:
        triangulated = np.array(
            [_triangulate_one(polygon) for polygon in polygons], dtype=object
        )
    triangles, part = shapely.get_parts(triangulated, return_index=True)
    if not len(triangles):
        return np.zeros(0, dtype=np.uint32)
    corners = shapely.get_coordinates(triangles).reshape(-1, 4, 2)[:, :3, :]

    vertex_start = ring_offsets[part_ring_offsets]
    vertex_part = np.repeat(np.arange(n_parts), np.diff(vertex_start))
    vertices = pd.DataFrame(
        {
            'part': vertex_part,
            'x': coords[:, 0],
            'y': coords[:, 1],
            'vertex': np.arange(len(coords)),
        }
    ).drop_duplicates(['part', 'x', 'y'])
    found = pd.DataFrame(
        {
            'part': np.repeat(part, 3),
            'x': corners[:, :, 0].ravel(),
            'y': corners[:, :, 1].ravel(),
        }
    ).merge(vertices, on=['part', 'x', 'y'], how='left')['vertex']
    index = found.to_numpy(dtype=float).reshape(-1, 3)
    complete = ~np.isnan(index).any(axis=1)
    return index[complete].astype(np.uint32).ravel()


def _triangulate_one(polygon):
    try:
        return shapely.constrained_delaunay_triangles(polygon)
    except shapely.errors.GEOSException:
        return shapely.GeometryCollection()


def encode_scene_file(geometry, base, height, value, stack=None) -> tuple[bytes, list]:
    """Encode one unit's features in the binary layout (module docstring).

    Parameters
    ----------
    geometry : array-like of shapely (Multi)Polygon
        In longitude and latitude (EPSG:4326). When any geometry has Z
        (a parcel draped on terrain), the vertices' Z is written as a
        per-vertex `z` array, and the page extrudes from it instead of
        from `base`; a 2D geometry in such a set reads as Z = 0.
    base, height : array-like of float
        Per feature, meters: where the extrusion starts and how tall it is.
    value : array-like of float
        Per feature, the value per area unit the page colors by; NaN
        where there is none.
    stack : array-like of float, optional
        Per feature, the part of `base` that is the extrusion of the
        layer below (a building on its parcel). With that layer
        switched off, the page lowers the feature by it.

    Returns
    -------
    bytes
        The uncompressed file.
    list of float
        The origin, `[longitude, latitude]`.
    """
    # Outer rings counterclockwise, holes clockwise: the page tells
    # deck.gl so (`_windingOrder: 'CCW'`) instead of having it check.
    geometry = shapely.orient_polygons(np.asarray(geometry), exterior_cw=False)
    coords, ring_offsets, part_ring_offsets, part_feature = _ragged_parts(geometry)
    triangles = _triangulate_parts(coords, ring_offsets, part_ring_offsets)
    z = coords[:, 2] if coords.shape[1] == 3 else None
    coords = coords[:, :2]
    origin = coords.mean(axis=0) if len(coords) else np.zeros(2)
    # Round the origin so that it reads back exactly from the JSON.
    origin = np.round(origin, 6)
    arrays = {
        'xy': (coords - origin).astype('<f4'),
        **({} if z is None else {'z': np.nan_to_num(z).astype('<f4')}),
        'ring_offsets': np.asarray(ring_offsets, dtype='<u4'),
        'part_ring_offsets': np.asarray(part_ring_offsets, dtype='<u4'),
        'part_feature': np.asarray(part_feature, dtype='<u4'),
        'triangles': np.asarray(triangles, dtype='<u4'),
        'base': np.asarray(base, dtype='<f4'),
        'height': np.asarray(height, dtype='<f4'),
        'value': np.asarray(value, dtype='<f4'),
        **({} if stack is None else {'stack': np.asarray(stack, dtype='<f4')}),
    }
    entries, offset = [], 0
    for name, array in arrays.items():
        entries.append(
            {
                'name': name,
                'dtype': array.dtype.str.lstrip('<|'),
                'offset': offset,
                'count': int(array.size),
            }
        )
        offset += array.nbytes
        offset += -offset % 8
    header = {
        'format': 'openplaces-value-scene',
        'version': FORMAT_VERSION,
        'origin': origin.tolist(),
        'n_features': int(len(geometry)),
        'arrays': entries,
    }
    header_bytes = json.dumps(header).encode('utf-8')
    header_bytes += b' ' * (-(len(MAGIC) + 4 + len(header_bytes)) % 8)
    chunks = [MAGIC, struct.pack('<I', len(header_bytes)), header_bytes]
    for array in arrays.values():
        data = array.tobytes()
        chunks.append(data + b'\0' * (-len(data) % 8))
    return b''.join(chunks), origin.tolist()


def decode_scene_file(data: bytes) -> tuple[dict, dict]:
    """Read a file written by `encode_scene_file` (for tests and checks).

    Returns
    -------
    dict
        The header.
    dict of str to numpy.ndarray
        The arrays by name.
    """
    if data[:4] != MAGIC:
        raise ValueError('Not an openplaces value-scene file.')
    (header_length,) = struct.unpack('<I', data[4:8])
    header = json.loads(data[8 : 8 + header_length])
    start = 8 + header_length
    arrays = {}
    for entry in header['arrays']:
        dtype = np.dtype(entry['dtype']).newbyteorder('<')
        begin = start + entry['offset']
        arrays[entry['name']] = np.frombuffer(
            data, dtype=dtype, count=entry['count'], offset=begin
        )
    return header, arrays


def _tooltip_values(gdf, layer: SceneLayer) -> dict:
    """Columnar tooltip values of *layer*, JSON-ready."""
    values = {}
    for tip in layer.tooltip:
        if tip.column not in gdf.columns:
            raise KeyError(
                f'Scene layer {layer.name!r} shows {tip.column!r}, which the '
                'rendered table lacks.'
            )
        column = gdf[tip.column]
        if tip.kind in ('money', 'money_per_area'):
            column = column.astype(float).round(0 if tip.kind == 'money' else 2)
        elif tip.kind in ('year', 'count'):
            column = column.astype('Int64')
        cells = column.astype(object).where(column.notna(), None).tolist()
        values[tip.column] = cells
        if tip.imputed_from is not None:
            source = (
                gdf[tip.imputed_from]
                if tip.imputed_from in gdf.columns
                else pd.Series(None, index=gdf.index)
            )
            values[f'{tip.column}__imputed'] = is_imputed(source).astype(int).tolist()
    return values


def layer_cmap(layer) -> mpl.colors.Colormap:
    """The colormap of *layer* (a SceneLayer or PaletteLayer), cut to
    its `cmap_range`. `cmap` is a colormap name or a tuple of color
    stops, interpolated evenly."""
    if isinstance(layer.cmap, tuple | list):
        cmap = mpl.colors.LinearSegmentedColormap.from_list('stops', list(layer.cmap))
    else:
        cmap = get_diverging_colormap(layer.cmap)
    low, high = layer.cmap_range
    if (low, high) == (0.0, 1.0):
        return cmap
    return mpl.colors.ListedColormap(
        cmap(np.linspace(low, high, 256)), name=f'{layer.cmap}_{low:g}_{high:g}'
    )


@dataclass(frozen=True)
class PaletteLayer:
    """How one scene layer is colored under one palette.

    Parameters
    ----------
    cmap : str or tuple of str
        Colormap name (`viz.colors.get_diverging_colormap`), or color
        stops interpolated evenly.
    vmin, vmax : float, optional
        Color-scale bounds, in dollars per the layer's area unit, on a
        log1p scale like `show_value_terrain_layer`'s. None (default)
        takes the scene's own bounds (`scene_bounds`), so one palette
        reads in a city and in a rural county alike.
    cmap_range : tuple of float
        The part of the colormap used (`SceneLayer.cmap_range`).
    """

    cmap: str
    vmin: float | None = None
    vmax: float | None = None
    cmap_range: tuple[float, float] = (0.0, 1.0)


def _nice(value: float, up: bool) -> float:
    """Round to 1, 2 or 5 times a power of ten, down or up."""
    if not value > 0:
        return 0.0
    power = 10 ** np.floor(np.log10(value))
    steps = (1, 2, 5, 10)
    if up:
        return float(next(s for s in steps if s * power >= value) * power)
    return float(next(s for s in reversed(steps) if s * power <= value) * power)


def scene_bounds(per_area, low_q=0.02, high_q=0.995) -> tuple[float, float]:
    """Color bounds from a scene's own values per area.

    Fixed bounds failed across places. Measured 2026-10-15 on US-NC-HYD:
    with Boston's $50K-$50M per acre, rural land fell almost wholly
    below the lower bound and drew in the ramp's darkest color, next
    to invisible on the dark page. The 2nd and 99.5th percentiles,
    rounded outward to 1-2-5 numbers, keep a scene's spread on the
    ramp and leave its extremes saturated.
    """
    values = np.asarray(per_area, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if not len(values):
        return 1.0, 10.0
    low, high = np.quantile(values, [low_q, high_q])
    low, high = _nice(low, up=False), _nice(high, up=True)
    return (low, high) if high > low else (low, low * 10)


@dataclass(frozen=True)
class Palette:
    """One color option of the page.

    Parameters
    ----------
    id : str
        Short id, used in the page's address (`?palette=<id>`).
    label : str
        Name on the page's palette switch.
    layers : dict of str to PaletteLayer
        Coloring per scene layer name.
    no_value_rgba : tuple of int
        Color of a feature without a value.
    background : str
        CSS background of the page.
    boundary_rgba : tuple of int
        Color of the town borders.
    """

    id: str
    label: str
    layers: dict
    no_value_rgba: tuple[int, int, int, int] = ZERO_VALUE_RGBA
    background: str = (
        'radial-gradient(ellipse 80% 70% at 50% 45%, #13213d 0%, #0a1222 45%, '
        '#05070d 100%)'
    )
    boundary_rgba: tuple[int, int, int, int] = (235, 240, 255, 90)


def default_palettes(layers=VALUE_SCENE_LAYERS) -> list[Palette]:
    """The page's color options (maintainer's request, 2026-10-15).

    The first is the scene's own coloring (`SceneLayer.cmap`). The
    others keep the cool-ground-under-warm-buildings idea in other
    hues. The notebook's pair of rainbows, which the page first
    shipped, was dropped as too loud (maintainer, 2026-10-15); Pastel
    takes its place as the playful option.
    """
    own = Palette(
        id='gold',
        label='Gold',
        layers={
            layer.name: PaletteLayer(layer.cmap, cmap_range=layer.cmap_range)
            for layer in layers
        },
    )
    # Kept after a review on three NC counties and Boston (maintainer,
    # 2026-10-16); Ember (its ground went black in rural counties),
    # Aurora (a heavy violet ground) and Slate were dropped.
    others = [
        # More hue than Gold, still in order of lightness (maintainer,
        # 2026-10-15: Gold liked, Slate too plain, Original too loud).
        Palette(
            id='sunset',
            label='Sunset',
            layers={
                'parcels': PaletteLayer('cmr.ocean', cmap_range=(0.05, 0.85)),
                'buildings': PaletteLayer('plasma', cmap_range=(0.3, 1.0)),
            },
        ),
        # Between Gold and the notebook's rainbows (maintainer,
        # 2026-10-15): soft, playful hues, but each ramp still rises in
        # lightness so the order reads. Ground: deep slate to sage.
        # Buildings: lavender, pink, peach, butter.
        Palette(
            id='pastel',
            label='Pastel',
            layers={
                'parcels': PaletteLayer(
                    # Topped at a mid sage: a paler top washed out
                    # downtown buildings (US-NC-DUR, 2026-10-16).
                    ('#1f2a3d', '#2c4556', '#43696d', '#6b9a8c')
                ),
                'buildings': PaletteLayer(
                    ('#a98bd6', '#d79bd0', '#f2a7b8', '#ffc4a3', '#ffe08f', '#fff4b8')
                ),
            },
        ),
    ]
    # Natural land colors (maintainer, 2026-10-16): green and blue out
    # in the country, through sand, to terracotta and brick red where
    # land is dearest, in muted tones. Unlike the other grounds this
    # ramp is not monotone in lightness (sand is its lightest step), so
    # land order reads by hue, as on a relief map. Buildings take the
    # cool side, dusty lavender to near-white, so they never share a
    # hue with the red center and stay ordered by lightness.
    others.append(
        Palette(
            id='landscape',
            label='Landscape',
            layers={
                'parcels': PaletteLayer(
                    (
                        '#3f6e8c',
                        '#5f9a8f',
                        '#8db88a',
                        '#d9d3a0',
                        '#e8a77c',
                        '#cf6a55',
                        '#a8443f',
                    )
                ),
                # Gold's buildings (maintainer, 2026-10-16: lavender to
                # white read as all white, plum to lilac as dull).
                'buildings': PaletteLayer('cmr.amber', cmap_range=(0.2, 0.95)),
            },
        )
    )
    names = {layer.name for layer in layers}
    order = ['landscape', 'sunset', 'pastel']
    others = sorted(others, key=lambda p: order.index(p.id))
    return [own] + [p for p in others if set(p.layers) == names]


def palette_manifest(palette: Palette, layers, bounds=None, stops: int = 256) -> dict:
    """A palette as the page reads it: per layer a 256-step lookup
    table (also drawn as the legend) and the bounds it spans.

    Parameters
    ----------
    bounds : dict of str to tuple of float, optional
        The scene's own bounds per layer (`scene_bounds`), used where
        the palette sets none.
    """
    alphas = {layer.name: layer.render.get('alpha', 1.0) for layer in layers}
    defaults = {layer.name: (layer.vmin, layer.vmax) for layer in layers}
    bounds = bounds or {}
    out = {}
    for name, spec in palette.layers.items():
        cmap = layer_cmap(spec)
        rgb = (cmap(np.linspace(0, 1, stops))[:, :3] * 255).round().astype(int)
        auto = bounds.get(name, defaults.get(name, (1.0, 10.0)))
        out[name] = {
            'vmin': spec.vmin if spec.vmin is not None else auto[0],
            'vmax': spec.vmax if spec.vmax is not None else auto[1],
            'alpha': round(255 * alphas.get(name, 1.0)),
            'lut': [mpl.colors.to_hex(c / 255) for c in rgb],
            'scale': 'log1p',
        }
    return {
        'id': palette.id,
        'label': palette.label,
        'layers': out,
        'no_value': list(palette.no_value_rgba),
        'background': palette.background,
        'boundary': list(palette.boundary_rgba),
    }


def _admin_names(admin_ids) -> pd.DataFrame:
    """Names and boundaries of the scene's units (level 3)."""
    admin = get_admin(list(admin_ids), level=3, geom=True)
    if 'admin3_id' not in admin.columns:
        admin = admin.reset_index()
    return admin[['admin3_id', 'name', 'geometry']].to_crs(4326)


def land_borders(admin_ids, land=None, tolerance_m=5.0, land_buffer_m=25.0) -> list:
    """Border lines the scene's units share with another unit, on land.

    A unit's whole outline also runs along the coast, which reads on
    the page as a line drawn in the sea. Only the stretches where a
    unit meets a neighbor (inside the scene or not) are kept, so the
    coastline drops out. Census units also extend into open water and
    meet there (Boston and its harbor neighbors), so a shared border is
    further kept only within `land_buffer_m` of the scene's parcels:
    a border on land has parcels beside it, one through the harbor
    does not. Neighbors are all units at the same level under the same
    parent (level 2); a border with a unit across a state line is
    therefore not drawn.

    Parameters
    ----------
    admin_ids : sequence of str
        The scene's level-3 units.
    land : geopandas.GeoDataFrame, optional
        Parcels marking land. Without it, every shared border is kept.
    tolerance_m : float
        Two outlines closer than this count as one border (the
        sources' shared edges rarely coincide exactly).
    land_buffer_m : float
        How close to a parcel a border must run to count as on land;
        about a street's width, so a border along a road is kept.

    Returns
    -------
    list of shapely.LineString
        In longitude and latitude, each shared border once.
    """
    from openplaces.core.schema import AdminId

    admin_ids = list(admin_ids)
    parents = sorted({str(AdminId(a).truncate_to_level(2)) for a in admin_ids})
    units = get_admin(parents, level=3, geom=True)
    if 'admin3_id' not in units.columns:
        units = units.reset_index()
    units = units[['admin3_id', 'geometry']]
    units = units.to_crs(units.estimate_utm_crs())
    scene = units[units['admin3_id'].isin(admin_ids)]
    seen, lines = set(), []
    for row in scene.itertuples():
        outline = row.geometry.boundary
        near = units[
            (units['admin3_id'] != row.admin3_id)
            & units.intersects(row.geometry.buffer(tolerance_m))
        ]
        for other in near.itertuples():
            pair = frozenset((row.admin3_id, other.admin3_id))
            if pair in seen:
                continue
            seen.add(pair)
            shared = outline.intersection(other.geometry.buffer(tolerance_m))
            shared = shapely.line_merge(shared) if not shared.is_empty else shared
            lines.extend(
                part
                for part in shapely.get_parts(shared)
                if part.geom_type == 'LineString' and part.length > tolerance_m
            )
    if not lines:
        return []
    borders = gpd.GeoSeries(lines, crs=units.crs)
    if land is not None and len(land):
        # Each border is cut into short pieces and a piece is kept when
        # a parcel lies within the buffer, by a nearest-neighbor query.
        # Parcels are never merged into a land shape: that would treat
        # them as one unit, which this display step has no need for.
        parcels = gpd.GeoSeries(land.geometry.to_crs(units.crs).to_numpy())
        kept = []
        for line in borders:
            pieces = shapely.segmentize(line, land_buffer_m)
            coords = np.asarray(pieces.coords)
            segments = shapely.linestrings(np.stack([coords[:-1], coords[1:]], axis=1))
            hit, _ = parcels.sindex.nearest(
                shapely.line_interpolate_point(segments, 0.5, normalized=True),
                max_distance=land_buffer_m,
                return_all=False,
            )
            on_land = np.zeros(len(segments), dtype=bool)
            on_land[hit] = True
            # Runs of consecutive pieces on land become one line.
            run = []
            for segment, keep in zip(segments, on_land, strict=True):
                if keep:
                    run.append(segment)
                elif run:
                    kept.append(shapely.line_merge(shapely.union_all(run)))
                    run = []
            if run:
                kept.append(shapely.line_merge(shapely.union_all(run)))
        borders = gpd.GeoSeries(
            [
                part
                for geom in kept
                for part in shapely.get_parts(geom)
                if part.geom_type == 'LineString' and part.length > land_buffer_m
            ],
            crs=units.crs,
        )
    return list(borders.to_crs(4326))


def tour_foci(
    gdf,
    value_column,
    names,
    n_foci=6,
    cell_m=400.0,
    min_separation_m=1500.0,
    min_parcels=20,
) -> list[dict]:
    """Points of greatest value density, for the page's opening flyover.

    The tour circles places a visitor recognizes as centers. Rather
    than naming them (which would put a place's geography in code),
    they are read off the data: the scene is cut into square cells,
    each cell's value is summed over its area, and the densest cells
    are kept. Every unit contributes its densest cell, so each town
    center is visited; further cells, at least `min_separation_m` from
    any kept one, fill up to `n_foci`. The densest cell of all comes
    first, so the tour lands on the main downtown before anything
    else, and the rest follow in nearest-neighbor order, so the flights
    between them stay short.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The rendered bottom layer (parcels), with `admin3_id`.
    value_column : str
        Value to sum per cell. The bottom layer's value (land) picks
        a district of costly land, which need not be the center a
        visitor knows; so `total_value` (land and buildings) is used
        whenever the table has it, and this column otherwise.
        Measured 2026-10-15 on boston-inner-core: land value alone put
        the first stop in Kendall Square, total value in downtown
        Boston.
    names : dict of str to str
        Unit names by `admin3_id`, for labels.
    n_foci : int
        Most points to return.
    cell_m : float
        Cell size in meters; about a few city blocks.
    min_separation_m : float
        Least distance between two points. Measured 2026-10-15 on
        boston-inner-core: at 2500 m, Kendall Square (2.4 km from
        downtown Boston) was refused and Cambridge's stop fell on its
        edge.
    min_parcels : int
        Least parcels in a cell. A center is many parcels; without
        this, one vast parcel (an airport, a rail yard) wins its cell.

    Returns
    -------
    list of dict
        `name`, `longitude`, `latitude`, tour order first to last.
    """
    if gdf.empty or n_foci < 1:
        return []
    # Cells in meters: the layer's own CRS when it is projected (the
    # bundles are), its UTM zone otherwise.
    projected = (
        gdf
        if gdf.crs is not None and gdf.crs.is_projected
        else gdf.to_crs(gdf.estimate_utm_crs())
    )
    points = projected.geometry.centroid
    cells = pd.DataFrame(
        {
            'i': np.floor(points.x / cell_m).astype(int),
            'j': np.floor(points.y / cell_m).astype(int),
            'value': gdf[
                'total_value' if 'total_value' in gdf.columns else value_column
            ]
            .fillna(0)
            .to_numpy(dtype=float),
            'admin3_id': gdf['admin3_id'].to_numpy(),
        }
    )
    summed = (
        cells.groupby(['i', 'j'])
        .agg(
            value=('value', 'sum'),
            n=('value', 'size'),
            admin3_id=('admin3_id', lambda s: s.mode()[0]),
        )
        .reset_index()
        .query('n >= @min_parcels')
        .sort_values('value', ascending=False)
    )
    if summed.empty:
        return []
    summed['x'] = (summed['i'] + 0.5) * cell_m
    summed['y'] = (summed['j'] + 0.5) * cell_m

    best_per_unit = summed.drop_duplicates('admin3_id')
    chosen = []

    def far_enough(row):
        return all(
            np.hypot(row.x - c.x, row.y - c.y) >= min_separation_m for c in chosen
        )

    for row in best_per_unit.itertuples():
        if len(chosen) < n_foci and far_enough(row):
            chosen.append(row)
    for row in summed.itertuples():
        if len(chosen) >= n_foci:
            break
        if far_enough(row):
            chosen.append(row)

    # Densest first, then always the nearest one not yet visited.
    chosen.sort(key=lambda r: -r.value)
    ordered = [chosen.pop(0)]
    while chosen:
        last = ordered[-1]
        nearest = min(chosen, key=lambda r: np.hypot(r.x - last.x, r.y - last.y))
        chosen.remove(nearest)
        ordered.append(nearest)

    lonlat = gpd.GeoSeries(
        gpd.points_from_xy([r.x for r in ordered], [r.y for r in ordered]),
        crs=projected.crs,
    ).to_crs(4326)
    return [
        {
            'name': names.get(row.admin3_id, row.admin3_id),
            # The page loads this unit first when it is the first stop.
            'admin3_id': row.admin3_id,
            'longitude': round(point.x, 6),
            'latitude': round(point.y, 6),
        }
        for row, point in zip(ordered, lonlat, strict=True)
    ]


def build_web_scene(
    region,
    admin_ids,
    output_dir,
    layers=VALUE_SCENE_LAYERS,
    title=None,
    build_id=None,
    elevation_recipe=None,
    terrain_exaggeration=1.0,
    n_foci=6,
    palettes=None,
    verbose=True,
) -> Path:
    """Write the static web scene of a region's delivery bundles.

    Parameters
    ----------
    region : str or list of str
        Region id(s); each layer's recipe must deliver them, and each
        unit is read from the region that pools it. None builds a
        **local preview** from the curated per-unit files instead: no
        delivery withholding applies, so the manifest is marked
        `preview` and `deploy_web_scene` refuses to publish it.
    admin_ids : sequence of str
        Units (`admin3_id`) of the region to include. All of them load
        when the page opens.
    output_dir : str or Path
        Directory to write the site into. Existing files of an earlier
        build stay, so a deploy can keep serving them until the new
        page replaces the old.
    layers : sequence of SceneLayer
        The scene, bottom layer first. Defaults to parcels by land value
        with buildings by structure value on top.
    title : str, optional
        Page title. Defaults to `DEFAULT_TITLE`.
    build_id : str, optional
        Name of this build's data folder. Defaults to the UTC time.
    elevation_recipe : str, optional
        DEM recipe (e.g. `US_land-elevation-usgs-3dep`) to ground the
        scene in terrain. The bottom layer is draped (one polygon per
        feature, each vertex at its own ground height; the notebook's
        mesh mode falls back to the same at city scale), and stacked
        layers stand flat on it. None (default) keeps flat ground.
    terrain_exaggeration : float
        Multiplier on ground relief only, never on value height.
    n_foci : int
        Most points the opening flyover circles (`tour_foci`).
    palettes : sequence of Palette, optional
        Color options the page offers, the first one shown by default.
        Defaults to `default_palettes(layers)`.
    verbose : bool

    Returns
    -------
    Path
        The output directory.
    """
    # The import is here, not at module level, because lonboard (which
    # terrain imports) belongs to the optional viz-fast extra.
    from openplaces.viz.terrain import show_value_terrain_layer

    refuse_personal_fields(layers)
    palettes = list(palettes) if palettes else default_palettes(layers)
    admin_ids = list(admin_ids)
    output_dir = Path(output_dir)
    build_id = build_id or datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
    data_dir = output_dir / 'data' / build_id
    data_dir.mkdir(parents=True, exist_ok=True)

    admin = _admin_names(admin_ids)
    names = dict(zip(admin['admin3_id'], admin['name'], strict=True))

    # One datum for every layer, or they slide against each other
    # (`viz.elevation.get_elevation_datum`).
    ground = {}
    if elevation_recipe is not None:
        from openplaces.viz.elevation import get_elevation_datum

        ground = {
            'elevation_recipe': elevation_recipe,
            'elevation_datum': get_elevation_datum(
                admin, elevation_recipe, admin_id_column='admin3_id'
            ),
            'terrain_exaggeration': terrain_exaggeration,
        }

    rendered = {}
    bounds = {}
    manifest_layers = []
    notices = {}
    for layer in layers:
        if region is None:
            from openplaces.io.readers import get_entities

            gdf = get_entities(layer.recipe, admin_id=admin_ids, geom=True)
            notices[layer.recipe] = 'Local preview from curated files; not published.'
        else:
            # Several regions: each unit is read from the bundle of the
            # region it belongs to. An entity on a boundary between two
            # regions can sit in both bundles; the first copy is kept.
            regions = [region] if isinstance(region, str) else list(region)
            parts = []
            for name in regions:
                members = set(_region_members(layer.recipe, name))
                wanted = [a for a in admin_ids if a in members]
                if wanted:
                    parts.append(read_bundle(layer.recipe, name, wanted))
            if not parts:
                raise ValueError(f'No unit of {admin_ids} is in {regions}.')
            gdf = pd.concat(parts)
            gdf = gdf[~gdf.index.duplicated(keep='first')]
            notices[layer.recipe] = '\n\n'.join(
                bundle_notice(layer.recipe, name) for name in regions
            )
        stack_on = rendered[layer.stack_on] if layer.stack_on else None
        terrain = show_value_terrain_layer(
            gdf,
            None,
            value_column=layer.value_column,
            area_unit=layer.area_unit,
            vmin=layer.vmin,
            vmax=layer.vmax,
            cmap=layer_cmap(layer),
            stack_on=stack_on,
            elevation_mode='drape' if ground and stack_on is None else 'flat',
            missing_value='render',
            silent=not verbose,
            **ground,
            **layer.render,
        )
        rendered[layer.name] = terrain
        out = terrain.gdf
        base = terrain.total_elevation - terrain.rendered_elevation
        geometry = out.geometry
        if layer.simplify_m:
            metric = geometry.crs is not None and geometry.crs.is_projected
            work = geometry if metric else geometry.to_crs(out.estimate_utm_crs())
            work = gpd.GeoSeries(
                shapely.simplify(
                    work.to_numpy(), layer.simplify_m, preserve_topology=True
                ),
                index=work.index,
                crs=work.crs,
            )
            geometry = work
        geometry = geometry.to_crs(4326).to_numpy()
        values = _tooltip_values(out, layer)
        # No value: a dark slate that recedes into the page, not the
        # renderer's mid gray, which read as a dull patch in the scene.
        per_area = out[f'{layer.value_column}_per_{layer.area_unit}'].to_numpy(
            dtype=float
        )
        no_value = out[layer.value_column].fillna(0).to_numpy() == 0
        per_area = np.where(no_value, np.nan, per_area)
        bounds[layer.name] = scene_bounds(per_area)
        files_by_unit = {}
        for admin_id in admin_ids:
            rows = (out['admin3_id'] == admin_id).to_numpy()
            if not rows.any():
                continue
            data, _ = encode_scene_file(
                geometry[rows],
                base[rows],
                terrain.rendered_elevation[rows],
                per_area[rows],
                stack=terrain.stack_elevation[rows] if layer.stack_on else None,
            )
            stem = f'{layer.name}_{admin_id}'
            (data_dir / f'{stem}.bin').write_bytes(gzip.compress(data, 9))
            unit_values = {
                key: [v for v, keep in zip(column, rows, strict=True) if keep]
                for key, column in values.items()
            }
            (data_dir / f'{stem}_tooltip.bin').write_bytes(
                gzip.compress(json.dumps(unit_values).encode('utf-8'), 9)
            )
            files_by_unit[admin_id] = {
                'geometry': f'data/{build_id}/{stem}.bin',
                'tooltip': f'data/{build_id}/{stem}_tooltip.bin',
            }
            if verbose:
                print(f'{stem}: {int(rows.sum()):,} features')
        manifest_layers.append(
            {
                'name': layer.name,
                'unit': layer.area_unit,
                'label': layer.label,
                'tooltip': [
                    {
                        'column': tip.column,
                        'label': tip.label,
                        'kind': tip.kind,
                        'imputed': tip.imputed_from is not None,
                    }
                    for tip in layer.tooltip
                ],
                'files': files_by_unit,
            }
        )

    boundaries = [
        {'path': np.round(np.asarray(line.coords)[:, :2], 6).tolist()}
        for line in land_borders(admin_ids, land=rendered[layers[0].name].gdf)
    ]

    center = admin.union_all().centroid
    bottom = layers[0]
    foci = tour_foci(
        rendered[bottom.name].gdf,
        bottom.value_column,
        names,
        n_foci=n_foci,
    )
    manifest = {
        'title': title or DEFAULT_TITLE,
        'build_id': build_id,
        'built': datetime.now(UTC).isoformat(timespec='seconds'),
        'units': [names.get(a, a) for a in admin_ids],
        'view': {
            'longitude': round(center.x, 6),
            'latitude': round(center.y, 6),
            'zoom': 12,
            'pitch': 45,
            'bearing': 0,
        },
        'layers': manifest_layers,
        'boundaries': boundaries,
        # Per unit [west, south, east, north]: a device short of memory
        # loads only the units in view (viewer.js `constrained`).
        'unit_bounds': {
            row.admin3_id: [round(v, 5) for v in row.geometry.bounds]
            for row in admin.itertuples()
        },
        'notices': notices,
        'preview': region is None,
        'deck_gl_url': DECK_GL_URL,
        'palettes': [palette_manifest(p, layers, bounds) for p in palettes],
        'tour': foci,
        'terrain': {
            'elevation_recipe': elevation_recipe,
            'terrain_exaggeration': terrain_exaggeration if ground else None,
        },
    }
    (output_dir / 'scene.json').write_text(json.dumps(manifest), encoding='utf-8')
    for item in STATIC_DIR.iterdir():
        if item.name == 'landing.html':
            # The landing page's template, written by `landing.py`.
            continue
        target = '.htaccess' if item.name == 'htaccess' else item.name
        (output_dir / target).write_bytes(item.read_bytes())
    if verbose:
        size = sum(f.stat().st_size for f in data_dir.iterdir()) / 2**20
        print(f'Wrote {output_dir} (build {build_id}, {size:.1f} MB of data)')
    return output_dir


# The deploy follows the maintainer's other sites on the same web host
# (the LakeSense dashboard's web/deploy.py, the PLACES-FMV docs): the
# target comes from the environment or an env file kept outside the
# repository, and rclone reaches the host over SFTP through a remote
# defined only in environment variables, so no rclone.conf entry is
# needed.
DEPLOY_PREFIX = 'OPENPLACES_WEB_DEPLOY_'
DEPLOY_REQUIRED = ('HOST', 'USER', 'PATH')
# Builds whose data stay online: this one and the one before, so a
# page opened before the switch keeps loading its own files.
KEEP_BUILDS = 2
# Shared hosting refuses connections once too many SSH sessions are
# open (Bluehost's cPHulk), and rclone's SFTP backend opens one per
# transfer and checker. Files are compared on size and modification
# time: `--checksum` would make the server run md5sum on every file.
# cPanel's AutoSSL writes certificate challenges to /.well-known/, which
# a deploy must never touch.
RCLONE_SFTP_FLAGS = (
    '--filter',
    '- /.well-known/**',
    '--transfers',
    '2',
    '--checkers',
    '2',
    '--tpslimit',
    '4',
    '--retries',
    '3',
    '--low-level-retries',
    '10',
)
# cPHulk refuses SSH for a while after a burst of connections; every
# step of the deploy is safe to repeat after a pause.
RETRY_TRIES = 4
RETRY_WAIT_S = 60


def default_deploy_env_file() -> Path:
    """`web-deploy.env` beside the user's `credentials.yaml`."""
    from openplaces.config import cfg

    return cfg.credentials_path.parent / 'web-deploy.env'


def load_deploy_settings(env_file=None) -> dict:
    """Read the web host's deploy settings.

    Variables already in the environment win over the file, so a CI
    secret beats a stale local copy.

    Parameters
    ----------
    env_file : str or Path, optional
        `KEY=value` lines. Defaults to `default_deploy_env_file`, in the
        user's openplaces config folder, outside every repository.
        Keys, all prefixed `OPENPLACES_WEB_DEPLOY_`:
        `HOST`, `USER` and `PATH` (absolute path of the site's folder on
        the host) are required; `KEY` (private key file),
        `KNOWN_HOSTS` (pinned host key file) and `URL` (public address)
        are optional.

    Returns
    -------
    dict
        `host`, `user`, `path`, `key`, `known_hosts`, `url`.
    """
    import os

    env_file = Path(env_file) if env_file else default_deploy_env_file()
    values = {}
    if env_file.exists():
        for raw in env_file.read_text(encoding='utf-8').splitlines():
            line = raw.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, value = line.partition('=')
            values[key.strip()] = value.strip().strip('"\'')
    values.update({k: v for k, v in os.environ.items() if k.startswith(DEPLOY_PREFIX)})

    missing = [
        DEPLOY_PREFIX + k for k in DEPLOY_REQUIRED if not values.get(DEPLOY_PREFIX + k)
    ]
    if missing:
        raise ValueError(
            f'Missing deploy settings {", ".join(missing)}. Set them in the '
            f'environment or in {env_file}, e.g.\n'
            f'  {DEPLOY_PREFIX}HOST=<the host your other sites deploy to>\n'
            f'  {DEPLOY_PREFIX}USER=<ssh user>\n'
            f'  {DEPLOY_PREFIX}PATH=/home/<user>/public_html/<folder>\n'
            f'  {DEPLOY_PREFIX}KEY=C:/Users/<you>/.ssh/<deploy key>\n'
            f'  {DEPLOY_PREFIX}KNOWN_HOSTS=C:/Users/<you>/.ssh/<known hosts>\n'
            f'  {DEPLOY_PREFIX}URL=https://<domain>/<folder>'
        )

    def _file(name):
        value = values.get(DEPLOY_PREFIX + name, '')
        if not value:
            return ''
        path = Path(value).expanduser()
        if not path.exists():
            raise FileNotFoundError(f'{DEPLOY_PREFIX}{name} names no file: {value}')
        return str(path)

    return {
        'host': values[DEPLOY_PREFIX + 'HOST'],
        'user': values[DEPLOY_PREFIX + 'USER'],
        'path': values[DEPLOY_PREFIX + 'PATH'].rstrip('/'),
        'key': _file('KEY'),
        'known_hosts': _file('KNOWN_HOSTS'),
        'url': values.get(DEPLOY_PREFIX + 'URL', '').rstrip('/'),
    }


def _rclone_env(settings) -> dict:
    """Environment defining the `bh:` SFTP remote for one rclone call."""
    import os

    env = os.environ.copy()
    env['RCLONE_CONFIG_BH_TYPE'] = 'sftp'
    env['RCLONE_CONFIG_BH_HOST'] = settings['host']
    env['RCLONE_CONFIG_BH_USER'] = settings['user']
    # Skip rclone's interactive shell probe on shared hosting.
    env['RCLONE_CONFIG_BH_SHELL_TYPE'] = 'unix'
    if settings['key']:
        env['RCLONE_CONFIG_BH_KEY_FILE'] = settings['key']
    if settings['known_hosts']:
        path = Path(settings['known_hosts'])
        env['RCLONE_CONFIG_BH_KNOWN_HOSTS_FILE'] = str(path)
        algorithms = _known_hosts_algorithms(path)
        if algorithms:
            env['RCLONE_CONFIG_BH_HOST_KEY_ALGORITHMS'] = ' '.join(algorithms)
    return env


def _known_hosts_algorithms(path: Path) -> list[str]:
    """Host key algorithms a known_hosts file pins.

    rclone otherwise negotiates whichever algorithm the server offers
    first, which can differ from the pinned key's and then fails the
    host check. An `ssh-rsa` key is also offered under its SHA-2
    signature names, which current servers require.
    """
    algorithms = []
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except OSError:
        return []
    for line in lines:
        if line.startswith(('#', '@')):
            continue
        fields = line.split()
        if len(fields) >= 3 and fields[1] not in algorithms:
            algorithms.append(fields[1])
    if 'ssh-rsa' in algorithms:
        i = algorithms.index('ssh-rsa')
        algorithms[i : i + 1] = ['rsa-sha2-512', 'rsa-sha2-256', 'ssh-rsa']
    return algorithms


def deploy_web_scene(
    output_dir, env_file=None, go=False, keep_builds=KEEP_BUILDS, subpath=None
):
    """Upload a built scene to the web host with `rclone` over SFTP.

    Nothing is written unless `go` is True: the default is a dry run
    that lists what would be added, changed or deleted.

    Three phases, so a visitor never sees a half-updated site:

    1. the build's data go into their own folder, `data/<build_id>/`;
       nothing a page already loads changes;
    2. switch: the page, its script, the manifest and `.htaccess`,
       which point at the new folder;
    3. clean up: data folders of builds other than the last
       `keep_builds` are deleted, so a page opened just before the
       switch still finds its files.

    Parameters
    ----------
    output_dir : str or Path
        Directory written by `build_web_scene`.
    env_file : str or Path, optional
        Deploy settings (`load_deploy_settings`).
    go : bool
        Upload for real.
    keep_builds : int
        Builds whose data stay online, this one included.
    subpath : str, optional
        Folder under the configured path, e.g. a place's slug beside the
        landing page (`/values/boston/`).

    Returns
    -------
    str
        The public address, or the remote path when no URL is set.
    """
    output_dir = Path(output_dir)
    if not (output_dir / 'index.html').exists():
        raise FileNotFoundError(f'No built scene in {output_dir}.')
    manifest = json.loads((output_dir / 'scene.json').read_text(encoding='utf-8'))
    build_id = manifest['build_id']
    if manifest.get('preview'):
        raise ValueError(
            f'{output_dir} is a local preview built from curated files, '
            'without the delivery withholding; it is never published. '
            'Build from a delivery region to publish.'
        )
    settings = load_deploy_settings(env_file)
    env = _rclone_env(settings)
    if subpath:
        subpath = subpath.strip('/')
        settings = {
            **settings,
            'path': f'{settings["path"]}/{subpath}',
            'url': f'{settings["url"]}/{subpath}' if settings['url'] else '',
        }
    target = f'bh:{settings["path"]}'
    dry = [] if go else ['--dry-run']
    print(f'Target: {settings["user"]}@{settings["host"]}:{settings["path"]}')
    if settings['url']:
        print(f'URL   : {settings["url"]}/')
    print(f'Mode  : {"PUBLISH" if go else "dry run"}')

    # The build currently online, read before the switch, stays.
    previous = _rclone_output(env, 'cat', f'{target}/scene.json')
    try:
        previous_id = json.loads(previous)['build_id'] if previous else None
    except (ValueError, KeyError):
        previous_id = None

    print(f'\nPhase 1: data of build {build_id}')
    _rclone(
        env,
        'copy',
        str(output_dir / 'data' / build_id),
        f'{target}/data/{build_id}',
        *RCLONE_SFTP_FLAGS,
        *dry,
    )
    print('\nPhase 2: switch (index.html, viewer.js, scene.json, .htaccess)')
    _rclone(
        env,
        'copy',
        str(output_dir),
        target,
        '--max-depth',
        '1',
        *RCLONE_SFTP_FLAGS,
        '--filter',
        '+ /{index.html,viewer.js,scene.json,.htaccess}',
        '--filter',
        '- **',
        *dry,
    )

    online = _rclone_output(
        env,
        'lsf',
        '--dirs-only',
        f'{target}/data',
        '--checkers',
        '2',
        '--tpslimit',
        '4',
    )
    builds = sorted(n.rstrip('/') for n in (online or '').split() if n)
    keep = {build_id}
    if previous_id and keep_builds > 1:
        keep.add(previous_id)
    stale = [n for n in builds if n not in keep]
    print(f'\nPhase 3: clean up ({len(stale)} older build(s))')
    for name in stale:
        _rclone(
            env,
            'purge',
            f'{target}/data/{name}',
            '--checkers',
            '2',
            '--tpslimit',
            '4',
            *dry,
        )
    address = settings['url'] or settings['path']
    print(f'\n{"Published" if go else "Dry run complete"}: {address}/')
    return address


def deploy_landing(output_dir, env_file=None, go=False):
    """Upload a landing page (`landing.build_landing`) to the configured
    path: its page and card images only; the scenes in the folders
    beside it are never touched."""
    output_dir = Path(output_dir)
    settings = load_deploy_settings(env_file)
    env = _rclone_env(settings)
    dry = [] if go else ['--dry-run']
    print(f'Target: {settings["user"]}@{settings["host"]}:{settings["path"]}')
    _rclone(
        env,
        'copy',
        str(output_dir),
        f'bh:{settings["path"]}',
        *RCLONE_SFTP_FLAGS,
        '--filter',
        '+ /index.html',
        '--filter',
        '+ /thumbs/**',
        '--filter',
        '- **',
        *dry,
    )
    address = settings['url'] or settings['path']
    print(f'{"Published" if go else "Dry run complete"}: {address}/')


def deploy_redirect(old_path, new_url, env_file=None, go=False):
    """Point an old address at a new one: write an `.htaccess` with a
    permanent redirect into `old_path` on the host. Files already there
    stay (unreachable behind the redirect) until removed by hand."""
    import tempfile

    settings = load_deploy_settings(env_file)
    env = _rclone_env(settings)
    dry = [] if go else ['--dry-run']
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / '.htaccess').write_text(
            '# Moved; written by openplaces.viz.web_scene.deploy_redirect.\n'
            f'RedirectMatch 301 ^.*$ {new_url}\n',
            encoding='utf-8',
        )
        _rclone(env, 'copy', tmp, f'bh:{old_path}', *RCLONE_SFTP_FLAGS, *dry)
    print(f'{"Redirected" if go else "Dry run complete"}: {old_path} -> {new_url}')


def _rclone_bin() -> str:
    found = shutil.which('rclone')
    if found is None:
        raise RuntimeError(
            'rclone is not on the PATH; it is part of the openplaces conda environment.'
        )
    return found


def _rclone(env, *args) -> None:
    """Run rclone, retrying after a pause while the host refuses SSH."""
    import time

    command = [_rclone_bin(), *args]
    for attempt in range(1, RETRY_TRIES + 1):
        if subprocess.run(command, env=env, check=False).returncode == 0:
            return
        if attempt < RETRY_TRIES:
            print(
                f'  rclone {args[0]} failed (the host may be limiting SSH); '
                f'retrying in {RETRY_WAIT_S} s ({attempt}/{RETRY_TRIES - 1})'
            )
            time.sleep(RETRY_WAIT_S)
    raise RuntimeError(f'`rclone {args[0]}` failed {RETRY_TRIES} times.')


def _rclone_output(env, *args) -> str | None:
    """Output of a read-only rclone call, or None if it fails."""
    completed = subprocess.run(
        [_rclone_bin(), *args], env=env, capture_output=True, text=True, check=False
    )
    return completed.stdout if completed.returncode == 0 else None
