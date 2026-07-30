"""GPU-accelerated interactive rendering of large entity datasets.

Uses Lonboard/deck.gl, rendered client-side in the browser over WebGL2.
"""

import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import xyzservices.providers as xyz
from lonboard import BitmapTileLayer, Map, PathLayer, PolygonLayer, SolidPolygonLayer
from lonboard.layer_extension import PathStyleExtension

from openplaces.io.readers import get_admin, get_entities
from openplaces.viz.colors import (
    MISSING_LABEL,
    RESERVED_NEUTRAL_COLOR,
    resolve_category_colors,
    to_rgba_array,
)
from openplaces.viz.legend import categorical_legend_entries, legend_html

DEFAULT_FILL_COLOR = '#5a9e6f'
# Matches openplaces.viz.colors.continuous_to_rgba's own defaults -- kept as
# a literal here (rather than imported) since _numeric_to_rgba below
# reimplements that ramp directly rather than calling continuous_to_rgba.
NUMERIC_RAMP = ('#e5e7db', '#5a2828')

# Free, keyless XYZ tile providers spanning visually distinct basemap styles
# for use as the ground plane under an extruded scene (e.g.
# openplaces.viz.terrain) -- Map's own basemap_style only offers abstract
# CARTO vector styles, no photographic/topo raster option, so a raster tile
# layer is used instead. All confirmed keyless via provider.requires_token().
_BASEMAP_PROVIDERS = {
    'satellite': xyz.Esri.WorldImagery,
    'osm': xyz.OpenStreetMap.Mapnik,
    'positron': xyz.CartoDB.Positron,
    'dark_matter': xyz.CartoDB.DarkMatter,
    'topo': xyz.Esri.WorldTopoMap,
}


def show_entities_interactive(
    recipe,
    admin_id=None,
    bbox: tuple[float, float, float, float] | None = None,
    color_by: str | None = None,
    simplified: bool = False,
    opacity: float = 0.8,
    default_color: str = DEFAULT_FILL_COLOR,
    missing: str = 'raise',
    legend: bool = True,
) -> Map:
    """Render entity polygons as an interactive GPU map via Lonboard/deck.gl.

    Loads the recipe's output through :func:`openplaces.io.readers.get_entities`
    and renders it with a `SolidPolygonLayer` rather than the composite
    `PolygonLayer` — the latter's extra outline draw pass is unnecessary
    overhead at million-polygon scale. Best suited for a WebGL2-capable
    client; there is no reliable way for a Jupyter kernel to detect GPU/WebGL
    availability, so choosing between this and
    :func:`openplaces.viz.raster.show_entities_raster` is left to the caller.

    Parameters
    ----------
    recipe : str or dict
        Recipe that defines the entity, passed to `get_entities`.
    admin_id : str, AdminId, or sequence, optional
        Administrative unit(s) to load. Defaults to the recipe admin ID.
    bbox : tuple of (minx, miny, maxx, maxy), optional
        Spatial bounding box filter in EPSG:4326, forwarded to `get_entities`
        for viewport-bounded loading across a recipe's per-admin-unit files.
    color_by : str, optional
        Column to color polygons by, via
        `openplaces.viz.colors.resolve_category_colors` for categorical
        columns (curated colors from `openplaces.viz.colors.CATEGORY_COLORS`
        where available, a deterministic per-label color for everything
        else -- never a single shared fallback, so the same label always
        gets the same color regardless of admin-unit scope). Numeric columns
        use a linear light-to-dark ramp over their observed range.
    simplified : bool
        If True, load simplified geometries (``geom='simplified'``) where a
        sidecar exists, trading vertex fidelity for faster transport/render.
    opacity : float
        Fill opacity in [0, 1].
    default_color : str
        Hex fill color used when `color_by` is None (a plain solid-color
        map).
    missing : {'raise', 'warn', 'ignore'}
        How to handle missing output files, forwarded to `get_entities`.
        Useful for rendering whatever subset of a multi-admin-unit recipe is
        already processed (e.g. a state where only some counties are done).
    legend : bool
        If True and `color_by` is given, display an HTML legend above the
        map via `IPython.display.display`. deck.gl's WebGL canvas can't
        host a DOM overlay the way a raster image can be drawn on directly,
        so the legend renders as a separate widget rather than being
        composited into the map itself. No legend is shown when `color_by`
        is None (a plain solid-color map has nothing to key a legend to).

    Returns
    -------
    lonboard.Map
        Renders automatically in Jupyter; remains user-customizable (add
        layers, reassign ``layers[0].get_fill_color``, etc.). Any legend is
        displayed as a side effect, not part of the returned object.
    """
    gdf = get_entities(
        recipe,
        admin_id=admin_id,
        geom='simplified' if simplified else True,
        bbox=bbox,
        missing=missing,
    )
    is_polygonal = gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])
    if not is_polygonal.all():
        warnings.warn(
            f'Dropping {(~is_polygonal).sum()} row(s) with non-polygon geometry '
            '(SolidPolygonLayer only renders Polygon/MultiPolygon).',
            stacklevel=2,
        )
        gdf = gdf[is_polygonal]

    if gdf.empty:
        raise ValueError(
            f'No polygon geometries to render for admin_id={admin_id!r}'
            f'{f", bbox={bbox!r}" if bbox is not None else ""} -- '
            'the selection returned zero rows.'
        )

    alpha = round(opacity * 255)
    fill_color, legend_kwargs = _resolve_fill_color(gdf, color_by, default_color, alpha)

    if legend and legend_kwargs is not None:
        from IPython.display import HTML, display

        display(HTML(legend_html(color_by, **legend_kwargs)))

    layer = SolidPolygonLayer.from_geopandas(gdf, get_fill_color=fill_color)
    return Map(layer)


def _resolve_fill_color(gdf: gpd.GeoDataFrame, color_by, default_color, alpha):
    """Resolve a per-row fill color, plus `legend_html` kwargs (or None)."""
    if color_by is None:
        return _hex_to_rgba(default_color, alpha), None

    values = gdf[color_by]
    if pd.api.types.is_numeric_dtype(values):
        finite = values[np.isfinite(values)]
        if finite.empty:
            return _hex_to_rgba(DEFAULT_FILL_COLOR, alpha), None
        legend_kwargs = {
            'numeric': True,
            'vmin': finite.min(),
            'vmax': finite.max(),
            'low_color': NUMERIC_RAMP[0],
            'high_color': NUMERIC_RAMP[1],
        }
        return _numeric_to_rgba(values, alpha), legend_kwargs

    filled = values.astype('object').fillna(MISSING_LABEL).astype(str)
    palette = resolve_category_colors(values, col_name=color_by)
    fill_color = to_rgba_array(
        filled, palette, default=RESERVED_NEUTRAL_COLOR, alpha=alpha
    )
    entries = categorical_legend_entries(filled, palette)
    return fill_color, {'entries': entries}


def _hex_to_rgba(hex_color: str, alpha: int) -> tuple[int, int, int, int]:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return (r, g, b, alpha)


def _numeric_to_rgba(values: pd.Series, alpha: int) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.empty:
        return _hex_to_rgba(DEFAULT_FILL_COLOR, alpha)
    vmin, vmax = finite.min(), finite.max()
    span = vmax - vmin or 1.0
    scaled = ((values.fillna(vmin) - vmin) / span).clip(0, 1).to_numpy()
    r = (229 - scaled * (229 - 90)).astype(np.uint8)
    g = (231 - scaled * (231 - 40)).astype(np.uint8)
    b = (219 - scaled * (219 - 40)).astype(np.uint8)
    return np.column_stack([r, g, b, np.full(len(values), alpha, dtype=np.uint8)])


def get_basemap_layer(
    provider: str = 'satellite', opacity: float = 1.0
) -> BitmapTileLayer:
    """XYZ tile basemap as a lonboard basemap layer.

    Add as the *first* entry in a `lonboard.Map([...])` layer list so it
    renders as the ground plane beneath other layers (e.g. an extruded 3D
    scene from `openplaces.viz.terrain`) — `Map`'s own `basemap_style`
    parameter only offers abstract CARTO vector styles (Positron, DarkMatter,
    Voyager), no photographic or topo raster option, so a raster tile layer
    is used instead.

    Parameters
    ----------
    provider : str
        One of ``'satellite'`` (Esri World Imagery, photographic), ``'osm'``
        (OpenStreetMap Mapnik, streets/labels), ``'positron'`` (CartoDB
        Positron, light/minimal), ``'dark_matter'`` (CartoDB Dark Matter,
        dark/minimal), or ``'topo'`` (Esri World Topo Map, shaded
        relief/contours).
    opacity : float
        Layer opacity in [0, 1].

    Returns
    -------
    lonboard.BitmapTileLayer
    """
    if provider not in _BASEMAP_PROVIDERS:
        raise ValueError(
            f'Unknown basemap provider {provider!r}; must be one of '
            f'{sorted(_BASEMAP_PROVIDERS)}.'
        )
    tile_provider = _BASEMAP_PROVIDERS[provider]
    return BitmapTileLayer(
        data=tile_provider.build_url(x='{x}', y='{y}', z='{z}'),
        tile_size=256,
        max_requests=-1,
        min_zoom=0,
        max_zoom=tile_provider.get('max_zoom', 19),
        opacity=opacity,
    )


def get_admin_boundary_layer(
    admin_id=None,
    level=None,
    recipe=None,
    *,
    gdf: gpd.GeoDataFrame | None = None,
    elevation: float | np.ndarray | None = None,
    color='white',
    width: float = 3.0,
    style: str = 'dotted',
    dash_array: tuple[int, int] | list[int] | None = None,
    width_units: str = 'pixels',
    opacity: float = 1.0,
    fill_color='white',
    fill_opacity: float = 0.118,
    mode: str = 'floating_line',
) -> PathLayer | PolygonLayer:
    """Get a lonboard layer representing administrative boundary outlines.

    Parameters
    ----------
    admin_id : str, AdminId, or sequence, optional
        Administrative unit ID(s) to fetch boundaries for.
    level : int, optional
        Administrative level (e.g. 1, 2, 3, 4).
    recipe : str, optional
        Recipe ID to load geometries from (e.g. ``'US_admin-census-2021_admin4'``).
    gdf : GeoDataFrame, optional
        Pre-loaded GeoDataFrame of admin boundaries. If provided, `admin_id`,
        `level`, and `recipe` are ignored.
    elevation : float, array-like, or None, optional
        Elevation in meters at which the boundary outlines float (e.g. the 99th
        percentile height of an underlying land value terrain).
    color : str or sequence of int
        Line color (e.g. ``'white'``, ``'#ffffff'``, ``[255, 255, 255, 255]``).
    width : float
        Line stroke width.
    style : {'dotted', 'dashed', 'solid'}
        Line pattern style. Ignored if `dash_array` is specified. Only used in
        'floating_line' mode.
    dash_array : sequence of int, optional
        Explicit dash array (e.g. ``(3, 3)`` for dotted, ``(6, 6)`` for dashed).
    width_units : {'pixels', 'meters'}
        Units for `width`.
    opacity : float
        Line/outline opacity in [0.0, 1.0].
    fill_color : str or sequence of int
        Fence face/wall color (e.g. ``'white'``, ``'#ffffff'``, or RGBA list).
        Only used in 'fence' mode.
    fill_opacity : float
        Fence face/wall opacity in [0.0, 1.0]. Only used in 'fence' mode.
    mode : {'floating_line', 'fence'}
        Visual style mode:

        - ``'floating_line'``: A single 3D line floating at `elevation`.
        - ``'fence'``: A 3D fence with vertical walls filled with `fill_color` at
          `fill_opacity`, raised to `elevation` (default 10m if elevation is None).

    Returns
    -------
    lonboard.PathLayer or lonboard.PolygonLayer
    """
    if gdf is None:
        gdf = get_admin(admin_id, level=level, recipe=recipe, geom=True)

    rgba_color = _parse_line_color(color, opacity=opacity)
    rgba_fill_color = _parse_line_color(fill_color, opacity=fill_opacity)

    if mode == 'fence':
        # Raise walls to elevation (default to 10.0m if not specified)
        wall_height = 10.0 if elevation is None else elevation

        # Buffer the boundary to create a very narrow 3D wall footprint
        # Project to EPSG:3857 to buffer in meters, then project back
        boundary_gdf = gdf.copy()
        boundary_gdf.geometry = (
            gdf.geometry.boundary.to_crs('EPSG:3857').buffer(0.05).to_crs(gdf.crs)
        )

        return PolygonLayer.from_geopandas(
            boundary_gdf,
            extruded=True,
            filled=True,
            wireframe=True,
            get_elevation=wall_height,
            get_fill_color=rgba_fill_color,
            get_line_color=rgba_color,
        )

    # Default 'floating_line' mode using PathLayer for correct style and width
    extensions = []
    extra_kwargs = {}

    if dash_array is None and style != 'solid':
        if style == 'dotted':
            dash_array = (3, 3)
        elif style == 'dashed':
            dash_array = (6, 6)

    if dash_array is not None:
        extensions.append(PathStyleExtension(dash=True))
        extra_kwargs['get_dash_array'] = list(dash_array)

    boundary_gdf = gdf.copy()
    boundary_gdf.geometry = gdf.geometry.boundary

    if elevation is not None:
        import shapely

        boundary_gdf.geometry = shapely.force_3d(boundary_gdf.geometry, z=elevation)

    return PathLayer.from_geopandas(
        boundary_gdf,
        get_color=rgba_color,
        get_width=width,
        width_units=width_units,
        extensions=extensions,
        **extra_kwargs,
    )


def _parse_line_color(color, opacity: float = 1.0):
    alpha = int(round(min(max(opacity, 0.0), 1.0) * 255))
    import matplotlib.colors as mcolors

    try:
        rgba = mcolors.to_rgba(color)
        r = int(round(rgba[0] * 255))
        g = int(round(rgba[1] * 255))
        b = int(round(rgba[2] * 255))
        return [r, g, b, alpha]
    except Exception:
        pass

    if isinstance(color, (list, tuple, np.ndarray)):
        color_list = list(color)
        if len(color_list) == 3:
            return [*color_list, alpha]
        elif len(color_list) == 4:
            return color_list
    return [255, 255, 255, alpha]
