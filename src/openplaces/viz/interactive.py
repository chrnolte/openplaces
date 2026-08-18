"""GPU-accelerated interactive rendering of large entity datasets.

Uses Lonboard/deck.gl, rendered client-side in the browser over WebGL2.
"""

import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
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
# Matches openplaces.viz.colors.continuous_to_rgba's own defaults --
# kept as
# a literal here (rather than imported) since _numeric_to_rgba below
# reimplements that ramp directly rather than calling
# continuous_to_rgba.
NUMERIC_RAMP = ('#e5e7db', '#5a2828')

# Free, keyless XYZ tile providers spanning visually distinct basemap
# styles
# for use as the ground plane under an extruded scene (e.g.
# openplaces.viz.terrain) -- Map's own basemap_style only offers
# abstract
# CARTO vector styles, no photographic/topo raster option, so a raster
# tile
# layer is used instead. All confirmed keyless via
# provider.requires_token().
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


def set_camera_pitch(
    map_widget: Map,
    pitch: float | None = None,
    max_pitch: float = 85.0,
    min_pitch: float | None = None,
):
    """Raise a lonboard map's camera pitch ceiling, and keep it raised.

    lonboard's `MapView` caps the camera at `max_pitch=60` out of the box
    (deck.gl's convention: pitch 0 is straight down, larger tilts toward the
    horizon) — too shallow to read topography or extruded height edge-on.
    Raising it on the map's own `view_state` works, but only until the first
    camera move, which is why doing it once is not enough:

    deck.gl echoes the camera *pose* back to Python on every interaction —
    longitude, latitude, zoom, pitch, bearing — but not the *constraints*,
    which it treats as configuration rather than state. lonboard rebuilds
    the view state from that echo with `MapViewState(**echoed)`, so every
    field the frontend did not send falls back to its dataclass default, and
    `max_pitch` silently returns to 60. The raised ceiling therefore
    survives the initial render and vanishes the moment the user drags. It
    is not visible in Python either — the widget reports whatever came back
    last.

    This sets the constraint once and installs a `view_state` observer that
    re-applies it whenever a round-trip drops it. `pitch` (the starting
    tilt) is deliberately applied only once: re-asserting it on every change
    would fight the user's own dragging, whereas the ceiling is a constraint
    they never set by hand.

    Parameters
    ----------
    map_widget : lonboard.Map
        Map to constrain. Modified in place; also returned for chaining.
    pitch : float, optional
        Starting tilt in degrees, applied once. `None` (default) leaves the
        current tilt alone.
    max_pitch : float
        Pitch ceiling in degrees. Defaults to 85 rather than 90 because
        deck.gl's Web-Mercator projection degrades as the camera approaches
        true horizontal — the ground plane stretches to the horizon and
        starts z-fighting — so 90 is unusable in practice.
    min_pitch : float, optional
        Pitch floor in degrees. `None` (default) leaves it unmanaged.

    Returns
    -------
    lonboard.Map
        The same widget.
    """
    # Imported here, not at module scope: this is the only function that
    # needs the view-state types, and `interactive` is already lazily
    # imported for the sake of installs without the `viz-fast` extra.
    from dataclasses import replace

    from lonboard.view_state import MapViewState

    def _constrain(view_state, include_pitch: bool):
        changes = {'max_pitch': max_pitch}
        if min_pitch is not None:
            changes['min_pitch'] = min_pitch
        if include_pitch and pitch is not None:
            changes['pitch'] = pitch
        return replace(view_state, **changes)

    current = map_widget.view_state
    if not isinstance(current, MapViewState):
        warnings.warn(
            f'Cannot set camera pitch on a {type(current).__name__} view state; '
            'only MapViewState carries pitch constraints. Leaving it unchanged.',
            stacklevel=2,
        )
        return map_widget

    map_widget.view_state = _constrain(current, include_pitch=True)

    def _reassert(change):
        new = change['new']
        # Guard against recursing forever: assigning inside the observer
        # fires it again, and the second pass must find nothing to do.
        if not isinstance(new, MapViewState) or new.max_pitch == max_pitch:
            return
        map_widget.view_state = _constrain(new, include_pitch=False)

    map_widget.observe(_reassert, names='view_state')
    return map_widget


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


def _elevation_module():
    """Import `viz.elevation` lazily; it pulls in rasterio and the DEM stack."""
    from openplaces.viz import elevation as elevation_module

    return elevation_module


def _dem_admin_ids(gdf: gpd.GeoDataFrame, elevation_recipe) -> pd.Series:
    """Each row's admin id truncated to the DEM recipe's own save level.

    `elevation.drape_parcel_elevation` groups by an admin-id column matching
    the DEM's per-admin-unit tiling, because that is how it resolves which
    raster covers a row. An admin-boundary frame from `get_admin` is keyed
    by its own admin id at whatever level was requested, so the mapping is a
    truncation: a level-4 town boundary is covered by its level-3 county's
    DEM. Read the level off the recipe rather than assuming 3, so this keeps
    working if the DEM is ever re-tiled at a different granularity.
    """
    from openplaces.core.schema import AdminId
    from openplaces.recipe import get_recipe_by_id, get_save_admin_level

    dem_recipe = (
        get_recipe_by_id(elevation_recipe)
        if isinstance(elevation_recipe, str)
        else elevation_recipe
    )
    dem_level = get_save_admin_level(dem_recipe)

    try:
        ids = [AdminId(*str(v).split('-')) for v in gdf.index]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Cannot drape these boundaries: elevation_recipe needs each row's "
            'admin id to find the DEM covering it, and this frame is not '
            'indexed by admin id. Pass a `gdf` from `get_admin`, or omit '
            'elevation_recipe.',
        ) from exc

    truncated = []
    for admin, original in zip(ids, gdf.index, strict=True):
        parent = admin.truncate_to_level(dem_level)
        if parent is None:
            raise ValueError(
                f'Admin unit {original!r} is coarser than the level-{dem_level} '
                'tiling the DEM recipe is saved at, so no single DEM covers '
                'it. Drape a '
                'finer admin level, or omit elevation_recipe.',
            )
        truncated.append(str(parent))
    return pd.Series(truncated, index=gdf.index)


def _drape_boundary(
    gdf: gpd.GeoDataFrame,
    elevation_recipe,
    terrain_exaggeration: float,
) -> gpd.GeoDataFrame:
    """Set each boundary vertex's Z from the DEM, exaggerated to match."""
    if elevation_recipe is None:
        return gdf

    elevation_module = _elevation_module()
    draping = gdf.copy()
    column = '_dem_admin_id'
    draping[column] = _dem_admin_ids(draping, elevation_recipe)

    # cache=False is required, not an optimization. The elevation cache
    # is
    # keyed by row index, and is only safe because an entity index
    # encodes a
    # geometry-shape hash (see `geo.ids.get_geo_ids`), so a changed
    # shape
    # gets a different key. An admin id encodes no such thing: the
    # town's
    # own polygon, its boundary line, and the buffered fence footprint
    # are
    # three different geometries all keyed 'US-MA-WO-LA'. Caching would
    # serve whichever was draped first for all three. Boundaries are a
    # handful of rows anyway, so there is nothing to gain.
    draped_geometry, _ = elevation_module.drape_parcel_elevation(
        draping, elevation_recipe, admin_id_column=column, cache=False
    )
    draping = draping.drop(columns=column)
    draping.geometry = gpd.GeoSeries(
        elevation_module.scale_z(draped_geometry.to_numpy(), terrain_exaggeration),
        index=draping.index,
        crs=gdf.crs,
    )
    return draping


def get_terrain_basemap_layer(
    admin_id=None,
    level=None,
    recipe=None,
    *,
    gdf: gpd.GeoDataFrame | None = None,
    elevation_recipe,
    provider: str = 'satellite',
    resolution: float = 20.0,
    terrain_exaggeration: float = 1.0,
    opacity: float = 1.0,
    max_cells: int = 600_000,
    clip: bool = True,
    zoom: int | None = None,
) -> SolidPolygonLayer:
    """Basemap imagery draped over the DEM, as a colored quad mesh.

    `get_basemap_layer` pins its tiles to sea level. That is fine for a flat
    map and wrong for a tilted 3D one: everything else in the scene sits on
    real terrain, so at pitch `p` a feature at height `h` appears displaced
    from its own basemap position by `h * tan(p)`. Over terrain a few
    hundred meters up, viewed near the pitch ceiling, that is a kilometer or
    more — buildings float far from the streets they belong to.

    deck.gl solves this with `TerrainLayer`, but lonboard does not wrap it,
    and its `BitmapLayer` accepts only 2D bounds, so neither can be lifted.
    What lonboard does expose is `SolidPolygonLayer`, which takes arbitrary
    3D geometry and a per-feature color — so the surface is built here as
    one quad per grid cell, each colored by the *area mean* of the basemap
    pixels it covers and carrying its four corners' own sampled elevations.
    Corners are shared between neighbors, so the mesh is continuous rather
    than stepped.

    Averaging (rather than sampling) the pixels is what keeps this legible:
    a road narrower than a cell still darkens every cell it crosses, so the
    network stays continuous instead of breaking into dots. At 20 m cells
    the street network of a New England town remains readable; by 30 m it is
    soft, and by 60 m it is noise.

    This is a mesh, not a texture, so cost scales with area over resolution
    squared — see `resolution` and `max_cells`. It suits one admin unit at a
    time, not a whole state.

    Parameters
    ----------
    admin_id : str, AdminId, or sequence, optional
        Administrative unit ID(s) whose extent to cover.
    level : int, optional
        Administrative level, forwarded to `io.readers.get_admin`.
    recipe : str, optional
        Admin recipe ID, forwarded to `io.readers.get_admin`.
    gdf : geopandas.GeoDataFrame, optional
        Pre-loaded extent polygons. If given, `admin_id`, `level` and
        `recipe` are ignored.
    elevation_recipe : str or dict
        DEM dataset recipe (e.g. ``'US_land-elevation-usgs-3dep'``). Unlike
        the other layer builders this is required — a draped basemap with no
        DEM would just be `get_basemap_layer` at greater expense.
    provider : str
        Basemap style, one of the same names `get_basemap_layer` accepts.
    resolution : float
        Grid cell size in meters. Defaults to 20, the coarsest size at which
        a street network still reads as continuous lines.
    terrain_exaggeration : float
        Multiplier on ground elevation. Must match the value passed to
        `viz.terrain.show_value_terrain_layer` and
        `get_admin_boundary_layer`, or the basemap will sit at a different
        vertical scale than the scene it is meant to align.
    opacity : float
        Fill opacity in [0, 1].
    max_cells : int
        Refuse to build a mesh larger than this many quads, rather than
        hanging the browser. Defaults to 600,000. The error names the
        resolution that would fit.
    clip : bool
        If True (default), drop cells whose center falls outside the extent
        polygons, so the mesh follows the admin unit instead of hanging a
        rectangular slab over its neighbors.
    zoom : int, optional
        Tile zoom level to average down from. Defaults to the level whose
        pixels are about a quarter of a cell. Worth raising by hand: tile
        styles drop detail at low zoom (CartoDB omits residential streets
        below roughly z15), so a level chosen purely on pixel size can
        average down imagery that never drew the roads in the first place.
        Higher zoom means more tiles to fetch.

    Returns
    -------
    lonboard.SolidPolygonLayer
        Add as the *first* layer of a `lonboard.Map`, in place of
        `get_basemap_layer`.
    """
    import contextily as cx

    from openplaces.geo.raster import sample_raster_at_points

    if provider not in _BASEMAP_PROVIDERS:
        raise ValueError(
            f'Unknown basemap provider {provider!r}; must be one of '
            f'{sorted(_BASEMAP_PROVIDERS)}.'
        )
    if gdf is None:
        gdf = get_admin(admin_id, level=level, recipe=recipe, geom=True)

    metric_crs = gdf.estimate_utm_crs()
    metric = gdf.to_crs(metric_crs)
    minx, miny, maxx, maxy = metric.total_bounds

    n_x = max(1, int(np.ceil((maxx - minx) / resolution)))
    n_y = max(1, int(np.ceil((maxy - miny) / resolution)))
    if n_x * n_y > max_cells:
        needed = np.sqrt((maxx - minx) * (maxy - miny) / max_cells)
        raise ValueError(
            f'A {resolution} m mesh over this extent needs {n_x * n_y:,} quads, '
            f'over the {max_cells:,} limit. Use resolution >= {needed:.0f}, a '
            'smaller admin unit, or raise max_cells if the browser can take it.'
        )

    # Corner grid: (n_y + 1) x (n_x + 1), shared between adjacent quads
    # so
    # the surface is continuous.
    edge_x = minx + np.arange(n_x + 1) * resolution
    edge_y = miny + np.arange(n_y + 1) * resolution
    grid_x, grid_y = np.meshgrid(edge_x, edge_y)

    corner_z = _sample_corner_elevation(
        grid_x, grid_y, metric_crs, elevation_recipe, gdf, sample_raster_at_points
    )
    corner_z = _fill_missing_elevation(corner_z) * terrain_exaggeration

    center_x = (edge_x[:-1] + edge_x[1:]) / 2
    center_y = (edge_y[:-1] + edge_y[1:]) / 2
    keep = np.ones((n_y, n_x), dtype=bool)
    if clip:
        centers = gpd.GeoSeries(
            gpd.points_from_xy(*[a.ravel() for a in np.meshgrid(center_x, center_y)]),
            crs=metric_crs,
        )
        extent = metric.geometry.union_all()
        keep = centers.within(extent).to_numpy().reshape(n_y, n_x)
        if not keep.any():
            raise ValueError(
                'No grid cell center falls inside the extent -- the mesh would '
                'be empty. Use a finer `resolution`, or clip=False.'
            )

    colors = _basemap_cell_colors(
        cx, provider, metric, metric_crs, edge_x, edge_y, n_x, n_y, zoom
    )

    rows, cols = np.nonzero(keep)
    quads = [
        shapely.Polygon(
            [
                (grid_x[r, c], grid_y[r, c], corner_z[r, c]),
                (grid_x[r, c + 1], grid_y[r, c + 1], corner_z[r, c + 1]),
                (grid_x[r + 1, c + 1], grid_y[r + 1, c + 1], corner_z[r + 1, c + 1]),
                (grid_x[r + 1, c], grid_y[r + 1, c], corner_z[r + 1, c]),
            ]
        )
        for r, c in zip(rows, cols, strict=True)
    ]
    mesh = gpd.GeoDataFrame(geometry=quads, crs=metric_crs).to_crs(gdf.crs)

    rgba = np.empty((len(rows), 4), dtype=np.uint8)
    rgba[:, :3] = colors[rows, cols]
    rgba[:, 3] = round(min(max(opacity, 0.0), 1.0) * 255)

    return SolidPolygonLayer.from_geopandas(
        mesh,
        extruded=False,
        filled=True,
        wireframe=False,
        get_fill_color=rgba,
    )


def _sample_corner_elevation(
    grid_x, grid_y, metric_crs, elevation_recipe, gdf, sample_raster_at_points
):
    """Sample the DEM at every grid corner, grouped by covering admin unit."""
    from openplaces.io.readers import get_dataset

    corners = gpd.GeoSeries(
        gpd.points_from_xy(grid_x.ravel(), grid_y.ravel()), crs=metric_crs
    )
    dem_ids = _dem_admin_ids(gdf, elevation_recipe).unique()

    z = np.full(corners.shape[0], np.nan)
    for dem_id in dem_ids:
        dem_path = get_dataset(elevation_recipe, admin_id=dem_id)
        if not Path(dem_path).exists():
            _elevation_module()._ingest_missing_dem(
                elevation_recipe, dem_id, Path(dem_path), False
            )
        with rasterio.open(dem_path) as src:
            raster_crs = src.crs
        points = corners.to_crs(raster_crs)
        sampled = sample_raster_at_points(dem_path, points.x, points.y)
        # Several DEMs can cover one extent (an admin unit straddling
        # two
        # counties); each contributes only where it has data, so later
        # rasters fill the previous one's gaps instead of overwriting
        # it.
        z = np.where(np.isnan(z), sampled, z)
    return z.reshape(grid_x.shape)


def _fill_missing_elevation(corner_z: np.ndarray) -> np.ndarray:
    """Replace nodata corners with their nearest sampled neighbor's value.

    A DEM carries nodata over water and outside its own footprint. Left as
    NaN those corners would render at sea level, punching isolated spikes
    hundreds of meters deep through an otherwise smooth mesh -- far more
    visually wrong than the small error of borrowing the nearest real
    elevation, since a nodata pixel here is almost always a pond or river
    surrounded by ground at nearly its own height.
    """
    missing = np.isnan(corner_z)
    if not missing.any():
        return corner_z
    if missing.all():
        raise ValueError(
            'The DEM has no data anywhere over this extent -- every grid '
            'corner sampled nodata. Check that elevation_recipe covers this '
            'admin unit.'
        )
    from scipy import ndimage

    _, nearest = ndimage.distance_transform_edt(
        missing, return_distances=True, return_indices=True
    )
    return corner_z[tuple(nearest)]


def _basemap_cell_colors(
    cx, provider, metric, metric_crs, edge_x, edge_y, n_x, n_y, zoom=None
):
    """Mean basemap RGB per grid cell, from a tile mosaic of the extent.

    Averaging rather than point-sampling is what keeps sub-cell features
    (roads) visible: a road narrower than a cell still darkens every cell it
    crosses.
    """
    web = metric.to_crs(3857)
    west, south, east, north = web.total_bounds

    # Pick the zoom whose pixels are about four times finer than a cell,
    # so
    # each cell averages a real neighborhood rather than one or two
    # pixels.
    # Web-Mercator resolution halves per zoom level from ~156543 m/px at
    # the
    # equator; the latitude factor matters away from it.
    cell_m = (edge_x[-1] - edge_x[0]) / n_x
    if zoom is None:
        geographic_bounds = metric.to_crs(4326).total_bounds
        latitude = np.radians((geographic_bounds[1] + geographic_bounds[3]) / 2)
        target = cell_m / 4
        zoom = int(np.ceil(np.log2(156543.03392 * np.cos(latitude) / target)))
    zoom = int(np.clip(zoom, 1, 19))

    image, extent = cx.bounds2img(
        west, south, east, north, zoom=zoom, source=_BASEMAP_PROVIDERS[provider]
    )
    image = image[:, :, :3].astype(np.float64)
    img_west, img_east, img_south, img_north = extent

    # Cell centers -> fractional pixel coordinates in the mosaic.
    # Averaging
    # is done by bincount over the pixel->cell assignment, which is much
    # faster than slicing a window per cell.
    px_h, px_w = image.shape[:2]
    px_x = np.linspace(img_west, img_east, px_w, endpoint=False)
    px_y = np.linspace(img_north, img_south, px_h, endpoint=False)
    mesh_px_x, mesh_px_y = np.meshgrid(px_x, px_y)

    pixels = gpd.GeoSeries(
        gpd.points_from_xy(mesh_px_x.ravel(), mesh_px_y.ravel()), crs=3857
    ).to_crs(metric_crs)
    col = np.floor((pixels.x.to_numpy() - edge_x[0]) / cell_m).astype(np.int64)
    row = np.floor(
        (pixels.y.to_numpy() - edge_y[0]) / ((edge_y[-1] - edge_y[0]) / n_y)
    ).astype(np.int64)
    inside = (col >= 0) & (col < n_x) & (row >= 0) & (row < n_y)

    flat_cell = (row[inside] * n_x + col[inside]).astype(np.int64)
    counts = np.bincount(flat_cell, minlength=n_x * n_y)
    colors = np.zeros((n_x * n_y, 3))
    for band in range(3):
        total = np.bincount(
            flat_cell, weights=image[:, :, band].ravel()[inside], minlength=n_x * n_y
        )
        colors[:, band] = np.divide(
            total, counts, out=np.zeros_like(total), where=counts > 0
        )
    return colors.reshape(n_y, n_x, 3).astype(np.uint8)


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
    elevation_recipe: str | dict | None = None,
    terrain_exaggeration: float = 1.0,
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
        percentile height of an underlying land value terrain). With
        `elevation_recipe` set this becomes a height *above the terrain*
        rather than above sea level.
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
    elevation_recipe : str or dict, optional
        DEM dataset recipe (e.g. ``'US_land-elevation-usgs-3dep'``) to drape
        the boundary over, so it follows real terrain instead of tracing a
        flat ring at sea level. Each boundary vertex gets its own sampled
        elevation, which matters here more than it does for a compact
        polygon: an admin boundary is long enough to cross hundreds of
        meters of relief, so a flat ring cuts through hillsides at one end
        and floats far above the ground at the other. `None` (default)
        keeps the sea-level behavior.

        Pass the same recipe and `terrain_exaggeration` used for the
        terrain layers it is drawn over — see
        `viz.terrain.show_value_terrain_layer`.
    terrain_exaggeration : float
        Multiplier on real-world ground elevation, matching
        `show_value_terrain_layer`'s argument of the same name. Must match
        the value used there, or the boundary will sit at a different
        vertical scale than the terrain it outlines. Ignored without
        `elevation_recipe`. Defaults to 1 (true-to-scale).

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
        # Drape the wall's footprint, not its top: `get_elevation` then
        # raises a constant-height wall from wherever the ground is, so
        # the fence follows the terrain instead of being sliced by it.
        boundary_gdf = _drape_boundary(
            boundary_gdf, elevation_recipe, terrain_exaggeration
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

    # Default 'floating_line' mode using PathLayer for correct style and
    # width
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
    draped = elevation_recipe is not None
    boundary_gdf = _drape_boundary(boundary_gdf, elevation_recipe, terrain_exaggeration)

    if elevation is not None:
        import shapely

        if draped:
            # `elevation` is a clearance above the draped ground, so it
            # has
            # to be added to each vertex's own terrain Z rather than
            # replacing it -- force_3d would leave an already-3D
            # geometry
            # untouched and silently drop the offset.
            boundary_gdf.geometry = gpd.GeoSeries(
                _elevation_module().add_z_offset(
                    boundary_gdf.geometry.to_numpy(),
                    np.broadcast_to(
                        np.asarray(elevation, dtype=float), (len(boundary_gdf),)
                    ),
                ),
                index=boundary_gdf.index,
                crs=boundary_gdf.crs,
            )
        else:
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
