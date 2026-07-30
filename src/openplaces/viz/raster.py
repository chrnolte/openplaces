"""CPU-only raster rendering of large entity datasets.

Rasterizes entity polygons server-side to a fixed-resolution image via
Datashader, for headless/CPU-only environments that can't meet
:func:`openplaces.viz.interactive.show_entities_interactive`'s WebGL2
client requirement.
"""

import warnings

import datashader as ds
import datashader.transfer_functions as tf
import pandas as pd
from PIL import Image

from openplaces.io.readers import get_admin, get_entities
from openplaces.recipe import get_recipe_by_id
from openplaces.viz.colors import (
    MISSING_LABEL,
    RESERVED_NEUTRAL_COLOR,
    resolve_category_colors,
)
from openplaces.viz.legend import categorical_legend_entries, draw_legend

DEFAULT_CMAP = ['#e8e8e8', '#5a9e6f']
DEFAULT_BOUNDARY_COLOR = '#000000'


def show_entities_raster(
    recipe,
    admin_id=None,
    bbox: tuple[float, float, float, float] | None = None,
    color_by: str | None = None,
    plot_width: int = 900,
    plot_height: int = 600,
    simplified: bool = False,
    missing: str = 'raise',
    legend: bool = True,
    show_boundaries: bool = False,
    boundary_admin_level: int = 3,
    boundary_color: str = DEFAULT_BOUNDARY_COLOR,
) -> Image.Image:
    """Rasterize entity polygons to a fixed-resolution image via Datashader.

    Loads the recipe's output through :func:`openplaces.io.readers.get_entities`
    and aggregates it with a single `datashader.Canvas.polygons` call — no
    Dask, no chunked aggregation; the whole (bbox-filtered) selection is
    rasterized in one pass, matching the scale this codebase already handles
    in memory elsewhere (`get_entities` itself concatenates a recipe's
    per-admin-unit files eagerly). For datasets too large to hold in RAM at
    once, chunked aggregation across files would need to be added separately.

    Parameters
    ----------
    recipe : str or dict
        Recipe that defines the entity, passed to `get_entities`.
    admin_id : str, AdminId, or sequence, optional
        Administrative unit(s) to load. Defaults to the recipe admin ID.
    bbox : tuple of (minx, miny, maxx, maxy), optional
        Spatial bounding box filter in EPSG:4326, forwarded to `get_entities`.
    color_by : str, optional
        Column to aggregate by. Categorical columns are rendered with
        `datashader.by`, colored via
        `openplaces.viz.colors.resolve_category_colors` (curated colors from
        `openplaces.viz.colors.CATEGORY_COLORS` where available, a
        deterministic per-label color for everything else -- never a single
        shared fallback, so the same label always gets the same color
        regardless of admin-unit scope or whether it's this track or
        `openplaces.viz.interactive.show_entities_interactive` rendering
        it). Numeric columns are rendered as a mean-value heatmap. If None,
        renders a plain density (count) heatmap.
    plot_width, plot_height : int
        Output image resolution in pixels.
    simplified : bool
        If True, load simplified geometries (``geom='simplified'``) instead
        of full-resolution ones. Raster output doesn't need per-vertex
        fidelity, so this is worth enabling when a ``_geo_simplified``
        sidecar exists for the recipe — currently only written for admin
        geometries, not footprint/building recipes, hence the default of
        False rather than True.
    missing : {'raise', 'warn', 'ignore'}
        How to handle missing output files, forwarded to `get_entities`.
    legend : bool
        If True and `color_by` is given, draw a legend box into the
        lower-right corner of the returned image. No legend is drawn when
        `color_by` is None (a plain density heatmap has no discrete value
        to key a legend to).
    show_boundaries : bool
        If True, overlay administrative boundary lines (e.g. county lines)
        on top of the rasterized entities, rendered with their own
        `datashader.Canvas.line` pass on the same `x_range`/`y_range` so
        they align pixel-for-pixel with the entity raster, then stacked
        over it with `datashader.transfer_functions.stack`.
    boundary_admin_level : int
        Administrative level for `show_boundaries` (3 = county). Boundaries
        are loaded via `openplaces.io.readers.get_admin` for the same
        `admin_id` passed to this function (or the recipe's own admin ID
        if `admin_id` is None), so a state-level call draws every county
        line within it while a single-county call draws just its own
        outline.
    boundary_color : str
        Hex color for `show_boundaries` lines.

    Returns
    -------
    PIL.Image.Image
        Displays directly in Jupyter.
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
            '(Canvas.polygons only rasterizes Polygon/MultiPolygon).',
            stacklevel=2,
        )
        gdf = gdf[is_polygonal]

    if gdf.empty:
        raise ValueError(
            f'No polygon geometries to render for admin_id={admin_id!r}'
            f'{f", bbox={bbox!r}" if bbox is not None else ""} -- '
            'the selection returned zero rows.'
        )

    # Canvas.polygons rasterizes raw x/y with no notion of projection; data
    # at rest is EPSG:4326 (lon/lat degrees), which visibly distorts aspect
    # ratio away from the equator. Reproject to match the convention already
    # used for basemap-style plots in openplaces.viz.maps.
    gdf = gdf.to_crs(epsg=3857)

    # Pin x_range/y_range explicitly (rather than letting Canvas auto-range
    # per call) so the boundary-line pass below rasterizes on the exact same
    # grid as the entity fill and the two stack pixel-for-pixel.
    xmin, ymin, xmax, ymax = gdf.total_bounds
    canvas = ds.Canvas(
        plot_width=plot_width,
        plot_height=plot_height,
        x_range=(xmin, xmax),
        y_range=(ymin, ymax),
    )

    legend_kwargs = None
    if color_by is None:
        agg = canvas.polygons(gdf, geometry='geometry', agg=ds.count())
        img = tf.shade(agg, cmap=DEFAULT_CMAP, how='eq_hist')
    elif pd.api.types.is_numeric_dtype(gdf[color_by]):
        values = gdf[color_by]
        agg = canvas.polygons(gdf, geometry='geometry', agg=ds.mean(color_by))
        img = tf.shade(agg, cmap=DEFAULT_CMAP, how='eq_hist')
        finite = values[pd.notna(values)]
        if not finite.empty:
            legend_kwargs = {
                'numeric': True,
                'vmin': finite.min(),
                'vmax': finite.max(),
                'low_color': DEFAULT_CMAP[0],
                'high_color': DEFAULT_CMAP[-1],
            }
    else:
        img, legend_kwargs = _shade_categorical(canvas, gdf, color_by)

    if show_boundaries:
        img = tf.stack(
            img,
            _boundary_line_image(
                canvas, recipe, admin_id, boundary_admin_level, boundary_color
            ),
        )

    pil_img = img.to_pil()

    if legend and legend_kwargs is not None:
        pil_img = draw_legend(pil_img, color_by, **legend_kwargs)

    return pil_img


def _boundary_line_image(canvas: ds.Canvas, recipe, admin_id, level: int, color: str):
    """Rasterize administrative boundary lines onto `canvas`'s exact grid."""
    if admin_id is None:
        recipe_dict = get_recipe_by_id(recipe) if isinstance(recipe, str) else recipe
        admin_id = recipe_dict['admin_id']

    boundaries = get_admin(admin_id, level=level, geom=True).to_crs(epsg=3857)
    boundaries = boundaries.assign(geometry=boundaries.geometry.boundary)

    agg = canvas.line(boundaries, geometry='geometry')
    return tf.shade(agg, cmap=[color, color], how='linear')


def _shade_categorical(canvas: ds.Canvas, gdf, color_by: str):
    """Render a categorical column; returns ``(image, legend_kwargs)``."""
    palette = resolve_category_colors(gdf[color_by], col_name=color_by)

    # ds.by requires a categorical column with every rendered value present
    # as a category, and tf.shade's color_key must cover every category
    # present in the aggregation or it raises KeyError. Stringified to match
    # resolve_category_colors' str(label) keys.
    cat_col = (
        gdf[color_by]
        .astype('object')
        .fillna(MISSING_LABEL)
        .astype(str)
        .astype('category')
    )
    gdf = gdf.assign(**{color_by: cat_col})
    agg = canvas.polygons(gdf, geometry='geometry', agg=ds.by(color_by, ds.count()))

    color_key = {
        cat: palette.get(cat, RESERVED_NEUTRAL_COLOR) for cat in cat_col.cat.categories
    }
    image = tf.shade(agg, color_key=color_key, how='eq_hist')
    entries = categorical_legend_entries(cat_col, palette)
    return image, {'entries': entries}
