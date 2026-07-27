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

from openplaces.io.readers import get_entities
from openplaces.viz.colors import match_palette

DEFAULT_CMAP = ['#e8e8e8', '#5a9e6f']
DEFAULT_FALLBACK_COLOR = '#999999'


def show_entities_raster(
    recipe,
    admin_id=None,
    bbox: tuple[float, float, float, float] | None = None,
    color_by: str | None = None,
    plot_width: int = 900,
    plot_height: int = 600,
    simplified: bool = False,
    missing: str = 'raise',
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
        Column to aggregate by. Categorical columns are matched against
        `openplaces.viz.colors.CATEGORY_COLORS` via `match_palette` and
        rendered with `datashader.by`; values without a palette match
        (including missing data) fall back to `DEFAULT_FALLBACK_COLOR`.
        Numeric columns are rendered as a mean-value heatmap. If None,
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

    # Canvas.polygons rasterizes raw x/y with no notion of projection; data
    # at rest is EPSG:4326 (lon/lat degrees), which visibly distorts aspect
    # ratio away from the equator. Reproject to match the convention already
    # used for basemap-style plots in openplaces.viz.maps.
    gdf = gdf.to_crs(epsg=3857)

    canvas = ds.Canvas(plot_width=plot_width, plot_height=plot_height)

    if color_by is None:
        agg = canvas.polygons(gdf, geometry='geometry', agg=ds.count())
        img = tf.shade(agg, cmap=DEFAULT_CMAP, how='eq_hist')
    elif pd.api.types.is_numeric_dtype(gdf[color_by]):
        agg = canvas.polygons(gdf, geometry='geometry', agg=ds.mean(color_by))
        img = tf.shade(agg, cmap=DEFAULT_CMAP, how='eq_hist')
    else:
        img = _shade_categorical(canvas, gdf, color_by)

    return img.to_pil()


def _shade_categorical(canvas: ds.Canvas, gdf, color_by: str):
    counts = gdf[color_by].value_counts(dropna=True)
    palette = match_palette(counts.index, col_name=color_by, weights=counts.to_numpy())

    # ds.by requires a categorical column with every rendered value present
    # as a category, and tf.shade's color_key must cover every category
    # present in the aggregation or it raises KeyError.
    cat_col = gdf[color_by].astype('object').fillna('(missing)').astype('category')
    gdf = gdf.assign(**{color_by: cat_col})
    agg = canvas.polygons(gdf, geometry='geometry', agg=ds.by(color_by, ds.count()))

    if palette is None:
        return tf.shade(agg, how='eq_hist')

    color_key = {
        cat: palette.get(cat, DEFAULT_FALLBACK_COLOR) for cat in cat_col.cat.categories
    }
    return tf.shade(agg, color_key=color_key, how='eq_hist')
