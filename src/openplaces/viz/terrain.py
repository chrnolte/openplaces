"""3D value-per-area "terrain" rendering.

Extrudes and colors entity polygons by a value column normalized to each
polygon's own area, e.g. building value per square foot or parcel land value
per hectare. Uses Lonboard/deck.gl, rendered client-side over WebGL2.
"""

import warnings
from typing import NamedTuple

import geopandas as gpd
import matplotlib as mpl
import numpy as np
import pandas as pd
import shapely
from lonboard import PolygonLayer, SolidPolygonLayer

from openplaces.core.constants import M2_PER_AREA_UNIT
from openplaces.geo.polygon import resolve_area
from openplaces.io.readers import get_entities
from openplaces.io.transform import convert_area_unit
from openplaces.viz.colors import adjust_brightness, continuous_to_rgba

_EQUAL_AREA_CRS = 'epsg:6933'

# Recovered from calibrating a real_height_column='n_stories' reference on
# real Boston-area CHEER footprint data: at ~0.00207 m per $1/m2, the median
# building rendered at ~1x its real story height (physically grounded, not
# an arbitrary target). Rounded up slightly and hardcoded so every call
# shares the exact same $/m2 -> meters conversion by default, with no
# calibration step needed -- pass `elevation_scale=` explicitly to override.
DEFAULT_ELEVATION_SCALE = 0.0025  # meters per $1/m2 of value


class TerrainLayer(NamedTuple):
    """Result of `show_value_terrain_layer`.

    Attributes
    ----------
    layer : lonboard.SolidPolygonLayer
        The extruded, colored layer.
    elevation_scale : float
        The `elevation_scale` argument actually used (the explicit value, or
        `DEFAULT_ELEVATION_SCALE` if not overridden) — returned for
        convenience/debugging, not something that needs reading back into a
        companion layer's own call: both default to the same constant.
    value_range : tuple of float
        Min/max of the raw (un-logged) value-per-area actually rendered, in
        `area_unit` — color is always log1p-transformed regardless of
        `log_height`, so build a matching colorbar with
        `np.log1p(value_range)` as its bounds.
    gdf : geopandas.GeoDataFrame
        The filtered geometries actually rendered, row-aligned with
        `rendered_elevation` — pass both to a subsequent call's `stack_on`
        to stack another layer on top of this one.
    rendered_elevation : numpy.ndarray
        Each row's actual rendered elevation in meters
        (`height_value * elevation_scale`, before any `stack_on` base
        offset of its own).
    outline_layer : lonboard.PolygonLayer or None
        A companion wireframe-only layer outlining each polygon in a darker
        shade of its own fill color, or `None` when `outline_width=0`. Add
        it to the `Map` alongside `layer` if present.
    clipped_layer : lonboard.SolidPolygonLayer or None
        A companion wireframe-only layer marking rows whose *height* was
        capped by `height_clip_percentile` (in `clipped_color`, regardless
        of `outline_width`) — `None` when `mark_clipped=False`,
        `log_height=True`, `height_clip_percentile=None`, or no row
        actually exceeded the cap. Add it to the `Map` alongside `layer` if
        present.
    """

    layer: SolidPolygonLayer
    elevation_scale: float
    value_range: tuple[float, float]
    gdf: gpd.GeoDataFrame
    rendered_elevation: np.ndarray
    outline_layer: PolygonLayer | None
    clipped_layer: SolidPolygonLayer | None


def show_value_terrain_layer(
    recipe,
    admin_id,
    value_column: str,
    area_unit: str | None = None,
    unit_system: str = 'metric',
    area_m2_column: str | None = None,
    log_height: bool = False,
    cmap: str | mpl.colors.Colormap = 'price',
    vmin: float | None = None,
    vmax: float | None = None,
    height_clip_percentile: float | None = 99.9,
    height_clip_value: float | None = None,
    elevation_scale: float = DEFAULT_ELEVATION_SCALE,
    stack_on: TerrainLayer | None = None,
    brightness: float = 1.0,
    outline_width: float = 0,
    outline_darken: float = 0.6,
    mark_clipped: bool = False,
    clipped_color: str = '#00ffff',
    clipped_fill_rgba: tuple[int, int, int, int] | None = (255, 0, 255, 63),
    alpha: float = 1.0,
    missing: str = 'raise',
    silent: bool = False,
) -> TerrainLayer:
    """Render entity polygons extruded and colored by value per unit area.

    Both the fill color and the extrusion height encode the same underlying
    quantity: `value_column` divided by each polygon's own area (never a raw
    total) — e.g. building value per square foot, or parcel land value per
    hectare. Color (and the reported `value_range`) use `area_unit` — a
    display choice, e.g. buildings in $/sqft and land in $/ha, each
    independently min-max normalized for its own layer by default (or
    pinned explicitly via `vmin`/`vmax`), so the display unit choice
    doesn't need to match across layers. *Height*, however, is always
    computed on a common $/m² basis internally regardless of `area_unit`:
    since a rendered polygon's true footprint area is a physical quantity
    (square meters, effectively, via the map projection), `area_m2 *
    height_value` recovers `value` with the *same* proportionality constant
    for every layer only if `height_value` itself is $/m²-based. Using
    `area_unit`-based numbers directly for height instead (e.g. mixing
    $/sqft for buildings with $/ha for land, a ~107,639x unit artifact)
    would break the correspondence between rendered "volume" (footprint
    area x height) and actual value across layers — `elevation_scale`
    defaults to the *same* constant (`DEFAULT_ELEVATION_SCALE`) for every
    call specifically so that correspondence holds without having to
    manage it by hand. Color has no such constraint — each layer's color is
    independently normalized — so it stays in `area_unit` for readability.

    Color is always `log1p`-transformed (monetary value-per-area
    distributions have long high-value tails, and a colormap needs that
    compression to stay legible); height is `log1p`-transformed only when
    `log_height` is True — the *actual* per-area value by default. Combine
    two calls' layers (e.g. one per entity) in a single
    `lonboard.Map([layer_a, layer_b])`, optionally passing the first call's
    `TerrainLayer` as the second call's `stack_on` to physically stack one
    on top of the other rather than have both extrude from elevation 0.

    Parameters
    ----------
    recipe : str or dict
        Recipe that defines the entity, passed to `get_entities`.
    admin_id : str, AdminId, or sequence
        Administrative unit(s) to load.
    value_column : str
        Column to divide by area and map to color/height.
    area_unit : str, optional
        Display area unit for *color* and the reported `value_range` (e.g.
        'sqft', 'ha') — one of `core.constants.M2_PER_AREA_UNIT`'s keys.
        When omitted (None, the default), defaults are resolved from
        `unit_system` and detected area columns: `'ha'` for parcel-shaped
        data (only `area_ha` present), `'m2'` (metric) or `'sqft'`
        (imperial) otherwise. Height always uses a $/m² basis regardless of
        this choice; see above.
    unit_system : str
        Control the imperial/metric default when `area_unit` is omitted
        (either 'metric' for m2/ha, or 'imperial' for sqft/ac). Defaults to
        'metric'; ignored when `area_unit` is explicitly passed, which always
        takes precedence. Ensures defaults never mix unit systems across
        calls with this parameter.
    area_m2_column : str, optional
        An existing column already holding each polygon's area in square
        meters (always m² — deliberately not named/typed to accept
        `area_unit`-flavored columns, since mixing units here would silently
        break the height/volume guarantee described above), reused instead
        of recomputing via `resolve_area`. When present, takes precedence
        over auto-detected `area_m2` or `area_ha` columns; this explicit-pass
        contract is unchanged from before the rename. Formerly referenced
        column `'m2'`, now references the renamed `'area_m2'` or an explicit
        custom area column.
    log_height : bool
        Log1p-transform the per-area value before mapping to elevation.
        Color is always log1p-transformed regardless of this flag. With
        `log_height=False` (the default), a few extreme value-per-area
        outliers can dominate the elevation range; see
        `height_clip_percentile` to cap them.
    cmap : str or matplotlib.colors.Colormap
        Colormap for the fill color, applied to the log1p-transformed
        per-area value. A name in `openplaces.viz.colors.DIVERGING_COLORMAPS`
        (default `'price'`) or a `Colormap` instance.
    vmin, vmax : float, optional
        Explicit lower/upper bounds for the color scale, in the same raw
        (un-logged) `area_unit` terms as `value_range` — log1p-transformed
        internally before being passed on (color is always log1p-transformed
        regardless of `log_height`; see above). Default (`None`) is each
        call's own observed min/max, matching `value_range`; pass both
        explicitly to pin two layers (or repeated calls over different
        admin units) to the same color scale instead of each auto-ranging
        independently.
    height_clip_percentile : float, optional
        Only applies when `log_height=False`: clip the $/m² value used for
        *height* (not color, which stays log-scaled and needs no clipping)
        at this percentile before rendering. Real value-per-area data has
        extreme enough outliers (observed: a parcel at ~$107M/ha against a
        county median of ~$340K/ha) that an uncapped linear height renders
        that one row at astronomical, unusable elevations. Deliberately
        high (99.9, not a rounder 99.0) so only the genuine
        extreme-extremes get capped, not the top percentile of otherwise-
        ordinary rows; see `mark_clipped` for flagging exactly which rows
        this affects. Set to None to disable.
    height_clip_value : float, optional
        Only applies when `log_height=False`: clip the value used for *height*
        at this specific value cutoff (expressed in raw, un-logged `area_unit`
        terms, e.g. $/sqft or $/ha) before rendering. Takes precedence over
        `height_clip_percentile` when explicitly provided. Set to None to disable.
    elevation_scale : float
        Elevation multiplier: meters per $1/m² of `value_column` (always
        $/m² internally, regardless of `area_unit` — see above). Defaults
        to `DEFAULT_ELEVATION_SCALE`, recovered from real data (see its
        definition) so every call shares the same conversion out of the
        box; pass an explicit value to override, e.g. to exaggerate one
        entity's layer relative to another's, or to match a specific
        real-world reference height by hand.
    stack_on : TerrainLayer, optional
        A previously built layer to stack this one on top of: each row here
        is matched to whichever `stack_on.gdf` row it overlaps with the
        largest area (or, when this call's own entity already carries a
        column matching `stack_on.gdf`'s index name — e.g. footprints'
        `parcel_id` against a parcels layer, whose index is itself
        `parcel_id` — an id-based lookup reusing the harmonizer's own
        already-resolved link instead, orders of magnitude faster than a
        live spatial join at large scale; see `_match_largest_overlap`).
        Its base is set to that row's `stack_on.rendered_elevation` (0 for
        rows with no match) via a 3D polygon Z-coordinate — deck.gl adds
        `get_elevation` on top of a polygon's own Z when one is present, so
        this layer's rendered zmin lines up with the matched `stack_on`
        row's zmax. Requires that base layer's `TerrainLayer` to already
        exist (built in an earlier call), for its geometry/
        `rendered_elevation` to match against.
    brightness : float
        HSV-value multiplier applied to the fill color after `cmap`
        (`openplaces.viz.colors.adjust_brightness`) — `1.0` leaves colors
        unchanged; use this to make two layers rendered together (e.g. land
        vs. buildings) distinguishable by shade in addition to colormap
        family, e.g. one call with `brightness=0.75` (darker) and another
        with `brightness=1.15` (brighter).
    outline_width : float
        Width in pixels of a companion wireframe outline layer (see
        `outline_layer` under Returns), colored a darker shade of each
        polygon's own fill color. Defaults to `0` (no outline layer built at
        all): deck.gl can't combine solid fill and wireframe on one
        `SolidPolygonLayer` (they're mutually exclusive render modes), so
        any nonzero value means building and uploading a *second* layer with
        the same geometry — real overhead this function's general-purpose
        default should not assume away, since it may be used at far larger
        (potentially million-row) scale than a single-county demo. Set
        explicitly where the polygon count makes the cost negligible.
    outline_darken : float
        `brightness` factor (see `adjust_brightness`) applied to each row's
        *fill* color to derive its outline color, when `outline_width > 0`.
    mark_clipped : bool
        Add a companion wireframe layer (see `clipped_layer` under Returns)
        marking, in `clipped_color`, exactly which rows had their *height*
        capped by `height_clip_percentile`. Defaults to `False` since the
        fill recoloring via `clipped_fill_rgba` (50% magenta by default)
        already signals clipping clearly. Set to `True` to also render a
        cyan wireframe outline as an additional marker. Independent of
        `outline_width`: shown whenever there are clipped rows and this flag
        is True, regardless of whether the general per-row outline is on.
        Only meaningful when `log_height=False` and `height_clip_percentile`
        is not None.
    clipped_color : str
        Hex color for `clipped_layer` wireframe outline. Defaults to a
        saturated cyan chosen to not collide with any `DIVERGING_COLORMAPS`
        entry's own range.
    clipped_fill_rgba : tuple of (int, int, int, int)
        RGBA color for the fill color of clipped rows (rows whose height
        exceeded `height_clip_percentile`), as a tuple of (R, G, B, A) with
        values 0-255. Defaults to (255, 0, 255, 128) — 50% transparent
        magenta — to signal that height was capped while preserving the
        underlying colormap for comparison. Set to `None` to disable fill-
        color capping and only use the wireframe outline (when
        `mark_clipped=True`).
    alpha : float
        Fill opacity, 0-1 (matplotlib convention — 0 fully transparent, 1
        fully opaque; converted internally to the 0-255 scale
        `continuous_to_rgba` expects). Defaults to `1.0` since overlapping
        extruded faces compound any translucency and read as visibly
        see-through well before `alpha` reaches 0; lower it deliberately
        for intentional translucency.
    missing : {'raise', 'warn', 'ignore'}
        How to handle missing output files, forwarded to `get_entities`.
    silent : bool
        If True, suppress warnings about zero/missing area, non-polygonal
        geometries, and missing values. Defaults to False.

    Returns
    -------
    TerrainLayer
    """
    gdf = get_entities(recipe, admin_id=admin_id, geom=True, missing=missing)

    if area_unit is None:
        is_parcel_like = 'area_ha' in gdf.columns and 'area_m2' not in gdf.columns
        if unit_system == 'metric':
            area_unit = 'ha' if is_parcel_like else 'm2'
        elif unit_system == 'imperial':
            area_unit = 'ac' if is_parcel_like else 'sqft'
        else:
            raise ValueError(
                f"unit_system must be 'metric' or 'imperial', got {unit_system!r}."
            )

    if area_unit not in M2_PER_AREA_UNIT:
        raise ValueError(
            f'Unsupported area_unit {area_unit!r}; must be one of '
            f'{sorted(M2_PER_AREA_UNIT)}.'
        )

    is_polygonal = gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])
    if not is_polygonal.all():
        if not silent:
            warnings.warn(
                f'Dropping {(~is_polygonal).sum()} row(s) with non-polygon geometry '
                '(SolidPolygonLayer only renders Polygon/MultiPolygon).',
                stacklevel=2,
            )
        gdf = gdf[is_polygonal]

    area_m2 = _area_m2(gdf, area_m2_column)
    has_area = area_m2 > 0
    if not has_area.all():
        if not silent:
            warnings.warn(
                f'Dropping {(~has_area).sum()} row(s) with zero/missing area.',
                stacklevel=2,
            )
        gdf = gdf[has_area]
        area_m2 = area_m2[has_area]

    n_missing_value = gdf[value_column].isna().sum()
    if n_missing_value:
        if not silent:
            warnings.warn(
                f'Setting {n_missing_value} row(s) with missing {value_column!r} to '
                '0 (e.g. accessory structures with no assessed value) so they still '
                'render, at the bottom of the color/height scale.',
                stacklevel=2,
            )

    value = gdf[value_column].fillna(0).to_numpy(dtype=float)
    area_m2 = np.asarray(area_m2, dtype=float)

    height_clip_value_m2 = None
    if height_clip_value is not None:
        height_clip_value_m2 = convert_area_unit(height_clip_value, area_unit, 'm2')

    per_area_m2, height_value, is_clipped = _height_value(
        value,
        area_m2,
        log_height,
        height_clip_percentile,
        height_clip_value_m2=height_clip_value_m2,
    )
    # Display unit for color/value_range only -- a pure unit conversion from
    # the same per_area_m2, so it never diverges from the height basis below.
    per_area_display = convert_area_unit(per_area_m2, 'm2', area_unit)
    log_per_area_display = np.log1p(per_area_display)
    log_vmin = np.log1p(vmin) if vmin is not None else None
    log_vmax = np.log1p(vmax) if vmax is not None else None

    rendered_elevation = height_value * elevation_scale

    alpha_255 = round(min(max(alpha, 0.0), 1.0) * 255)
    fill_color = continuous_to_rgba(
        log_per_area_display, cmap=cmap, alpha=alpha_255, vmin=log_vmin, vmax=log_vmax
    )
    if brightness != 1.0:
        fill_color = adjust_brightness(fill_color, brightness)

    is_zero = value == 0
    if is_zero.any():
        fill_color = fill_color.copy()
        fill_color[is_zero] = [128, 128, 128, alpha_255]

    if is_clipped.any() and clipped_fill_rgba is not None:
        fill_color = fill_color.copy()
        fill_color[is_clipped] = np.array(clipped_fill_rgba, dtype=np.uint8)

    # Surface the value the map is actually encoding in the click popup --
    # Lonboard's side panel lists every non-geometry column of `gdf` -- with
    # `address` and this value leading so they're the first thing shown.
    gdf = gdf.copy()
    # Reflect the same missing -> 0 fill applied to `value` above, so the
    # popup shows the rendered value (0), not a stale NaN.
    gdf[value_column] = gdf[value_column].fillna(0)
    per_area_column = f'{value_column}_per_{area_unit}'
    gdf[per_area_column] = per_area_display
    lead_columns = [c for c in ('address', per_area_column) if c in gdf.columns]
    other_columns = [
        c for c in gdf.columns if c not in lead_columns and c != 'geometry'
    ]
    gdf = gdf[[*lead_columns, *other_columns, 'geometry']]

    if stack_on is not None:
        base_z = _match_largest_overlap(gdf, stack_on.gdf, stack_on.rendered_elevation)
        gdf['geometry'] = shapely.force_3d(gdf.geometry.to_numpy(), z=base_z)

    layer = SolidPolygonLayer.from_geopandas(
        gdf,
        extruded=True,
        filled=True,
        wireframe=False,
        get_elevation=rendered_elevation,
        get_fill_color=fill_color,
    )

    outline_layer = None
    if outline_width > 0:
        # SolidPolygonLayer's wireframe mode has no line-width control at
        # all (deck.gl draws a fixed hairline) -- PolygonLayer is the
        # composite layer with a real stroke accessor, used here only for
        # this opt-in companion layer, not the main fill layer where its
        # extra draw pass would cost more at large scale (see module docs).
        outline_layer = PolygonLayer.from_geopandas(
            gdf,
            extruded=True,
            filled=False,
            stroked=True,
            wireframe=True,
            get_elevation=rendered_elevation,
            get_line_color=adjust_brightness(fill_color, outline_darken),
            get_line_width=outline_width,
            line_width_units='pixels',
        )

    clipped_layer = None
    if mark_clipped and is_clipped.any():
        warnings.warn(
            f'{is_clipped.sum()} row(s) exceeded height_clip_percentile '
            f'({height_clip_percentile}) and had their height (not color) '
            f"capped -- see the returned TerrainLayer's clipped_layer.",
            stacklevel=2,
        )
        clipped_layer = SolidPolygonLayer.from_geopandas(
            gdf[is_clipped],
            extruded=True,
            filled=False,
            wireframe=True,
            get_elevation=rendered_elevation[is_clipped],
            get_line_color=clipped_color,
        )

    value_range = (per_area_display.min(), per_area_display.max())
    return TerrainLayer(
        layer,
        elevation_scale,
        value_range,
        gdf,
        rendered_elevation,
        outline_layer,
        clipped_layer,
    )


def _height_value(
    value: np.ndarray,
    area_m2: np.ndarray,
    log_height: bool,
    height_clip_percentile: float | None,
    height_clip_value_m2: float | None = None,
):
    """Per-area value ($/m2) and the height basis derived from it (possibly
    log1p'd or percentile-clipped)."""
    # Guard rare negative reconciled values -- not a meaningful value-per-area.
    per_area_m2 = np.clip(value / area_m2, a_min=0, a_max=None)

    is_clipped = np.zeros(len(per_area_m2), dtype=bool)
    if log_height:
        height_value = np.log1p(per_area_m2)
    elif height_clip_value_m2 is not None:
        is_clipped = per_area_m2 > height_clip_value_m2
        height_value = np.clip(per_area_m2, a_min=None, a_max=height_clip_value_m2)
    elif height_clip_percentile is not None:
        cap = np.percentile(per_area_m2, height_clip_percentile)
        is_clipped = per_area_m2 > cap
        height_value = np.clip(per_area_m2, a_min=None, a_max=cap)
    else:
        height_value = per_area_m2

    return per_area_m2, height_value, is_clipped


def _area_m2(gdf: gpd.GeoDataFrame, area_m2_column: str | None):
    if area_m2_column is not None:
        return gdf[area_m2_column].astype(float).to_numpy()
    return resolve_area(gdf, unit='m2')


def _match_largest_overlap(
    top_gdf: gpd.GeoDataFrame, base_gdf: gpd.GeoDataFrame, base_elevation: np.ndarray
) -> np.ndarray:
    """For each row in `top_gdf`, return the `base_elevation` of whichever
    `base_gdf` row it overlaps with the largest area (0 for no overlap).

    If `top_gdf` already carries a column matching `base_gdf`'s index name
    (e.g. footprints' `parcel_id` column, against a parcels layer whose
    index is itself named `parcel_id`) — the harmonizer's own
    already-resolved footprint-to-parcel link (`io/harmonizer/links.py`,
    `classify_footprint_priority`) — reuse that id-based match via a hash
    join instead of recomputing the same relationship from scratch with a
    live spatial join: measured on real Boston-area data (~926K footprints
    x ~755K parcels), the geometric join alone took ~590s, almost entirely
    in `gpd.sjoin` itself (the subsequent exact-overlap-area computation
    was ~25s) — the id lookup below is effectively instant by comparison.
    Only entity pairs without such a pre-resolved link fall through to the
    spatial join.
    """
    link_column = base_gdf.index.name
    if link_column and link_column in top_gdf.columns:
        elevation_by_id = pd.Series(np.asarray(base_elevation), index=base_gdf.index)
        return top_gdf[link_column].map(elevation_by_id).fillna(0.0).to_numpy()

    top = top_gdf.reset_index(drop=True).to_crs(_EQUAL_AREA_CRS)
    base = base_gdf.reset_index(drop=True).to_crs(_EQUAL_AREA_CRS)
    base_elevation = np.asarray(base_elevation)

    pairs = gpd.sjoin(
        top[['geometry']], base[['geometry']], predicate='intersects', how='inner'
    )
    result = np.zeros(len(top))
    if pairs.empty:
        return result

    top_idx = pairs.index.to_numpy()
    base_idx = pairs['index_right'].to_numpy()
    overlap_area = (
        top.geometry.iloc[top_idx]
        .reset_index(drop=True)
        .intersection(base.geometry.iloc[base_idx].reset_index(drop=True))
        .area.to_numpy()
    )
    matches = pd.DataFrame(
        {'top_idx': top_idx, 'base_idx': base_idx, 'area': overlap_area}
    )
    best = matches.loc[matches.groupby('top_idx')['area'].idxmax()]
    result[best['top_idx'].to_numpy()] = base_elevation[best['base_idx'].to_numpy()]
    return result
