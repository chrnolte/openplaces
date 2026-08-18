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
from openplaces.geo.polygon import find_corridors, resolve_area
from openplaces.io.readers import get_entities
from openplaces.io.transform import convert_area_unit
from openplaces.viz import elevation
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
        (`height_value * elevation_scale`, before any `elevation_column` or
        `stack_on` base offset of its own).
    total_elevation : numpy.ndarray
        Each row's absolute top-of-stack elevation in meters: `rendered_elevation`
        plus its own base (`elevation_column`'s ground elevation, if given,
        plus any `stack_on` base of its own). Pass this (via `gdf`, not
        `rendered_elevation`) as a subsequent call's `stack_on` so that call's
        base lines up with this layer's true top, not just its value height.
    outline_layer : lonboard.PolygonLayer or None
        A companion wireframe-only layer outlining each polygon in a darker
        shade of its own fill color, or `None` when `outline_width=0`. Add
        it to the `Map` alongside `layer` if present.
    ghost_layer : lonboard.PolygonLayer or None
        A companion outline-only layer holding the rows with no value, drawn
        as a flat ring floating `ghost_offset` meters above whatever they
        sit on — `None` unless `missing_value='ghost'` and at least one row
        actually lacks a value. These rows are excluded from `layer` and
        `outline_layer`, so they are drawn exactly once. Add it to the `Map`
        alongside `layer` if present.
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
    total_elevation: np.ndarray
    outline_layer: PolygonLayer | None
    clipped_layer: SolidPolygonLayer | None
    ghost_layer: PolygonLayer | None


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
    elevation_column: str | None = None,
    elevation_recipe: str | dict | None = None,
    elevation_mode: str = 'flat',
    terrain_exaggeration: float = 1.0,
    elevation_datum: float = 0.0,
    stack_on: TerrainLayer | None = None,
    brightness: float = 1.0,
    outline_width: float = 0,
    outline_darken: float = 0.6,
    missing_value: str = 'render',
    ghost_offset: float = 2.0,
    ghost_rgba: tuple[int, int, int, int] = (255, 255, 255, 140),
    ghost_width: float = 1.0,
    drop_corridors: bool | dict = False,
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
    Pass `elevation_column` to additionally (or instead) ground a layer at
    its own real-world elevation rather than extruding from sea level.

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
    elevation_column : str, optional
        Column already present in the loaded entity holding each row's
        real-world ground elevation in meters (e.g. `'elevation'` on curated
        parcels — see `core.attribute_registry`). Applied as a base
        z-offset from 0 to the *bottom* of each polygon, i.e. the ground the
        geometry physically sits on, via a 3D polygon Z-coordinate — additive
        with `stack_on`'s own base if both are given (see `total_elevation`
        under Returns). Missing values fill to 0 (sea level), with a warning
        unless `silent=True`. Defaults to `None`: every polygon's base stays
        at 0 (unaffected by real-world terrain).
    elevation_recipe : str or dict, optional
        DEM dataset recipe (e.g. `'US_land-elevation-usgs-3dep'`) to sample
        real-world ground elevation from directly, via `viz.elevation`. An
        alternative, additional source to `elevation_column` feeding the
        same z-offset base -- both can be given together, though that's
        unusual. Requires the rows loaded by `recipe` to carry an
        `'admin3_id'` column (curated parcels/footprints already do) to
        resolve each row's DEM tile. Results are cached to disk per admin
        unit under `cfg.cache_dir`, keyed by each row's own index, and
        reused across calls whose rows haven't changed shape -- see the
        `viz.elevation` module docstring. Defaults to `None`: no DEM-based
        elevation.
    elevation_mode : {'flat', 'drape'}
        How `elevation_recipe` grounds this layer; ignored when
        `elevation_recipe` is `None`. `'flat'` (the default -- use for
        buildings) samples one zonal-mean elevation per polygon
        (`viz.elevation.get_building_elevation`) and applies it uniformly,
        like `elevation_column` -- appropriate since a real building's
        base is flat. `'drape'` (use for land/parcel layers) samples
        elevation at every polygon vertex and follows the terrain along
        each polygon's own boundary (`viz.elevation.drape_parcel_elevation`)
        instead of sitting at one flat elevation -- geometrically correct
        for land, which spans real, unevenly sloped terrain. Any
        `elevation_column`/`stack_on` scalar offset is still added on top
        of the per-vertex terrain in `'drape'` mode (see
        `viz.elevation.add_z_offset`), not used to overwrite it.
    terrain_exaggeration : float
        Multiplier applied to real-world ground elevation (from
        `elevation_column` and/or `elevation_recipe`, including every
        vertex of a `'drape'`-mode geometry, via `viz.elevation.scale_z`)
        before rendering. A visualization aid only -- real terrain relief
        across one parcel is often just a few meters, easy to miss next to
        the value-based extrusion height, which this leaves untouched
        (`elevation_scale` controls that independently). Defaults to `1.0`
        (true-to-scale, no exaggeration). Not reapplied to a `stack_on`
        layer's own inherited ground offset, which already reflects
        whatever exaggeration that base layer applied to itself.
    elevation_datum : float
        Ground elevation in meters to treat as z=0, subtracted before
        `terrain_exaggeration` is applied. Defaults to 0 (sea level, the
        original behavior).

        Set this when a basemap is in the scene. Extruding from sea level
        lifts everything as far above the flat basemap as the land is above
        the ocean, and with the camera tilted to pitch `p` a feature `h`
        meters up appears displaced from its own basemap position by
        `h * tan(p)` — at pitch 75 over terrain 345 m up, roughly 1.3 km.
        Referencing the scene to the ground under it removes that term,
        leaving only the relief the terrain is meant to show.

        Ground elevation is clamped at z=0 afterward, so nothing sinks
        below a flat basemap and out of sight even if the datum is set
        above part of the extent. Only the *ground* is clamped; value
        height and `stack_on` offsets are added on top of the clamped
        ground and are unaffected.

        Compute it once with `viz.elevation.get_elevation_datum` and pass
        the same value to every layer of the scene — parcels, buildings,
        boundaries, basemap. A datum that differs between layers slides
        them vertically against each other, exactly like a mismatched
        `terrain_exaggeration`.
    stack_on : TerrainLayer, optional
        A previously built layer to stack this one on top of: each row here
        is matched to whichever `stack_on.gdf` row it overlaps with the
        largest area (or, when this call's own entity already carries a
        column matching `stack_on.gdf`'s index name — e.g. footprints'
        `parcel_id` against a parcels layer, whose index is itself
        `parcel_id` — an id-based lookup reusing the harmonizer's own
        already-resolved link instead, orders of magnitude faster than a
        live spatial join at large scale; see `_match_largest_overlap`).
        Its base is set to that row's `stack_on.total_elevation` (0 for
        rows with no match) via a 3D polygon Z-coordinate — deck.gl adds
        `get_elevation` on top of a polygon's own Z when one is present, so
        this layer's rendered zmin lines up with the matched `stack_on`
        row's true top (its own ground elevation, if any, plus its rendered
        value height). Requires that base layer's `TerrainLayer` to already
        exist (built in an earlier call), for its geometry/`total_elevation`
        to match against. When this call also supplies its own ground
        elevation (`elevation_column` or `elevation_recipe`), the match uses
        `stack_on.rendered_elevation` (value height only, no ground) instead,
        so `stack_on`'s own ground elevation isn't added a second time on
        top of this layer's independently sourced one.
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
    missing_value : {'render', 'drop', 'ghost'}
        What to do with rows whose `value_column` is null or zero. Note this
        is about missing *values*; `missing` (below) is about missing output
        *files*.

        - `'render'` (default) — keep them in `layer` as flat, zero-height
          gray polygons.
        - `'drop'` — remove them entirely, leaving the basemap visible. Most
          useful for parcels, where the unvalued polygons are typically
          rights-of-way whose geometry traces a whole road network as one
          feature: at town zoom such a corridor is narrower than a pixel and
          collapses to a hairline criss-crossing the map.
        - `'ghost'` — move them to `ghost_layer`, an outline-only ring
          floating `ghost_offset` meters above whatever they sit on. Most
          useful for buildings, where a missing value usually means a real
          structure nobody assessed separately (a secondary footprint, say)
          rather than nothing being there — visible as present, without
          claiming a value it does not have.

        Null and zero are treated alike: they are indistinguishable
        downstream (both land in the `is_zero` gray branch), and a source
        holding no record for a polygon is the case these modes exist for.
    ghost_offset : float
        Meters to float `ghost_layer` above each row's own base elevation.
        Defaults to 2 — enough to read as hovering rather than as a ground
        marking, without detaching from the feature it belongs to. Applies
        only when `missing_value='ghost'`.
    ghost_rgba : tuple of int
        RGBA line color for `ghost_layer`. Defaults to translucent white.
    ghost_width : float
        `ghost_layer` line width in pixels. Defaults to 1.
    drop_corridors : bool or dict
        Drop road, rail, and water right-of-way polygons identified by shape
        alone, via `geo.polygon.find_corridors` — no value column, land-use
        code, or source-specific field involved, so it works on any parcel
        source including ones publishing nothing that marks a right-of-way.
        Defaults to False. Pass True for the calibrated defaults, or a dict
        of keyword arguments to override them (e.g.
        `{'max_elongation': 100}`). Complements `missing_value='drop'`,
        which only reaches corridors that happen to carry no value.
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

    if elevation_mode not in ('flat', 'drape'):
        raise ValueError(
            f"elevation_mode must be 'flat' or 'drape', got {elevation_mode!r}."
        )

    if missing_value not in ('render', 'drop', 'ghost'):
        raise ValueError(
            f"missing_value must be 'render', 'drop' or 'ghost', got {missing_value!r}."
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

    if drop_corridors is not False:
        corridor_kwargs = drop_corridors if isinstance(drop_corridors, dict) else {}
        is_corridor = find_corridors(gdf, **corridor_kwargs)
        if is_corridor.any():
            if not silent:
                warnings.warn(
                    f'Dropping {is_corridor.sum()} row(s) whose shape reads as a '
                    'road/rail/water corridor (drop_corridors).',
                    stacklevel=2,
                )
            gdf = gdf[~is_corridor]

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

    # Zero counts as missing, not as a real "worth nothing" reading: the
    # two are indistinguishable downstream (both land in the `is_zero`
    # gray branch below), and a source with no record for a polygon at
    # all is exactly the case `missing_value` exists for.
    has_value = gdf[value_column].notna() & (gdf[value_column] != 0)
    if missing_value == 'drop':
        if not has_value.all():
            if not silent:
                warnings.warn(
                    f'Dropping {(~has_value).sum()} row(s) with missing/zero '
                    f"{value_column!r} (missing_value='drop').",
                    stacklevel=2,
                )
            gdf = gdf[has_value]
            area_m2 = area_m2[has_value]
            has_value = has_value[has_value]
    elif missing_value == 'render':
        n_missing_value = gdf[value_column].isna().sum()
        if n_missing_value and not silent:
            warnings.warn(
                f'Setting {n_missing_value} row(s) with missing {value_column!r} '
                'to 0 (e.g. accessory structures with no assessed value) so they '
                'still render, at the bottom of the color/height scale.',
                stacklevel=2,
            )
    # 'ghost' keeps its rows all the way through the elevation and
    # `stack_on` math below -- a ghost still has to know what it floats
    # above -- and is split out of the fill layers at the very end.
    is_ghost = (~has_value).to_numpy() if missing_value == 'ghost' else None

    # deck.gl cannot build a layer from zero features, so a filter that
    # empties the selection has to say so, rather than failing later
    # inside lonboard's geometry encoding.
    if gdf.empty:
        raise ValueError(
            f'No rows left to render for {recipe!r} after filtering '
            f'(missing_value={missing_value!r}, drop_corridors='
            f'{drop_corridors!r}); nothing would be drawn.'
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

    has_own_ground_elevation = (
        elevation_column is not None or elevation_recipe is not None
    )

    # `base_z` is the full ground-elevation total, used for the returned
    # `total_elevation` (and as the `stack_on` target for whatever's stacked
    # on this layer). `extra_z` is what actually gets added to the rendered
    # geometry's own Z -- identical to `base_z` except it excludes drape's
    # `mean_elevation`, which is already baked into each vertex of the
    # draped geometry itself and must not be added a second time.
    base_z = np.zeros(len(gdf))
    extra_z = np.zeros(len(gdf))

    if elevation_column is not None:
        elev = gdf[elevation_column]
        n_missing_elev = elev.isna().sum()
        if n_missing_elev and not silent:
            warnings.warn(
                f'Setting {n_missing_elev} row(s) with missing {elevation_column!r} '
                'to 0 (sea level).',
                stacklevel=2,
            )
        contribution = elev.fillna(0).to_numpy(dtype=float) * terrain_exaggeration
        base_z = base_z + contribution
        extra_z = extra_z + contribution

    draped = False
    if elevation_recipe is not None and elevation_mode == 'drape':
        drape_geometry, mean_elevation = elevation.drape_parcel_elevation(
            gdf, elevation_recipe, silent=silent
        )
        # Reference to the datum, clamp at the ground plane, then
        # exaggerate -- in that order. Exaggerating first would scale
        # the datum offset too, and clamping last would let a negative
        # Z survive multiplied.
        referenced = elevation.add_z_offset(
            drape_geometry.to_numpy(), np.full(len(gdf), -float(elevation_datum))
        )
        gdf['geometry'] = elevation.scale_z(
            elevation.clamp_z(referenced, lower=0.0), terrain_exaggeration
        )
        base_z = base_z + (
            np.maximum(mean_elevation - elevation_datum, 0.0) * terrain_exaggeration
        )
        draped = True
    elif elevation_recipe is not None:
        building_elev = elevation.get_building_elevation(
            gdf, elevation_recipe, silent=silent
        )
        n_missing_building_elev = int(np.isnan(building_elev).sum())
        if n_missing_building_elev and not silent:
            warnings.warn(
                f'Setting {n_missing_building_elev} row(s) with missing DEM elevation '
                "(from 'elevation_recipe') to the elevation_datum.",
                stacklevel=2,
            )
        # nan fills to the datum, not to 0: with a datum set, "no DEM
        # here" should put the row on the reference plane, not drop it
        # to sea level far below the rest of the scene.
        contribution = (
            np.maximum(
                np.nan_to_num(building_elev, nan=float(elevation_datum))
                - elevation_datum,
                0.0,
            )
            * terrain_exaggeration
        )
        base_z = base_z + contribution
        extra_z = extra_z + contribution

    if stack_on is not None:
        stack_source = (
            stack_on.rendered_elevation
            if has_own_ground_elevation
            else stack_on.total_elevation
        )
        contribution = _match_largest_overlap(gdf, stack_on.gdf, stack_source)
        base_z = base_z + contribution
        extra_z = extra_z + contribution

    if draped:
        gdf['geometry'] = elevation.add_z_offset(gdf.geometry.to_numpy(), extra_z)
    elif (
        elevation_column is not None
        or elevation_recipe is not None
        or stack_on is not None
    ):
        gdf['geometry'] = shapely.force_3d(gdf.geometry.to_numpy(), z=extra_z)

    # Ghost rows are drawn once, by `ghost_layer` alone -- `solid` masks
    # them out of the fill and outline layers so they don't also show up
    # as zero-height gray polygons under their own floating outline.
    solid = slice(None) if is_ghost is None else ~is_ghost
    if is_ghost is not None and is_ghost.all():
        raise ValueError(
            f'Every row of {recipe!r} has a missing/zero {value_column!r}, so '
            "missing_value='ghost' would leave the fill layer empty. Use "
            "missing_value='render' to draw them, or 'drop' to skip them."
        )

    layer = SolidPolygonLayer.from_geopandas(
        gdf[solid],
        extruded=True,
        filled=True,
        wireframe=False,
        get_elevation=rendered_elevation[solid],
        get_fill_color=fill_color[solid],
    )

    ghost_layer = None
    if is_ghost is not None and is_ghost.any():
        # Floats the ring `ghost_offset` above the row's own base:
        # `extra_z` is already in the geometry's Z (set by the elevation
        # block above), so only the offset is added here.
        # `extruded=True` with a flat `get_elevation=0` is what makes
        # deck.gl honor that Z -- the same configuration `outline_layer`
        # uses -- while drawing no walls.
        ghost_gdf = gdf[is_ghost].copy()
        ghost_gdf['geometry'] = elevation.add_z_offset(
            ghost_gdf.geometry.to_numpy(),
            np.full(int(is_ghost.sum()), float(ghost_offset)),
        )
        ghost_layer = PolygonLayer.from_geopandas(
            ghost_gdf,
            extruded=True,
            filled=False,
            stroked=True,
            wireframe=True,
            get_elevation=0,
            get_line_color=list(ghost_rgba),
            get_line_width=ghost_width,
            line_width_units='pixels',
        )

    outline_layer = None
    if outline_width > 0:
        # SolidPolygonLayer's wireframe mode has no line-width control at
        # all (deck.gl draws a fixed hairline) -- PolygonLayer is the
        # composite layer with a real stroke accessor, used here only for
        # this opt-in companion layer, not the main fill layer where its
        # extra draw pass would cost more at large scale (see module docs).
        outline_layer = PolygonLayer.from_geopandas(
            gdf[solid],
            extruded=True,
            filled=False,
            stroked=True,
            wireframe=True,
            get_elevation=rendered_elevation[solid],
            get_line_color=adjust_brightness(fill_color[solid], outline_darken),
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

    # Ghost rows carry no value and are never colored, so they must not
    # widen the range a caller builds its colorbar from.
    colored = per_area_display[solid]
    value_range = (colored.min(), colored.max())
    total_elevation = base_z + rendered_elevation
    return TerrainLayer(
        layer,
        elevation_scale,
        value_range,
        gdf,
        rendered_elevation,
        total_elevation,
        outline_layer,
        clipped_layer,
        ghost_layer,
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
    elif height_clip_percentile is not None and len(per_area_m2):
        # `len` guard: a filter (`missing_value='drop'` or
        # `drop_corridors`) can legitimately empty an admin unit, and
        # np.percentile raises on an empty array rather than returning
        # a no-op cap.
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
