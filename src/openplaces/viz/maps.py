"""Core mapping interface for quick visualization.

This module provides the main entry points for creating maps with sensible
defaults and automatic performance optimization.
"""

import textwrap
import warnings

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import requests
from pyproj import Transformer
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

from openplaces.core.schema import AdminId
from openplaces.geo.polygon import get_areas
from openplaces.io.readers import get_admin, get_entities
from openplaces.recipe import find_admin_recipe_id


def show_geometry_context(
    gdf: gpd.GeoDataFrame,
    idx: int | str,
    buffer_factor: float = 3.0,
    basemap_source: str = 'Esri.WorldImagery',
    figsize: tuple = (12, 8),
    max_attrs: int = 20,
    title: str = None,
    min_buffer_m: float = 25.0,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """
    Plot any geometry type (Point, LineString, Polygon, and their Multi-
    variants) in geographic context with its attributes.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Source geodataframe.
    idx : int or str
        Integer position or index label of the target feature.
    buffer_factor : float
        Buffer size as a multiple of the feature's maximum dimension.
        For point geometries (zero extent) the buffer is derived from
        `min_buffer_m` instead.
    basemap_source : str
        Contextily basemap provider string.
    figsize : tuple
        Figure size (width, height).
    max_attrs : int
        Maximum number of attributes to display in the table.
    title : str
        Overrides the default title ('Feature {idx}').
    min_buffer_m : float
        Minimum plot half-width in metres, applied when the feature's
        projected extent is smaller than this value (most relevant for
        point geometries).  Default is 1 000 m.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax_map, ax_table : matplotlib.axes.Axes
    """
    from pyproj import CRS

    if not isinstance(gdf, gpd.GeoDataFrame):
        raise ValueError('`gdf` is not a gpd.GeoDataFrame.')

    # Extract target row (avoid copying the entire GeoDataFrame)
    if isinstance(idx, int):
        target = gdf.iloc[[idx]].copy()
    else:
        target = gdf.loc[[idx]].copy()

    geom = target.geometry.iloc[0]

    _SUPPORTED = (
        Point,
        MultiPoint,
        LineString,
        MultiLineString,
        Polygon,
        MultiPolygon,
    )
    if not isinstance(geom, _SUPPORTED):
        raise ValueError(
            f'Geometry type not supported: {type(geom).__name__}. '
            f'Expected one of: {", ".join(t.__name__ for t in _SUPPORTED)}.'
        )

    is_line = isinstance(geom, LineString | MultiLineString)
    is_poly = isinstance(geom, Polygon | MultiPolygon)

    # Generous .cx bounding box in the original CRS for context query
    bounds = target.total_bounds  # (minx, miny, maxx, maxy)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    max_dim_orig = max(width, height)

    warnings.filterwarnings('ignore', 'Geometry is in a geographic CRS')
    centroid = target.geometry.centroid.iloc[0]
    warnings.filterwarnings('default', 'Geometry is in a geographic CRS')

    # For zero-extent geometries use a small nominal value so the buffer
    # arithmetic doesn't collapse to zero.
    # Minimum cx radius matching the minimum plot half-width (min_buffer_m *
    # buffer_factor metres), converted to approximate degrees, with the same
    # 1.5x generous factor so context always covers the full plot extent.
    _min_cx = min_buffer_m * buffer_factor * 1.5 / 111_320

    if max_dim_orig == 0:
        buffer_dist_cx = _min_cx
    else:
        buffer_dist_cx = max(max_dim_orig * buffer_factor * 1.5 / 2, _min_cx)

    cx_bounds = [
        centroid.x - buffer_dist_cx,
        centroid.y - buffer_dist_cx,
        centroid.x + buffer_dist_cx,
        centroid.y + buffer_dist_cx,
    ]
    context = gdf.cx[
        cx_bounds[0] : cx_bounds[2],
        cx_bounds[1] : cx_bounds[3],
    ].copy()

    # Reproject to a local orthographic CRS centred on the feature
    lon, lat = centroid.x, centroid.y
    ortho_crs = CRS.from_proj4(
        f'+proj=ortho +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 +datum=WGS84 +units=m'
    )

    target_ortho = target.to_crs(ortho_crs)
    context_ortho = context.to_crs(ortho_crs)

    # Recalculate max_dim in the metric orthographic CRS
    bounds_ortho = target_ortho.total_bounds
    width_ortho = bounds_ortho[2] - bounds_ortho[0]
    height_ortho = bounds_ortho[3] - bounds_ortho[1]
    max_dim_ortho = max(width_ortho, height_ortho)

    # Apply minimum buffer floor (important for point / tiny geometries)
    buffer_dist_plot = max(
        max_dim_ortho * buffer_factor / 2, min_buffer_m * buffer_factor
    )

    # Build figure
    fig, (ax_map, ax_table) = plt.subplots(
        1,
        2,
        figsize=figsize,
        gridspec_kw={'width_ratios': [7, 3], 'wspace': 0},
    )

    if is_poly:
        context_ortho.plot(
            ax=ax_map,
            facecolor='none',
            edgecolor='yellow',
            linewidth=1,
            alpha=0.7,
        )
        # Fill with semi-transparent color + thick boundary
        target_ortho.plot(
            ax=ax_map,
            facecolor='none',
            edgecolor='yellow',
            linewidth=2.5,
            alpha=0.9,
        )
    elif is_line:
        context_ortho.plot(
            ax=ax_map,
            color='yellow',
            linewidth=1,
            alpha=0.7,
        )
        target_ortho.plot(
            ax=ax_map,
            color='yellow',
            linewidth=2.5,
            alpha=0.9,
        )
    else:  # Point / MultiPoint
        context_ortho.plot(
            ax=ax_map,
            facecolor='yellow',
            edgecolor='red',
            markersize=20,
            linewidth=1,
            alpha=0.9,
        )
        target_ortho.plot(
            ax=ax_map,
            facecolor='yellow',
            edgecolor='red',
            markersize=40,
            linewidth=1.5,
        )

    # Axis limits
    target_centroid_ortho = target_ortho.geometry.centroid.iloc[0]
    ax_map.set_xlim(
        target_centroid_ortho.x - buffer_dist_plot,
        target_centroid_ortho.x + buffer_dist_plot,
    )
    ax_map.set_ylim(
        target_centroid_ortho.y - buffer_dist_plot,
        target_centroid_ortho.y + buffer_dist_plot,
    )

    # Basemap
    cx.add_basemap(
        ax_map,
        crs=ortho_crs.to_string(),
        source=basemap_source,
        attribution=False,
    )

    ax_map.set_axis_off()

    # Default title includes geometry type for clarity
    geom_label = type(geom).__name__
    ax_map.set_title(
        title if title else f'{geom_label} {idx}',
        fontsize=12,
        pad=10,
    )

    # Attribute table (unchanged logic)
    attrs = target.iloc[0].drop('geometry')

    interesting = [
        (col, str(val))
        for col, val in attrs.items()
        if val is not None and val != '' and str(val).lower() != 'nan'
    ]
    interesting = interesting[:max_attrs]

    ax_table.axis('off')

    if interesting:
        table = ax_table.table(
            cellText=[[k, v] for k, v in interesting],
            colLabels=['Attribute', 'Value'],
            cellLoc='left',
            loc='upper left',
            bbox=[0, 0, 1, 1],
            colWidths=[0.4, 0.6],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        for key, cell in table.get_celld().items():
            cell.set_text_props(wrap=True)
            cell.PAD = 0.05

        for i in range(2):
            table[(0, i)].set_facecolor('#E0E0E0')
            table[(0, i)].set_text_props(weight='bold')
    else:
        ax_table.text(
            0.5,
            0.5,
            'No attributes to display',
            ha='center',
            va='center',
            fontsize=10,
        )

    ax_table.set_title('Attributes', fontsize=12, pad=10)

    plt.tight_layout(w_pad=0)

    return fig, (ax_map, ax_table)


_DEFAULT_COLORS = ['#00ffff', '#ff66ff', '#66ff66', '#ffaa00', '#ff6666', '#aaaaff']
# (ha, va) positions for per-dataset text boxes; cycles if > 3 non-parcel datasets
_LABEL_POSITIONS = [
    ('right', 'top'),
    ('left', 'bottom'),
    ('right', 'bottom'),
]


def _is_polygon(gdf):
    return gdf.geometry.dropna().iloc[0].geom_type in ('Polygon', 'MultiPolygon')


def _default_label_columns(gdf):
    return [
        c
        for c in gdf.columns
        if c != 'geometry'
        and not c.startswith(('geo_', 'source_'))
        and not c.endswith(('_wkt',))
    ][:12]


def _make_label_text(rows, columns):
    texts = []
    for _, row in rows.iterrows():
        lines = []
        for col in columns:
            if col not in row:
                continue
            val = row[col]
            if pd.isna(val) or val == '':
                continue
            label = col.replace('_', ' ').title()[:16]
            if isinstance(val, float):
                value = f'{val:,.2f}'
            else:
                value = str(val)
            if len(value) > 24:
                value = value[:24] + '...'
            lines.append(f'{label}: {value}')
        if lines:
            texts.append('\n'.join(lines))
    return '\n\n'.join(texts)


def show_building(
    location,
    geodatasets,
    styles=None,
    radius=100,
    size=10,
    show_basemap=True,
    show_crosshair=True,
    show_location=True,
    return_fig_ax=False,
    verbose=False,
):
    """Show building in its context with basemap

    Parameters
    ----------
    location : tuple or gpd.GeoDataFrame
        A single-row GeoDataFrame
    geodatasets : dict of GeoDataFrames
        Loaded geodatasets to be plotted alongside location.
        The ``'parcels'`` key is special: drawn with inner border and
        attribute text box at top-left. Any other key is treated as a
        building footprint or point GeoDataFrame — polygons are drawn as
        boundaries, points as markers, labels from available columns.
    styles : dict of dict, optional
        Per-dataset style overrides, keyed by geodatasets keys. Supported
        sub-keys: 'color' (str), 'n_max' (int, default 4),
        'columns' (list[str]), 'marker' (str, default 'o'),
        'markersize' (int, default 10).
    radius : float
        Radius of plot in EPSG:3857 "meters" (~1.3m in NY)
    size : float
        Size of plot in inches (both height and width of `figsize`)
    show_basemap : bool
        If True, a basemap is added, and colors of parcels are adjusted.
    show_crosshair : bool
        Display crosshair of the centerpoint
    show_location : bool
        Display latitude/longitude of location
    return_fig_ax : bool
        If True, return the plot's Figure and Axes objects
    verbose : bool
        If True, prints counts of matched entities per dataset.
    """

    MIN_OVERLAP_M2 = 10

    to_3857 = Transformer.from_crs('EPSG:4326', 'EPSG:3857').transform
    to_4326 = Transformer.from_crs('EPSG:3857', 'EPSG:4326').transform

    # Get centroid coordinates in EPSG:3857
    if isinstance(location, tuple):
        lat_center, long_center = location[0], location[1]
    elif isinstance(location, gpd.GeoDataFrame):
        # Compute centroid of first geometry
        warnings.filterwarnings('ignore', '.*geographic CRS.*')
        location = location.iloc[[0]]['geometry'].centroid
        warnings.filterwarnings('default', '.*geographic CRS.*')
        if location.crs != 'epsg:4326':
            location = location.to_crs('epsg:4326')
        lat_center, long_center = location.iloc[0].y, location.iloc[0].x
    else:
        raise TypeError(f'Type of `location` is not yet supported: {type(location)}.')

    # Get outer bounds in EPSG:3857
    x, y = to_3857(lat_center, long_center)
    xmin, xmax = x - radius, x + radius
    ymin, ymax = y - radius, y + radius

    # Convert outer bounds to lat/long for quick spatial selection with .cx
    lat_min, long_min = to_4326(xmin, ymin)
    lat_max, long_max = to_4326(xmax, ymax)

    fig, ax = plt.subplots(figsize=(size, size))
    ax.set_xlim(xmin, xmax)  # Needs to happen before adding a basemap
    ax.set_ylim(ymin, ymax)

    # Try to show basemap
    if show_basemap:
        try:
            cx.add_basemap(
                ax,
                crs='epsg:3857',
                source=cx.providers.Esri.WorldImagery,
                alpha=0.8,
            )
        except requests.exceptions.ConnectionError:
            print('No internet connection: could not add basemap. ')
            show_basemap = False

    if show_crosshair:
        crosshair_color = 'white' if show_basemap else 'black'
        ax.plot([xmin, xmax], [ymin, ymax], color=crosshair_color, ls=':', linewidth=1)
        ax.plot([xmin, xmax], [ymax, ymin], color=crosshair_color, ls=':', linewidth=1)

    if show_location:
        ax.text(
            x,
            ymin + radius / 15,
            f'lat: {lat_center:.3f}, long: {long_center:.3f}',
            backgroundcolor='#ffffffcc',
            va='bottom',
            ha='center',
            bbox=dict(
                facecolor='#ffffffdd',
                edgecolor='crimson',
                boxstyle='round',
                linewidth=1,
                pad=0.5,
            ),
        )

    color_parcel = 'yellow' if show_basemap else 'gold'

    # Resolve per-dataset styles
    styles = styles or {}
    non_parcel_keys = [k for k in geodatasets if k != 'parcels']
    resolved = {}
    for i, key in enumerate(non_parcel_keys):
        s = styles.get(key, {})
        resolved[key] = {
            'color': s.get('color', _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]),
            'n_max': s.get('n_max', 4),
            'columns': s.get('columns', _default_label_columns(geodatasets[key])),
            'marker': s.get('marker', 'o'),
            'markersize': s.get('markersize', 10),
        }

    # Draw property boundaries for all properties
    if 'parcels' in geodatasets:
        parcels = (
            geodatasets['parcels']
            .cx[long_min:long_max, lat_min:lat_max]
            .to_crs('epsg:3857')
        )
        parcels.boundary.plot(ax=ax, linewidth=0.5, color=color_parcel)

    # Drawing pass: all non-parcel datasets
    for key in non_parcel_keys:
        gdf = geodatasets[key]
        s = resolved[key]
        local = gdf.cx[long_min:long_max, lat_min:lat_max].to_crs('epsg:3857')
        if local.empty:
            continue
        if _is_polygon(local):
            local.boundary.plot(ax=ax, color=s['color'], linewidth=1)
        else:
            local.plot(
                ax=ax,
                color='white',
                marker=s['marker'],
                edgecolor=s['color'],
                markersize=s['markersize'],
            )

    # Add parcel info
    parcel_found = False
    if 'parcels' in geodatasets:
        parcel = geodatasets['parcels'].cx[
            long_center:long_center, lat_center:lat_center
        ]
        if len(parcel) == 0:
            print('No parcel found.')
        else:
            parcel_found = True
            if len(parcel) > 1:
                print('Multiple (overlapping) parcels found at location.')

    if parcel_found:
        parcel_3857 = parcel.to_crs('epsg:3857')

        # Draw property boundaries for parcel
        parcel_3857.boundary.plot(ax=ax, color=color_parcel, linewidth=1)

        # Add inner border to boundary of selected parcel
        # (to show which side is "inside" without filling the polygon)
        parcel_3857_inner = parcel_3857.copy()
        parcel_3857_inner['geometry'] = parcel_3857_inner['geometry'].difference(
            parcel_3857_inner['geometry'].buffer(-2).rename('geometry').to_frame()
        )
        parcel_3857_inner.plot(ax=ax, color=color_parcel, alpha=0.15)
        parcel_3857_inner.plot(
            ax=ax,
            color='none',
            edgecolor=color_parcel,
            linewidth=0,
            hatch='/////',
            alpha=0.5,
        )

        N_MAX_PARCEL_TEXT = 4
        txt_p_list = [parcel.iloc[0]['geo_id']]
        for _gid, _p_txt in parcel.head(N_MAX_PARCEL_TEXT).iterrows():
            if verbose:
                print(f'Parcel GID: {_gid}')

            txt_p = f'Parcel ID: {_p_txt["parcel_id_admin3"]}\n'
            for var in ['address', 'purpose_group', 'purpose_subgroup']:
                if _p_txt[var]:
                    txt_p += (
                        f'{_p_txt[var].title()[:25]}'
                        + ('...' if len(_p_txt[var].title()) > 25 else '')
                        + '\n'
                    )
            if 'year_built' in _p_txt:
                txt_p += f'Year built: {int(_p_txt["year_built"]):d}\n'
            txt_p += f'Value: ${int(_p_txt["value"]):,d}\n'
            txt_p += f'Improv. value: ${int(_p_txt["improvement_value"]):,d}\n'
            if 'land_value' in _p_txt:
                txt_p += f'Land value: ${int(_p_txt["land_value"]):,d}\n'
            legal_desc = (
                _p_txt['legal_description']
                .title()[:50]
                .replace('|', ' | ')
                .replace('  ', ' ')
            )
            txt_p += 'Legal description:\n  ' + '\n  '.join(
                textwrap.wrap(legal_desc, 25)
            )
            txt_p_list += [txt_p]
        n_omitted = len(parcel) - N_MAX_PARCEL_TEXT
        if n_omitted > 0:
            txt_p_list += [f'... and {n_omitted} more']
        txt_parcel = '\n\n'.join(txt_p_list)
        ax.text(
            xmin + radius / 25,
            ymax - radius / 25,
            txt_parcel,
            va='top',
            bbox=dict(
                facecolor='#ffffffdd',
                edgecolor=color_parcel,
                linewidth=1,
                boxstyle='round',
                pad=0.5,
            ),
        )

    # Labeling pass: all non-parcel datasets
    label_pos_idx = 0
    for key in non_parcel_keys:
        gdf = geodatasets[key]
        s = resolved[key]
        color = s['color']
        n_max = s['n_max']
        columns = s['columns']

        location_in_dataset = (
            isinstance(location, gpd.GeoSeries)
            and location.index.name == gdf.index.name
            and location.index[0] in gdf.index
        )

        if not parcel_found and not location_in_dataset:
            continue

        candidates_list = []

        if location_in_dataset:
            crosshair_row = gdf.loc[[location.index[0]]].reset_index()
            candidates_list.append(crosshair_row)

        if parcel_found:
            bbox_gdf = gdf.cx[long_min:long_max, lat_min:lat_max]
            if not bbox_gdf.empty:
                if _is_polygon(gdf):
                    on_parcel = gpd.overlay(
                        parcel[['geometry']].iloc[[0]].reset_index(),
                        bbox_gdf.reset_index(),
                    )
                    on_parcel['_overlap_m2'] = get_areas(on_parcel, 'm2')
                    on_parcel = on_parcel[
                        on_parcel['_overlap_m2'].ge(MIN_OVERLAP_M2)
                    ].sort_values('_overlap_m2', ascending=False)
                else:
                    on_parcel = gpd.sjoin(
                        parcel[['geometry']].iloc[[0]],
                        bbox_gdf.reset_index(),
                    )

                if not on_parcel.empty:
                    if location_in_dataset:
                        idx_col = gdf.index.name
                        not_crosshair = on_parcel[idx_col].ne(location.index[0])
                        candidates_list.append(on_parcel[not_crosshair])
                    else:
                        candidates_list.append(on_parcel)

        if not candidates_list:
            continue
        to_label = pd.concat(candidates_list).head(n_max)
        if to_label.empty:
            print(f'No {key} entities found at location.')
            continue

        if verbose:
            print(f'{len(to_label)} {key} entities at location')

        label_text = _make_label_text(to_label, columns)
        if not label_text:
            continue

        ha, va = _LABEL_POSITIONS[label_pos_idx % len(_LABEL_POSITIONS)]
        label_pos_idx += 1
        x_text = xmax - radius / 25 if ha == 'right' else xmin + radius / 25
        y_text = ymax - radius / 25 if va == 'top' else ymin + radius / 15
        ax.text(
            x_text,
            y_text,
            label_text,
            va=va,
            ha=ha,
            bbox=dict(
                facecolor='#ffffffdd',
                edgecolor=color,
                boxstyle='round',
                linewidth=1,
                pad=0.5,
            ),
        )

    if return_fig_ax:
        return fig, ax


def show_ingested_geometries(
    ingester,
    admin_recipe_id: str | None = None,
    fill: bool = True,
    color: str = 'skyblue',
    edgecolor: str = 'blue',
    point_markersize: float = 2,
    max_plot: int = 250_000,
    basemap_source: str = 'Esri.WorldImagery',
    figsize: tuple = (10, 10),
) -> tuple[plt.Figure, plt.Axes] | None:
    """Plot the last ingested layer for visual inspection.

    Reads entities and admin boundary from the ingester, applies a sample cap
    for large datasets, and renders a basemap.

    Parameters
    ----------
    ingester : openplaces.io.ingester.Ingester
        Completed ingester whose last saved admin unit is shown.
    admin_recipe_id : str, optional
        Recipe ID for the admin boundary dataset. Auto-detected from the
        recipe's admin_id when omitted.
    fill : bool
        If True (default), fill polygon geometries. If False, draw boundary only.
    color : str
        Fill color for polygons or face color for points.
    edgecolor : str
        Edge/boundary color for polygons.
    point_markersize : float
        Marker size when entities are points.
    max_plot : int
        Maximum number of polygon features to render; a random sample is taken
        when exceeded.
    basemap_source : str
        Contextily basemap provider string (e.g. 'Esri.WorldImagery').
    figsize : tuple
        Figure size (width, height) in inches.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes, or None if nothing to plot.
    """
    if not ingester.admin_ids_to_save:
        return None

    admin_id = ingester.admin_ids_to_save[-1]
    print(admin_id)

    admin = None
    if admin_id is not None:
        level = AdminId(admin_id).get_level()
        if admin_recipe_id is None and level > 1:
            admin_recipe_id = find_admin_recipe_id(ingester.recipe['admin_id'], level)
        if admin_recipe_id:
            admin = get_admin(admin_id, level=level, recipe=admin_recipe_id, geom=True)

    entities = get_entities(ingester.recipe, admin_id, geom=True)

    if entities.empty:
        print(f'No entities found for {admin_id}.')
        return None

    entity_label = str(ingester.recipe['entity'].entity_type) + 's'
    if admin is not None:
        title = (
            f'{len(entities):,d} {entity_label} in '
            + admin.loc[admin_id, 'name']
            + f' ({admin_id})'
        )
    else:
        title = f'{len(entities):,d} {entity_label}'

    fig, ax = plt.subplots(figsize=figsize)

    geom0 = entities.geometry.iloc[0]

    if isinstance(geom0, Point | MultiPoint):
        entities.plot(
            ax=ax,
            facecolor=color,
            markersize=point_markersize,
            linewidth=0,
            alpha=0.7,
        )
    elif isinstance(geom0, Polygon | MultiPolygon):
        to_plot = entities
        if len(entities) > max_plot:
            print(f'>{max_plot:,d} polygon features to plot. Taking sample.')
            to_plot = entities.sample(max_plot)
        if fill:
            to_plot.plot(
                ax=ax,
                color=color,
                edgecolor=edgecolor,
                linewidth=0.5,
                alpha=0.5,
            )
        else:
            to_plot.boundary.plot(
                ax=ax,
                color=edgecolor,
                linewidth=0.3,
                alpha=0.5,
            )

    if admin is not None:
        admin.boundary.plot(ax=ax, color='black', linewidth=0.25)

    ax.set_title(title)
    ax.axis('off')

    basemap_provider = basemap_source.split('.')
    source = cx.providers
    for part in basemap_provider:
        source = source[part]
    cx.add_basemap(ax, crs=entities.crs, source=source, alpha=0.5)

    return fig, ax


def show_random_entity(
    ingester,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Plot a random entity from the last ingested admin unit with its attributes.

    Delegates to :func:`show_geometry_context`.

    Parameters
    ----------
    ingester : openplaces.io.ingester.Ingester
        Completed ingester whose last saved admin unit is sampled.

    Returns
    -------
    fig, (ax_map, ax_table) : matplotlib Figure and Axes pair.
    """
    admin_id = ingester.admin_ids_to_save[-1] if ingester.admin_ids_to_save else None
    entities = get_entities(ingester.recipe, admin_id, geom=True)
    idx = entities.sample().index[0]
    print(idx)
    return show_geometry_context(entities, idx)
