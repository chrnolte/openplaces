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

from openplaces.geo.polygon import get_areas


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
    (ax_map, ax_table) : tuple of matplotlib.axes.Axes
    """
    from pyproj import CRS

    if not isinstance(gdf, gpd.GeoDataFrame):
        raise ValueError('`gdf` is not a gpd.GeoDataFrame.')

    # ------------------------------------------------------------------
    # Extract target row (avoid copying the entire GeoDataFrame)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Generous .cx bounding box in the original CRS for context query
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Reproject to a local orthographic CRS centred on the feature
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Build figure
    # ------------------------------------------------------------------
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

    # ---- Axis limits ---------------------------------------------------
    target_centroid_ortho = target_ortho.geometry.centroid.iloc[0]
    ax_map.set_xlim(
        target_centroid_ortho.x - buffer_dist_plot,
        target_centroid_ortho.x + buffer_dist_plot,
    )
    ax_map.set_ylim(
        target_centroid_ortho.y - buffer_dist_plot,
        target_centroid_ortho.y + buffer_dist_plot,
    )

    # ---- Basemap -------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Attribute table (unchanged logic)
    # ------------------------------------------------------------------
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


def show_building(
    location,
    geodatasets,
    radius=100,
    size=10,
    show_basemap=True,
    show_crosshair=True,
    show_location=True,
    return_fig_ax=False,
    verbose=False,
):
    """Show building in its context with basemap

    Current version is for US building inventories

    Parameters
    ----------
    location : tuple or gpd.GeoDataFrame
        A single-row GeoDataFrame
    geodatasets: dict of lists of GeoDataFrames
        Loaded geodatasets to be plotted alongside location.
        'parcels': parcel GeoDataFrame
        'buildings_nsi': building GeoDataFrame (USACE: NSI)
        'buildings_fema': building GeoDataFrame (FEMA: USA Structures)
        'buildings_microsoft': building GeoDataFrame (Microsoft)
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
        If True, prints IDs of parcels and FEMA polygons.
    """

    # Maximum number of entities for which text info is shown
    # ('and [x] more...' will be added if entities are ommitted)
    N_MAX_PARCEL_TEXT = 4
    N_MAX_BUILDINGS_NSI_TEXT = 4
    N_MAX_BUILDINGS_FEMA_TEXT = 3
    N_MAX_BUILDINGS_LOCAL_TEXT = 3

    # Minimum size of overlap between parcels & FEMA (ignore slivers)
    MIN_SQFT_FEMA = 10 * 10.7639  # 10 m2 in sqft

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

    # Draw property boundaries for all properties
    if 'parcels' in geodatasets:
        parcels = (
            geodatasets['parcels']
            .cx[long_min:long_max, lat_min:lat_max]
            .to_crs('epsg:3857')
        )
        parcels.boundary.plot(ax=ax, linewidth=0.5, color=color_parcel)

    # Draw Microsoft footprints
    if 'buildings_microsoft' in geodatasets:
        buildings_microsoft = (
            geodatasets['buildings_microsoft']
            .cx[long_min:long_max, lat_min:lat_max]
            .to_crs('epsg:3857')
        )
        if len(buildings_microsoft) > 0:
            buildings_microsoft.boundary.plot(ax=ax, color='magenta', linewidth=1)

    # Draw FEMA footprints
    if 'buildings_fema' in geodatasets:
        buildings_fema = (
            geodatasets['buildings_fema']
            .cx[long_min:long_max, lat_min:lat_max]
            .to_crs('epsg:3857')
        )
        if len(buildings_fema) > 0:
            buildings_fema.boundary.plot(ax=ax, color='#00ffff', linewidth=1)

    # Draw local footprints
    if 'buildings_local' in geodatasets:
        buildings_local = (
            geodatasets['buildings_local']
            .cx[long_min:long_max, lat_min:lat_max]
            .to_crs('epsg:3857')
        )
        if len(buildings_local) > 0:
            buildings_local.boundary.plot(ax=ax, color='#66ff66', linewidth=1)

    # Draw NSI footprints
    if 'buildings_nsi' in geodatasets:
        buildings_nsi = (
            geodatasets['buildings_nsi']
            .cx[long_min:long_max, lat_min:lat_max]
            .to_crs('epsg:3857')
        )
        if len(buildings_nsi) > 0:
            buildings_nsi.plot(
                ax=ax,
                color='white',
                marker='D',
                edgecolor='#00ff00',
                markersize=15,
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

    location_is_nsi = (
        isinstance(location, gpd.GeoSeries) and location.index.name == 'building_id_nsi'
    )
    if 'buildings_nsi' in geodatasets and (parcel_found or location_is_nsi):
        buildings_nsi_to_label_list = []
        if location_is_nsi:
            buildings_nsi_in_crosshair = (
                geodatasets['buildings_nsi'].loc[[location.index[0]]].reset_index()
            )
            buildings_nsi_to_label_list = [buildings_nsi_in_crosshair]
        if parcel_found:
            buildings_nsi_on_parcel = gpd.sjoin(
                parcel[['geometry']].iloc[[0]],
                geodatasets['buildings_nsi'].cx[long_min:long_max, lat_min:lat_max],
            ).sort_values('area_sqft', ascending=False)

            if location_is_nsi:
                nsi_id_column = geodatasets['buildings_nsi'].index.name
                not_in_crosshair = buildings_nsi_on_parcel[nsi_id_column].ne(
                    buildings_nsi_in_crosshair[nsi_id_column].iloc[0]
                )
                buildings_nsi_to_label_list += [
                    buildings_nsi_on_parcel[not_in_crosshair]
                ]
            else:
                buildings_nsi_to_label_list += [buildings_nsi_on_parcel]
        buildings_nsi_to_label = pd.concat(buildings_nsi_to_label_list)

        if len(buildings_nsi_to_label) == 0:
            print('No NSI points found on parcel in plot frame.')
        else:
            if verbose:
                print(
                    f'{len(buildings_nsi_to_label)} NSI points on parcel: '
                    + ', '.join(
                        buildings_nsi_to_label['nsi_id'].astype(int).astype(str)
                    )
                )
            txt_nsi_list = []
            for _, _building_nsi in buildings_nsi_to_label.head(
                N_MAX_BUILDINGS_NSI_TEXT
            ).iterrows():
                txt_nsi_list += [
                    f'NSI ID {_building_nsi["building_id_nsi"]}\n'
                    + f'{_building_nsi["purpose_subgroup"]}\n'
                    + f'Construction: {_building_nsi["construction_type"]}\n'
                    + f'Foundation: {_building_nsi["foundation_type"]}\n'
                    + 'Bldg value (2021): '
                    + f'${int(_building_nsi["structure_value"]):,d}\n'
                    + f'Source: {_building_nsi["source"]}'
                ]

            n_omitted = len(buildings_nsi_to_label) - N_MAX_BUILDINGS_NSI_TEXT
            if n_omitted > 0:
                txt_nsi_list += [f'... and {n_omitted} more']
            txt_nsi = '\n\n'.join(txt_nsi_list)
            ax.text(
                xmin + radius / 25,
                ymin + radius / 15,
                txt_nsi,
                va='bottom',
                bbox=dict(
                    facecolor='#ffffffdd',
                    edgecolor='#00ff00',
                    linewidth=1,
                    boxstyle='round',
                    pad=0.5,
                ),
            )

    if parcel_found and 'buildings_fema' in geodatasets:
        building_fema_in_crosshair = geodatasets['buildings_fema'].cx[
            long_center, lat_center
        ]

        buildings_fema_on_parcel = gpd.overlay(
            parcel[['geometry']].iloc[[0]].reset_index(),
            geodatasets['buildings_fema']
            .cx[long_min:long_max, lat_min:lat_max]
            .reset_index(),
        )
        buildings_fema_on_parcel['overlap_area_sqft'] = get_areas(
            buildings_fema_on_parcel, unit='sqft'
        )
        buildings_fema_on_parcel = buildings_fema_on_parcel[
            buildings_fema_on_parcel['overlap_area_sqft'].ge(MIN_SQFT_FEMA)
        ].copy()
        buildings_fema_on_parcel['frac_sqft'] = (
            buildings_fema_on_parcel['overlap_area_sqft']
            / buildings_fema_on_parcel['footprint_area_sqft']
        )
        buildings_fema_on_parcel = buildings_fema_on_parcel.sort_values(
            'overlap_area_sqft', ascending=False
        )

        if len(building_fema_in_crosshair) == 0:
            buildings_fema_to_label = buildings_fema_on_parcel
        else:
            not_in_crosshair = buildings_fema_on_parcel[
                building_fema_in_crosshair.index.name
            ].ne(building_fema_in_crosshair.index[0])
            buildings_fema_to_label = pd.concat(
                [
                    building_fema_in_crosshair.reset_index(),
                    buildings_fema_on_parcel[not_in_crosshair],
                ]
            )

        if len(buildings_fema_to_label) == 0:
            print('No FEMA polygons identified by location.')
        else:
            if verbose:
                print(
                    f'{len(buildings_fema_to_label)} FEMA footprints on parcel: '
                    + ', '.join(
                        buildings_fema_to_label['fema_id'].astype(int).astype(str)
                    )
                )

            txt_fema_list = []
            city = None
            for (
                _,
                building_fema_to_label,
            ) in buildings_fema_to_label.head(N_MAX_BUILDINGS_FEMA_TEXT).iterrows():
                txt_use = building_fema_to_label['purpose_subgroup'] or 'No primary use'
                address = (
                    building_fema_to_label['address']
                    if isinstance(building_fema_to_label['address'], str)
                    else 'No address'
                )
                txt_fema_list += [
                    f'FEMA ID {int(building_fema_to_label["building_id_fema"])}\n'
                    + f'{address.title()}\n'
                    + f'{txt_use.title()}\n'
                    + (
                        f'{building_fema_to_label["frac_sqft"]:,.0%} of '
                        if building_fema_to_label['frac_sqft'] < 0.99
                        else ''
                    )
                    + f'{building_fema_to_label["footprint_area_sqft"]:,.0f} sqft\n'
                    + 'Validation:'
                    f' {building_fema_to_label["validation_method"].title()}'
                ]
                city = building_fema_to_label['city'] or city
            n_omitted = len(buildings_fema_to_label) - N_MAX_BUILDINGS_FEMA_TEXT
            if n_omitted > 0:
                txt_fema_list += [f'... and {n_omitted} more']
            txt_fema = '\n\n'.join(txt_fema_list)
            ax.text(
                xmax - radius / 25,
                ymax - radius / 25,
                txt_fema,
                va='top',
                ha='right',
                bbox=dict(
                    facecolor='#ffffffdd',
                    edgecolor='#00ffff',
                    boxstyle='round',
                    linewidth=1,
                    pad=0.5,
                ),
            )

            if isinstance(city, str):
                ax.text(
                    x,
                    ymax - radius / 25,
                    'City: ' + city.title(),
                    backgroundcolor='#ffffffcc',
                    va='top',
                    ha='center',
                    bbox=dict(
                        facecolor='#ffffffdd',
                        edgecolor='#00ffff',
                        boxstyle='round',
                        linewidth=1,
                        pad=0.5,
                    ),
                )

    if parcel_found and 'buildings_local' in geodatasets:
        building_local_in_crosshair = geodatasets['buildings_local'].cx[
            long_center, lat_center
        ]

        buildings_local_on_parcel = gpd.overlay(
            parcel[['geometry']].iloc[[0]].reset_index(),
            geodatasets['buildings_local']
            .cx[long_min:long_max, lat_min:lat_max]
            .reset_index(),
        )
        buildings_local_on_parcel['overlap_area_sqft'] = get_areas(
            buildings_local_on_parcel, unit='sqft'
        )
        buildings_local_on_parcel = buildings_local_on_parcel[
            buildings_local_on_parcel['overlap_area_sqft'].ge(MIN_SQFT_FEMA)
        ].copy()
        # buildings_local_on_parcel['frac_sqft'] = (
        #     buildings_local_on_parcel['overlap_area_sqft']
        #     / buildings_local_on_parcel['footprint_area_sqft']
        # )
        buildings_local_on_parcel = buildings_local_on_parcel.sort_values(
            'overlap_area_sqft', ascending=False
        )

        if len(building_local_in_crosshair) == 0:
            buildings_local_to_label = buildings_local_on_parcel
        else:
            not_in_crosshair = buildings_local_on_parcel[
                building_local_in_crosshair.index.name
            ].ne(building_local_in_crosshair.index[0])
            buildings_local_to_label = pd.concat(
                [
                    building_local_in_crosshair.reset_index(),
                    buildings_local_on_parcel[not_in_crosshair],
                ]
            )

        if len(buildings_local_to_label) == 0:
            print('No local footprints found at location.')
        else:
            if verbose:
                print(
                    f'{len(buildings_local_to_label)} local footprints on parcel: '
                    + ', '.join(
                        buildings_local_to_label['local_id'].astype(int).astype(str)
                    )
                )

            txt_local_list = []
            city = None
            for (
                _,
                building_local_to_label,
            ) in buildings_local_to_label.head(N_MAX_BUILDINGS_FEMA_TEXT).iterrows():
                _txt_local = ''
                for column in [
                    'occupancy_type',
                    'construction_type',
                    'year_built',
                    'building_value',
                    'purpose_group',
                    'purpose_subgroup',
                ]:
                    if column in building_local_to_label:
                        col_label = (
                            (column[:12] + ('...' if len(column) > 12 else ''))
                            .replace('_', ' ')
                            .title()
                        )

                        col_value = building_local_to_label[column]
                        if isinstance(col_value, float):
                            col_value = f'{col_value:.3f}'
                        col_value = str(col_value).title()
                        if len(col_value) > 20:
                            col_value = col_value[:20] + '...'
                        _txt_local += f'{col_label}: {col_value}\n'
                txt_local_list += [_txt_local[:-2]]  # cut off last \n
            n_omitted = len(buildings_local_to_label) - N_MAX_BUILDINGS_LOCAL_TEXT
            if n_omitted > 0:
                txt_local_list += [f'... and {n_omitted} more']
            txt_local = '\n\n'.join(txt_local_list)
            ax.text(
                xmax - radius / 25,
                ymin + radius / 15,
                txt_local,
                va='bottom',
                ha='right',
                bbox=dict(
                    facecolor='#ffffffdd',
                    edgecolor='#66ff66',
                    boxstyle='round',
                    linewidth=1,
                    pad=0.5,
                ),
            )

    if return_fig_ax:
        return fig, ax
