"""Core mapping interface for quick visualization.

This module provides the main entry points for creating maps with sensible
defaults and automatic performance optimization.
"""

import warnings

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
from pyproj import Transformer
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon

from openplaces.geo.vector import get_areas
from openplaces.recipe import get_recipe_by_id


def show_polygon_context(
    gdf: gpd.GeoDataFrame,
    idx: int | str,
    buffer_factor: float = 3.0,
    basemap_source: str = "Esri.WorldImagery",
    figsize: tuple = (12, 8),
    max_attrs: int = 20,
    title: str = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """
    Plot a polygon in geographic context with its attributes.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Source geodataframe
    idx : int or str
        Integer position or index label of target polygon
    buffer_factor : float
        Buffer size as multiple of polygon's max dimension
    basemap_source : str
        Contextily basemap provider
    figsize : tuple
        Figure size (width, height)
    max_attrs : int
        Maximum attributes to display
    title : str
        Set to overwrite the default title ('Polygon {idx}')

    Returns
    -------
    fig : Figure
    (ax_map, ax_table) : tuple of Axes
    """
    import contextily as ctx
    import matplotlib.pyplot as plt
    from pyproj import CRS

    if not isinstance(gdf, gpd.GeoDataFrame):
        raise ValueError('`gdf` is not a gpd.GeoDataFrame.')

    # Get target polygon (avoid copying entire gdf)
    if isinstance(idx, int):
        target = gdf.iloc[[idx]].copy()
    else:
        target = gdf.loc[[idx]].copy()

    if isinstance(target.geometry.iloc[0], Point):
        raise ValueError('`plot_polygon_context` is not for point geometries.')
    elif not isinstance(
        target.geometry.iloc[0], (Polygon, MultiPolygon, LineString, MultiLineString)
    ):
        raise ValueError(
            '`geometry type not recognized: ' + str(type(target.geometry.iloc[0]))
        )

    # Compute buffer in original CRS for .cx query
    # Use 1.5x generous buffer to account for projection distortion
    bounds = target.total_bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    max_dim_orig = max(width, height)
    buffer_dist_cx = max_dim_orig * buffer_factor * 1.5 / 2

    warnings.filterwarnings('ignore', 'Geometry is in a geographic CRS')
    centroid = target.geometry.centroid.iloc[0]
    warnings.filterwarnings('default', 'Geometry is in a geographic CRS')

    # Create generous bounding box for .cx in original CRS
    cx_bounds = [
        centroid.x - buffer_dist_cx,
        centroid.y - buffer_dist_cx,
        centroid.x + buffer_dist_cx,
        centroid.y + buffer_dist_cx,
    ]

    # Extract context polygons before reprojection
    context = gdf.cx[cx_bounds[0] : cx_bounds[2], cx_bounds[1] : cx_bounds[3]].copy()

    # Now reproject only the subset
    lon, lat = centroid.x, centroid.y
    ortho_crs = CRS.from_proj4(
        f"+proj=ortho +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 +datum=WGS84 +units=m"
    )

    target_ortho = target.to_crs(ortho_crs)
    context_ortho = context.to_crs(ortho_crs)

    # Calculate max_dim in orthographic CRS
    bounds_ortho = target_ortho.total_bounds
    width_ortho = bounds_ortho[2] - bounds_ortho[0]
    height_ortho = bounds_ortho[3] - bounds_ortho[1]
    max_dim_ortho = max(width_ortho, height_ortho)

    # Create figure with two panels
    fig, (ax_map, ax_table) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={'width_ratios': [7, 3], 'wspace': 0}
    )

    # Plot context polygons
    context_ortho.plot(
        ax=ax_map, facecolor='none', edgecolor='white', linewidth=0.5, alpha=0.6
    )

    # Plot target polygon
    target_ortho.boundary.plot(ax=ax_map, edgecolor='yellow', linewidth=2)

    # Add basemap
    ctx.add_basemap(
        ax_map, crs=ortho_crs.to_string(), source=basemap_source, attribution=False
    )

    # Set axis limits to desired buffer (tighter than .cx query)
    target_centroid_ortho = target_ortho.geometry.centroid.iloc[0]
    buffer_dist_plot = max_dim_ortho * buffer_factor / 2
    ax_map.set_xlim(
        target_centroid_ortho.x - buffer_dist_plot,
        target_centroid_ortho.x + buffer_dist_plot,
    )
    ax_map.set_ylim(
        target_centroid_ortho.y - buffer_dist_plot,
        target_centroid_ortho.y + buffer_dist_plot,
    )

    ax_map.set_axis_off()
    ax_map.set_title(title if title else f"Polygon {idx}", fontsize=12, pad=10)

    # Prepare attribute table
    attrs = target.iloc[0].drop('geometry')

    # Select "interesting" columns: non-null, convert to string, take first N
    interesting = []
    for col, val in attrs.items():
        if val is not None and val != '' and str(val).lower() != 'nan':
            interesting.append((col, str(val)))

    interesting = interesting[:max_attrs]

    # Display attribute table
    ax_table.axis('off')

    if interesting:
        table_data = [[k, v] for k, v in interesting]
        table = ax_table.table(
            cellText=table_data,
            colLabels=['Attribute', 'Value'],
            cellLoc='left',
            loc='upper left',
            bbox=[0, 0, 1, 1],
            colWidths=[0.4, 0.6],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        # Enable text wrapping for all cells
        for key, cell in table.get_celld().items():
            cell.set_text_props(wrap=True)
            cell.PAD = 0.05

        # Style header
        for i in range(2):
            table[(0, i)].set_facecolor('#E0E0E0')
            table[(0, i)].set_text_props(weight='bold')
    else:
        ax_table.text(
            0.5, 0.5, 'No attributes to display', ha='center', va='center', fontsize=10
        )

    ax_table.set_title('Attributes', fontsize=12, pad=10)

    plt.tight_layout(w_pad=0)

    return fig, (ax_map, ax_table)


def show_building(
    location,
    geodatasets,
    radius=150,
    size=10,
    show_basemap=True,
    return_fig_ax=False,
    verbose=False,
):
    """Show building in its context with basemap

    Parameters
    ----------
    location : tuple or gpd.GeoDataFrame
        A single-row GeoDataFrame
    geodatasets: dict of gpd.GeoDataFrame or dict of lists of GeoDataFrames
        Loaded geodatasets to be plotted alongside location.
        'parcels': parcel GeoDataFrame
        'buildings': building GeoDataFrame
    radius : float
        Radius of plot in EPSG:3857 "meters" (~1.3m in NY)
    size : float
        Size of plot in inches (both height and width of `figsize`)
    show_basemap : bool
        If True, a basemap is added, and colors of parcels are adjusted.
    return_fig_ax : bool
        If True, return the plot's Figure and Axes objects
    verbose : bool
        If True, prints IDs of parcels and FEMA polygons.
    """

    to_3857 = Transformer.from_crs('EPSG:4326', 'EPSG:3857').transform
    to_4326 = Transformer.from_crs('EPSG:3857', 'EPSG:4326').transform

    # Get centroid coordinates in EPSG:3857
    if isinstance(location, tuple):
        lat, long = location[0], location[1]
    elif isinstance(location, gpd.GeoDataFrame):
        # Compute centroid of first geometry
        warnings.filterwarnings('ignore', '.*geographic CRS.*')
        location = location.iloc[[0]]['geometry'].centroid
        warnings.filterwarnings('default', '.*geographic CRS.*')
        if location.crs != 'epsg:4326':
            location = location['geometry'].to_crs('epsg:4326')
        lat, long = location.iloc[0].y, location.iloc[0].x
    else:
        raise TypeError(f'Type of `location` is not yet supported: {type(location)}.')

    # Get outer bounds in EPSG:3857
    x, y = to_3857(lat, long)
    xmin, xmax = x - radius, x + radius
    ymin, ymax = y - radius, y + radius

    # Convert outer bounds to lat/long for quick spatial selection with .cx
    lat_min, long_min = to_4326(xmin, ymin)
    lat_max, long_max = to_4326(xmax, ymax)
    # if verbose:
    #     print(f'cx[{long_min}:{long_max}, {lat_min}:{lat_max}]')
    #     print(f'({xmin}, {xmax}), ({ymin}, {ymax})')

    fig, ax = plt.subplots(figsize=(size, size))
    ax.set_xlim(xmin, xmax)  # Needs to happen before adding a basemap
    ax.set_ylim(ymin, ymax)

    # Try to show basemap
    if show_basemap:
        # try:
        cx.add_basemap(
            ax,
            crs='epsg:3857',
            source=cx.providers.Esri.WorldImagery,
            alpha=0.8,
        )
        # except:
        # print(
        #     'No internet connection: could not add basemap. '
        #     'Set `show_basemap` to `False`.'
        # )
        # show_basemap = False

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
        parcel = geodatasets['parcels'].cx[long:long, lat:lat]
        if len(parcel) == 0:
            print('No parcel found.')
        else:
            parcel_found = True
            if len(parcel) > 1:
                print('Warning: multiple overlapping parcels found.')

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

        txt_p_list = []
        for _gid, _p_txt in parcel.iterrows():
            if verbose:
                print(f'Parcel GID: {_gid}')
            txt_p_list += [
                f'Parcel ID {_p_txt['parcel_id_admin2']}\n'
                # + f'{_p_txt['geo_zone_name'].title().split(' - ')[1][:25]}\n'
                + f'{_p_txt['purpose_group'].title()[:25]}\n'
                + f'{_p_txt['address'].title()}\n'
                + f'Value (2025): ${int(_p_txt['value']):,d}\n'
                + f'Bldg value (2025): ${int(_p_txt['building_value']):,d}\n'
                + f'Owner: {_p_txt['owner_name'].title()[:25]}'
            ]
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

    # Add NSI info
    N_MAX_NSI = 4
    if parcel_found and 'buildings_nsi' in geodatasets:

        nsi_use_codes = (
            get_recipe_by_id('US_building-usace-2022_purpose-group-codebook')
            .set_index('code')['label']
            .to_dict()
        )
        nsi_use_codes = {
            k: v.replace(' Residential', '').replace(' housing', '')
            for k, v in nsi_use_codes.items()
        }

        buildings_nsi_on_parcel = gpd.sjoin(
            parcel[['geometry']].iloc[[0]],
            geodatasets['buildings_nsi'].cx[long_min:long_max, lat_min:lat_max],
        ).sort_values('area_sqft', ascending=False)
        if len(buildings_nsi_on_parcel) == 0:
            print('No NSI points found on parcel in plot frame.')
        else:
            if verbose:
                print(
                    f'{len(buildings_nsi_on_parcel)} NSI points on parcel: '
                    + ', '.join(
                        buildings_nsi_on_parcel['nsi_id'].astype(int).astype(str)
                    )
                )
            txt_nsi_list = []
            for (
                _,
                buildings_nsi_on_parcel_item
            ) in buildings_nsi_on_parcel.head(N_MAX_NSI).iterrows():
                nsi_use_code = nsi_use_codes[
                    buildings_nsi_on_parcel_item['purpose_subgroup']
                ]
                txt_nsi_list += [
                    f'NSI ID {buildings_nsi_on_parcel_item['building_id_usace']}\n'
                    + f'{nsi_use_code}\n'
                    + 'Bldg value (2021): '
                    + f'${int(buildings_nsi_on_parcel_item['structure_value']):,d}'
                ]
            if len(buildings_nsi_on_parcel) > N_MAX_NSI:
                txt_nsi_list += [
                    f'... and {len(buildings_nsi_on_parcel)-N_MAX_NSI} more'
                ]
            txt_nsi = '\n\n'.join(txt_nsi_list)
            ax.text(
                xmin + radius / 25,
                ymin + radius / 15,
                txt_nsi,
                # backgroundcolor='#ffffffcc',
                va='bottom',
                bbox=dict(
                    facecolor='#ffffffdd',
                    edgecolor='#00ff00',
                    linewidth=1,
                    boxstyle='round',
                    pad=0.5,
                ),
            )

    # Maximum listings of footprints
    N_MAX_FEMA = 4

    # Minimum size over overlap (ignore slivers)
    M2_TO_SQFT = 10.7639
    MIN_SQFT_FEMA = 10 * M2_TO_SQFT

    if parcel_found and 'buildings_fema' in geodatasets:
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

        if len(buildings_fema_on_parcel) == 0:
            print('No FEMA polygons points found on parcel in plot frame.')
        else:
            if verbose:
                print(
                    f'{len(buildings_fema_on_parcel)} FEMA footprints on parcel: '
                    + ', '.join(
                        buildings_fema_on_parcel['fema_id'].astype(int).astype(str)
                    )
                )

            txt_fema_list = []
            city = None
            for (
                _,
                fema_building_on_parcel,
            ) in buildings_fema_on_parcel.head(N_MAX_FEMA).iterrows():
                txt_use = (
                    fema_building_on_parcel['purpose_subgroup']
                    or 'No primary use'
                )
                address = (fema_building_on_parcel['address'] or 'No address')
                txt_fema_list += [
                    f'FEMA ID {int(fema_building_on_parcel['building_id_fema'])}\n'
                    + f'{txt_use.title()}\n'
                    + f'{address.title()}\n'
                    + (
                        f'{fema_building_on_parcel['frac_sqft']:,.0%} of '
                        if fema_building_on_parcel['frac_sqft'] < 0.99
                        else ''
                    )
                    + f'{fema_building_on_parcel['footprint_area_sqft']:,.0f} ft²\n'
                    + f'{fema_building_on_parcel['validation_method'].title()}'
                ]
                city = fema_building_on_parcel['city'] or city
            if len(buildings_fema_on_parcel) > N_MAX_FEMA:
                txt_fema_list += [
                    f'... and {len(buildings_fema_on_parcel) - N_MAX_FEMA} more.'
                ]
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
            if city:
                ax.text(
                    x,
                    ymax - radius / 25,
                    'City: ' + city.title(),
                    backgroundcolor='#ffffffcc',
                    va='top',
                    ha='center',
                )
