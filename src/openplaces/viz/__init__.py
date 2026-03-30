"""Visualization tools for geospatial data."""

import urllib

from openplaces.viz.colors import CATEGORY_COLORS, match_palette
from openplaces.viz.maps import (
    show_building,
    show_geometry_context,
    show_ingested_geometries,
    show_random_entity,
)
from openplaces.viz.tabulation import plot_tabulation, tabulate

__all__ = [
    'CATEGORY_COLORS',
    'create_street_view_link',
    'match_palette',
    'plot_tabulation',
    'show_building',
    'show_geometry_context',
    'show_ingested_geometries',
    'show_random_entity',
    'tabulate',
]


def create_street_view_link(address):
    """Create a Google street view link from a sequence

    Parameters
    ----------
    address: list | pd.Series
        Address, ideally in the order: street, city, zip, state, country
    """

    # Combine the address components into a single string
    full_address = ', '.join(address)

    # URL encode the string (e.g., spaces become '+')
    encoded_address = urllib.parse.quote_plus(full_address)

    # Construct the final URL
    base_url = 'https://www.google.com/maps/search/?api=1&query='
    return f'{base_url}{encoded_address}'
