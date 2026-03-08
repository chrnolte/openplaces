"""Visualization tools for geospatial data."""

import urllib

from openplaces.viz.maps import show_building, show_geometry_context

__all__ = ['create_street_view_link', 'show_building', 'show_geometry_context']


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
