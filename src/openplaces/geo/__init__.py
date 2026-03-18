# src/openplaces/geo/__init__.py

import warnings

import pyogrio


def get_crs(filepath, layer=None):
    """Get the CRS from the metadata of a file using `pyogrio`"""
    geo_metadata = pyogrio.read_info(filepath, layer=layer)
    if 'crs' not in geo_metadata:
        warnings.warn('No CRS found in input data.')
        return None
    return geo_metadata['crs']


# Submodules:
# - src/openplaces/geo/raster.py
# - src/openplaces/geo/polygon.py   — core geometry operations
# - src/openplaces/geo/overlay.py   — spatial overlay and admin-ID joins
# - src/openplaces/geo/link.py      — entity linking
# - src/openplaces/geo/vector.py    — backwards-compatible re-exports
