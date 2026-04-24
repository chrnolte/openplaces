# src/openplaces/geo/__init__.py

import warnings
from pathlib import Path

import pyogrio


def get_crs(filepath, layer=None):
    """Get the CRS from the metadata of a file."""
    if Path(filepath).suffix == '.parquet':
        import json

        import pyarrow.parquet as pq

        meta = pq.read_schema(filepath).metadata or {}
        geo = meta.get(b'geo')
        if geo is None:
            warnings.warn('No CRS found in input data.')
            return None
        from pyproj import CRS

        wkt = json.loads(geo).get('columns', {})
        for col_meta in wkt.values():
            if 'crs' in col_meta:
                return CRS.from_user_input(col_meta['crs'])
        warnings.warn('No CRS found in input data.')
        return None
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
