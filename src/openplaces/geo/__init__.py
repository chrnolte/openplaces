"""
Geoprocessing functionality of openplaces.

Most functionality is in the submodules:
- src/openplaces/geo/ids.py        parcel_id, UBID, openlocationcode
- src/openplaces/geo/link.py       entity linking
- src/openplaces/geo/overlay.py    spatial overlay and admin-ID joins
- src/openplaces/geo/polygon.py    core geometry operations
- src/openplaces/geo/raster.py     zonal statistics (exact_extract)
"""

import json
import warnings
from pathlib import Path

import pyarrow.parquet as pq
import pyogrio

from openplaces.geo.tiles import add_tile_utm_derivatives

__all__ = ['get_crs', 'add_tile_utm_derivatives']


def get_crs(filepath, layer=None):
    """Get the CRS from the metadata of a file."""
    if Path(filepath).suffix == '.parquet':
        meta = pq.read_schema(filepath).metadata or {}
        geo = meta.get(b'geo')
        if geo is None:
            warnings.warn('No CRS found in input data.')
            return None
        from pyproj import CRS

        wkt = json.loads(geo).get('columns', {})
        for col_meta in wkt.values():
            # GeoParquet separates two cases this used to conflate. An
            # absent `crs` key means OGC:CRS84, so return that rather
            # than None, which callers pass straight to `to_crs`. An
            # explicit null (what geopandas writes for a frame with no
            # CRS) means the data is not georeferenced, so warn and
            # return None instead of handing None to `from_user_input`,
            # which raises.
            crs = col_meta.get('crs', 'OGC:CRS84')
            if crs is None:
                break
            return CRS.from_user_input(crs)
        warnings.warn('No CRS found in input data.')
        return None
    geo_metadata = pyogrio.read_info(filepath, layer=layer)
    if geo_metadata.get('crs') is None:
        warnings.warn('No CRS found in input data.')
        return None
    return geo_metadata['crs']
