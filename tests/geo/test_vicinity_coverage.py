"""Tests for `compute_vicinity_coverage` at a raster edge and over nodata.

Both cases come up on the same kind of admin unit: a coastal or border
county whose bounding box reaches past the source raster, and whose
offshore pixels are nodata rather than 0.
"""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import from_origin

from openplaces.geo.raster import compute_vicinity_coverage


def _write_raster(path, data, nodata, dtype):
    """Write *data* as a 1-band north-up GeoTIFF with 1-unit pixels."""
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=dtype,
        crs='EPSG:3857',
        transform=from_origin(0, data.shape[0], 1, 1),
        nodata=nodata,
    ) as dst:
        dst.write(data.astype(dtype), 1)
    return path


def test_bounds_reaching_past_the_raster_stay_georeferenced(tmp_path):
    """A window starting left of and above the raster must not wrap."""
    data = np.ones((20, 20))
    path = _write_raster(tmp_path / 'ones.tif', data, None, 'float32')

    # Bounds overhang the raster by 5 pixels on the left and the top.
    array, transform, _ = compute_vicinity_coverage(path, (-5, 5, 10, 25), px_radius=2)

    # The output covers only the part of the bounds inside the raster,
    # and its origin is that clipped corner, not the overhang corner.
    assert array.shape == (15, 10)
    assert (transform.c, transform.f) == (0.0, 20.0)
    assert (array == 100).all()


def test_nodata_is_excluded_from_the_neighborhood(tmp_path):
    """A uint8 nodata sentinel must not be read as a covered pixel."""
    data = np.full((20, 20), 255, dtype='uint8')
    data[:, :10] = 1
    data[:, 10:15] = 0
    path = _write_raster(tmp_path / 'nodata.tif', data, 255, 'uint8')

    array, _, _ = compute_vicinity_coverage(path, (0, 0, 20, 20), px_radius=2)

    # Nothing may exceed 100%: before the fix the 255 sentinel
    # counted as covered, and the uint8 cast wrapped modulo 256.
    observed = array[array != 255]
    assert observed.max() <= 100
    assert array[10, 2] == 100  # deep in the covered half
    assert array[10, 12] == 0  # deep in the observed-but-uncovered band
    assert array[10, 18] == 255  # nothing observed nearby at all
