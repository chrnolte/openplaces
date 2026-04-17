"""
Raster ingestion from VRT sources (e.g. USGS 3DEP) via window reads.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.mask
from rasterio.io import MemoryFile
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

from openplaces.io.readers import get_admin
from openplaces.recipe import get_output_path

_GDAL_ENV = {
    'AWS_NO_SIGN_REQUEST': 'YES',
    'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
    'GDAL_NUM_THREADS': 'ALL_CPUS',
    'GDAL_TIFF_OVR_BLOCKSIZE': '512',
}


def fetch_raster_from_vrt(
    boundary: gpd.GeoDataFrame,
    output_path: str | Path,
    vrt_url: str,
) -> Path:
    """Download a raster window from a VRT and save as a Cloud Optimized GeoTIFF.

    Reads directly from the source VRT via window reads — no temporary file on
    disk. The raster is kept in its native CRS; if the input geometry is in a
    different CRS it is reprojected to match the raster.

    Parameters
    ----------
    boundary
        Clipping boundary. Must have a valid CRS.
    output_path
        Destination path for the output COG (parent dirs must exist).
    vrt_url
        URL or path of the source VRT (e.g. ``s3://prd-tnm/...``).

    Returns
    -------
    Path
        Resolved path of the saved COG.
    """
    output_path = Path(output_path)

    with rasterio.Env(**_GDAL_ENV):
        with rasterio.open(vrt_url) as src:
            clipping = boundary.to_crs(src.crs.wkt)
            shapes = [geom.__geo_interface__ for geom in clipping.geometry]

            data, transform = rasterio.mask.mask(
                src,
                shapes,
                crop=True,
                nodata=np.nan,
                all_touched=True,
            )
            profile = src.profile.copy()

        profile.update(
            driver='GTiff',
            height=data.shape[1],
            width=data.shape[2],
            transform=transform,
            nodata=np.nan,
            dtype='float32',
            count=1,
        )

        with MemoryFile() as mem:
            with mem.open(**profile) as ds:
                ds.write(data.astype('float32'))

            cog_translate(
                mem.name,
                str(output_path),
                cog_profiles.get('deflate'),
                config=_GDAL_ENV,
                use_cog_driver=True,
                in_memory=True,
                quiet=True,
            )

    return output_path


def fetch_rasters_by_admin(ingester) -> None:
    """Fetch raster windows from a VRT and save one COG per admin unit.

    Called by `Ingester._ingest_download_partition` when
    ``recipe['dataset'].is_raster`` is True.

    Parameters
    ----------
    ingester
        `openplaces.io.ingester.Ingester` instance with resolved
        ``admin_ids_to_process`` and ``download_partition``.
    """
    vrt_url = ingester.download_partition['download_url']

    for admin_id in ingester.admin_ids_to_process:
        if ingester.verbose:
            print(f'Ingesting raster for {admin_id}...')

        admin = get_admin(admin_id, geom=True)
        output_path = get_output_path(ingester.recipe, admin_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fetch_raster_from_vrt(admin, output_path, vrt_url)

        if ingester.verbose:
            print(f'Saved → {output_path}')
