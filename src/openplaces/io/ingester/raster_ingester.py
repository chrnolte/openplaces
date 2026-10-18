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
    nodata: int | float | None = None,
) -> Path:
    """Download a raster window from a VRT and save as a Cloud Optimized GeoTIFF.

    Reads directly from the source VRT via window reads — no temporary file on
    disk. The raster is kept in its native CRS and dtype; if the input geometry
    is in a different CRS it is reprojected to match the raster.

    Nodata resolution order: ``nodata`` argument → ``src.nodata`` → dtype
    sentinel (max for unsigned int, min for signed int, NaN for float).

    Parameters
    ----------
    boundary
        Clipping boundary. Must have a valid CRS.
    output_path
        Destination path for the output COG (parent dirs must exist).
    vrt_url
        URL or path of the source VRT (e.g. ``s3://prd-tnm/...``).
    nodata
        Override nodata value. If None, the source nodata or a dtype-based
        sentinel is used.

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

            # Priority: caller nodata → src.nodata → dtype sentinel
            out_nodata = nodata if nodata is not None else src.nodata
            if out_nodata is None:
                _np_dtype = np.dtype(src.dtypes[0])
                if np.issubdtype(_np_dtype, np.unsignedinteger):
                    out_nodata = int(np.iinfo(_np_dtype).max)
                elif np.issubdtype(_np_dtype, np.signedinteger):
                    out_nodata = int(np.iinfo(_np_dtype).min)
                else:
                    out_nodata = float('nan')

            data, transform = rasterio.mask.mask(
                src,
                shapes,
                crop=True,
                nodata=out_nodata,
                all_touched=True,
            )
            profile = src.profile.copy()
            descriptions = src.descriptions

        _write_cog(data, transform, profile, out_nodata, output_path, descriptions)

    return output_path


def _write_cog(data, transform, profile, nodata, output_path, descriptions=None):
    """Write a (bands, rows, cols) array as a deflate COG.

    Every band is kept: a multi-band source (one band per year, as in an
    annual land-cover stack) clips to a multi-band COG whose band
    descriptions are those of the source.
    """
    profile = dict(profile)
    profile.update(
        driver='GTiff',
        height=data.shape[1],
        width=data.shape[2],
        transform=transform,
        nodata=nodata,
        count=data.shape[0],
    )
    with MemoryFile() as mem:
        with mem.open(**profile) as ds:
            ds.write(data)
            for i, text in enumerate(descriptions or (), start=1):
                if text and i <= data.shape[0]:
                    ds.set_band_description(i, text)

        cog_translate(
            mem.name,
            str(output_path),
            cog_profiles.get('deflate'),
            config=_GDAL_ENV,
            use_cog_driver=True,
            in_memory=True,
            quiet=True,
        )


def fetch_raster_from_tiles(
    boundary: gpd.GeoDataFrame,
    output_path: str | Path,
    tile_paths: list[Path],
    nodata: int | float | None = None,
) -> Path:
    """Clip a product shipped as several local tiles to one boundary.

    A product exported in tiles (Google Earth Engine splits a large export
    into a grid of GeoTIFFs) is mosaicked only where the boundary needs
    it: the tiles that intersect the boundary's bounds are merged over
    those bounds, then masked to the boundary. Building a VRT would do the
    same, but GDAL's Python bindings are not a dependency of this
    package, and `rasterio.merge` reads only the window it is given.

    Parameters
    ----------
    boundary
        Clipping boundary. Must have a valid CRS.
    output_path
        Destination path for the output COG.
    tile_paths
        The tiles, all in one CRS, resolution and band layout.
    nodata
        Value for pixels outside the boundary or outside every tile; the
        source nodata, else 0, when not given.

    Returns
    -------
    Path
        Resolved path of the saved COG.
    """
    from rasterio.coords import disjoint_bounds
    from rasterio.merge import merge

    output_path = Path(output_path)
    with rasterio.Env(**_GDAL_ENV):
        sources = [rasterio.open(p) for p in tile_paths]
        try:
            crs = sources[0].crs
            clipping = boundary.to_crs(crs.wkt)
            bounds = tuple(clipping.total_bounds)
            hits = [s for s in sources if not disjoint_bounds(bounds, s.bounds)]
            if not hits:
                raise ValueError(f'No tile intersects the boundary of {output_path}.')
            out_nodata = nodata if nodata is not None else (hits[0].nodata or 0)
            data, transform = merge(hits, bounds=bounds, nodata=out_nodata)
            profile = hits[0].profile.copy()
            descriptions = hits[0].descriptions
        finally:
            for s in sources:
                s.close()

        with MemoryFile() as mem:
            profile.update(
                driver='GTiff',
                height=data.shape[1],
                width=data.shape[2],
                transform=transform,
                nodata=out_nodata,
                count=data.shape[0],
            )
            with mem.open(**profile) as ds:
                ds.write(data)
                shapes = [g.__geo_interface__ for g in clipping.geometry]
                data, transform = rasterio.mask.mask(
                    ds, shapes, crop=True, nodata=out_nodata, all_touched=True
                )

        _write_cog(data, transform, profile, out_nodata, output_path, descriptions)

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
    dataset = ingester.recipe.get('dataset')
    recipe_nodata = dataset.nodata if dataset is not None else None

    # A product placed by hand as a set of tiles names them with a glob in
    # `tile_files`, relative to the recipe's external directory.
    tile_glob = ingester.recipe.get('tile_files')
    tile_paths = None
    vrt_url = None
    if tile_glob:
        tile_paths = sorted(ingester.recipe_external_dir.glob(tile_glob))
        if not tile_paths:
            raise FileNotFoundError(
                f'No tiles match {tile_glob!r} in {ingester.recipe_external_dir}.'
            )
    elif ingester.download_partition.get('data_path') is not None:
        # Recipe specifies a compressed source (e.g. tar.gz): download and
        # extract to heap first, then open the local file with rasterio.
        ingester._download_and_unzip_recipe_data()
        vrt_url = str(ingester.download_partition['data_path'])
    else:
        vrt_url = ingester.download_partition['download_url']

    for admin_id in ingester.admin_ids_to_process:
        if ingester.verbose:
            print(f'Ingesting raster for {admin_id}...')

        admin = get_admin(admin_id, geom=True)
        # `clip_buffer_m` widens the clip beyond the unit, for a raster a
        # neighborhood statistic reads (a vicinity share needs the pixels
        # across the boundary). Buffered in the unit's UTM zone.
        buffer_m = ingester.recipe.get('clip_buffer_m')
        if buffer_m:
            utm = admin.estimate_utm_crs()
            admin = admin.to_crs(utm).buffer(buffer_m).to_frame('geometry')
            admin = admin.set_geometry('geometry').to_crs(4326)
        output_path = get_output_path(ingester.recipe, admin_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if tile_paths is not None:
            fetch_raster_from_tiles(
                admin, output_path, tile_paths, nodata=recipe_nodata
            )
        else:
            fetch_raster_from_vrt(admin, output_path, vrt_url, nodata=recipe_nodata)

        if ingester.verbose:
            print(f'Saved → {output_path}')

        derive_recipe_layers(ingester.recipe, admin_id, output_path, ingester.verbose)


def derived_layer_path(recipe, admin_id, layer: str, year: int):
    """Path of one derived (layer, year) raster of a raster dataset recipe.

    Derived layers sit beside the recipe's own output as partitions named
    ``{layer}_{year}``, so ``get_dataset(recipe, admin_id,
    partition_id='forest_2005')`` reads them and the DAG's output path
    rules apply unchanged.
    """
    return get_output_path(recipe, admin_id, partition_id=f'{layer}_{year}')


def derive_recipe_layers(recipe, admin_id, source_path, verbose=False):
    """Run a raster recipe's ``derived_layers`` against its fetched output.

    A no-op for recipes without the key. ``band_first_year`` on the recipe
    names the year of band 1 in a multi-band source.
    """
    specs = recipe.get('derived_layers')
    if not specs:
        return {}
    from openplaces.geo.raster_derive import derive_layers

    return derive_layers(
        source_path,
        specs=specs,
        first_year=recipe.get('band_first_year'),
        verbose=verbose,
        path_fn=lambda layer, year: derived_layer_path(recipe, admin_id, layer, year),
    )
