"""
Enrichment step for vicinity coverage.

A vicinity-coverage raster encodes, per pixel, the % of some neighborhood
(e.g. a 60px-radius circle) that is positive in a source boolean raster.
Computes vicinity coverage from a raster via
:func:`~openplaces.geo.raster.compute_vicinity_coverage`, using an
FFT-convolution algorithm windowed to one admin unit and padded by
`px_radius` pixels so neighboring admin units are still accounted for.

"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import rasterio

from openplaces.geo.raster import compute_vicinity_coverage
from openplaces.io.enricher import EnrichState, _register
from openplaces.io.enricher.zonal import sample_raster
from openplaces.path import resolve_raster_path


def _covers(raster_path: Path, bounds: tuple) -> bool:
    """Whether the raster at *raster_path* extends over *bounds*.

    The cached vicinity raster is keyed on the admin unit, but it is
    computed over the bounds of whatever spine subset the run handed in:
    a run naming two towns writes a raster of just those towns. Reusing
    that file for the whole county would sample every other parcel
    against a window that does not contain it and return nodata without
    a word, so the extent is checked before reuse rather than trusted.
    One pixel of tolerance absorbs the snapping of bounds to the grid.
    """
    with rasterio.open(raster_path) as src:
        left, bottom, right, top = src.bounds
        tolerance = max(abs(src.res[0]), abs(src.res[1]))
    return (
        left - tolerance <= bounds[0]
        and bottom - tolerance <= bounds[1]
        and right + tolerance >= bounds[2]
        and top + tolerance >= bounds[3]
    )


def _admin_vicinity_raster_path(source_raster_path, admin_id, px_radius: int) -> Path:
    """Path for an admin-unit-scoped vicinity raster created from
    *source_raster_path*."""
    source_raster_path = Path(source_raster_path)
    out_dir = source_raster_path.parent / 'vicinity_coverage' / str(admin_id)
    return out_dir / f'{source_raster_path.stem}Vicinity{px_radius}px.tif'


@_register('vicinity_coverage')
def vicinity_coverage(
    state: EnrichState,
    raster_key: str,
    source_raster_path: str,
    px_radius: int = 60,
    reprocess: bool = False,
    stat: str = 'max',
    column: str | None = None,
    method: str = 'exactextract',
) -> EnrichState:
    """Create and sample a vicinity-coverage raster over parcel polygons.

    Parameters
    ----------
    source_raster_path : str
        Path to a raw boolean raster; a vicinity-coverage raster is computed
        for just the current admin unit's extent (via
        :func:`~openplaces.geo.raster.compute_vicinity_coverage`) and written
        to ``{source_raster_path's dir}/vicinity_coverage/{admin_id}/
        {stem}Vicinity{px_radius}px.tif`` before being sampled, so it can be
        reused rather than recomputed on the next run (skipped when it
        already exists and extends over the current spine's bounds,
        unless *reprocess*). A raster written by a run restricted to a
        few sub-admin units is therefore recomputed, and overwritten, by
        the next run over a wider extent.
    px_radius : int
        Neighborhood radius, in source-raster pixels.
    reprocess : bool
        Recompute and overwrite the admin-scoped vicinity raster even if it
        already exists.
    stat : str
        Statistic to compute (e.g. ``'max'``, ``'mean'``).
    column : str, optional
        Evidence column name. Defaults to ``f'{raster_key}_{stat}'``.
    method : {'exactextract', 'rasterstats', 'rasterized'}
        Zonal-statistics implementation used for the final sampling step
        (creating the vicinity raster itself is unaffected). See
        :func:`~openplaces.io.enricher.zonal.sample_raster`.
    """
    spine = state.spine
    if not isinstance(spine, gpd.GeoDataFrame):
        raise ValueError(
            "vicinity_coverage requires spine geometry; set 'spine_geom: "
            "true' on the enrich recipe."
        )

    source_raster_path = resolve_raster_path(source_raster_path)
    out_path = _admin_vicinity_raster_path(
        source_raster_path, state.admin_id, px_radius
    )

    with rasterio.open(source_raster_path) as src:
        raster_crs = src.crs
    bounds = tuple(spine.to_crs(raster_crs).total_bounds)

    if reprocess or not out_path.exists() or not _covers(out_path, bounds):
        array, transform, crs = compute_vicinity_coverage(
            source_raster_path, bounds, bounds_crs=raster_crs, px_radius=px_radius
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            out_path,
            'w',
            driver='GTiff',
            height=array.shape[0],
            width=array.shape[1],
            count=1,
            dtype='uint8',
            crs=crs,
            transform=transform,
            nodata=255,
            compress='lzw',
        ) as dst:
            dst.write(array, 1)

    return sample_raster(state, str(out_path), raster_key, stat, column, method)
