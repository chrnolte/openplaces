"""
Processes a single table (layer) from an already-resolved source file
into an openplaces entity output file.
"""

from pathlib import Path

import geopandas as gpd

from openplaces.config import cfg
from openplaces.geo.polygon import (
    reproject,
)
from openplaces.io.ingester.table_ids import _TableIdMixin
from openplaces.io.ingester.table_preprocess import _TablePreprocessMixin
from openplaces.io.ingester.table_read import _TableReadMixin
from openplaces.io.ingester.table_save import _TableSaveMixin
from openplaces.io.ingester.table_scope import (
    _admin_level_of,  # noqa: F401
    _TableScopeMixin,
)


class TableIngester(
    _TableScopeMixin,
    _TableReadMixin,
    _TablePreprocessMixin,
    _TableIdMixin,
    _TableSaveMixin,
):
    """
    Processes a single table from an already-resolved source file into an
    openplaces entity output file.

    Handles reading, preprocessing, and saving one layer or table (a named
    layer in a GDB or GeoPackage, or the entirety of a single-table file).
    Does not own download or chunking logic — those belong to the parent
    Ingester.

    Parameters
    ----------
    table_recipe : dict
        Recipe for this specific table. For the primary entity this is the
        full recipe dict; for additional entities it is the result of
        build_table_recipe().
    download_partition : dict
        Shared mutable state from the parent Ingester (data_path,
        admin_id_to_download, admin_id_crosswalk, table_fids,
        admin_geometries, etc.). Mutated in place to cache results.
    processing_chunk : dict
        Mutable dict holding 'admin_id_to_process'. Shared with Ingester;
        mutated in place by the tile-partition loop so that TableIngester
        always sees the current admin unit.
    recipe_heap_dir : Path
        Heap directory for the primary entity. Used as the unzip target in
        the compressed-file fallback path of _read_recipe_data.
    timer : Timer
    verbose : bool
    admin_ids_to_save : list
        Admin ID strings (or [None]) at the save level. Required only when
        the recipe uses save_to: admin_level to split output by admin unit.
    """

    def __init__(
        self,
        table_recipe: dict,
        download_partition: dict,
        processing_chunk: dict,
        recipe_heap_dir: Path,
        timer,
        verbose: bool = False,
        admin_ids_to_save: list = None,
    ):
        self.recipe = table_recipe
        self.download_partition = download_partition
        self.processing_chunk = processing_chunk
        self.recipe_heap_dir = recipe_heap_dir
        self.timer = timer
        self.verbose = verbose
        self.admin_ids_to_save = admin_ids_to_save or []

    @property
    def table_name(self) -> str:
        """Stable key for this table in per-table caches (e.g. table_fids)."""
        return self.recipe.get('layer') or str(
            self.recipe.get('entity') or self.recipe.get('dataset')
        )

    @staticmethod
    def _mark_suffix(*parts):
        present = [str(p) for p in parts if p is not None]
        return (': ' + ' | '.join(present)) if present else ''

    # Public entry point

    def process(self, process_in_chunks: bool = False, bbox=None):
        """Read, preprocess, and save data for this table.

        Parameters
        ----------
        process_in_chunks : bool
            If True, filter data to the current
            processing_chunk['admin_id_to_process'].
        bbox : tuple, optional
            Bounding-box filter (minx, miny, maxx, maxy) used instead of
            FID-based filtering when process_in_chunks is True.
        """
        admin_id_to_process = self.processing_chunk['admin_id_to_process']
        partition_id = self.download_partition.get('partition_id_to_download')
        suffix = self._mark_suffix(admin_id_to_process, partition_id)

        read_kwargs = {}
        data_path_override = None
        file_pattern = (self.recipe.get('process_by') or {}).get('file_pattern')
        if process_in_chunks and file_pattern:
            # Some sources (e.g. FL DOR's NAL rolls) ship one file per admin
            # unit already, rather than one shared file to split by FID.
            data_path_override = self._resolve_file_pattern_path(file_pattern)
            if data_path_override is None:
                # process_by.file_pattern_missing: skip - this partition's
                # download genuinely lacks this admin unit (FL DOR's 2008
                # roll ships 43 of 67 counties), the same soft gap as a
                # missing download URL. Nothing to read, nothing written.
                return None
        elif process_in_chunks:
            if bbox is not None:
                read_kwargs['bbox'] = bbox
            elif (self.recipe.get('process_by') or {}).get('use_spatial_mask'):
                # A masked chunk needs no FID map: restrict the read to
                # the admin unit's bounding box (cheap through a format
                # with a spatial index, e.g. a state GDB read per town)
                # and let the mask overlay downstream cut the exact
                # boundary. The geometries are already in the data's CRS
                # (_load_admin_geometries reprojects when loading).
                if 'admin_geometries' not in self.download_partition:
                    self._load_admin_geometries()
                read_kwargs['bbox'] = tuple(
                    self.download_partition['admin_geometries']
                    .loc[[admin_id_to_process]]
                    .total_bounds
                )
            else:
                self._ensure_table_fid_filter()
                scope = self.download_partition.get('admin_id_crosswalk_scope')
                if scope is not None and admin_id_to_process not in scope:
                    # process_by.admin_id_crosswalk named a scoped sidecar
                    # table (recipe_id form) rather than a universal
                    # census-style crosswalk -- e.g. a source that only
                    # covers a subset of a state's counties. A requested
                    # admin unit outside that scope has no rows to match
                    # by construction, so filtering would silently write
                    # an empty output; raise instead so a caller learns
                    # this admin unit is out of scope, not that the
                    # source happened to have nothing for it.
                    raise KeyError(
                        f'{admin_id_to_process} is not covered by this '
                        "recipe's process_by.admin_id_crosswalk scope. "
                        f'Covered admin units: {sorted(scope)}'
                    )
                fids_series = self.download_partition['table_fids'][self.table_name]
                read_kwargs['fids'] = list(
                    fids_series[fids_series.eq(admin_id_to_process)].index
                )

        gdf = self._read_recipe_data(
            data_path_override=data_path_override, **read_kwargs
        )

        # Some source shapefiles ship a blank/unreadable .prj (no CRS on
        # read) despite genuinely being in a known projection -- declare it
        # via the recipe's `source_crs` rather than guessing. Only applied
        # when the read geometry truly has no CRS, so it can't silently
        # override a valid one that merely differs from `source_crs`.
        if (
            isinstance(gdf, gpd.GeoDataFrame)
            and gdf.crs is None
            and self.recipe.get('source_crs')
        ):
            gdf = gdf.set_crs(self.recipe['source_crs'])

        if isinstance(gdf, gpd.GeoDataFrame) and gdf.crs != cfg.crs:
            gdf = reproject(gdf, cfg.crs)
            self.timer.mark(f'Reproject to {cfg.crs}{suffix}')

        gdf = self._preprocess_recipe_data(gdf)
        self.timer.mark(f'Preprocess{suffix}')

        gdf, properties = self._split_stacked_units(gdf, suffix)
        self._save_recipe_data(gdf)
        if properties is not None:
            self._save_recipe_data(properties, recipe=self._stacked_units_recipe())
        self.timer.mark(f'Save{suffix}')

    # FID filter helpers (per-table, cached in download_partition)
