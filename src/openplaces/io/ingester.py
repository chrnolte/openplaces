import glob
import os
import re
import ssl
import urllib
import warnings
from itertools import product
from pathlib import Path

import geopandas as gpd
import pandas as pd

from openplaces.config import cfg
from openplaces.core.attribute_registry import load_registry as _load_attr_registry
from openplaces.core.constants import (
    REGEX_FILENAME_IN_URL,
    REGEX_HAS_GLOB_WILDCARDS,
    ZIP_EXTENSIONS,
)
from openplaces.core.schema import AdminId
from openplaces.io import (
    delete_data,
    delete_parquet,
    download,
    find_latest_file_or_gdb,
    read_parquet,
    save_parquet,
    unzip,
)
from openplaces.io.aggregate import aggregate_to_admin_level
from openplaces.io.raster_ingester import fetch_rasters_by_admin
from openplaces.io.readers import get_admin, get_entities
from openplaces.io.table_ingester import TableIngester
from openplaces.path import (
    external_dir,
    heap_dir,
    path_matches_pattern,
)
from openplaces.recipe import (
    build_table_recipe,
    find_admin_recipe_id,
    get_download_admin_level,
    get_layers,
    get_output_path,
    get_partition_ids,
    get_process_admin_level,
    get_recipe_by_id,
    get_save_admin_level,
    get_table_recipe,
)
from openplaces.timing import Timer, get_timer
from openplaces.utils import format_list


def _warn_registry_type_mismatches(gdf) -> None:
    """Warn when a column's dtype disagrees with the attribute registry."""
    reg = _load_attr_registry()
    for col in gdf.columns:
        if col not in reg.index:
            continue
        expected = reg.at[col, 'data_type']
        actual = gdf[col].dtype
        if expected == 'categorical' and str(actual) not in ('category', 'object'):
            warnings.warn(
                f"Column '{col}' expected dtype categorical but got {actual}.",
                stacklevel=2,
            )
        elif expected in ('float', 'int') and not pd.api.types.is_numeric_dtype(actual):
            warnings.warn(
                f"Column '{col}' expected numeric dtype but got {actual}.",
                stacklevel=2,
            )


class Ingester:
    """
    Smart data ingester for `openplaces` ingestion recipes.

    Handles downloads, unzipping, loading, and preprocessing.
    """

    def __init__(
        self,
        recipe: str | dict = None,
        admin_ids: str | list | None = None,
        partition_ids: str | list | None = None,
        timer: Timer | None = None,
        verbose: bool = False,
    ):
        """
        Initialize an Ingester.

        Parameters
        ----------
        recipe : dict
            Data ingestion recipe (string ID or resolved recipe)
        admin_ids : str or list of str
            Identifier(s) of the administrative unit(s) to ingest.
            Required if the download links vary by admin unit.
            Can also be used to query large geodatabases/data files.
        partition_ids : str or list of str
            Identifier(s) of other partitions to download
            (e.g. year-month, tile)
        timer : openplaces.timing.Timer
            Timer
        verbose : bool
            If True, will print additional outputs during processing
        """

        if isinstance(recipe, str):
            recipe = get_recipe_by_id(recipe)
        if recipe is None:
            raise ValueError(
                '`recipe` must be a recipe dict or a valid recipe ID string.'
            )
        self.recipe = recipe

        if isinstance(admin_ids, str) or admin_ids is None:
            self.admin_ids = [AdminId(admin_ids)]
        elif isinstance(admin_ids, AdminId):
            self.admin_ids = [admin_ids]
        elif isinstance(admin_ids, list):
            self.admin_ids = [AdminId(admin_id) for admin_id in admin_ids]
        else:
            raise ValueError(
                f'Admin ID type not supported: {type(admin_ids)} ({admin_ids})'
            )

        if partition_ids is None:
            self.partition_ids = None
        elif isinstance(partition_ids, str):
            self.partition_ids = [partition_ids]
        elif isinstance(partition_ids, list | set):
            self.partition_ids = partition_ids
        else:
            raise ValueError(
                'Partition ID type not supported: '
                f'{type(partition_ids)} ({partition_ids})'
            )

        if timer is None:
            timer = get_timer('Ingester', verbose=True)
        self.timer = timer

        self.verbose = verbose

        recipe_admin_id = recipe['admin_id']
        invalid = [
            aid
            for aid in self.admin_ids
            if aid.get_level() > 0 and not recipe_admin_id.is_parent_or_equal_of(aid)
        ]
        if invalid:
            raise ValueError(
                f'Admin IDs are not children of recipe admin_id '
                f'({recipe_admin_id}): {[str(a) for a in invalid]}'
            )

        self._early_warnings()

    @property
    def _is_tile_partition(self):
        return (self.recipe.get('download_by') or {}).get('partition') == 'tile_id'

    @property
    def _process_level(self):
        """Max admin level across download_by and process_by."""
        level = self.recipe['admin_id'].get_level()
        for by in ('download_by', 'process_by'):
            if by in self.recipe and 'admin_level' in self.recipe[by]:
                level = max(level, self.recipe[by]['admin_level'])
        return level

    @staticmethod
    def _make_temp_recipe(recipe):
        """Strip save_to: admin_level so TableIngester saves one file per chunk.

        In aggregate mode the primary loop writes one parquet file per
        processing chunk.  Removing the explicit save_to.admin_level causes
        get_output_path to resolve to the process-level path instead of the
        (coarser) save-level path.
        """
        temp = dict(recipe)
        if temp.get('save_to') and 'admin_level' in temp['save_to']:
            temp['save_to'] = {
                k: v for k, v in temp['save_to'].items() if k != 'admin_level'
            }
        return temp

    @property
    def _is_aggregate_mode(self):
        """True when save_to.admin_level is coarser than the process level.

        When process_by (or download_by) is at a finer granularity than
        save_to, the ingester processes at that finer level and then
        aggregates the intermediate files up to save_to.admin_level.
        """
        save_admin_level = (self.recipe.get('save_to') or {}).get('admin_level')
        return save_admin_level is not None and save_admin_level < self._process_level

    def _early_warnings(self):
        """Warnings to throw if the initialization looks problematic"""

        if 'process_by' in self.recipe and any(
            '.gpkg' in self.recipe[key]
            for key in ['uncompressed_filename', 'compressed_filename']
            if key in self.recipe
        ):
            warnings.warn(
                "Geopackages should not be combined with 'process_by'."
                'Querying with `fids=` can run extremely slow.'
                'Read in bulk or write code to use `where=`, if faster.'
            )

    def ingest(self, reprocess=False, redownload=False, keep_unzipped=False):
        """Run the full data ingestion

        Parameters
        ----------
        reprocess : bool
            If True, re-runs the data ingestion from the original file
            even if the output data already exists.
        redownload : bool
            If True, re-downloads the original data file even if it
            already exists. Also sets `reprocess` to `True`.
        keep_unzipped : bool
            If True, keeps unzipped files in 'heap' folder after
            the download partition has been processed.
        """

        if redownload:
            reprocess = True

        self._resolve_admin_ids(reprocess)

        self._resolve_partition_ids(reprocess)

        # Partition first so all admin units within the same partition are
        # processed consecutively — this allows the downloaded file to be
        # read once and cached (e.g. for tile-partitioned recipes).
        for partition_id_to_download, admin_id_to_download in product(
            self.partition_ids_to_download, self.admin_ids_to_download
        ):
            if self.verbose and (admin_id_to_download or partition_id_to_download):
                print_txt = 'Ingesting data for '
                if admin_id_to_download is not None:
                    print_txt += f'geography: {admin_id_to_download}, '
                if partition_id_to_download is not None:
                    print_txt += f'partition: {partition_id_to_download}, '
                print(print_txt[:-2])
            self._ingest_download_partition(
                admin_id_to_download=admin_id_to_download,
                partition_id_to_download=partition_id_to_download,
                redownload=redownload,
                keep_unzipped=keep_unzipped,
            )

        if self._is_tile_partition:
            self._merge_tile_partials()

        self._aggregate_to()

    def show_ingested_geometries(self, **kwargs):
        """Plot the last ingested layer for visual inspection.

        Delegates to :func:`openplaces.viz.maps.show_ingested_geometries`.
        See that function for the full list of keyword arguments.
        """
        from openplaces.viz.maps import show_ingested_geometries

        return show_ingested_geometries(self, **kwargs)

    def show_random_entity(self):
        """Plot a random entity from the last ingested admin unit with its attributes.

        Delegates to :func:`openplaces.viz.maps.show_random_entity`.
        """
        from openplaces.viz.maps import show_random_entity

        return show_random_entity(self)

    def sample_layer(self, n=5):
        """Return a transposed sample of the principal entity DataFrame.

        Parameters
        ----------
        n : int
            Number of rows to sample.

        Returns
        -------
        pd.DataFrame
            Transposed sample of the principal entity table.
        """
        admin_id_by_level = {
            AdminId(aid).get_level(): aid
            for aid in (self.admin_ids_to_save or [])
            + (self.admin_ids_to_process or [])
            if aid is not None
        }

        layer_level = get_save_admin_level(self.recipe)
        data = get_entities(self.recipe, admin_id_by_level[layer_level])
        return data.sample(n).T

    def sample_additional_layer(self, n=5):
        """Return a transposed sample of the first additional layer.

        Parameters
        ----------
        n : int
            Number of rows to sample.

        Returns
        -------
        pd.DataFrame
            Transposed sample of the additional layer table.
        """
        layers = get_layers(self.recipe)
        if not layers:
            return None

        admin_id_by_level = {
            AdminId(aid).get_level(): aid
            for aid in (self.admin_ids_to_save or [])
            + (self.admin_ids_to_process or [])
            if aid is not None
        }

        layer = layers[0]
        print(layer)
        layer_level = get_save_admin_level(get_table_recipe(self.recipe, layer))
        data = get_entities(self.recipe, admin_id_by_level[layer_level], layer=layer)
        return data.sample(n).T

    def _merge_tile_partials(self):
        """Merge per-tile partial files into final per-admin-id output files.

        For each admin_id in `admin_ids_to_save`, reads all tile partial
        files written during the tile loop, concatenates them, writes the
        merged result to the final output path, and deletes the partials.
        Admin IDs whose data came from a single tile are handled by a simple
        rename rather than a concat.
        """
        for admin_id in self.admin_ids_to_save:
            final_path = get_output_path(self.recipe, admin_id)

            # Tile IDs that were actually downloaded for this admin_id.
            tile_ids = [
                tile_id
                for tile_id in self.partition_ids_to_download
                if admin_id in self.tile_admin_link.xs(tile_id, level=0).index
            ]
            output_tile_paths = [
                get_output_path(self.recipe, admin_id, tile_id) for tile_id in tile_ids
            ]
            existing_output_tile_paths = [
                _path for _path in output_tile_paths if _path.exists()
            ]

            if not existing_output_tile_paths:
                continue

            try:
                if len(existing_output_tile_paths) == 1:
                    existing_output_tile_paths[0].replace(final_path)
                    _geo_partial = existing_output_tile_paths[0].with_stem(
                        existing_output_tile_paths[0].stem + '_geo'
                    )
                    if _geo_partial.exists():
                        _geo_partial.replace(
                            final_path.with_stem(final_path.stem + '_geo')
                        )
                else:
                    _geo_path = existing_output_tile_paths[0].with_stem(
                        existing_output_tile_paths[0].stem + '_geo'
                    )
                    gdf_tile_list = [
                        read_parquet(_path, geom=_geo_path.exists())
                        for _path in existing_output_tile_paths
                    ]
                    gdf_merged = gpd.GeoDataFrame(pd.concat(gdf_tile_list).sort_index())
                    _warn_registry_type_mismatches(gdf_merged)
                    save_parquet(gdf_merged, final_path)
                    for _path in existing_output_tile_paths:
                        delete_parquet(_path)
            except PermissionError as e:
                raise PermissionError(
                    f'Cannot write to {final_path.name}.\n\n'
                    '\033[1m→ Close the file in QGIS / ArcGIS / Dropbox sync '
                    'and re-run.\033[0m'
                ) from e

            if self.verbose and len(existing_output_tile_paths) > 1:
                print(
                    f'Merged {len(existing_output_tile_paths)} '
                    f'tile partial(s) → {final_path.name}'
                )

    def _aggregate_to(self):
        """Aggregate process-level intermediate files into save-level outputs.

        Only runs in aggregate mode (save_to.admin_level coarser than
        process_by.admin_level).  Delegates to aggregate_to_admin_level()
        using the already-resolved admin_ids_to_save and admin_ids_to_process.
        """
        if not self._is_aggregate_mode or not self.admin_ids_to_save:
            return

        aggregate_to_admin_level(
            self.recipe,
            admin_ids_to_process=self.admin_ids_to_process,
            verbose=self.verbose,
        )

        for layer_spec in self.recipe.get('additional_layers', []):
            table_recipe = build_table_recipe(self.recipe, layer_spec)
            if not (table_recipe.get('save_to') or {}).get('admin_level'):
                continue
            aggregate_to_admin_level(
                table_recipe,
                admin_ids_to_process=self.admin_ids_to_process,
                verbose=self.verbose,
            )

    def _resolve_admin_ids(self, reprocess):
        """Resolve admin IDs to save, process, and download"""

        self._resolve_admin_ids_to_save(reprocess)

        if not self.admin_ids_to_save:
            if self.verbose:
                print('All output files found. Processing skipped.\n')
            self.admin_ids_to_process = []
            self.admin_ids_to_download = []
            return

        if self.verbose:
            print('Admin IDs of output files:', format_list(self.admin_ids_to_save))

        self._resolve_admin_ids_to_process()
        if self.verbose and self.admin_ids_to_process != self.admin_ids_to_save:
            print(
                'Admin IDs of processing chunks:',
                format_list(self.admin_ids_to_process),
            )

        self._resolve_admin_ids_to_download()
        if self.verbose:
            print(
                'Admin IDs of download partitions:',
                format_list(self.admin_ids_to_download),
            )

    def _resolve_output_admin_ids(
        self,
        operation_keys=('download_by', 'process_by', 'save_to'),
        reprocess=False,
        partition_id=None,
    ):
        """Return admin IDs for which output files should be (re-)created.

        Parameters
        ----------
        operation_keys : tuple of str
            Recipe sections to inspect for 'admin_level'. 'save_to' is
            included by default; override when calling from other recipe runners.
        reprocess : bool
            If True, include admin IDs whose output files already exist.
        partition_id : str, optional
            Forwarded to `get_output_path` when checking file existence.

        Returns
        -------
        list of str or list of None
            Admin ID strings at the save level, or `[None]` if not admin-split.
        """
        save_level = get_save_admin_level(self.recipe, operation_keys)

        if save_level == 0:
            admin_ids_all = [None]
        else:
            admin_ids_all = list(
                dict.fromkeys(get_admin(self.recipe['admin_id'], save_level).index)
            )

        admin_ids_to_save = [
            admin_id_str
            for admin_id_requested in self.admin_ids
            for admin_id_str in admin_ids_all
            if admin_id_requested.is_parent_or_equal_of(AdminId(admin_id_str))
            or AdminId(admin_id_str).is_parent_of(admin_id_requested)
        ]
        admin_ids_to_save = list(dict.fromkeys(admin_ids_to_save))

        if not reprocess:
            admin_ids_to_save = [
                admin_id
                for admin_id in admin_ids_to_save
                if not get_output_path(self.recipe, admin_id, partition_id).exists()
            ]

        return admin_ids_to_save

    def _resolve_admin_ids_to_save(self, reprocess):
        """Create list of admin_ids for which to create output files

        Used to check which files already exist.

        Parameters
        ----------
        reprocess : bool
            If False, drop admin_ids for which files already exist
            (as a result, they won't be re-processed).
            If True, keep all admin_ids, as all will be re-processed.
        """
        if not reprocess and self._is_aggregate_mode:
            all_candidates = self._resolve_output_admin_ids(
                operation_keys=('download_by', 'process_by', 'save_to'),
                reprocess=True,
            )
            self.admin_ids_to_save = [
                sid
                for sid in all_candidates
                if not get_output_path(self.recipe, sid).exists()
                or self._output_is_incomplete(sid)
            ]
        else:
            self.admin_ids_to_save = self._resolve_output_admin_ids(
                operation_keys=('download_by', 'process_by', 'save_to'),
                reprocess=reprocess,
            )

    def _output_is_incomplete(self, admin_id_to_save):
        """Is the existing output file missing any requested process-level chunks?"""
        process_level = self._process_level
        process_col = f'admin{process_level}_id'

        all_process_ids = [
            str(pid)
            for pid in get_admin(AdminId(admin_id_to_save), process_level).index
        ]
        expected = {
            pid
            for pid in all_process_ids
            if any(
                req.is_parent_or_equal_of(AdminId(pid))
                or AdminId(pid).is_parent_or_equal_of(req)
                for req in self.admin_ids
            )
        }
        if not expected:
            return False

        save_path = get_output_path(self.recipe, admin_id_to_save)
        try:
            present = set(
                pd.read_parquet(save_path, columns=[process_col])[process_col].unique()
            )
        except Exception:
            return False  # Column absent (pre-stamp file): leave as-is

        return not expected.issubset(present)

    def _resolve_admin_ids_to_process(self):
        """Create list of admin_ids to process

        Finds admin_ids to process based on admin_ids to save.

        Returns admin_ids at the level at which data is "chunked" for
        processing purposes (e.g. downloading admin-level partitions of
        partitioned downloads or querying a large geodatabase by admin).
        """

        # For tile partitions each admin unit is processed independently
        # from whichever tile(s) cover it; admin_ids_to_process == admin_ids_to_save.
        if self._is_tile_partition:
            self.admin_ids_to_process = self.admin_ids_to_save
            return

        process_by_admin_level = get_process_admin_level(self.recipe)

        if process_by_admin_level == 0:
            if self.admin_ids_to_save in ([None], []):
                admin_ids_to_process = self.admin_ids_to_save
            else:
                raise ValueError(
                    'Error not captured self.admin_ids_to_save ='
                    + self.admin_ids_to_save
                )
        elif self._is_aggregate_mode:
            # save_to is coarser than process_by: expand save-level IDs
            # to process-level via get_admin (cannot truncate upward).
            all_process_admin_ids = list(
                dict.fromkeys(
                    str(admin_id)
                    for admin_id_to_save in self.admin_ids_to_save
                    for admin_id in get_admin(
                        AdminId(admin_id_to_save), process_by_admin_level
                    ).index
                )
            )
            # If the caller specified a sub-county filter, keep only the
            # process-level IDs that overlap with the requested admin_ids.
            if any(
                requested_admin_id.get_level() > 0
                for requested_admin_id in self.admin_ids
            ):
                admin_ids_to_process = [
                    process_admin_id
                    for process_admin_id in all_process_admin_ids
                    if any(
                        requested_admin_id.is_parent_or_equal_of(
                            AdminId(process_admin_id)
                        )
                        or AdminId(process_admin_id).is_parent_or_equal_of(
                            requested_admin_id
                        )
                        for requested_admin_id in self.admin_ids
                    )
                ]
            else:
                admin_ids_to_process = all_process_admin_ids
        else:
            # List of unique admin_ids, preserving order
            admin_ids_to_process = list(
                dict.fromkeys(
                    str(AdminId(*AdminId(admin_id).levels[:process_by_admin_level]))
                    for admin_id in self.admin_ids_to_save
                )
            )

        self.admin_ids_to_process = admin_ids_to_process

    def _resolve_admin_ids_to_download(self):
        """Make list of admin_ids for which files need to be downloaded

        Finds admin_ids to download based on admin_ids to save

        Returns admin_ids at the level at which data is chunked for
        download (e.g., state-level downloads, county-level downloads).

        Used to check whether all files have been downloaded.
        """

        # For tile partitions tiles are downloaded as a whole, not split by admin.
        if self._is_tile_partition:
            self.admin_ids_to_download = [None]
            return

        download_by_admin_level = get_download_admin_level(self.recipe)

        if download_by_admin_level == 0:
            if self.admin_ids_to_process == [None]:
                admin_ids_to_download = [None]
            elif self.admin_ids_to_process == []:
                admin_ids_to_download = []
            else:
                raise ValueError(
                    'Error not captured self.admin_ids_to_download ='
                    + self.admin_ids_to_download
                )

        else:
            # List of unique admin_ids, preserving order.
            # Use admin_ids_to_process (not admin_ids_to_save) as the source so
            # that aggregate-mode recipes (save_to.admin_level coarser than
            # download_by.admin_level) correctly expand to the download level.
            admin_ids_to_download = list(
                dict.fromkeys(
                    str(AdminId(*AdminId(admin_id).levels[:download_by_admin_level]))
                    for admin_id in self.admin_ids_to_process
                )
            )

        self.admin_ids_to_download = admin_ids_to_download

    def _resolve_partition_ids(self, reprocess):
        """Resolve partition IDs to save, process, and download

        By convention, all partition IDs are strings.
        """
        download_by = self.recipe.get('download_by') or {}

        if self._is_tile_partition:
            self._resolve_tile_ids()
            return

        all_partition_ids = get_partition_ids(self.recipe)

        # Filter to requested subset if caller specified partition_ids
        if isinstance(self.partition_ids, list | set):
            self.partition_ids_to_download = [
                x for x in self.partition_ids if x in all_partition_ids
            ]
        else:
            self.partition_ids_to_download = all_partition_ids

        if self.verbose and self.partition_ids_to_download != [None]:
            partition = download_by.get('partition')
            print(
                f'Partitioned by `{partition}`:',
                format_list(self.partition_ids_to_download),
            )
            if isinstance(self.partition_ids, list | set):
                print('Selected:', format_list(self.partition_ids_to_download))

    def _resolve_tile_ids(self):
        """Resolve tile partition IDs filtered to admin_ids_to_save.

        Loads the tile-admin link table produced by overlay_polygons() and
        keeps only the tile IDs that intersect at least one admin ID in
        self.admin_ids_to_save. The link table is stored on self for reuse
        in _ingest_download_partition.
        """
        admin_overlay_specs = self.recipe['overlay_admin_ids']
        admin_recipe_id = admin_overlay_specs.get(
            'admin_recipe_id'
        ) or find_admin_recipe_id(
            self.recipe['admin_id'], admin_overlay_specs['admin_level']
        )

        tile_recipe_id = self.recipe['download_by']['tile_recipe_id']
        tiles_path = get_output_path(get_recipe_by_id(tile_recipe_id))
        tile_admin_link_path = tiles_path.with_name(
            tiles_path.stem + f'_{admin_recipe_id}.parquet'
        )
        if not tile_admin_link_path.exists():
            raise ValueError(
                'File linking {tile_recipe_id} and {admin_recipe_id} not found:\n\n'
                + str(tile_admin_link_path)
                + '\n\nDid you process the overlay?'
            )
        self.tile_admin_link = pd.read_parquet(tile_admin_link_path)

        # Level 0 = tile index, level 1 = admin index (overlay_polygons order)
        admin_ids_set = set(self.admin_ids_to_save)
        relevant = self.tile_admin_link.index.get_level_values(1).isin(admin_ids_set)
        tile_ids = (
            self.tile_admin_link[relevant].index.get_level_values(0).unique().tolist()
        )

        if isinstance(self.partition_ids, list | set):
            tile_ids = [t for t in tile_ids if t in set(self.partition_ids)]

        self.partition_ids_to_download = tile_ids

        if self.verbose:
            print(
                'Partitioned by `tile_id`:', format_list(self.partition_ids_to_download)
            )

    def _ingest_download_partition(
        self,
        admin_id_to_download=None,
        partition_id_to_download=None,
        redownload=False,
        keep_unzipped=False,
    ):
        """Run data ingestion for a download partition of the data"""

        # Initialize download partition (will be used by many functions)
        self.download_partition = {
            'admin_id_to_download': admin_id_to_download,
            'partition_id_to_download': partition_id_to_download,
        }

        self._resolve_download_url()
        if self.verbose:
            print('Download URL:', self.download_partition['download_url'])

        self._resolve_downloaded_and_data_paths()
        if self.verbose:
            print('Downloaded path:', self.download_partition['downloaded_path'])
            print('Data path:', self.download_partition['data_path'])

        if self.recipe.get('dataset') and self.recipe['dataset'].is_raster:
            fetch_rasters_by_admin(self)
            return

        self._download_and_unzip_recipe_data(redownload=redownload)

        if self._is_tile_partition:
            admin_ids_in_tile_set = set(
                self.tile_admin_link.xs(partition_id_to_download, level=0).index
            )
            admin_ids_in_tile = [
                aid for aid in self.admin_ids_to_save if aid in admin_ids_in_tile_set
            ]
            # Stored for use in _load_admin_geometries: restrict to admin
            # units in this tile so the overlay only works with relevant bounds.
            self.download_partition['admin_ids_in_tile'] = admin_ids_in_tile

            # Read, reproject, and overlay once for all admin units in this tile.
            self.processing_chunk = {'admin_id_to_process': None}
            table_ingester = self._make_table_ingester(self.recipe)
            gdf = table_ingester._read_recipe_data()
            if isinstance(gdf, gpd.GeoDataFrame) and gdf.crs != cfg.crs:
                gdf = gdf.to_crs(cfg.crs)
                self.timer.mark(f'Reproject to {cfg.crs}: {partition_id_to_download}')
            gdf = table_ingester._preprocess_recipe_data(gdf)
            self.timer.mark(f'Preprocess: {partition_id_to_download}')

            # Save one file per admin unit.
            # Mutate processing_chunk in place so table_ingester sees the updated value.
            for admin_id_to_process in admin_ids_in_tile:
                self.processing_chunk['admin_id_to_process'] = admin_id_to_process
                table_ingester._save_recipe_data(gdf)
                self.timer.mark(f'Save: {admin_id_to_process}')
        else:
            admin_ids_to_process_in_partition = [
                admin_id
                for admin_id in self.admin_ids_to_process
                if AdminId(admin_id_to_download).is_parent_or_equal_of(
                    AdminId(admin_id)
                )
            ]

            for admin_id_to_process in admin_ids_to_process_in_partition:
                if self.verbose and admin_id_to_process is not None:
                    print(
                        f'Processing {self.recipe["entity"]} for {admin_id_to_process}:'
                    )
                self._process_recipe_data(admin_id_to_process)

        # Delete unzipped files in heap folder
        if not keep_unzipped and self.download_partition['data_path'].is_relative_to(
            self.recipe_heap_dir
        ):
            if self.verbose:
                print('Deleting unzipped data.')
            delete_data(self.download_partition['data_path'])

    def _catch_missing_partition_ids_error(self):
        # Error checks
        if 'download_by' not in self.recipe:
            return
        download_by = self.recipe['download_by']
        if 'entity' in self.recipe:
            source = self.recipe['entity'].source
            _type = self.recipe['entity'].entity_type
        elif 'dataset' in self.recipe:
            source = self.recipe['dataset'].source
            _type = self.recipe['dataset'].theme

        if (
            'admin_level' in download_by
            and self.download_partition['admin_id_to_download'] is None
        ):
            raise ValueError(
                f'Download of `{_type}` from `{source}` is by admin level '
                + str(download_by['admin_level'])
                + '.\n\n'
                'Use `admin_ids` argument to identify the admin unit'
                ' to download.'
            )
        elif (
            'partition' in download_by
            and self.download_partition['partition_id_to_download'] is None
        ):
            raise ValueError(
                f'Download of `{_type}` from `{source}` is by partition `'
                + download_by['partition']
                + '`.\n\n'
                'Use `partition_ids` argument to identify the '
                'partition ID to download.'
            )

    def _get_admin_partition_key(self, placeholder):
        """Get the key of an administrative unit used by a dataset partition

        Used to obtain the correct filename for partitioned datasets.

        Example: 'US-ND' -> 'NorthDakota'

        Parameters
        ----------
        placeholder: str
            Data partition placeholder as used in the `download_url`
        """
        # Get partition key from admin data
        admin_level = self.recipe['download_by']['admin_level']
        admin_recipe_id = self.recipe['download_by'].get('admin_recipe_id')

        # Translate placeholders to admin columns / identifiers by cutting
        # off 'adminX_' prefixes unless the prefix is 'adminX_id'
        # 'admin2_name' -> 'name'
        # 'admin2_id_leaf', 'admin2_id_admin1' -> keep as is
        if placeholder.startswith(
            f'admin{admin_level}_'
        ) and not placeholder.startswith(f'admin{admin_level}_id'):
            column = placeholder.replace(f'admin{admin_level}_', '')
        else:
            column = placeholder

        if column == f'admin{admin_level}_id_leaf':
            # Special case, e.g. 'MA' for Massachusetts
            partition_key = AdminId(
                self.download_partition['admin_id_to_download']
            ).levels[-1]
        else:
            try:
                # Find the key in an official admin dataset
                if admin_recipe_id is None and admin_level > 1:
                    admin_recipe_id = find_admin_recipe_id(
                        self.recipe['admin_id'], admin_level
                    )
                partition_key = get_admin(
                    self.download_partition['admin_id_to_download'],
                    admin_level,
                    recipe=admin_recipe_id,
                    columns=column,
                ).iloc[0, 0]
            except IndexError:
                partition_key = get_admin(
                    self.download_partition['admin_id_to_download'],
                    admin_level,
                    columns=column,
                ).iloc[0, 0]

        # Transform Admin key if needed
        if (
            'admin_key_transform' in self.recipe['download_by']
            and placeholder in self.recipe['download_by']['admin_key_transform']
        ):
            key_transform = self.recipe['download_by']['admin_key_transform'][
                placeholder
            ]
            # Temporary hack: supporting only one type of transformation
            if key_transform == 'remove_spaces':
                partition_key = partition_key.replace(' ', '')
            else:
                raise NotImplementedError(
                    f'key_transform == {key_transform} not yet supported.'
                )

        return partition_key

    def _get_placeholders(self, url):
        """Extract placeholders ('{placeholder}') from URL."""
        return list(dict.fromkeys(re.findall(r'\{([a-zA-Z0-9_-]+)\}', url)))

    def _resolve_placeholders(
        self,
        url_or_path,
    ):
        """Resolve placeholders (partition keys) in a URL or filepath

        Example: {admin2_name}.geojson.zip => NorthCarolina.geojson.zip
        """

        # If partition keys aren't resolved yet, initiate empty dict
        if 'partition_key_dict' not in self.download_partition:
            self.download_partition['partition_key_dict'] = {}

        placeholders = self._get_placeholders(url_or_path)

        # Iterate through placeholders in passed path and resolve them
        for placeholder in placeholders:
            if placeholder in self.download_partition['partition_key_dict']:
                _partition_key = self.download_partition['partition_key_dict'][
                    placeholder
                ]
            else:
                if placeholder.startswith('admin'):
                    _partition_key = self._get_admin_partition_key(placeholder)
                elif (
                    self.recipe.get('download_by')
                    and self.recipe['download_by'].get('partition') == placeholder
                ):
                    _partition_key = self.download_partition['partition_id_to_download']
                else:
                    raise NotImplementedError(
                        'Custom placeholder has not yet been implemented:\n'
                        f'Placeholder: `{placeholder}`, '
                        f'`download_by`: `{self.recipe["download_by"]}`'
                    )

                self.download_partition['partition_key_dict'][placeholder] = (
                    _partition_key
                )

            url_or_path = url_or_path.replace(
                '{' + placeholder + '}', str(_partition_key)
            )

        return url_or_path

    def _resolve_download_url(
        self,
    ):
        """Resolve the download URL of a recipe

        Identifies placeholders of partitions and substitutes them

        Example
        -------

        Download URL with placeholder:
            https://[...]/nsi_2022/nsi_2022_{admin2_id_admin1}.gpkg.zip
        Resolved download URL (for North Carolina)
            https://[...]/nsi_2022/nsi_2022_37.gpkg.zip

        """
        if 'entity' in self.recipe:
            source = self.recipe['entity'].source
        elif 'dataset' in self.recipe:
            source = self.recipe['dataset'].source
        else:
            raise ValueError(
                'recipe needs an `entity` or `dataset` with a `source` for the '
                'download.'
            )

        if source.download_url is not None:
            download_url = source.download_url

            if 'download_by' in self.recipe:
                self._catch_missing_partition_ids_error()

                download_url = self._resolve_placeholders(download_url)
            else:
                # Catch error if the URL has placeholder
                placeholders_in_url = self._get_placeholders(download_url)
                if placeholders_in_url:
                    raise ValueError(
                        'Set `download_by` in ingestion recipe to resolve partition '
                        f'placeholders in download URL:\n{placeholders_in_url}'
                    )
        elif source.download_url_source is not None:
            if not self.recipe['download_by']:
                raise ValueError(
                    '`download_url_source` was provided, but '
                    '`download_by` is not defined.'
                )
            # Scrape website providing download URLs
            req = urllib.request.Request(
                source.download_url_source, headers={'User-Agent': 'Mozilla/5.0'}
            )
            ssl_context = (
                None if source.verify_ssl else ssl._create_unverified_context()
            )
            with urllib.request.urlopen(req, context=ssl_context) as response:
                html = response.read().decode('utf8')

            download_url_source_regex = self._resolve_placeholders(
                source.download_url_source_regex
            )
            download_url_found = re.compile(download_url_source_regex).findall(html)

            if not download_url_found:
                raise ValueError(
                    f'Could not extract {download_url_source_regex} from html:\n{html}'
                )

            download_url = download_url_found[0]

            if download_url.startswith('/'):
                from urllib.parse import urlparse

                # Filepaths are relative: add URL structure from download_url_source
                parsed = urlparse(source.download_url_source)
                domain_url = f'{parsed.scheme}://{parsed.netloc}'

                download_url = domain_url + download_url

        else:
            download_url = None

        self.download_partition['download_url'] = download_url

    def _resolve_downloaded_and_data_paths(self):
        """Get the paths for the data ingestion files of a recipe"""

        # Set compressed file name, if given
        compressed_file_name = None
        if 'compressed_file_name' in self.recipe:
            compressed_file_name = self.recipe['compressed_file_name']
            if 'download_by' in self.recipe:
                compressed_file_name = self._resolve_placeholders(compressed_file_name)
        if self.verbose:
            print('Compressed file name:', compressed_file_name)

        # Set uncompressed file name, if given
        uncompressed_file_name = None
        if 'uncompressed_file_name' in self.recipe:
            uncompressed_file_name = self.recipe['uncompressed_file_name']
            if 'download_by' in self.recipe:
                uncompressed_file_name = self._resolve_placeholders(
                    uncompressed_file_name
                )
        if self.verbose:
            print('Uncompressed file name:', uncompressed_file_name)

        # Set external and heap directories
        if 'entity' not in self.recipe and 'dataset' not in self.recipe:
            raise NotImplementedError(
                'Either an `entity` or a `dataset` must be defined in the recipe.'
            )
        self.recipe_heap_dir = heap_dir(
            self.recipe.get('admin_id'),
            self.recipe.get('entity'),
            self.recipe.get('dataset'),
        )
        self.recipe_external_dir = external_dir(
            self.recipe.get('admin_id'),
            self.recipe.get('entity'),
            self.recipe.get('dataset'),
        )

        # Identify path of file to import (to see whether it's already saved)
        if compressed_file_name is not None:
            if uncompressed_file_name is not None:
                # Assume that uncompressed file will be read
                data_path = self.recipe_heap_dir / uncompressed_file_name
            else:
                # Assume that compressed file will be read
                data_path = self.recipe_external_dir / compressed_file_name
        elif uncompressed_file_name is not None:
            data_path = self.recipe_external_dir / uncompressed_file_name
        else:
            data_path = None

        # Identify path of file that has been downloaded
        downloaded_path = None
        if data_path is None or not data_path.exists():
            # Find the path of the downloaded file.
            if compressed_file_name is not None:
                downloaded_path = self.recipe_external_dir / compressed_file_name
            elif uncompressed_file_name is not None:
                downloaded_path = self.recipe_external_dir / uncompressed_file_name
            elif 'download_url' in self.download_partition:
                # Try to extract filename from URL
                re_match = re.search(
                    REGEX_FILENAME_IN_URL, self.download_partition['download_url']
                )
                if re_match:
                    filename = re_match.group(1)
                    print(f'Name of downloaded file inferred from URL: {filename}')
                    downloaded_path = self.recipe_external_dir / filename

            # If the downloaded path contains wildcards, search for it
            if (
                downloaded_path is not None
                and not downloaded_path.exists()
                and re.search(REGEX_HAS_GLOB_WILDCARDS, str(downloaded_path))
            ):
                filepaths = glob.glob(str(downloaded_path))
                if len(filepaths) > 0:
                    downloaded_path = Path(max(filepaths, key=os.path.getmtime))
                    if len(filepaths) > 1:
                        print(
                            'Found more than one file. Selected most recent one:\n\n'
                            f'{downloaded_path}\n\n'
                            'Others:\n\n'
                            + '\n'.join([x for x in filepaths if x != downloaded_path])
                        )

        self.download_partition['downloaded_path'] = downloaded_path
        self.download_partition['data_path'] = data_path

    def _download_and_unzip_recipe_data(self, redownload=False):
        """Download and unzip dataset from the original source

        redownload : bool
            Set to True to skip checking for existing files (overwrite)
        """

        # Skip if the data path exists and no redownload is requested
        if (
            self.download_partition['data_path'] is not None
            and self.download_partition['data_path'].exists()
            and not redownload
        ):
            if self.verbose:
                print('Data file found. Download and unzipping skipped.')
            return

        # Download if neither downloaded file nor data file exist
        if redownload or (
            (
                self.download_partition['downloaded_path'] is None
                or not self.download_partition['downloaded_path'].exists()
            )
            and (
                self.download_partition['data_path'] is None
                or not self.download_partition['data_path'].exists()
            )
        ):
            self._catch_missing_download_url_error()

            if self.verbose:
                print('Downloading...')

            entity_or_dataset = self.recipe.get('entity') or self.recipe.get('dataset')
            verify_ssl = (
                entity_or_dataset.source.verify_ssl if entity_or_dataset else True
            )
            downloaded_path = download(
                self.download_partition['download_url'],
                self.recipe_external_dir,
                verify_ssl=verify_ssl,
            )
            self.timer.mark('Download')

            if self.download_partition['downloaded_path'] is None:
                self.download_partition['downloaded_path'] = downloaded_path
            elif self.download_partition['downloaded_path'] != downloaded_path:
                if not path_matches_pattern(
                    downloaded_path, self.download_partition['downloaded_path']
                ):
                    raise ValueError(
                        'Downloaded path from recipe does not match downloaded file\n'
                        + f'Expected:\n{self.download_partition["downloaded_path"]}\n'
                        + f'Got:\n{downloaded_path}\n'
                    )
                self.download_partition['downloaded_path'] = downloaded_path
        elif self.verbose:
            print('Downloaded data found. Skipping download.')

        if redownload or (
            self.download_partition['downloaded_path'] is not None
            and self.download_partition['downloaded_path']
            != self.download_partition['data_path']
            and Path(self.download_partition['downloaded_path']).suffix.lower()
            in ZIP_EXTENSIONS
        ):
            if self.verbose:
                print('Unzipping...')

            unzip(self.download_partition['downloaded_path'], self.recipe_heap_dir)
            self.timer.mark('Unzip')

        # Identify last extracted file if the data path is unknown
        # or contains wildcards
        if (
            self.download_partition['data_path'] is None
            or re.search(
                REGEX_HAS_GLOB_WILDCARDS, str(self.download_partition['data_path'])
            )
        ) and self.recipe_heap_dir.exists():
            self.download_partition['data_path'] = find_latest_file_or_gdb(
                self.recipe_heap_dir
            )
            if self.download_partition['data_path'] is None:
                raise ValueError(
                    f'Did not find a valid dataset in {self.recipe_heap_dir}.\n'
                    'Searched for: ' + str(self.download_partition['data_path'])
                )
            location = self.download_partition['data_path'].relative_to(
                self.recipe_heap_dir
            )
            if self.verbose:
                print(f'Inferred file to read: {location}')

        if (
            self.download_partition['data_path'] is not None
            and not self.download_partition['data_path'].exists()
        ):
            raise FileNotFoundError(
                'Did not succeed in downloading and unzipping:\n\n'
                + str(self.download_partition['data_path'])
            )

    def _catch_missing_download_url_error(self):
        entity_or_dataset = self.recipe.get('entity') or self.recipe.get('dataset')
        source = entity_or_dataset.source
        if not source.download_url and not source.download_url_source:
            error_message = ''
            if self.download_partition['downloaded_path'] is not None:
                filename = self.download_partition['downloaded_path'].relative_to(
                    cfg.data_root
                )
                error_message += (
                    '\n\nDownloaded file not found in the `openplaces` filesystem:'
                    f'\n\n{filename}\n\n'
                )
                location = str(self.download_partition['downloaded_path'])
            else:
                location = external_dir(
                    self.recipe.get('admin_id'),
                    self.recipe.get('entity'),
                    self.recipe.get('dataset'),
                )
            error_message += (
                f'Recipe for `{entity_or_dataset}` has no download URL.\n\n'
                '1. Download the data manually here:\n\n'
                + f'{source.portal_url}'
                + '\n\n2. Save it in this location:\n\n'
                + location
                + '\n\n3. Re-run this data ingestion script.'
            )
            raise FileNotFoundError(error_message)

    def _process_recipe_data(self, admin_id_to_process=None):
        """Process data from a downloaded (and unzipped) data file"""

        self.processing_chunk = {'admin_id_to_process': admin_id_to_process}

        admin_id_to_download = self.download_partition['admin_id_to_download']
        suffix = '' if admin_id_to_process is None else f': {admin_id_to_process}'

        process_in_chunks = (
            admin_id_to_download is not None
            and admin_id_to_process is not None
            and AdminId(admin_id_to_download).is_parent_of(AdminId(admin_id_to_process))
        )

        if process_in_chunks and 'process_by' not in self.recipe:
            raise ValueError(
                f'Admin ID to process ({admin_id_to_process}) is below '
                f'Admin ID to download ({admin_id_to_download}), '
                "but no 'process_by' argument found in `recipe`."
            )

        # For spatial-mask chunking, load admin geometries once and derive
        # the bounding box before creating any TableIngester.
        bbox = None
        if process_in_chunks and 'use_spatial_mask' in self.recipe.get(
            'process_by', {}
        ):
            if self.verbose:
                print('Reading with bounding box. This can be slow.')
            if 'admin_geometries' not in self.download_partition:
                self._make_table_ingester(self.recipe)._load_admin_geometries()
            bbox = (
                self.download_partition['admin_geometries']
                .loc[admin_id_to_process]
                .geometry.bounds
            )

        # Process primary table.  In aggregate mode use a temp recipe (no
        # save_to.admin_level) so each chunk is written to its own file.
        primary_recipe = (
            self._make_temp_recipe(self.recipe)
            if self._is_aggregate_mode
            else self.recipe
        )
        primary_table_ingester = self._make_table_ingester(primary_recipe)
        primary_table_ingester.process(process_in_chunks=process_in_chunks, bbox=bbox)
        self.timer.mark(f'Wrap up{suffix}')

        # Process additional tables from the same source file
        for table_spec in self.recipe.get('additional_layers', []):
            table_recipe = build_table_recipe(self.recipe, table_spec)
            if self.verbose and admin_id_to_process is not None:
                print(f'Processing {table_recipe["entity"]} for {admin_id_to_process}:')
            table_in_chunks = process_in_chunks and 'process_by' in table_recipe
            temp_table_recipe = (
                self._make_temp_recipe(table_recipe)
                if self._is_aggregate_mode
                else table_recipe
            )
            additional_table_ingester = self._make_table_ingester(temp_table_recipe)
            additional_table_ingester.process(
                process_in_chunks=table_in_chunks,
                bbox=bbox if table_in_chunks else None,
            )
            self.timer.mark(f'Wrap up {table_recipe["entity"]}{suffix}')

    def _make_table_ingester(self, table_recipe) -> TableIngester:
        """Construct a TableIngester bound to the current partition state."""
        return TableIngester(
            table_recipe,
            self.download_partition,
            self.processing_chunk,
            self.recipe_heap_dir,
            self.timer,
            self.verbose,
            self.admin_ids_to_save,
        )


def ingest(
    recipe: str | dict,
    admin_ids: str | list | None = None,
    partition_ids: str | list | None = None,
    reprocess: bool = False,
    redownload: bool = False,
    keep_unzipped: bool = False,
    verbose: bool = False,
) -> None:
    """Instantiate and run ingestion for *recipe*.

    Convenience wrapper around ``Ingester(recipe, ...).ingest()``.

    Parameters
    ----------
    recipe : str or dict
        Recipe ID string or loaded recipe dict.
    admin_ids : str, list, or None
        Admin IDs to process (passed to the Ingester constructor).
    partition_ids : str, list, or None
        Partition IDs to process (passed to the Ingester constructor).
    reprocess : bool
        If True, re-run even if output already exists.
    redownload : bool
        If True, re-download even if source file already exists.
        Also sets ``reprocess`` to ``True``.
    keep_unzipped : bool
        If True, keep unzipped files in the heap folder after processing.
    verbose : bool
        Print progress messages.
    """
    Ingester(
        recipe,
        admin_ids=admin_ids,
        partition_ids=partition_ids,
        verbose=verbose,
    ).ingest(
        reprocess=reprocess,
        redownload=redownload,
        keep_unzipped=keep_unzipped,
    )
