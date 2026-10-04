"""
Orchestrates download, unzip, and processing of a recipe into output parquet files.

`Ingester` keeps its constructor, `ingest`, the per-partition driver and
table processing here; its other methods live in mixins, one module per
concern since 2026-09-29 (until then one 2,518-line module): `scope`
(which units are saved, processed, downloaded), `partitions` (partition,
tile and placeholder ids), `download` (URLs, paths, download, unzip,
scrapers) and `outputs` (entity links, tile merges, table joins,
aggregation). A test that patches a name a method looks up patches the
module the method lives in: `openplaces.io.ingester.download.unzip`, not
`openplaces.io.ingester.unzip` (AGENTS.md, module layer hierarchy).
"""

import re
import warnings
from itertools import product

import geopandas as gpd

from openplaces.config import cfg
from openplaces.core.schema import AdminId
from openplaces.io import delete_data
from openplaces.io.ingester._helpers import (  # noqa: F401
    _SCRAPER_MODULE_CACHE,
    _match_extracted_file,
    _transform_partition_key,
    _warn_registry_type_mismatches,
)
from openplaces.io.ingester.download import _DownloadMixin
from openplaces.io.ingester.outputs import _OutputMixin
from openplaces.io.ingester.partitions import _PartitionMixin
from openplaces.io.ingester.raster_ingester import fetch_rasters_by_admin
from openplaces.io.ingester.scope import _AdminScopeMixin
from openplaces.io.ingester.table_ingester import TableIngester
from openplaces.io.readers import get_entities
from openplaces.recipe import (
    STACKED_UNITS_LAYER_KEY,
    build_table_recipe,
    get_layers,
    get_partition_ids,
    get_process_admin_level,
    get_recipe_by_id,
    get_save_admin_level,
    get_table_recipe,
)
from openplaces.timing import Timer, get_timer
from openplaces.utils import inspect_table


class Ingester(_OutputMixin, _AdminScopeMixin, _PartitionMixin, _DownloadMixin):
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
        stage = recipe.get('stage', 'ingest')
        if stage != 'ingest':
            raise ValueError(
                f"Recipe stage is '{stage}', expected 'ingest'. "
                'Pass an ingestion recipe.'
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

        self._owns_timer = timer is None
        if timer is None:
            timer = get_timer('Ingester', verbose=verbose, overwrite=True)
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
        partition = (self.recipe.get('download_by') or {}).get('partition')
        return partition in ('tile_id', 'latlon_tile')

    @property
    def _is_joined_table_partition(self):
        """True when the recipe declares `join_partitions_by` -- column-wise join
        of `download_by: {partition: table}` outputs, distinct from `aggregate_by`
        (row-concat) and from tile-partition merging."""
        return bool(self.recipe.get('join_partitions_by'))

    @property
    def _process_level(self):
        """Max admin level across download_by and process_by.

        Delegated rather than recomputed: the local copy omitted the
        deprecated `cache_by: admin_level` term, so on a recipe still
        carrying it `_is_aggregate_mode` and
        `_resolve_admin_ids_to_process` disagreed about the level.
        """
        return get_process_admin_level(self.recipe)

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

    @staticmethod
    def _mark_suffix(*parts):
        present = [str(p) for p in parts if p is not None]
        return (': ' + ' | '.join(present)) if present else ''

    def _recipe_id(self):
        """Human-readable recipe id, e.g. 'US-WI_transaction-widor-2026'."""
        entity_or_dataset = self.recipe.get('entity') or self.recipe.get('dataset')
        return self.recipe['admin_id'].to_prefix() + str(entity_or_dataset)

    def _chunk_label(self, admin_id, n_admins):
        """Identifier(s) distinguishing the current chunk, for log headers.

        Shows admin_id when several admins are processed, partition_id when
        several partitions are, both when both vary, else the single relevant
        one. Lets a one-admin / many-partition run read 'for <partition_id>'.
        """
        partition_id = self.download_partition.get('partition_id_to_download')
        parts = getattr(self, 'partition_ids_to_download', None) or []
        n_parts = len([p for p in parts if p is not None])
        bits = []
        if n_admins > 1 and admin_id is not None:
            bits.append(str(admin_id))
        if n_parts > 1 and partition_id is not None:
            bits.append(str(partition_id))
        if not bits:
            bits = [str(x) for x in (admin_id, partition_id) if x is not None][:1]
        return ' | '.join(bits) or '(all)'

    @property
    def _first_partition_id(self):
        if self.partition_ids:
            return self.partition_ids[0]
        try:
            return get_partition_ids(self.recipe)[0]
        except NotImplementedError:
            return None

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

        if self.recipe.get('join_partitions_by'):
            download_by = self.recipe.get('download_by') or {}
            if download_by.get('partition') != 'table' or not download_by.get(
                'table_names'
            ):
                raise ValueError(
                    "'join_partitions_by' requires "
                    'download_by: {partition: table, table_names: [...]}.'
                )

    def ingest(
        self,
        reprocess=False,
        redownload=False,
        keep_unzipped=False,
        target_recipe_id: str | None = None,
    ):
        """Run the full data ingestion

        Parameters
        ----------
        reprocess : bool
            If True, re-runs the data ingestion from the original file
            even if the output data already exists.
        redownload : bool
            If True, re-downloads the original data file even if it
            already exists. Also sets `reprocess` to `True`. For image
            recipes, missing imagery is always fetched on first ingest;
            `redownload` only re-fetches images that already exist on disk
            (otherwise cached images are reused).
        keep_unzipped : bool
            If True, keeps unzipped files in 'heap' folder after
            the download partition has been processed.
        target_recipe_id : str, optional
            For image recipes only: recipe ID of the harmonized entity to
            photograph (e.g. ``'US_building-nsi-2022'``).  Overrides
            ``entity_recipe`` in the image recipe YAML.
        """

        if redownload:
            reprocess = True

        try:
            if self.recipe.get('image_scraper'):
                # Imagery has no ingest stage. Google's Static API policy
                # prohibits storing or caching content, so images are fetched
                # in memory by the enrichment step that consumes them (see
                # io.ingester.image_ingester.fetch_images_in_memory) and are
                # never written to disk. An image recipe therefore carries
                # camera configuration only and produces no output.
                if self.verbose:
                    print(
                        f'{self._recipe_id()}: imagery is fetched on the fly '
                        'during enrichment; nothing to ingest.'
                    )
                return

            self._resolve_admin_ids(reprocess)

            self._resolve_partition_ids(reprocess)

            # Partition first so all admin units within the same partition
            # are processed consecutively — this allows the downloaded file
            # to be read once and cached (e.g. for tile-partitioned recipes).
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

            if self._is_joined_table_partition:
                self._join_table_partitions()

            self._aggregate_to()

            self._aggregate_partitions(reprocess)

            self.timer.mark('Aggregate')

            self._create_entity_links(reprocess)

            self._record_units_the_source_does_not_cover(reprocess)
        finally:
            if self._owns_timer:
                self.timer.finish()

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

        admin_id = self.admin_ids_to_save[0] if self.admin_ids_to_save else None
        return show_random_entity(self.recipe, admin_id, self._first_partition_id)

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
        layer_level = get_save_admin_level(self.recipe)
        admin_id = next(
            (
                aid
                for aid in (self.admin_ids_to_save or [])
                + (self.admin_ids_to_process or [])
                if aid is not None and AdminId(aid).get_level() == layer_level
            ),
            None,
        )
        partition_id = self._first_partition_id
        data = get_entities(self.recipe, admin_id, partition_id=partition_id)
        return inspect_table(data, n=n)

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

        layer = layers[0]
        print(layer)
        table_recipe = get_table_recipe(self.recipe, layer)
        layer_level = get_save_admin_level(table_recipe)
        admin_id = next(
            (
                aid
                for aid in (self.admin_ids_to_save or [])
                + (self.admin_ids_to_process or [])
                if aid is not None and AdminId(aid).get_level() == layer_level
            ),
            None,
        )
        partition_id = self._first_partition_id
        data = get_entities(
            self.recipe, admin_id, layer=layer, partition_id=partition_id
        )
        return inspect_table(data, n=n)

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

        # A download scraper may report this partition has no published file
        # (e.g. a not-yet-generated month); skip processing and move on.
        if self.download_partition.get('unavailable'):
            return

        if self._is_tile_partition and (
            self.download_partition['data_path'] is None
            or not self.download_partition['data_path'].exists()
        ):
            return

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
                suffix = self._mark_suffix(
                    admin_id_to_process, partition_id_to_download
                )
                self.timer.mark(f'Save{suffix}')
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
                    target = self._chunk_label(
                        admin_id_to_process, len(self.admin_ids_to_process or [])
                    )
                    print(f'Processing {self._recipe_id()} for {target}:')
                self._process_recipe_data(admin_id_to_process)

        # Delete unzipped files in heap folder. A process_by.file_pattern
        # recipe reads a different physical file per admin unit (see
        # TableIngester._resolve_file_pattern_path), but download_partition
        # ['data_path'] stays pinned to the single sentinel file used for
        # the "already unzipped" check -- deleting only that path would
        # leave every other admin unit's file behind forever. Delete every
        # file matching this partition instead, regardless of which admin
        # subset was actually requested (the source zip stays on disk, so a
        # later run touching a different admin unit just re-extracts).
        if not keep_unzipped:
            process_by = self.recipe.get('process_by') or {}
            if process_by.get('file_pattern'):
                partition_glob = process_by['file_pattern'].replace(
                    '{partition_id}',
                    str(self.download_partition.get('partition_id_to_download')),
                )
                partition_glob = re.sub(r'\{[^}]+\}', '*', partition_glob)
                # Collapse any '**' created by a placeholder substitution
                # landing next to a literal '*' in the pattern (e.g. an
                # admin-code placeholder immediately followed by a wildcard
                # absorbing the rest of a filename) -- pathlib rejects '**'
                # unless it's an entire path component on its own.
                partition_glob = re.sub(r'\*+', '*', partition_glob)
                if self.verbose:
                    print('Deleting unzipped data.')
                for matched_path in self.recipe_heap_dir.glob(partition_glob):
                    delete_data(matched_path)
            elif self.download_partition['data_path'].is_relative_to(
                self.recipe_heap_dir
            ):
                if self.verbose:
                    print('Deleting unzipped data.')
                delete_data(self.download_partition['data_path'])

    def _process_recipe_data(self, admin_id_to_process=None):
        """Process data from a downloaded (and unzipped) data file"""

        self.processing_chunk = {'admin_id_to_process': admin_id_to_process}

        admin_id_to_download = self.download_partition['admin_id_to_download']
        partition_id = self.download_partition.get('partition_id_to_download')
        suffix = self._mark_suffix(admin_id_to_process, partition_id)

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
        # Truthiness, not key presence: `use_spatial_mask: false` was read
        # as "masked" here and as "not masked" by the TableIngester, which
        # gave a bbox-clipped read whose out-of-unit rows were never dropped.
        if process_in_chunks and (self.recipe.get('process_by') or {}).get(
            'use_spatial_mask'
        ):
            if self.verbose:
                print('Reading with bounding box. This can be slow.')
            if 'admin_geometries' not in self.download_partition:
                self._make_table_ingester(self.recipe)._load_admin_geometries()
            # `admin_geometries` is a GeoSeries, so .loc[...] is already the
            # shapely geometry; .bounds on it is the (minx, miny, maxx, maxy)
            # tuple the readers want.
            bbox = (
                self.download_partition['admin_geometries']
                .loc[admin_id_to_process]
                .bounds
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

        # Process additional tables from the same source file. The
        # stacked-units property layer is not read from the file: the
        # primary parcel table's own processing wrote it.
        for table_spec in self.recipe.get('additional_layers', []):
            if table_spec.get(STACKED_UNITS_LAYER_KEY):
                continue
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
    years: int | list[int] | None = None,
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
    years : int, list of int, or None
        Four-digit calendar years to process. Registry recipes receive this
        filter directly. For standard recipes partitioned by year or
        year-month, it is converted to ``partition_ids``.
    """
    if isinstance(recipe, str):
        recipe = get_recipe_by_id(recipe)
    if years and partition_ids:
        raise ValueError('Pass either partition_ids or years, not both.')

    # A `scraper` block names a dedicated ingester orchestrator (the scraper
    # crawls records directly rather than producing a downloadable file, so it
    # cannot use the standard download->TableIngester flow). The value of
    # `scraper.ingester` (or a bare `scraper` scalar) selects which one. This
    # is distinct from `source.download_url_scraper`, which is a file-producing
    # browser scraper handled inside the standard Ingester.
    scraper = recipe.get('scraper')
    if scraper is not None:
        ingester_name = (
            scraper.get('ingester') if isinstance(scraper, dict) else scraper
        )
        if ingester_name == 'registry':
            from openplaces.io.ingester.registry_ingester import RegistryIngester

            RegistryIngester(
                recipe,
                admin_ids=admin_ids,
                partition_ids=partition_ids,
                years=years,
                reprocess=reprocess,
                verbose=verbose,
            ).ingest()
            return
        raise ValueError(
            f"Unknown scraper ingester '{ingester_name}'. Supported: 'registry'."
        )

    if years:
        partition = (recipe.get('download_by') or {}).get('partition')
        if partition not in ('year', 'year_month'):
            raise ValueError(
                'The years filter requires a recipe partitioned by year or '
                f"year_month, not '{partition}'."
            )

        available = get_partition_ids(recipe)
        selected = []
        year_values = [years] if isinstance(years, int) else years
        for value in dict.fromkeys(str(year) for year in year_values):
            if len(value) != 4 or not value.isdigit():
                raise ValueError(
                    f"Invalid year '{value}'. Use four-digit YYYY values; "
                    'use partition_ids for specific YYYYMM months.'
                )
            matches = [
                partition_id
                for partition_id in available
                if partition_id == value
                or (partition == 'year_month' and partition_id.startswith(value))
            ]
            if not matches:
                raise ValueError(
                    f"No partitions matching '{value}' are available in this recipe."
                )
            selected.extend(
                partition_id for partition_id in matches if partition_id not in selected
            )
        partition_ids = selected

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
