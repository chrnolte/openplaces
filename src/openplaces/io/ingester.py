"""
src/openplaces/io/ingester.py
"""

import glob
import importlib
import os
import re
import urllib
import warnings
from itertools import product
from pathlib import Path

import geopandas as gpd
import pandas as pd
from pyogrio.errors import DataSourceError

from openplaces.api import get_admin
from openplaces.config import cfg
from openplaces.core.constants import (
    GEOPANDAS_EXTENSIONS,
    PANDAS_EXTENSIONS,
    REGEX_FILENAME_IN_URL,
    REGEX_HAS_GLOB_WILDCARDS,
    ZIP_EXTENSIONS,
)
from openplaces.core.schema import AdminId
from openplaces.geo.ids import get_geo_ids
from openplaces.geo.vector import (
    get_crs,
    overlay_admin_ids,
)
from openplaces.io import (
    delete_data,
    download,
    find_latest_file_or_gdb,
    read_gdb_with_domains,
    save_parquet,
    unzip,
)
from openplaces.io.admin import find_admin_recipe_id
from openplaces.io.transform import (
    add_unique_suffix,
    apply_transformation,
    apply_transformations,
    get_crosswalk,
)
from openplaces.path import (
    cache_path,
    external_dir,
    heap_dir,
    path_matches_pattern,
    recipe_path,
)
from openplaces.recipe import get_recipe_by_id
from openplaces.timing import Timer, get_timer, log_step
from openplaces.utils import format_list


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

        self._early_warnings()

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

        for admin_id_to_download, partition_id_to_download in product(
            self.admin_ids_to_download, self.partition_ids_to_download
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

    def _resolve_admin_ids(self, reprocess):
        """Resolve admin IDs to save, process, and download"""

        self._resolve_admin_ids_to_save(reprocess)
        if self.verbose:
            if not self.admin_ids_to_save:
                print('All output files found. Processing skipped.\n')
                return
            else:
                print('Admin IDs of output files:', format_list(self.admin_ids_to_save))

        self._resolve_admin_ids_to_process()
        if self.verbose:
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
        save_by_admin_level = self.recipe['admin_id'].get_level()
        for by in ['download_by', 'process_by', 'cache_by']:
            if by in self.recipe and 'admin_level' in self.recipe[by]:
                save_by_admin_level = max(
                    save_by_admin_level, self.recipe[by]['admin_level']
                )

        # Get all admin units at the level of where data will be saved
        if save_by_admin_level == 0:
            admin_ids_to_save = [None]

        else:
            admin_all = get_admin(self.recipe['admin_id'], save_by_admin_level)

            # Pick Admin IDs from the level where data is to be saved
            # that are related to the Admin IDs requested to be ingested
            admin_ids_to_save = [
                admin_id_str
                for admin_id_requested in self.admin_ids
                for admin_id_str in admin_all.index
                if admin_id_requested.is_parent_or_equal_of(AdminId(admin_id_str))
                or AdminId(admin_id_str).is_parent_of(admin_id_requested)
            ]
            # Retain only unique IDs
            admin_ids_to_save = list(dict.fromkeys(admin_ids_to_save))

        if reprocess:
            self.admin_ids_to_save = admin_ids_to_save
        else:
            # Keep only Admin IDs for which output files do not exist
            admin_ids_to_save = [
                admin_id
                for admin_id in admin_ids_to_save
                if not self._get_output_path(admin_id).exists()
            ]
            self.admin_ids_to_save = admin_ids_to_save

    def _get_output_path(self, admin_id):
        """Returns path of destination parquet file for admin unit

        Parameters
        ----------
        admin_id : AdminId
            Admin ID of file to save
        """
        output_path = cache_path(
            admin_id,
            self.recipe.get('entity'),
            self.recipe.get('dataset'),
            filename=self.recipe.get('cache_filename'),
        )
        return output_path

    def _resolve_admin_ids_to_process(self):
        """Create list of admin_ids to process

        Finds admin_ids to process based on admin_ids to save.

        Returns admin_ids at the level at which data is "chunked" for
        processing purposes (e.g. downloading admin-level partitions of
        partitioned downloads or querying a large geodatabase by admin).
        """

        process_by_admin_level = self.recipe['admin_id'].get_level()
        for by in ['download_by', 'process_by']:
            if by in self.recipe and 'admin_level' in self.recipe[by]:
                process_by_admin_level = max(
                    process_by_admin_level, self.recipe[by]['admin_level']
                )

        if process_by_admin_level == 0:
            if self.admin_ids_to_save == [None]:
                admin_ids_to_process = [None]
            elif self.admin_ids_to_save == []:
                admin_ids_to_process = []
            else:
                raise ValueError(
                    'Error not captured self.admin_ids_to_save ='
                    + self.admin_ids_to_save
                )
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

        download_by_admin_level = self.recipe['admin_id'].get_level()
        if 'download_by' in self.recipe and 'admin_level' in self.recipe['download_by']:
            download_by_admin_level = self.recipe['download_by']['admin_level']

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
            # List of unique admin_ids, preserving order
            admin_ids_to_download = list(
                dict.fromkeys(
                    str(AdminId(*AdminId(admin_id).levels[:download_by_admin_level]))
                    for admin_id in self.admin_ids_to_save
                )
            )

        self.admin_ids_to_download = admin_ids_to_download

    def _resolve_partition_ids(self, reprocess):
        """Resolve partition IDs to save, process, and download

        By convention, all partition IDs are strings.
        """

        # Initialize default: list with no partition
        self.partition_ids_to_download = [None]

        download_by = self.recipe.get('download_by')
        if not download_by:
            return

        partition = download_by.get('partition')
        if not partition:
            return

        # Generate pool of partition IDs
        if partition == 'year':
            first = download_by.get('first')
            last = download_by.get('last')
            if first is None or last is None:
                raise ValueError(
                    'If `download_by` has `partition: year`, define `first` and `last`.'
                )
            self.partition_ids_to_download = [
                str(year) for year in list(range(first, last + 1))
            ]
        else:
            raise NotImplementedError(
                f'Partition not yet interpreted by openplaces.io.ingester: {partition}.'
            )
        if self.verbose:
            print(
                f'Partitioned by `{partition}`:',
                format_list(self.partition_ids_to_download),
            )

        # Filter partition IDs to those requested
        if isinstance(self.partition_ids, list | set):
            self.partition_ids_to_download = [
                x for x in self.partition_ids if x in self.partition_ids_to_download
            ]
            if self.verbose:
                print(
                    'Selected:',
                    format_list(self.partition_ids_to_download),
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

        self._download_and_unzip_recipe_data(redownload=redownload)

        if self.recipe.get('dataset') and self.recipe['dataset'].is_raster:
            if self.verbose:
                print('Raster is in heap folder. No further processing.')
            return

        admin_ids_to_process_in_partition = [
            admin_id
            for admin_id in self.admin_ids_to_process
            if AdminId(admin_id_to_download).is_parent_or_equal_of(AdminId(admin_id))
        ]

        for admin_id_to_process in admin_ids_to_process_in_partition:
            if self.verbose and admin_id_to_process is not None:
                print(f'Processing data for {admin_id_to_process}:')
            self._process_recipe_data(admin_id_to_process)
            self.timer.mark(
                'Wrap up'
                + (
                    f': {admin_id_to_process}'
                    if admin_id_to_process is not None
                    else ''
                )
            )

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
            with urllib.request.urlopen(req) as response:
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

            downloaded_path = download(
                self.download_partition['download_url'], self.recipe_external_dir
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

        # Initiate processing chunk
        self.processing_chunk = {'admin_id_to_process': admin_id_to_process}

        admin_id_to_download = self.download_partition['admin_id_to_download']

        timer_text_suffix = (
            '' if admin_id_to_process is None else f': {admin_id_to_process}'
        )

        # If the processing is partitioned by a data column in the
        # data file, prepare the crosswalk.
        process_data_in_chunks = (
            admin_id_to_download is not None
            and admin_id_to_process is not None
            and AdminId(admin_id_to_download).is_parent_of(AdminId(admin_id_to_process))
        )

        read_data_kwargs = {}
        if process_data_in_chunks:
            if 'process_by' not in self.recipe:
                raise ValueError(
                    f'Admin ID to process ({admin_id_to_process}) is below '
                    f'Admin ID to download ({admin_id_to_download}), '
                    "but no 'process_by' argument found in `recipe`."
                )

            if 'use_spatial_mask' in self.recipe['process_by']:
                # Reading part of geodatabase by passing polygon
                if self.verbose:
                    print('Reading with bounding box. This can be slow.')
                if 'admin_geometries' not in self.download_partition:
                    self._load_admin_geometries()
                admin_bbox_bounds = (
                    self.download_partition['admin_geometries']
                    .loc[admin_id_to_process]
                    .geometry.bounds
                )
                read_data_kwargs['bbox'] = admin_bbox_bounds
            else:
                # Reading part of geodatabase by passing feature IDs
                if 'admin_id_fids' not in self.download_partition:
                    self._prepare_admin_id_crosswalk(admin_id_to_process)
                    self._prepare_admin_id_filter_column(admin_id_to_process)
                _fids = self.download_partition['admin_id_fids']
                read_data_kwargs['fids'] = list(
                    _fids[_fids.eq(admin_id_to_process)].index
                )

        gdf = self._read_recipe_data(**read_data_kwargs)

        # Reproject vector data to default CRS
        if isinstance(gdf, gpd.GeoDataFrame) and gdf.crs != cfg.crs:
            gdf = gdf.to_crs(cfg.crs)
            self.timer.mark(f'Reproject to {cfg.crs}{timer_text_suffix}')

        # Preprocess recipe data
        # (column names, indices, remapping, NAs, categoricals)
        gdf = self._preprocess_recipe_data(gdf)
        self.timer.mark(f'Preprocess recipe data{timer_text_suffix}')

        # Save recipe data
        self._save_recipe_data(gdf)
        self.timer.mark(f'Save recipe data{timer_text_suffix}')

    def _prepare_admin_id_crosswalk(self, admin_id_to_process):
        """Get crosswalk from openplaces AdminIds to recipe AdminId"""
        # Attribute entities to administrative unit IDs
        if 'admin_id_crosswalk' in self.recipe['process_by']:
            # Use custom crosswalk
            admin_id_crosswalk_dict = self.recipe['process_by']['admin_id_crosswalk']
        else:
            admin_id_crosswalk_dict = None

        if admin_id_crosswalk_dict:
            admin_id_crosswalk_dict['admin_id'] = self.download_partition[
                'admin_id_to_download'
            ]
            self.download_partition['admin_id_crosswalk'] = get_crosswalk(
                admin_id_crosswalk_dict, flip=True
            )
        else:
            raise ValueError('No crosswalk recipe found')

    def _prepare_admin_id_filter_column(self, admin_id_to_process):
        """Prepare the mapping of row IDs (FIDs) to admin_ids"""
        admin_level_to_process = self.recipe['process_by']['admin_level']
        admin_id_column_source = self.recipe['process_by']['admin_id_column']

        admin_id_filter = self._read_recipe_data(
            columns=[admin_id_column_source],
            read_geometry=False,
            fid_as_index=True,
        )

        if 'admin_id_transformation' in self.recipe['process_by']:
            transform_config = self.recipe['process_by']['admin_id_transformation']
            transform_config['input'] = self.recipe['process_by']['admin_id_column']
            admin_id_filter = apply_transformation(admin_id_filter, transform_config)
            join_column = transform_config['output']
        else:
            join_column = admin_id_column_source

        admin_id_filter = admin_id_filter.join(
            self.download_partition['admin_id_crosswalk'],
            on=join_column,
        )

        self.download_partition['admin_id_fids'] = admin_id_filter[
            f'admin{admin_level_to_process}_id'
        ]

    def _load_admin_geometries(self):
        if (
            'process_by' in self.recipe
            and 'use_spatial_index' in self.recipe['process_by']
            and self.recipe['process_by']['use_spatial_index']
        ):
            admin_specs = self.recipe['process_by']
        elif 'overlay_admin_ids' in self.recipe:
            admin_specs = self.recipe['overlay_admin_ids']

        admin_geometries = get_admin(
            self.download_partition['admin_id_to_download'],
            admin_specs['admin_level'],
            recipe=admin_specs.get('admin_recipe_id'),
            geom=True,
        )['geometry']
        data_crs = get_crs(self.download_partition['data_path'])
        if data_crs != admin_geometries.crs:
            admin_geometries = admin_geometries.to_crs(data_crs)
        self.download_partition['admin_geometries'] = admin_geometries

    def _read_recipe_data(self, columns=None, **kwargs):
        """Read a recipe dataset

        Parameters
        ----------
        columns : list
            List of selected column names to read. Enables the reading
            of single columns, e.g. for partitioning datasets
        kwargs : dict
            Will be passed to reading function, e.g., gpd.read_file(),
            gpd.read_parquet(), pd.read_parquet()
            Can include: `read_geometry`, `fids_as_index`, `fids`
        """
        # Silence warnings from reading complex polygons
        warnings.filterwarnings('ignore', 'received a polygon with more than 100 parts')

        if 'encoding' in self.recipe:
            kwargs['encoding'] = self.recipe['encoding']

        if columns:
            timer_message_suffix = (
                ', '
                + str(len(columns))
                + ' column'
                + ('(s)' if len(columns) > 1 else '')
            )
        else:
            timer_message_suffix = ''

        layer = self.recipe['layer'] if 'layer' in self.recipe else None

        data_path = self.download_partition['data_path']
        if data_path.suffix == '.parquet':
            if 'fids' in kwargs:
                raise ValueError('`fid`-based selection might not work with `parquet`.')
            gdf = gpd.read_parquet(data_path, columns=columns, **kwargs)
            self.timer.mark('Read parquet file' + timer_message_suffix, path=data_path)
        elif data_path.suffix == '.gdb':
            gdf = read_gdb_with_domains(
                data_path, columns=columns, layer=layer, **kwargs
            )
        elif data_path.suffix in GEOPANDAS_EXTENSIONS:
            try:
                gdf = gpd.read_file(data_path, layer=layer, columns=columns, **kwargs)
            except DataSourceError:
                raise OSError(
                    f'Failed to read data file:\n\n{data_path}\n\n'
                    'Possibly an incompletely unzipped file? '
                    'If so, delete manually, and re-run unzipping.'
                )
            self.timer.mark('Read vector file' + timer_message_suffix, path=data_path)
        elif data_path.suffix in PANDAS_EXTENSIONS:
            gdf = pd.read_file(data_path, columns=columns)
            self.timer.mark('Read data table' + timer_message_suffix, path=data_path)
        elif data_path.suffix in ZIP_EXTENSIONS:
            try:
                # Try to read compressed file with `geopandas`
                # (hoping that it might be in a readable zipped format)
                gdf = gpd.read_file(data_path, layer=layer, columns=columns, **kwargs)
                self.timer.mark(
                    'Read compressed file' + timer_message_suffix, path=data_path
                )
            except (RuntimeWarning, Exception):
                unzip(data_path, self.recipe_heap_dir)

                data_path = find_latest_file_or_gdb(self.recipe_heap_dir)
                self.download_partition['data_path'] = data_path
                if data_path is None:
                    raise OSError(
                        f'`geopandas` could not read compressed file:\n\n{data_path}.'
                        '\n\n'
                        f'Could not find a dataset after unzipping to:\n\n{heap_dir}'
                    )

                gdf = gpd.read_file(data_path, layer=layer, columns=columns, **kwargs)
                self.timer.mark(
                    'Read unzipped file' + timer_message_suffix, path=data_path
                )
        else:
            raise ValueError(f'Filepath suffix not yet interpreted: {data_path.suffix}')

        warnings.filterwarnings(
            'default', 'received a polygon with more than 100 parts'
        )

        return gdf

    def _preprocess_recipe_data(self, df):
        """Preprocess imported dataset

        Handles column renaming, indexing, querying, null value filling

        Parameters
        ----------
        df : DataFrame or GeoDataFrame
            Dataframe with unprocessed data.
        """

        # Rename columns
        if 'columns' in self.recipe:
            # Rename columns
            df = df.rename(columns={v: k for k, v in self.recipe['columns'].items()})

        # Replace known NA value strings with `None`.
        if 'null_value_strings' in self.recipe:
            # Exclude country identifiers ("NA" is Namibia)
            columns_to_convert = [
                v for v in df.columns if not v.startswith('admin1_id')
            ]
            for col, na_value in product(
                columns_to_convert, self.recipe['null_value_strings']
            ):
                i_has_na_value = df[col].eq(na_value)
                if i_has_na_value.sum():
                    df.loc[i_has_na_value, col] = None

        # Filter rows
        if 'query' in self.recipe:
            df = df.query(self.recipe['query'])

        # Cast columns to categorical
        if 'columns_to_categorical' in self.recipe:
            columns_to_cast = [
                v for v in self.recipe['columns_to_categorical'] if v in df
            ]
            for column_to_cast in columns_to_cast:
                # If labels are provided, use labels as categories
                # (more human-readable)
                labels = self._get_labels(column_to_cast)
                if labels is not None:
                    values = df[column_to_cast].replace(labels)
                    categories = labels.values()
                    ordered = True
                else:
                    values = df[column_to_cast]
                    categories = None
                    ordered = False

                df[column_to_cast] = pd.Series(
                    pd.Categorical(values, categories, ordered),
                    index=values.index,
                )

        # Apply variable transformations
        # (Before crosswalks, to permit extraction of parent admin IDs)
        if 'transformations' in self.recipe:
            cols_before = list(df)
            df = apply_transformations(df, self.recipe)
            cols_added = [v for v in df if v not in cols_before]
        else:
            cols_added = []

        # Attribute entities to administrative unit IDs
        # (Before admin ID index creation, which needs parent Admin ID)
        use_spatial_mask = (
            'process_by' in self.recipe
            and 'use_spatial_mask' in self.recipe['process_by']
            and self.recipe['process_by']['use_spatial_mask']
        )
        if 'admin_id_crosswalk' in self.recipe:
            admin_id_crosswalk_dict = self.recipe['admin_id_crosswalk']
            admin_id_crosswalk_dict['admin_id'] = self.processing_chunk[
                'admin_id_to_process'
            ]
            admin_id_crosswalk = get_crosswalk(admin_id_crosswalk_dict, flip=True)

            missing_crosswalk_ids = set(df[admin_id_crosswalk.index.name]) - set(
                admin_id_crosswalk.index
            )
            if missing_crosswalk_ids:
                mask_unmatched = df[admin_id_crosswalk.index.name].isin(
                    missing_crosswalk_ids
                )
                if self.verbose:
                    warnings.warn(
                        f'\n\nImperfect crosswalk: {mask_unmatched.sum():,d} '
                        'admin IDs were not linked and will be dropped:\n\n'
                        + str(df[mask_unmatched][[v for v in df if 'name' in v]])
                        + '\n'
                    )
            df = df.join(
                admin_id_crosswalk, on=admin_id_crosswalk.index.name, how='inner'
            )
            cols_added += (
                [admin_id_crosswalk.name]
                if isinstance(admin_id_crosswalk, pd.Series)
                else list(admin_id_crosswalk)
            )

        elif use_spatial_mask or ('overlay_admin_ids' in self.recipe):
            if self.verbose:
                print(
                    'Overlaying polygons with administrative boundaries. '
                    'This can take a while.'
                )
            if 'admin_geometries' not in self.download_partition:
                self._load_admin_geometries()
            if use_spatial_mask:
                admin_specs = self.recipe['process_by']
                # Spatial bounding box already applied. Intersect only
                # with the administrative unit of interest.
                admin_geometries = (
                    self.download_partition['admin_geometries']
                    .loc[[self.processing_chunk['admin_id_to_process']]]
                    .copy()
                )
            elif 'overlay_admin_ids' in self.recipe:
                admin_specs = self.recipe['overlay_admin_ids']
                admin_geometries = self.download_partition['admin_geometries']

            kwargs_overlay = {
                k: v for k, v in admin_specs.items() if k != 'admin_recipe_id'
            }
            cols_before = set(df.columns)
            df = overlay_admin_ids(
                df,
                admin_geometries=admin_geometries,
                timer=self.timer,
                **kwargs_overlay,
            )
            cols_added += [v for v in df.columns if v not in cols_before]

        # Set index
        if str(self.recipe['entity'].entity_type) == 'parcel' and isinstance(
            df, gpd.GeoDataFrame
        ):
            # Parcels get a standardized ID: 'geo_ids' without duplicates
            # Column 'geo_id' (with possible duplicates) is also stored
            # to link up geometries and parcels.
            df['geo_id'] = get_geo_ids(df, handle_duplicates=False)
            df.index = pd.Index(add_unique_suffix(df['geo_id']), name='parcel_id')
        elif 'set_index' in self.recipe:
            # Set column as index
            if self.recipe['set_index'] not in df:
                raise ValueError(
                    'Column not found to use as index: ' + str(self.recipe['set_index'])
                )
            if df[self.recipe['set_index']].duplicated().any():
                raise ValueError(
                    f"Duplicates found in '{self.recipe['set_index']}'. "
                    'Choose other index.\n\n'
                    + str(
                        df[df[self.recipe['set_index']].duplicated(keep=False)][
                            self.recipe['set_index']
                        ]
                        .sort_values()
                        .head(5)
                    )
                )
            df = df.set_index(self.recipe['set_index'])
        elif 'create_index' in self.recipe:
            if 'function' in self.recipe['create_index']:
                if not self.recipe['create_index']['function'].startswith(
                    'openplaces.'
                ):
                    raise ValueError(
                        'Function in `create_index` must start with `openplaces.`\n'
                        'Changing this would create a security risk (run any function).'
                    )
                index_function = self._load_function(
                    self.recipe['create_index']['function']
                )
                index_function_kwargs = (
                    self.recipe['create_index']['args']
                    if 'args' in self.recipe['create_index']
                    else {}
                )
                df = index_function(df, **index_function_kwargs)
            elif 'method' in self.recipe['create_index']:
                if self.recipe['create_index']['method'] == 'prefix':
                    df.index = pd.Index(
                        self.recipe['create_index']['prefix']
                        + df[self.recipe['create_index']['column']],
                        name=self.recipe['create_index']['name'],
                    )
        elif 'index_function' in self.recipe:
            # Create index with custom function

            with log_step('Generate indices', timer=self.timer):
                if not self.recipe['index_function'].startswith('openplaces.'):
                    raise ValueError(
                        'Function in `index_function` must start with `openplaces.`\n'
                        'Changing this would create a security risk (run any function).'
                    )
                index_function = self._load_function(self.recipe['index_function'])

                df = index_function(df)

        # Drop observations by index
        if 'drop' in self.recipe:
            df = df.drop(self.recipe['drop'])

        # Double-check that the index has no duplicates
        if df.index.duplicated().any():
            raise ValueError(
                'Duplicated indices are not allowed in imported data.\n'
                'Change `index_function`, `create_index` or `set_index` column:\n'
                + str(df[df.index.duplicated(keep=False)].sort_index().head())
            )

        # Reorder columns
        if 'columns' in self.recipe:
            # Start with named columns
            cols_order = (
                [c for c in list(self.recipe['columns']) if c in df]
                + [c for c in df if c.startswith('admin') and c.endswith('_id_source')]
                + cols_added
            )
            # If requested, add any other non-geometry columns
            if self.recipe.get('keep_unnamed_columns'):
                cols_order += [
                    c for c in df if c not in cols_order + ['geo_id', 'geometry']
                ]
            # Finish by adding geometry columns
            for geo_col in ['geo_id', 'geometry']:
                if geo_col in df:
                    cols_order += [geo_col]
            df = df[cols_order]

        self.timer.mark('Preprocessing')

        return df

    def _load_function(self, path):
        module, name = path.rsplit('.', 1)
        return getattr(importlib.import_module(module), name)

    def _get_labels(self, column):
        """Get dictionary of codes > labels for a column in a recipe

        Parameters
        ----------
        column : str
            Column for which to find a CSV file with labels near the recipe
            Underscores will be converted to dashes.
            Example: if the column is 'purpose_group', the label CSV is:
            '<recipe_id>_purpose-group-labels.csv'
        """
        labels_recipe_path = recipe_path(
            self.recipe['admin_id'],
            self.recipe['entity'],
            filename=column.replace('_', '-') + '-labels.csv',
        )
        if labels_recipe_path.exists():
            labels = pd.read_csv(labels_recipe_path)
            labels = labels.set_index(labels.columns[0])[labels.columns[1]].to_dict()
            return labels
        else:
            return None

    def _save_recipe_data(self, gdf):
        """Save data from an ingested recipe

        Parameters
        ----------
        gdf : GeoDataFrame or DataFrame
            Data ready to be saved
        """

        split_dataset_by_admin = (
            'cache_by' in self.recipe and 'admin_level' in self.recipe['cache_by']
        )
        if split_dataset_by_admin:
            admin_level = self.recipe['cache_by']['admin_level']
            admin_id_col = f'admin{admin_level}_id'
            if admin_id_col not in gdf:
                raise ValueError(
                    f"Recipe says 'cache_by: admin_level: {admin_level}', but column "
                    f"'{admin_id_col}' does not exist in DataFrame:\n\n"
                    + str(gdf.sample(1).T)
                )
            admin_ids_in_data = sorted(set(gdf[admin_id_col].dropna()))
            admin_ids_to_save_in_data = [
                admin_id
                for admin_id in self.admin_ids_to_save
                if admin_id in admin_ids_in_data
            ]
            admin_ids_to_save_expected = [
                admin_id
                for admin_id in self.admin_ids_to_save
                if admin_id.startswith(self.processing_chunk['admin_id_to_process'])
            ]

            missing_admin_ids = set(admin_ids_to_save_expected) - set(
                admin_ids_to_save_in_data
            )
            if missing_admin_ids:
                txt_warnings = (
                    f'\n\n{len(missing_admin_ids)} AdminIds to save not found in data:'
                    '\n' + ', '.join(sorted(missing_admin_ids)[:15]) + '\n'
                )
                warnings.warn(txt_warnings)
        else:
            admin_ids_to_save_in_data = [self.processing_chunk['admin_id_to_process']]

        print_admin_id_progress = (
            self.verbose and not len(admin_ids_to_save_in_data) == 1
        )
        if print_admin_id_progress:
            print('Saving ', end='')
        for admin_id_to_save in admin_ids_to_save_in_data:
            if print_admin_id_progress:
                end = ', ' if admin_id_to_save != admin_ids_to_save_in_data[-1] else ''
                print(admin_id_to_save, end=end)
            if split_dataset_by_admin:
                redundant_admin_id_columns = [
                    v for v in gdf if v.startswith(f'admin{admin_level}_id')
                ]
                gdf_to_save = (
                    gdf[gdf[admin_id_col].eq(admin_id_to_save)]
                    .copy()
                    .drop(columns=redundant_admin_id_columns)
                )
            else:
                gdf_to_save = gdf.copy()

            output_path = self._get_output_path(admin_id_to_save)
            if output_path.suffix == '.parquet':
                save_parquet(gdf_to_save, output_path)
            else:
                raise NotImplementedError(
                    f'Output file type not yet supported: {output_path.suffix}'
                )

        if print_admin_id_progress:
            print('')
