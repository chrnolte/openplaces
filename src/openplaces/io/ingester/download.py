"""Resolving a download URL and the downloaded and data paths,
downloading and unzipping, and running a download scraper.
"""

from __future__ import annotations

import glob
import os
import re
import ssl
import urllib
import warnings
from pathlib import Path

from openplaces.config import cfg
from openplaces.core.constants import (
    REGEX_FILENAME_IN_URL,
    REGEX_HAS_GLOB_WILDCARDS,
    ZIP_EXTENSIONS,
)
from openplaces.io import (
    download,
    find_latest_file_or_gdb,
    request_headers,
    unzip,
)
from openplaces.io.ingester._helpers import (  # noqa: F401
    _SCRAPER_MODULE_CACHE,
    _match_extracted_file,
    _transform_partition_key,
    _warn_registry_type_mismatches,
)
from openplaces.io.usage_profile import require_usage_compatible
from openplaces.path import (
    external_dir,
    heap_dir,
    path_matches_pattern,
)
from openplaces.recipe import (
    get_recipe_id,
)


class _DownloadMixin:
    """Methods of :class:`~openplaces.io.ingester.Ingester` (download).

    State lives on the Ingester instance.
    """

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
            if not self.recipe.get('download_by'):
                raise ValueError(
                    '`download_url_source` was provided, but '
                    '`download_by` is not defined.'
                )
            # Scrape website providing download URLs
            req = urllib.request.Request(
                source.download_url_source, headers=request_headers()
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

        if (self.recipe.get('download_by') or {}).get('partition') == 'latlon_tile':
            entity_or_dataset = self.recipe.get('entity') or self.recipe.get('dataset')
            source_id = entity_or_dataset.source.source_id
            version = str(entity_or_dataset.version or 'latest')
            type_str = str(
                entity_or_dataset.entity_type
                if hasattr(entity_or_dataset, 'entity_type')
                else entity_or_dataset.theme
            )
            recipe_prefix = f'{type_str}-{source_id}-{version}'
            tile_id = self.download_partition.get('partition_id_to_download')
            self.recipe_heap_dir = heap_dir(
                self.recipe.get('admin_id'), entity_or_dataset
            )
            self.recipe_external_dir = external_dir(
                self.recipe.get('admin_id'), entity_or_dataset
            )
            cache_path = self.recipe_external_dir / f'{recipe_prefix}_{tile_id}.parquet'
            self.download_partition['downloaded_path'] = None
            self.download_partition['data_path'] = cache_path
            return

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
        # Both directories are keyed on the same unit, so the "already
        # downloaded" and "already unzipped" checks agree about which
        # partition a file on disk belongs to.
        #
        # That unit is the partition's own download unit. A national recipe
        # whose source ships one file per state (`download_by:
        # {admin_level: 2}`) keeps each under that state's directory;
        # building the path from the recipe's `admin_id` sent every
        # partition to the country-level path, so only one state's file
        # could ever be found, and a recipe with a fixed
        # `uncompressed_file_name` resolved to one heap path shared by
        # every state, where a leftover extraction satisfied the next
        # state's skip check and it processed the previous state's data.
        #
        # A `download_by.partition_key_transformation` is the exception: it
        # deliberately maps several download units onto one download (NSI's
        # New England towns each resolve to their county's file). Keying on
        # the town hid the sibling's copy and re-fetched the same
        # multi-hundred-MB file once per town, about 1,500 fetches across
        # New England. Those recipes keep their files in the recipe's own
        # directory, where the resolved file name is what tells the
        # partitions apart.
        shares_download_across_units = bool(
            (self.recipe.get('download_by') or {}).get('partition_key_transformation')
        )
        partition_admin_id = (
            self.recipe.get('admin_id')
            if shares_download_across_units
            else (
                self.download_partition.get('admin_id_to_download')
                or self.recipe.get('admin_id')
            )
        )
        self.recipe_heap_dir = heap_dir(
            partition_admin_id,
            self.recipe.get('entity'),
            self.recipe.get('dataset'),
        )
        self.recipe_external_dir = external_dir(
            partition_admin_id,
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
            elif self.download_partition.get('download_url'):
                # Try to extract filename from URL. `download_url` is present
                # but None whenever the source declares no URL at all (a
                # manually placed file, or a browser-driven scraper), which
                # `re.search` cannot take.
                re_match = re.search(
                    REGEX_FILENAME_IN_URL, self.download_partition['download_url']
                )
                if re_match:
                    filename = re_match.group(1)
                    if self.verbose:
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
                elif self.download_partition.get('download_url'):
                    # No existing file matches the wildcard pattern (nothing
                    # downloaded yet). Save under the concrete filename from
                    # the resolved download URL instead of the literal
                    # wildcard name, which some filesystems (e.g. NTFS)
                    # reject as invalid.
                    re_match = re.search(
                        REGEX_FILENAME_IN_URL,
                        self.download_partition['download_url'],
                    )
                    if re_match:
                        downloaded_path = self.recipe_external_dir / re_match.group(1)

        self.download_partition['downloaded_path'] = downloaded_path
        self.download_partition['data_path'] = data_path

    def _download_and_unzip_recipe_data(self, redownload=False):
        """Download and unzip dataset from the original source

        redownload : bool
            Set to True to skip checking for existing files (overwrite)
        """

        if (self.recipe.get('download_by') or {}).get('partition') == 'latlon_tile':
            from openplaces.io.ingester.cloud_geoparquet_ingester import (
                fetch_latlon_tile_to_cache,
            )

            download_by = self.recipe.get('download_by') or {}
            fetch_latlon_tile_to_cache(
                download_url=self.download_partition['download_url'],
                tile_id=self.download_partition['partition_id_to_download'],
                tile_size_deg=float(download_by.get('tile_size_deg', 1.0)),
                bbox_column=download_by.get('bbox_column', 'bbox'),
                cache_path=self.download_partition['data_path'],
                s3_anonymous=download_by.get('s3_anonymous', False),
                s3_region=download_by.get('s3_region'),
                redownload=redownload,
                verbose=self.verbose,
            )
            return

        # Skip if the data path exists and no redownload is requested
        if (
            self.download_partition['data_path'] is not None
            and self.download_partition['data_path'].exists()
            and not redownload
        ):
            if self.verbose:
                print('Data file found. Download and unzipping skipped.')
            return

        # Usage gate: a source whose terms condition access on who is
        # asking (non-commercial only, a restricted environment, a
        # jurisdictional interest) is checked against the declared usage
        # profile before any download mechanism runs. Placed here, once,
        # rather than per-mechanism, because every mechanism below passes
        # this point. A resolved decline reuses the unavailable-partition
        # soft skip, so a batch run moves on instead of dying; only the
        # unresolvable unattended case raises (inside the gate itself).
        entity_or_dataset = self.recipe.get('entity') or self.recipe.get('dataset')
        source = entity_or_dataset.source if entity_or_dataset else None
        if (
            source is not None
            and getattr(source, 'usage_requirement', None) is not None
            and not source.usage_requirement.is_empty()
        ):
            compatible = require_usage_compatible(
                source,
                recipe_id=get_recipe_id(self.recipe),
                admin_id=str(self.recipe['admin_id']),
                verbose=self.verbose,
            )
            if not compatible:
                self.download_partition['unavailable'] = True
                return

        # Browser-driven acquisition: when the source names a download scraper,
        # drive a real browser to produce the per-partition file. This is the
        # dynamic-portal analog of download_url_source (which extracts a link
        # from static HTML); the scraper writes the final file directly, so no
        # HTTP download or unzip step follows.
        if source is not None and getattr(source, 'download_url_scraper', None):
            self._run_download_scraper(
                source.download_url_scraper, redownload=redownload
            )
            return

        _dl_suffix = self._mark_suffix(
            self.download_partition.get('admin_id_to_download'),
            self.download_partition.get('partition_id_to_download'),
        )

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
            # A partitioned recipe (e.g. one zip per year) with no download
            # mechanism at all -- typically request-only historical data,
            # manually placed on disk -- treats a missing partition as
            # "not yet available" rather than an error: skip it and move on,
            # reusing the same unavailable-partition short-circuit that
            # download scrapers use for an unpublished period.
            entity_or_dataset = self.recipe.get('entity') or self.recipe.get('dataset')
            source = entity_or_dataset.source if entity_or_dataset else None
            if (
                (self.recipe.get('download_by') or {}).get('partition')
                and source is not None
                and not source.download_url
                and not source.download_url_source
                and not getattr(source, 'download_url_scraper', None)
            ):
                warnings.warn(
                    f'\n\nNo download URL and no local file found for partition '
                    f"'{self.download_partition.get('partition_id_to_download')}'. "
                    'Skipping (assumed not yet available).\n',
                    stacklevel=2,
                )
                self.download_partition['unavailable'] = True
                return

            self._catch_missing_download_url_error()

            if self.verbose:
                print('Downloading...')

            entity_or_dataset = self.recipe.get('entity') or self.recipe.get('dataset')
            verify_ssl = (
                entity_or_dataset.source.verify_ssl if entity_or_dataset else True
            )
            _download_target = (
                self.download_partition['downloaded_path']
                if self.download_partition['downloaded_path'] is not None
                else self.recipe_external_dir
            )
            downloaded_path = download(
                self.download_partition['download_url'],
                _download_target,
                verify_ssl=verify_ssl,
                headers=self.recipe.get('download_headers'),
            )
            self.timer.mark(f'Download{_dl_suffix}')

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

        # Unzip if the downloaded file is a container that wraps the actual
        # data file (downloaded_path != data_path and extension is an archive
        # format). When downloaded_path IS data_path the source is a flat file
        # (e.g. a GeoTIFF or GeoParquet) fetched directly, so no extraction
        # is needed. Check both the final suffix and recognized compound
        # suffixes so names such as data.gpkg.bz2 and data.shp.zip work.
        _dl_path = self.download_partition['downloaded_path']
        _dl_suffixes = (
            [suffix.lower() for suffix in Path(_dl_path).suffixes]
            if _dl_path is not None
            else []
        )
        _archive_suffixes = set(_dl_suffixes)
        if len(_dl_suffixes) >= 2:
            _archive_suffixes.add(''.join(_dl_suffixes[-2:]))
        _is_archive = (
            _dl_path is not None
            and _dl_path != self.download_partition['data_path']
            and bool(_archive_suffixes & ZIP_EXTENSIONS)
        )
        # Extraction is driven purely by whether the download IS an archive.
        # `redownload` must not force it: unzip() has no
        # skip-if-already-extracted branch, so archives re-extract on every
        # run regardless, and forcing it on a flat file (a .geojson or .csv
        # fetched directly) fed unzip a non-archive and raised BadZipFile.
        if _is_archive:
            if self.verbose:
                print('Unzipping...')

            # `extract_members` names the members this recipe reads,
            # so a roll whose archive also holds multi-GB tables no
            # recipe reads does not unpack them into the heap, where
            # the cleanup below (which deletes only `data_path`)
            # would leave them behind.
            unzip(
                self.download_partition['downloaded_path'],
                self.recipe_heap_dir,
                members=self.recipe.get('extract_members'),
                verbose=self.verbose,
            )
            self.timer.mark(f'Unzip{_dl_suffix}')

        # If the expected flat path wasn't created by extraction, search
        # recursively (tar archives often add nested subdirectories).
        if (
            self.download_partition['data_path'] is not None
            and not re.search(
                REGEX_HAS_GLOB_WILDCARDS,
                str(self.download_partition['data_path']),
            )
            and not self.download_partition['data_path'].exists()
            and self.recipe_heap_dir.exists()
        ):
            _found = _match_extracted_file(
                self.recipe_heap_dir, self.download_partition['data_path']
            )
            if _found is not None:
                self.download_partition['data_path'] = _found
                if self.verbose:
                    _rel = _found.relative_to(self.recipe_heap_dir)
                    print(f'Extracted file found at: {_rel}')

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

    def _run_download_scraper(self, scraper_name, redownload=False):
        """Produce a partition's source file via a browser-driven scraper.

        Invoked from `_download_and_unzip_recipe_data` when the source defines
        ``download_url_scraper``.  For data reachable only through an
        interactive web portal (a JavaScript app behind a terms-of-use gate,
        with download URLs generated on the fly), the named scraper drives a
        browser to obtain the file for the current partition and writes it to
        the partition's ``downloaded_path`` (== ``data_path`` for a flat file
        such as a CSV).  Processing then continues through the normal
        `TableIngester` path.

        Parameters
        ----------
        scraper_name : str
            Value of ``source.download_url_scraper``. Names a module file in
            ``openplaces/io/scrapers/`` (stem only, e.g.
            ``'US-WI_transaction-widor-2026_scraper'``). The module is loaded by
            path — geography-specific scrapers are named like their recipe ID
            and so contain hyphens that a normal import cannot handle — and must
            expose a ``fetch(partition_id, target_path, portal_url,
            admin_id_to_download, redownload, **options)`` entrypoint
            (``admin_id_to_download`` is the current admin unit for recipes
            partitioned by ``admin_level`` rather than a partition key;
            scrapers that don't need it, or don't need ``redownload``, should
            still accept and ignore the keyword).
        redownload : bool
            Forwarded to the scraper's ``fetch`` so it can bypass any of its
            own "already downloaded/extracted" shortcuts. This method is only
            reached when the caller has already decided a fetch is needed
            (`_download_and_unzip_recipe_data`'s own exists-and-not-redownload
            skip happens before this is called), so scrapers that fetch
            unconditionally on every call don't need to consult this flag
            themselves.
        """
        data_path = self.download_partition['data_path']
        target_path = self.download_partition['downloaded_path'] or data_path
        if target_path is None:
            raise ValueError(
                'A `download_url_scraper` source must define a resolvable target '
                'filename (e.g. via `uncompressed_file_name`) so the scraper '
                'knows where to save the downloaded file.'
            )
        target_path.parent.mkdir(parents=True, exist_ok=True)

        partition_id = self.download_partition.get('partition_id_to_download')
        entity_or_dataset = self.recipe.get('entity') or self.recipe.get('dataset')
        source = entity_or_dataset.source if entity_or_dataset else None
        options = dict(self.recipe.get('scraper_options') or {})

        if self.verbose:
            target = self._chunk_label(
                self.download_partition.get('admin_id_to_download'),
                len(self.admin_ids_to_download or []),
            )
            print(f'Scrape download {self._recipe_id()} for {target}')

        fetch = self._load_scraper_fetch(scraper_name)
        result = fetch(
            partition_id=partition_id,
            target_path=target_path,
            portal_url=options.pop('portal_url', None)
            or (source.portal_url if source else None),
            admin_id_to_download=self.download_partition.get('admin_id_to_download'),
            label=options.pop('label', None) or scraper_name.removesuffix('_scraper'),
            redownload=redownload,
            verbose=self.verbose,
            **options,
        )

        _dl_suffix = self._mark_suffix(
            self.download_partition.get('admin_id_to_download'), partition_id
        )
        self.timer.mark(f'Downloaded{_dl_suffix}')

        # A scraper returns None when this partition has no published file yet
        # (e.g. a not-yet-generated month). Mark it so the partition is skipped
        # without aborting a multi-partition run.
        if result is None:
            if self.verbose:
                print(f'No source file available; skipping partition: {partition_id}')
            self.download_partition['unavailable'] = True
            unit = self.download_partition.get('admin_id_to_download')
            if unit is not None:
                self.__dict__.setdefault('_unavailable_units', set()).add(str(unit))
            return

        if not target_path.exists():
            raise FileNotFoundError(
                f"Scraper '{scraper_name}' did not produce the expected file:\n\n"
                f'{target_path}\n\n'
                'The portal layout may have changed; check scraper selectors.'
            )

        self.download_partition['downloaded_path'] = target_path
        if data_path is None:
            self.download_partition['data_path'] = target_path

    @staticmethod
    def _load_scraper_fetch(scraper_name):
        """Load a download scraper module by file path and return its ``fetch``.

        Scraper modules live in ``openplaces/io/scrapers/``. Geography-specific
        scrapers are named like their recipe ID (e.g.
        ``US-WI_transaction-widor-2026_scraper``) and contain hyphens, so they
        are loaded by file path rather than imported by dotted name. The
        loaded module is cached in `_SCRAPER_MODULE_CACHE` by *scraper_name*
        (same name always maps to the same file) so it is loaded and executed
        once per process, like a normal `import` -- this call runs once per
        (admin unit, partition), so without the cache a scraper's own
        module-level state (e.g. an in-process cache it keeps for itself)
        would be silently reset before every single call.

        Parameters
        ----------
        scraper_name : str
            Module file stem (without ``.py``).

        Returns
        -------
        callable
            The module's ``fetch`` entrypoint.
        """
        import importlib.util
        from pathlib import Path as _Path

        module = _SCRAPER_MODULE_CACHE.get(scraper_name)
        if module is None:
            scrapers_dir = _Path(__file__).parent.parent / 'scrapers'
            module_path = scrapers_dir / f'{scraper_name}.py'
            if not module_path.exists():
                raise ValueError(
                    f"Unknown download_url_scraper '{scraper_name}': no module at "
                    f'{module_path}.'
                )
            spec = importlib.util.spec_from_file_location(
                f'_op_scraper_{scraper_name}', module_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _SCRAPER_MODULE_CACHE[scraper_name] = module
        if not hasattr(module, 'fetch'):
            raise AttributeError(
                f"Scraper module '{scraper_name}' has no 'fetch' entrypoint."
            )
        return module.fetch

    def _catch_missing_download_url_error(self):
        entity_or_dataset = self.recipe.get('entity') or self.recipe.get('dataset')
        source = entity_or_dataset.source
        if (
            not source.download_url
            and not source.download_url_source
            and not getattr(source, 'download_url_scraper', None)
        ):
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
