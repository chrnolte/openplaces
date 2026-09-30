"""Download partitions: partition, tile and lat-lon tile ids, the
admin partition key and URL placeholders.
"""

from __future__ import annotations

import re

import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.io.ingester._helpers import (  # noqa: F401
    _SCRAPER_MODULE_CACHE,
    _match_extracted_file,
    _transform_partition_key,
    _warn_registry_type_mismatches,
)
from openplaces.io.readers import get_admin
from openplaces.recipe import (
    find_admin_recipe_id,
    get_output_path,
    get_partition_ids,
    get_recipe_by_id,
    get_save_admin_level,
)
from openplaces.utils import format_list


class _PartitionMixin:
    """Methods of :class:`~openplaces.io.ingester.Ingester` (partitions).

    State lives on the Ingester instance.
    """

    def _resolve_partition_ids(self, reprocess):
        """Resolve partition IDs to save, process, and download

        By convention, all partition IDs are strings.
        """
        download_by = self.recipe.get('download_by') or {}
        partition = download_by.get('partition')

        if partition == 'tile_id':
            self._resolve_tile_ids()
            return

        if partition == 'latlon_tile':
            self._resolve_latlon_tile_ids()
            return

        all_partition_ids = get_partition_ids(self.recipe)

        # Filter to requested subset if caller specified partition_ids
        if isinstance(self.partition_ids, list | set):
            self.partition_ids_to_download = [
                x for x in self.partition_ids if x in all_partition_ids
            ]
        else:
            self.partition_ids_to_download = all_partition_ids

        # When not reprocessing, drop partitions already ingested for every
        # admin unit to be saved. A partition is ingested if its per-partition
        # file is on disk or it is recorded in an aggregated file's footer
        # coverage (so the skip survives deletion of the per-partition files).
        if not reprocess and partition:
            save_admin_ids = self.admin_ids_to_save or [None]
            ingested = {
                admin_id: self.get_ingested_partition_ids(admin_id)
                for admin_id in save_admin_ids
            }
            self.partition_ids_to_download = [
                pid
                for pid in self.partition_ids_to_download
                if not all(pid in ingested[admin_id] for admin_id in save_admin_ids)
            ]

        if self.verbose and self.partition_ids_to_download != [None]:
            print(
                f'Partitioned by `{partition}`:',
                format_list(self.partition_ids_to_download),
            )
            if isinstance(self.partition_ids, list | set):
                print('Selected:', format_list(self.partition_ids_to_download))

    def get_ingested_partition_ids(self, admin_id) -> set[str]:
        """Return partition ids already ingested for *admin_id*.

        Combines two signals so skipping works whether or not the per-partition
        files are kept: partition ids whose per-partition output file is on
        disk, plus those recorded in the aggregated file's footer coverage
        (`openplaces:partitions`, written by `aggregate_partitions`).
        """
        from openplaces.io.aggregate import read_partition_coverage

        ingested = {
            str(pid)
            for pid in get_partition_ids(self.recipe)
            if pid is not None
            and get_output_path(self.recipe, admin_id, partition_id=pid).exists()
        }
        agg = self.recipe.get('aggregate_by') or {}
        if agg.get('single_file'):
            ingested |= read_partition_coverage(
                get_output_path(self.recipe, admin_id, partition_id='all')
            )
        return ingested

    def _resolve_tile_ids(self):
        """Resolve tile partition IDs filtered to admin_ids_to_save.

        Loads the tile-admin link table produced by overlay_polygons() and
        keeps only the tile IDs that intersect at least one admin ID in
        self.admin_ids_to_save. The link table is stored on self for reuse
        in _ingest_download_partition.
        """
        # `download_by.tile_admin_recipe_id` wins when present. The tile
        # crosswalk is built once, globally, by the tile recipe, so it can
        # only be keyed on an admin layer that recipe itself links to --
        # which is not necessarily the layer this recipe overlays against.
        # Those two pulled apart the moment the overlay moved to the
        # harmonized per-state reference while the global tile grid stayed
        # on the nationally-saved census layer: the consumer then asked for
        # a crosswalk nobody builds.
        admin_overlay_specs = self.recipe['overlay_admin_ids']
        admin_recipe_id = (
            (self.recipe.get('download_by') or {}).get('tile_admin_recipe_id')
            or admin_overlay_specs.get('admin_recipe_id')
            or find_admin_recipe_id(
                self.recipe['admin_id'], admin_overlay_specs['admin_level']
            )
        )

        tile_recipe_id = self.recipe['download_by']['tile_recipe_id']
        # One link per admin unit at the admin recipe's save level, built
        # from that unit's polygons and the global tile grid the first
        # time it is needed and reused after. A county build asks for its
        # own state's link, never for a crosswalk of the whole world.
        from openplaces.geo.link import ensure_scoped_tile_link

        level = get_save_admin_level(get_recipe_by_id(admin_recipe_id))
        units = list(
            dict.fromkeys(
                str(AdminId(str(aid)).truncate_to_level(level))
                for aid in self.admin_ids_to_save
            )
        )
        self.tile_admin_link = pd.concat(
            [
                ensure_scoped_tile_link(
                    tile_recipe_id, admin_recipe_id, unit, verbose=self.verbose
                )
                for unit in units
            ]
        )

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

    def _resolve_latlon_tile_ids(self):
        """Resolve 1°×1° lat/lon tile IDs from admin polygon bounds.

        Unlike :meth:`_resolve_tile_ids`, no pre-ingested tile dataset is needed.
        The ``tile_admin_link`` MultiIndex DataFrame is computed in memory.
        """
        from openplaces.io.ingester.cloud_geoparquet_ingester import tile_ids_for_admin

        download_by = self.recipe.get('download_by') or {}
        tile_size_deg = float(download_by.get('tile_size_deg', 1.0))

        rows = []
        for admin_id in self.admin_ids_to_save:
            for tile_id in tile_ids_for_admin(admin_id, tile_size_deg):
                rows.append((tile_id, admin_id))

        if not rows:
            self.tile_admin_link = pd.DataFrame(
                index=pd.MultiIndex.from_tuples([], names=[None, None])
            )
            self.partition_ids_to_download = []
            return

        idx = pd.MultiIndex.from_tuples(rows)
        self.tile_admin_link = pd.DataFrame(index=idx)

        admin_ids_set = set(self.admin_ids_to_save)
        relevant = self.tile_admin_link.index.get_level_values(1).isin(admin_ids_set)
        self.partition_ids_to_download = (
            self.tile_admin_link[relevant].index.get_level_values(0).unique().tolist()
        )

        if isinstance(self.partition_ids, list | set):
            self.partition_ids_to_download = [
                t
                for t in self.partition_ids_to_download
                if t in set(self.partition_ids)
            ]

        if self.verbose:
            print(
                'Partitioned by `latlon_tile`:',
                format_list(self.partition_ids_to_download),
            )

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
            if key_transform == 'remove_spaces':
                partition_key = partition_key.replace(' ', '')
            elif key_transform == 'slugify':
                # Lowercase, whitespace to single hyphens. Geofabrik and
                # similar publishers name state files this way
                # ('North Carolina' -> 'north-carolina').
                partition_key = '-'.join(partition_key.lower().split())
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

                # download_by.partition_key_transformation reshapes a
                # resolved key before it enters the URL/filename. The
                # motivating case: New England towns' admin3_id_admin1 is
                # the 10-digit COUSUB GEOID whose first five digits are
                # the county FIPS a county-keyed API (NSI) wants; a
                # substring makes every town of one county resolve to the
                # same download, which the filename cache then fetches
                # once. A no-op for units whose key already has the target
                # shape (a county's own 5-digit FIPS). Applied to every
                # placeholder, not only `admin*` ones: the documented
                # `year` case was silently ignored and the URL was built
                # from the untransformed key.
                _pkt = (self.recipe.get('download_by') or {}).get(
                    'partition_key_transformation'
                ) or {}
                _spec = _pkt.get(placeholder)
                if _spec:
                    _partition_key = _transform_partition_key(_partition_key, _spec)

                self.download_partition['partition_key_dict'][placeholder] = (
                    _partition_key
                )

            url_or_path = url_or_path.replace(
                '{' + placeholder + '}', str(_partition_key)
            )

        return url_or_path
