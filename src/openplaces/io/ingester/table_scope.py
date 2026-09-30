"""What a table ingest reads for an admin unit: the fid filter, the
admin-id crosswalks, file-pattern paths and admin geometries.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.geo import get_crs
from openplaces.io.readers import get_admin
from openplaces.io.transform import (
    apply_transformation,
    get_crosswalk,
)


def _admin_level_of(admin_id) -> int:
    """Depth of an admin id given as an AdminId, a string, or nothing.

    `AdminId(x)` raises when `x` is already an AdminId, so callers that
    accept either form cannot just reconstruct to ask for the level.
    """
    if admin_id is None or admin_id == '':
        return 0
    if not isinstance(admin_id, AdminId):
        admin_id = AdminId(admin_id)
    return admin_id.get_level()


class _TableScopeMixin:
    """Methods of :class:`~openplaces.io.ingester.table_ingester.TableIngester`
    (table_scope).

    State lives on the TableIngester instance.
    """

    def _ensure_table_fid_filter(self):
        """Build FID filter for this table if not already cached."""
        if 'admin_id_crosswalk' not in self.download_partition:
            self._prepare_admin_id_crosswalk()
        if 'table_fids' not in self.download_partition:
            self.download_partition['table_fids'] = {}
        if self.table_name not in self.download_partition['table_fids']:
            self._prepare_table_fid_filter()

    def _prepare_admin_id_crosswalk(self):
        """Build crosswalk from source admin column values to AdminIds.

        Uses process_by.admin_id_crosswalk from the recipe. The result is
        stored in download_partition and shared across all tables in the
        same download partition.
        """
        process_by = self.recipe.get('process_by', {})
        if 'admin_id_crosswalk' not in process_by:
            raise ValueError('No crosswalk recipe found in process_by.')
        admin_id_crosswalk_dict = dict(process_by['admin_id_crosswalk'])
        admin_id_crosswalk_dict['admin_id'] = self.download_partition[
            'admin_id_to_download'
        ]
        self.download_partition['admin_id_crosswalk'] = get_crosswalk(
            admin_id_crosswalk_dict, flip=True
        )
        if 'recipe_id' in process_by['admin_id_crosswalk']:
            # A recipe_id-form crosswalk is a scoped sidecar table (e.g.
            # the 7-county MetroGIS list, or a state parcel source that
            # only some counties have opted into), not a universal
            # census-style admin lookup. Record the admin units it
            # actually covers so process() can raise on a request outside
            # that scope instead of silently matching zero rows.
            self.download_partition['admin_id_crosswalk_scope'] = set(
                self.download_partition['admin_id_crosswalk'].iloc[:, 0]
            )

    def _prepare_reverse_admin_id_crosswalk(self):
        """Build crosswalk from AdminIds to source admin column values.

        The inverse of :meth:`_prepare_admin_id_crosswalk`: needed by
        ``process_by.file_pattern`` resolution, which looks up the raw
        per-admin code for the *current* processing admin unit to format
        into a filename, rather than mapping a column of raw codes found in
        the data to AdminIds.
        """
        process_by = self.recipe.get('process_by', {})
        if 'admin_id_crosswalk' not in process_by:
            raise ValueError('No crosswalk recipe found in process_by.')
        admin_id_crosswalk_dict = dict(process_by['admin_id_crosswalk'])
        admin_id_crosswalk_dict['admin_id'] = self.download_partition[
            'admin_id_to_download'
        ]
        self.download_partition['admin_id_crosswalk_reverse'] = get_crosswalk(
            admin_id_crosswalk_dict, flip=False
        )

    def _resolve_file_pattern_path(self, file_pattern: str) -> Path:
        """Resolve the per-admin-unit source file named by a file pattern.

        Some sources (e.g. FL DOR's NAL rolls) ship one already-split file
        per admin unit inside a single shared download, rather than one
        file with an in-data admin column to filter rows by.
        ``process_by.file_pattern`` names that file relative to the
        recipe's heap directory, with ``{partition_id}`` substituted for
        the current download partition id and the crosswalk's raw-code
        column name (e.g. ``{admin3_id_admin2}``) substituted for this
        admin unit's code, looked up via :meth:`_prepare_reverse_admin_id_crosswalk`.
        Any remaining glob wildcards (e.g. a trailing ``*``) absorb parts
        of the filename that vary independently of the admin code (e.g.
        the county name spelling).

        Parameters
        ----------
        file_pattern : str
            Pattern from ``process_by.file_pattern``, relative to
            :attr:`recipe_heap_dir`.
        """
        if 'admin_id_crosswalk_reverse' not in self.download_partition:
            self._prepare_reverse_admin_id_crosswalk()
        reverse_crosswalk = self.download_partition['admin_id_crosswalk_reverse']

        admin_id_to_process = self.processing_chunk['admin_id_to_process']
        if admin_id_to_process not in reverse_crosswalk.index:
            # The reverse crosswalk records no scope of its own (unlike the
            # forward one), so an admin unit a scoped sidecar does not cover
            # reached `.loc` and raised a bare KeyError naming only the id.
            raise KeyError(
                f"{admin_id_to_process} has no code in this recipe's "
                'process_by.admin_id_crosswalk, so no file name can be '
                'built for it.'
            )
        raw_code = reverse_crosswalk.loc[admin_id_to_process]

        pattern = file_pattern.replace(
            '{partition_id}',
            str(self.download_partition.get('partition_id_to_download')),
        )
        pattern = pattern.replace('{' + reverse_crosswalk.name + '}', str(raw_code))

        matches = list(self.recipe_heap_dir.glob(pattern))
        if len(matches) == 0 and (
            (self.recipe.get('process_by') or {}).get('file_pattern_missing') == 'skip'
        ):
            # Opt-in tolerance for a source whose per-unit files come and
            # go across partitions (FL DOR year rolls omit whole
            # counties in some years). Absent stays a hard error unless
            # the recipe declares the gap expected.
            warnings.warn(
                f"process_by.file_pattern '{pattern}' matched no file for "
                f'{admin_id_to_process}; skipping this chunk '
                '(file_pattern_missing: skip).'
            )
            return None
        if len(matches) != 1:
            raise ValueError(
                f"process_by.file_pattern '{pattern}' matched {len(matches)} "
                f'files under {self.recipe_heap_dir} for {admin_id_to_process} '
                '(expected exactly 1):\n' + '\n'.join(str(m) for m in matches)
            )
        return matches[0]

    def _prepare_table_fid_filter(self):
        """Build FID → admin_id mapping for this table's layer.

        FIDs are layer-specific in multi-layer formats (e.g. GDB), so each
        table builds and caches its own mapping under
        download_partition['table_fids'][table_name].
        """
        admin_level_to_process = self.recipe['process_by']['admin_level']
        admin_id_column_source = self.recipe['process_by']['admin_id_column']

        admin_id_filter = self._read_recipe_data(
            columns=[admin_id_column_source],
            read_geometry=False,
            fid_as_index=True,
        )

        if 'admin_id_transformation' in self.recipe['process_by']:
            transforms = self.recipe['process_by']['admin_id_transformation']
            if isinstance(transforms, list):
                transforms[0].setdefault('input', admin_id_column_source)
                for t in transforms:
                    admin_id_filter = apply_transformation(admin_id_filter, t)
                join_column = transforms[-1]['output']
            else:
                transforms['input'] = admin_id_column_source
                admin_id_filter = apply_transformation(admin_id_filter, transforms)
                join_column = transforms['output']
        else:
            join_column = admin_id_column_source

        admin_id_filter = admin_id_filter.join(
            self.download_partition['admin_id_crosswalk'],
            on=join_column,
        )

        self.download_partition['table_fids'][self.table_name] = admin_id_filter[
            f'admin{admin_level_to_process}_id'
        ]

    # Admin geometry loading (cached in download_partition)

    def _load_admin_geometries(self):
        """Load admin geometries for spatial overlay or spatial mask.

        Reads admin unit boundaries and stores them in download_partition
        for reuse across admin chunks and tables.
        """
        process_by = self.recipe.get('process_by') or {}
        if process_by.get('use_spatial_index') or process_by.get('use_spatial_mask'):
            admin_specs = self.recipe['process_by']
        elif 'overlay_admin_ids' in self.recipe:
            admin_specs = self.recipe['overlay_admin_ids']
        else:
            raise ValueError(
                'Cannot load admin geometries: recipe has neither '
                'process_by.use_spatial_index/use_spatial_mask nor '
                'overlay_admin_ids.'
            )

        admin_id = self.download_partition['admin_id_to_download'] or self.recipe.get(
            'admin_id'
        )
        admin_ids_in_tile = self.download_partition.get('admin_ids_in_tile')

        # A tile partition has no admin unit to download, and a global
        # recipe has none of its own, so both sources of `admin_id` come
        # back at level 0 -- which `get_admin` cannot resolve to a path
        # for a recipe that saves per admin unit. The tile already knows
        # which units it covers, so ask for those: it is the only
        # non-empty answer available, and it loads the units needed
        # rather than a whole country's to filter down afterwards.
        if admin_ids_in_tile and _admin_level_of(admin_id) == 0:
            admin_id = list(admin_ids_in_tile)

        def _read_admin_geometries(requested):
            return get_admin(
                requested,
                admin_specs['admin_level'],
                recipe=admin_specs.get('admin_recipe_id'),
                geom=True,
            )['geometry']

        admin_geometries = _read_admin_geometries(admin_id)

        if admin_ids_in_tile:
            # `get_admin` resolves a single output file per call, keyed on
            # the deepest id it was asked for, so a tile spanning two
            # states came back with only the first state's units. The rest
            # then got null geometry, the left join gave their buildings no
            # admin id, and their county files were never written. Ask
            # again for whatever is still missing: each round resolves at
            # least one more file, or stops.
            missing = [
                aid for aid in admin_ids_in_tile if aid not in admin_geometries.index
            ]
            while missing:
                more = _read_admin_geometries(missing)
                found = [aid for aid in missing if aid in more.index]
                if not found:
                    warnings.warn(
                        f'{len(missing)} admin unit(s) in this tile have no '
                        f'geometry at level {admin_specs["admin_level"]} and '
                        'will contribute no rows: ' + ', '.join(sorted(missing)[:5])
                    )
                    break
                admin_geometries = gpd.GeoSeries(
                    pd.concat([admin_geometries, more.loc[found]]),
                    crs=admin_geometries.crs,
                )
                missing = [aid for aid in missing if aid not in found]

            admin_geometries = admin_geometries.loc[
                [aid for aid in admin_ids_in_tile if aid in admin_geometries.index]
            ]

        data_crs = get_crs(
            self.download_partition['data_path'], layer=self.recipe.get('layer')
        )
        if data_crs != admin_geometries.crs:
            admin_geometries = admin_geometries.to_crs(data_crs)
        self.download_partition['admin_geometries'] = admin_geometries

    # Read
