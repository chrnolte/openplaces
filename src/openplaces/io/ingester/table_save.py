"""Saving a processed table to its output files."""

from __future__ import annotations

import warnings

from openplaces.core.schema import admin_scope_covers
from openplaces.io import (
    coerce_mixed_object_columns,
    save_parquet,
)
from openplaces.recipe import (
    get_output_path,
)


class _TableSaveMixin:
    """Methods of :class:`~openplaces.io.ingester.table_ingester.TableIngester`
    (table_save).

    State lives on the TableIngester instance.
    """

    def _save_recipe_data(self, gdf, recipe=None):
        """Save processed data to the entity's output path.

        Parameters
        ----------
        gdf : DataFrame or GeoDataFrame
            Preprocessed data ready to be saved.
        recipe : dict, optional
            The table recipe to resolve the output path from; defaults to
            this table's own. The stacked-units property layer is saved
            through its own table recipe.
        """
        recipe = self.recipe if recipe is None else recipe
        gdf = coerce_mixed_object_columns(gdf)

        save_to = recipe.get('save_to') or {}
        admin_level = save_to.get('admin_level') or (recipe.get('cache_by') or {}).get(
            'admin_level'
        )
        # Saving at the level the chunk is processed at is not a split.
        # The chunk already *is* that unit, so which unit a row belongs to
        # is carried by the output path and filename, not by a column --
        # which is why the splitting branch below drops the admin id
        # columns before writing. Only a coarser save is a real split,
        # where one file gathers rows from several units and the column is
        # needed to tell them apart.
        processing_level = (recipe.get('process_by') or {}).get('admin_level')
        split_dataset_by_admin = (
            admin_level is not None and admin_level != processing_level
        )

        if split_dataset_by_admin:
            admin_id_col = f'admin{admin_level}_id'
            if admin_id_col not in gdf and gdf.empty:
                # An empty chunk has nothing to split and no column to
                # report missing: the crosswalk join and the admin overlay
                # both skip an empty frame, so a `query` that excludes a
                # whole partition by design (New England in
                # `US_admin-census-2025_admin4`) arrived here without the
                # column, raised, and then raised a second, more confusing
                # error while building the message.
                if self.verbose:
                    print(f'  no rows to save for {self.table_name}.')
                return
            if admin_id_col not in gdf:
                raise ValueError(
                    f"Recipe says 'save_to: admin_level: {admin_level}', but column "
                    f"'{admin_id_col}' does not exist in DataFrame:\n\n"
                    + str(gdf.head(1).T)
                )
            admin_ids_in_data = sorted(set(gdf[admin_id_col].dropna()))
            # A raw prefix test is not containment: leaf codes mix two
            # and three characters, so the pre-2026 'US-NC-WA' (Wake)
            # would have gathered Warren's units into Wake's file.
            chunk_admin_id = self.processing_chunk['admin_id_to_process']
            admin_ids_to_save_expected = [
                admin_id
                for admin_id in self.admin_ids_to_save
                if admin_scope_covers(chunk_admin_id, admin_id)
            ]
            admin_ids_to_save_in_data = [
                admin_id
                for admin_id in admin_ids_to_save_expected
                if admin_id in admin_ids_in_data
            ]

            missing_admin_ids = set(admin_ids_to_save_expected) - set(
                admin_ids_to_save_in_data
            )
            if missing_admin_ids:
                warnings.warn(
                    f'\n\n{len(missing_admin_ids)} AdminIds to save not found in data:'
                    '\n' + ', '.join(sorted(missing_admin_ids)[:15]) + '\n'
                )
        else:
            admin_ids_to_save_in_data = [self.processing_chunk['admin_id_to_process']]

        if split_dataset_by_admin and admin_ids_to_save_in_data:
            print('Saving ' + ', '.join(admin_ids_to_save_in_data))

        for admin_id_to_save in admin_ids_to_save_in_data:
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

            output_path = get_output_path(
                recipe,
                admin_id_to_save,
                self.download_partition.get('partition_id_to_download'),
            )

            if output_path.suffix == '.parquet':
                save_parquet(gdf_to_save, output_path)
            else:
                raise NotImplementedError(
                    f'Output file type not yet supported: {output_path.suffix}'
                )

    # Utilities
