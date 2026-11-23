"""Saving a processed table to its output files."""

from __future__ import annotations

import warnings

from openplaces.core.schema import admin_scope_covers
from openplaces.io import (
    coerce_mixed_object_columns,
    save_parquet,
)
from openplaces.io.ingester.empty_outputs import (
    UnexplainedEmptyIngestError,
    save_empty_table,
)
from openplaces.recipe import (
    get_output_path,
)


class _TableSaveMixin:
    """Methods of :class:`~openplaces.io.ingester.table_ingester.TableIngester`
    (table_save).

    State lives on the TableIngester instance.
    """

    # Set by the Ingester on the TableIngester of its primary table (the
    # recipe's own entity, in a chunk or a tile): only its units are
    # filled by the after-run fill, so only its empty reasons are kept.
    is_primary = False

    def _note_read(
        self, rows_read, read_kwargs=None, data_path_override=None, layer_override=None
    ):
        """Record what the read before a save held, for an empty save.

        A save without rows has to say whether the source holds none or
        holds rows of which none are this unit's. The read is where that
        is known: the rows read, and the rows in the source before the
        unit filter (the FID map of the whole table, or the cheap count
        the download check took), when they differ.

        Parameters
        ----------
        rows_read : int
            Rows the read returned.
        read_kwargs : dict, optional
            The read's filter (`fids` or `bbox`), if any.
        data_path_override, layer_override : optional
            Set when the read was one unit's file or layer.
        """
        read_kwargs = read_kwargs or {}
        unit = (self.processing_chunk or {}).get('admin_id_to_process')
        partition = self.download_partition or {}
        rows_in_source = rows_read
        unit_filter = None
        if 'fids' in read_kwargs:
            column = (self.recipe.get('process_by') or {}).get('admin_id_column')
            fids = (partition.get('table_fids') or {}).get(self.table_name)
            rows_in_source = len(fids) if fids is not None else None
            code = None
            reverse = partition.get('admin_id_crosswalk_reverse')
            if reverse is not None and unit in reverse.index:
                code = reverse.loc[unit]
            unit_filter = f'{column} = {code!r}' if code is not None else f'{column}'
            unit_filter += f' for {unit}'
        elif 'bbox' in read_kwargs:
            counted = partition.get('rows_counted')
            same_file = counted is not None and counted[0] == str(
                partition.get('data_path')
            )
            rows_in_source = partition.get('source_rows') if same_file else None
            unit_filter = f"{unit}'s bounding box"
        elif data_path_override is not None or layer_override is not None:
            piece = layer_override or getattr(data_path_override, 'name', None)
            unit_filter = f'{unit} read from its own piece {piece}'
        self._read_note = {
            'rows_read': int(rows_read),
            'rows_in_source': rows_in_source,
            'unit_filter': unit_filter,
        }

    def _empty_save_reason(self, admin_id):
        """The reason and detail for saving *admin_id*'s table without rows.

        Decided from the read before the save (`_note_read`): a source
        with no rows at all is `source_has_none`; a source with rows of
        which the unit filter kept none, or of which preprocessing (a
        query, a filter) kept none, is `not_covered`.

        Returns
        -------
        tuple of str
            (reason, detail).
        """
        note = getattr(self, '_read_note', None) or {}
        rows_read = note.get('rows_read')
        rows_in_source = note.get('rows_in_source')
        unit_filter = note.get('unit_filter')
        table = self.table_name
        if rows_read is None:
            # No read was noted, so nothing says which case holds.
            raise UnexplainedEmptyIngestError(
                f'{table}: saving {admin_id} without rows, but the read that '
                'preceded the save was not noted, so the reason is unknown.'
            )
        if rows_in_source == 0:
            checked = (
                'allowed by empty_download_ok'
                if self.refuse_empty_read
                else 'a table the empty-download check does not cover'
            )
            where = f' ({unit_filter})' if unit_filter else ''
            detail = f'{table}: the source holds no rows{where}; {checked}'
            return 'source_has_none', detail
        if rows_read == 0:
            total = (
                f'{rows_in_source:,} rows in the source'
                if rows_in_source is not None
                else 'the source row count was not taken'
            )
            return 'not_covered', f'{table}: {total}; none matched {unit_filter}'
        return (
            'not_covered',
            f'{table}: {rows_read:,} rows read for {admin_id}; preprocessing '
            '(query, filters, admin overlay) kept none',
        )

    def _record_empty(self, admin_id, reason, detail):
        """Remember why a primary-table unit got no output this run.

        The after-run fill
        (`_OutputMixin._record_units_the_source_does_not_cover`) writes
        the unit's empty table with this reason. Only the primary table
        records: the fill writes nothing for other tables.
        """
        if not self.is_primary or admin_id is None:
            return
        reasons = self.download_partition.setdefault('empty_reasons', {})
        reasons.setdefault(str(admin_id), (reason, detail))

    def _save_recipe_data(self, gdf, recipe=None, empty_reason=None):
        """Save processed data to the entity's output path.

        A table without rows is written through
        `empty_outputs.save_empty_table`, stamped with why it has none.

        Parameters
        ----------
        gdf : DataFrame or GeoDataFrame
            Preprocessed data ready to be saved.
        recipe : dict, optional
            The table recipe to resolve the output path from; defaults to
            this table's own. The stacked-units property layer is saved
            through its own table recipe.
        empty_reason : tuple of str, optional
            (reason, detail) for a save without rows, where the caller
            knows it (the stacked-units layer); decided from the read
            otherwise (`_empty_save_reason`).
        """
        own_table = recipe is None
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
                if own_table and self.is_primary:
                    chunk_admin_id = self.processing_chunk['admin_id_to_process']
                    for admin_id in self.admin_ids_to_save:
                        if admin_scope_covers(chunk_admin_id, admin_id):
                            self._record_empty(
                                admin_id, *self._empty_save_reason(admin_id)
                            )
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
                if own_table:
                    # The chunk holds rows, none of them in these units:
                    # the source covers the chunk, not them.
                    for admin_id in missing_admin_ids:
                        self._record_empty(
                            admin_id,
                            'not_covered',
                            f'{self.table_name}: {len(gdf):,} rows saved for '
                            f'{chunk_admin_id}, none with {admin_id_col} = '
                            f'{admin_id}',
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

            if output_path.suffix == '.parquet' and gdf_to_save.empty:
                reason, detail = empty_reason or self._empty_save_reason(
                    admin_id_to_save
                )
                save_empty_table(output_path, reason, detail, frame=gdf_to_save)
            elif output_path.suffix == '.parquet':
                save_parquet(gdf_to_save, output_path)
            else:
                raise NotImplementedError(
                    f'Output file type not yet supported: {output_path.suffix}'
                )

    # Utilities
