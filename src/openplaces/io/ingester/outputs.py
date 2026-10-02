"""What an ingest writes besides its chunks: units the source does
not cover, entity links, merged tile partials, joined table
partitions and the aggregation to the save level.
"""

from __future__ import annotations

import geopandas as gpd

from openplaces.config import cfg
from openplaces.geo.link import create_entity_link, get_entity_link_path
from openplaces.io import (
    release_unused_memory,
    save_parquet,
)
from openplaces.io.aggregate import aggregate_to_admin_level
from openplaces.io.ingester._helpers import (  # noqa: F401
    _SCRAPER_MODULE_CACHE,
    _match_extracted_file,
    _transform_partition_key,
    _warn_registry_type_mismatches,
)
from openplaces.recipe import (
    build_table_recipe,
    get_output_path,
    get_recipe_id,
)


class _OutputMixin:
    """Methods of :class:`~openplaces.io.ingester.Ingester` (outputs).

    State lives on the Ingester instance.
    """

    def _record_units_the_source_does_not_cover(self, reprocess=False):
        """Write an empty table for a unit the source has no data for.

        A soft skip means the source does not publish this unit (the
        fmv parcel source has not published Tyrrell County, so all
        three of its table partitions report 'not published'). The
        ingest then succeeds, exits 0 and writes nothing, which is
        indistinguishable from a failed job to an orchestrator: it
        demands the declared output and fails the job, and everything
        downstream of it. Recording the empty answer is the honest
        form, and it matches what the harmonize stage does for a unit
        whose pipeline resolves no spine.

        Only reached after a run that raised nothing: a real failure
        propagates, so an output still missing here is an absence in
        the source rather than an error.

        On a reprocess, a unit whose partition the scraper reported
        unavailable this time also gets the empty table, in place of
        whatever an earlier run wrote for it: a source can stop
        covering a unit (the Wikidata harvest of Curacao emptied when
        its one candidate was ruled out), and the stale rows would
        otherwise read as current.

        Parameters
        ----------
        reprocess : bool, optional
            Whether this run was asked to rebuild existing outputs.
        """
        stale = getattr(self, '_unavailable_units', set()) if reprocess else set()
        for admin_id in self.admin_ids_to_save or []:
            out_path = get_output_path(self.recipe, admin_id)
            replacing = out_path.exists() and str(admin_id) in stale
            if out_path.exists() and not replacing:
                continue
            if self.verbose:
                what = 'no longer covered' if replacing else 'not covered'
                print(f'{admin_id}: {what} by the source; saved an empty table.')
            for old in (out_path, out_path.with_stem(out_path.stem + '_geo')):
                old.unlink(missing_ok=True)
            empty = gpd.GeoDataFrame(
                {'geometry': gpd.GeoSeries([], dtype='geometry')}, crs=cfg.crs
            )
            save_parquet(empty, out_path)

    def _create_entity_links(self, reprocess=False):
        """Persist the n:m entity links declared under `entity_links`.

        For each entry, computes the full intersection link to the
        referenced entity with create_entity_link() and saves it at the
        canonical get_entity_link_path() location (beside the finer
        entity's output). Existing link files are reused unless
        `reprocess` is True. Runs even when the primary output already
        existed, so missing links are backfilled without a reingest.
        """
        entries = self.recipe.get('entity_links') or []
        if not entries:
            return

        # Clean up memory-heavy ingestion caches to maximize available RAM
        if hasattr(self, 'tile_admin_link'):
            del self.tile_admin_link
        release_unused_memory()

        self_recipe_id = get_recipe_id(self.recipe)
        for entry in entries:
            other_recipe_id = entry['recipe_id']
            link_path = get_entity_link_path(self_recipe_id, other_recipe_id)
            if link_path.exists() and not reprocess:
                continue
            create_entity_link(self_recipe_id, other_recipe_id)
            if self.timer is not None:
                self.timer.mark(f'Create link: {self_recipe_id} - {other_recipe_id}')

    def _merge_tile_partials(self):
        """Merge per-tile partial files into final per-admin-id output files.

        For each admin_id in `admin_ids_to_save`, reads all tile partial
        files written during the tile loop, concatenates them, writes the
        merged result to the final output path, and deletes the partials.
        Admin IDs whose data came from a single tile are handled by a simple
        rename rather than a concat.

        Concatenation goes through the shared aggregation core
        (`io.aggregate._aggregate_to_file`) rather than a local `pd.concat`.
        The local copy had diverged: it did not union categorical category
        sets, so a county straddling two tiles whose observed values differ
        silently lost the categorical dtype that a single-tile county kept,
        and it skipped `coerce_mixed_object_columns`.
        """
        from openplaces.io.aggregate import _aggregate_to_file

        # Build the admin-unit-to-tiles map once. Asking the link table per
        # (admin unit x tile) was O(admins x tiles), roughly three million
        # lookups on a national run.
        downloaded_tile_ids = set(self.partition_ids_to_download)
        tiles_by_admin: dict[str, list[str]] = {}
        for tile_id, admin_id in self.tile_admin_link.index:
            if tile_id in downloaded_tile_ids:
                tiles_by_admin.setdefault(admin_id, []).append(tile_id)

        for admin_id in self.admin_ids_to_save:
            final_path = get_output_path(self.recipe, admin_id)

            # Tile IDs that were actually downloaded for this admin_id.
            tile_ids = sorted(set(tiles_by_admin.get(admin_id, [])))
            existing_tiles = [
                (tile_id, get_output_path(self.recipe, admin_id, tile_id))
                for tile_id in tile_ids
            ]
            existing_tiles = [(t, p) for t, p in existing_tiles if p.exists()]

            if not existing_tiles:
                continue

            if len(existing_tiles) == 1:
                partial_path = existing_tiles[0][1]
                try:
                    partial_path.replace(final_path)
                    _geo_partial = partial_path.with_stem(partial_path.stem + '_geo')
                    if _geo_partial.exists():
                        _geo_partial.replace(
                            final_path.with_stem(final_path.stem + '_geo')
                        )
                except PermissionError as e:
                    raise PermissionError(
                        f'Cannot write to {final_path.name}.\n\n'
                        '\033[1m→ Close the file in QGIS / ArcGIS / Dropbox sync '
                        'and re-run.\033[0m'
                    ) from e
                continue

            warned: list[bool] = []

            def _warn_once(df, _warned=warned):
                # The registry check is about the recipe's own columns, so
                # one input frame answers it; running it per tile would
                # repeat the same warning for every tile of every county.
                if not _warned:
                    _warned.append(True)
                    _warn_registry_type_mismatches(df)
                return df

            _aggregate_to_file(
                final_path,
                existing_tiles,
                transform=_warn_once,
                verbose=False,
            )

            if self.verbose:
                print(
                    f'Merged {len(existing_tiles)} tile partial(s) → {final_path.name}'
                )

    def _join_table_partitions(self):
        """Column-join per-table partition outputs per the recipe's
        `join_partitions_by` block (table-partition counterpart to
        `_merge_tile_partials`)."""
        from openplaces.io.aggregate import join_partitions_by_index

        join_cfg = self.recipe.get('join_partitions_by') or {}
        join_partitions_by_index(
            self.recipe,
            table_names=self.recipe['download_by']['table_names'],
            admin_ids=self.admin_ids_to_save,
            join_key_name=join_cfg.get('join_key_name'),
            keep_original=join_cfg.get('keep_original', False),
            verbose=self.verbose,
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

    def _aggregate_partitions(self, reprocess=False):
        """Roll up partition outputs per the recipe's `aggregate_by` block.

        For partitioned recipes that declare `aggregate_by`, concatenate the
        per-partition output files into one combined ``..._all.parquet`` (with
        ``single_file: true``) or into per-group files (e.g. the 12 months of
        each year into one per-year file, with ``partition: year``). Delegates
        to `aggregate_partitions`.

        A reprocess of every partition replaces the combined file. The
        default ``union`` merge keeps rows the new batch does not repeat
        exactly, so a reprocess that changed a value (a re-keyed
        ``parcel_id_local``) appended a second copy of every row: Orange
        County FL's sales file doubled to 3.1M rows on 2026-09-12, and
        the curate step's dedup then kept the stale first copies. A
        reprocess of only some partitions cannot be merged safely either
        way and is refused, since ``replace`` would narrow the file to
        those partitions and ``union`` would duplicate them.
        """
        agg = self.recipe.get('aggregate_by')
        if not agg or not (agg.get('single_file') or 'partition' in agg):
            return

        from openplaces.io.aggregate import aggregate_partitions

        how = agg.get('how', 'union')
        if reprocess:
            if getattr(self, 'partition_ids', None) is None:
                how = 'replace'
            elif how == 'union':
                raise ValueError(
                    f'{get_recipe_id(self.recipe)}: reprocessing some partitions '
                    'of a recipe that aggregates them into one file would leave '
                    'stale rows beside the new ones (union) or drop the other '
                    'partitions (replace). Reprocess every partition instead.'
                )

        aggregate_partitions(
            self.recipe,
            by=agg.get('partition', 'year'),
            single_file=agg.get('single_file', False),
            how=how,
            keep_original=agg.get('keep_partitions', False),
            verbose=self.verbose,
        )
