"""Which admin units an ingest saves, processes and downloads, and
whether a saved output is complete.
"""

from __future__ import annotations

import pandas as pd
import pyarrow as pa

from openplaces.core.schema import AdminId
from openplaces.io.cleanup import (
    discard_input_receipts,
    discard_receipt,
    receipt_justifies_skip,
)
from openplaces.io.ingester._helpers import (  # noqa: F401
    _SCRAPER_MODULE_CACHE,
    _match_extracted_file,
    _transform_partition_key,
    _warn_registry_type_mismatches,
)
from openplaces.io.readers import get_admin
from openplaces.io.transform import get_crosswalk
from openplaces.recipe import (
    get_download_admin_level,
    get_output_path,
    get_process_admin_level,
    get_save_admin_level,
)
from openplaces.utils import format_list


class _AdminScopeMixin:
    """Methods of :class:`~openplaces.io.ingester.Ingester` (scope).

    State lives on the Ingester instance.
    """

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
        if self.admin_ids and not admin_ids_to_save and save_level != 0:
            # Every requested id fell outside the spine. Before the
            # re-mint this was a quiet no-op that reported success; a
            # recycled or retired identifier then cost a full run that
            # looked fine while doing nothing.
            #
            # A unit the spine knows but that has no children at the save
            # level is a different case: legitimately empty scope, not a
            # renamed id. 83 of the 263 spine countries carry no admin3
            # rows, so a global recipe asked for one of them should do
            # nothing quietly rather than abort the run.
            unresolved = [
                admin_id
                for admin_id in self.admin_ids
                if not self._admin_id_is_in_spine(admin_id)
            ]
            if unresolved:
                requested = ', '.join(str(a) for a in unresolved)
                scope = self.recipe.get('admin_id')
                scope_text = repr(str(scope)) if scope and scope.levels else 'the world'
                raise ValueError(
                    f'None of the requested admin ids ({requested}) resolve '
                    f'to a unit at save level {save_level} within '
                    f'{scope_text}. A re-mint may have '
                    'renamed them; look the unit up in the current spine '
                    'by its name or national code.'
                )

        if not reprocess:
            # Skip if the output exists, or a tombstone receipt records its
            # deliberate deletion with all consumers intact (section 4.3 of
            # the lifecycle design; gated by retention.cleanup.honor_receipts
            # and voided under an orchestrator)
            admin_ids_to_save = [
                admin_id
                for admin_id in admin_ids_to_save
                if not get_output_path(self.recipe, admin_id, partition_id).exists()
                and not (
                    partition_id is None
                    and receipt_justifies_skip(self.recipe, admin_id)
                )
            ]

        return admin_ids_to_save

    def _admin_id_is_in_spine(self, admin_id) -> bool:
        """Is a requested admin unit itself present in the admin spine?

        Asked at the requested unit's own level, which is what separates a
        retired or renamed identifier from a unit that is simply childless
        at the level a recipe saves to.

        Parameters
        ----------
        admin_id : AdminId
            The requested unit. Level 0 (global) is always present.

        Returns
        -------
        bool
            False when the spine cannot be read at that level either, so
            an unreadable spine keeps the louder of the two outcomes.
        """
        level = admin_id.get_level()
        if level == 0:
            return True
        try:
            spine_ids = get_admin(self.recipe['admin_id'], level).index
        except (FileNotFoundError, KeyError, ValueError):
            return False
        return str(admin_id) in {str(spine_id) for spine_id in spine_ids}

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
            if reprocess:
                # A deliberate re-run supersedes any tombstone receipt,
                # this recipe's own and those of its inputs, which name
                # this output as one of the consumers that freed them
                for admin_id in self.admin_ids_to_save:
                    discard_receipt(get_output_path(self.recipe, admin_id))
                    discard_input_receipts(self.recipe, admin_id)

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
        except (ValueError, KeyError, pa.ArrowInvalid):
            # Column absent (a pre-stamp file): leave as-is. Narrow, so a
            # locked or unreadable parquet raises instead of reading as
            # "complete" and silently skipping the rebuild.
            return False

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
                    'A recipe processed at admin level 0 saves either '
                    'globally or not at all, but admin_ids_to_save is '
                    f'{self.admin_ids_to_save}.'
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

        self.admin_ids_to_process = self._limit_to_crosswalk_scope(admin_ids_to_process)

    def _limit_to_crosswalk_scope(self, admin_ids_to_process):
        """Drop process units a scoped crosswalk does not claim to cover.

        A `recipe_id`-form `process_by.admin_id_crosswalk` is a sidecar
        listing the admin units a source actually carries -- Maine's parcel
        layer covers organized towns only, so roughly two hundred of the
        state's units have no rows by design. Expanding a whole-state
        request to every unit and then failing on the first uncovered one
        would make such a source impossible to ingest in bulk.

        Only the *expanded* list is filtered. Naming an uncovered unit
        explicitly still raises in `TableIngester.process`, because there
        the caller has asserted the unit should be there and silence would
        hide a real mistake.
        """
        process_by = self.recipe.get('process_by') or {}
        crosswalk = process_by.get('admin_id_crosswalk') or {}
        if 'recipe_id' not in crosswalk:
            return admin_ids_to_process
        if any(
            admin_id.get_level() >= get_process_admin_level(self.recipe)
            for admin_id in self.admin_ids
        ):
            return admin_ids_to_process
        try:
            spec = dict(crosswalk)
            spec['admin_id'] = str(self.recipe['admin_id'])
            covered = set(get_crosswalk(spec, flip=True).iloc[:, 0])
        except (FileNotFoundError, KeyError, ValueError):
            # An absent or unreadable sidecar is not fatal here: the scope
            # is an optimization and `process()` reports a real mismatch.
            # Narrow, so an unexpected failure is not read as "unscoped".
            return admin_ids_to_process
        # Compare as strings: AdminId does not hash equal to its own
        # string form, so membership against the crosswalk's raw column
        # silently matches nothing if the cast is left out.
        kept = [a for a in admin_ids_to_process if str(a) in covered]
        dropped = len(admin_ids_to_process) - len(kept)
        if dropped and self.verbose:
            print(
                f'{dropped} admin unit(s) are outside '
                f'{crosswalk["recipe_id"]} and have no source rows; skipping.'
            )
        if not kept:
            # The crosswalk covers no unit at this level -- it is keyed to
            # a different one. Filtering here would drop the whole run, so
            # leave the list alone and let process() report the mismatch.
            return admin_ids_to_process
        return kept

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
                    'A recipe downloaded at admin level 0 has one global '
                    'download, but admin_ids_to_process is '
                    f'{self.admin_ids_to_process}.'
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
