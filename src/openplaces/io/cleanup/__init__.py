"""
Data lifecycle cleanup: tombstone receipts, consumption checks, and the
cleanup() and compact() entry points.

The on-disk state of downstream outputs is the consumption ledger: every
decision here is recomputed from disk (plus per-file receipts), so
sequential notebook runs, driver scripts, and parallel cluster jobs reach
the same conclusions from the same evidence.

An output with retention 'until_consumed' may be deleted iff every
consumer's output passes its completeness check. Deletion leaves a
tombstone receipt (.consumed.json) beside the would-be output path so
skip-if-exists logic can honor the deliberate deletion without
re-ingesting.

Split into a package on 2026-09-23. The package file re-exports
every name the one-file module defined, private names included,
so every import that worked before still does: receipts,
consumption, lock, walk, compaction.
"""

from openplaces.io.cleanup.compaction import (
    _MIN_RECIPES_FOR_ORPHAN_GC as _MIN_RECIPES_FOR_ORPHAN_GC,
)
from openplaces.io.cleanup.compaction import _bucket_of as _bucket_of
from openplaces.io.cleanup.compaction import _bucket_roots as _bucket_roots
from openplaces.io.cleanup.compaction import _classify_file as _classify_file
from openplaces.io.cleanup.compaction import _compact_action as _compact_action
from openplaces.io.cleanup.compaction import _compact_delete as _compact_delete
from openplaces.io.cleanup.compaction import (
    _enrich_suffix_index as _enrich_suffix_index,
)
from openplaces.io.cleanup.compaction import _layer_output_names as _layer_output_names
from openplaces.io.cleanup.compaction import (
    _match_recipe_for_file as _match_recipe_for_file,
)
from openplaces.io.cleanup.compaction import (
    _match_recipe_for_path as _match_recipe_for_path,
)
from openplaces.io.cleanup.compaction import _process_receipts as _process_receipts
from openplaces.io.cleanup.compaction import (
    _recipe_admin_covers as _recipe_admin_covers,
)
from openplaces.io.cleanup.compaction import _recipe_id_rest as _recipe_id_rest
from openplaces.io.cleanup.compaction import _recipe_token_index as _recipe_token_index
from openplaces.io.cleanup.compaction import compact as compact
from openplaces.io.cleanup.consumption import _all_recipe_ids as _all_recipe_ids
from openplaces.io.cleanup.consumption import _consumer_satisfies as _consumer_satisfies
from openplaces.io.cleanup.consumption import _consumers_complete as _consumers_complete
from openplaces.io.cleanup.consumption import _dependency_index as _dependency_index
from openplaces.io.cleanup.consumption import _DependencyIndex as _DependencyIndex
from openplaces.io.cleanup.consumption import (
    _geometry_sidecar_ok as _geometry_sidecar_ok,
)
from openplaces.io.cleanup.consumption import _parquet_schema_ok as _parquet_schema_ok
from openplaces.io.cleanup.consumption import (
    _path_conceptually_exists as _path_conceptually_exists,
)
from openplaces.io.cleanup.consumption import (
    _required_subadmin_ids as _required_subadmin_ids,
)
from openplaces.io.cleanup.consumption import _truncate_admin as _truncate_admin
from openplaces.io.cleanup.consumption import is_output_complete as is_output_complete
from openplaces.io.cleanup.consumption import (
    output_conceptually_exists as output_conceptually_exists,
)
from openplaces.io.cleanup.consumption import (
    receipt_justifies_skip as receipt_justifies_skip,
)
from openplaces.io.cleanup.lock import DataLock as DataLock
from openplaces.io.cleanup.lock import _cluster_busy as _cluster_busy
from openplaces.io.cleanup.lock import _touch_lock as _touch_lock
from openplaces.io.cleanup.receipts import _RECEIPT_FORMAT as _RECEIPT_FORMAT
from openplaces.io.cleanup.receipts import RECEIPT_SUFFIX as RECEIPT_SUFFIX
from openplaces.io.cleanup.receipts import _cleanup_config as _cleanup_config
from openplaces.io.cleanup.receipts import (
    _recipe_retention_override as _recipe_retention_override,
)
from openplaces.io.cleanup.receipts import _relative_posix as _relative_posix
from openplaces.io.cleanup.receipts import _resolve_relative as _resolve_relative
from openplaces.io.cleanup.receipts import _utc_now_iso as _utc_now_iso
from openplaces.io.cleanup.receipts import discard_receipt as discard_receipt
from openplaces.io.cleanup.receipts import is_orchestrated as is_orchestrated
from openplaces.io.cleanup.receipts import read_receipt as read_receipt
from openplaces.io.cleanup.receipts import receipt_path as receipt_path
from openplaces.io.cleanup.receipts import write_receipt as write_receipt
from openplaces.io.cleanup.walk import _REPORT_COLUMNS as _REPORT_COLUMNS
from openplaces.io.cleanup.walk import _admin_ids_with_output as _admin_ids_with_output
from openplaces.io.cleanup.walk import _cleanup_image_node as _cleanup_image_node
from openplaces.io.cleanup.walk import _cleanup_node as _cleanup_node
from openplaces.io.cleanup.walk import (
    _delete_output_with_receipt as _delete_output_with_receipt,
)
from openplaces.io.cleanup.walk import _node_admins as _node_admins
from openplaces.io.cleanup.walk import _tree_size_bytes as _tree_size_bytes
from openplaces.io.cleanup.walk import _walk_dag as _walk_dag
from openplaces.io.cleanup.walk import cleanup as cleanup
from openplaces.io.cleanup.walk import (
    cleanup_consumed_inputs as cleanup_consumed_inputs,
)
from openplaces.io.cleanup.walk import discard_input_receipts as discard_input_receipts
