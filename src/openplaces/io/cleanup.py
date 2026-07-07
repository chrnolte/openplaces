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
"""

import fnmatch
import getpass
import json
import os
import shutil
import socket
import subprocess
import time
from contextlib import nullcontext
from datetime import UTC, datetime
from functools import cache
from pathlib import Path

import pandas as pd

from openplaces.config import cfg
from openplaces.core.attribute_registry import get_data_type
from openplaces.core.constants import NEVER_DELETE, STANDARD_DIRS
from openplaces.core.schema import AdminId
from openplaces.io import delete_parquet
from openplaces.io.aggregate import COVERAGE_ALL, read_partition_coverage
from openplaces.recipe import (
    get_output_path,
    get_recipe_by_id,
    get_recipe_dependencies,
    get_recipe_id,
    get_recipe_retention,
    get_save_admin_level,
)

RECEIPT_SUFFIX = '.consumed.json'

_RECEIPT_FORMAT = 1

# Destructive orphan deletion is refused when fewer recipes than this
# loaded successfully (misconfigured recipe path, broken environment).
_MIN_RECIPES_FOR_ORPHAN_GC = 25

_REPORT_COLUMNS = [
    'path',
    'bucket',
    'class',
    'action',
    'size_mb',
    'recipe_id',
    'admin_id',
    'blocked_by',
]


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')


def _relative_posix(path) -> str:
    """Data-root-relative forward-slash path (portable across OSes)."""
    path = Path(path)
    try:
        return path.relative_to(cfg.data_root).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_relative(path_str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else cfg.data_root / p


def is_orchestrated() -> bool:
    """True when running under an orchestrator (e.g. Snakemake).

    Orchestrated runs must produce the physical output file, so
    receipt-based skips are voided.
    """
    return bool(
        os.environ.get('SNAKEMAKE') or os.environ.get('OPENPLACES_ORCHESTRATED')
    )


def _cleanup_config() -> dict:
    return (cfg.get('retention') or {}).get('cleanup') or {}


# RECEIPTS


def receipt_path(output_path) -> Path:
    """Path of the tombstone receipt for an output file or directory."""
    p = Path(output_path)
    return p.with_name(p.stem + RECEIPT_SUFFIX)


def write_receipt(output_path, receipt: dict) -> Path:
    """Atomically write a tombstone receipt beside an output path."""
    receipt = {'format': _RECEIPT_FORMAT, **receipt}
    rp = receipt_path(output_path)
    rp.parent.mkdir(parents=True, exist_ok=True)
    tmp = rp.with_name(rp.name + f'.tmp{os.getpid()}')
    tmp.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    os.replace(tmp, rp)
    return rp


def read_receipt(output_path) -> dict | None:
    """Return the receipt dict for an output path, or None if absent/corrupt."""
    rp = receipt_path(output_path)
    if not rp.exists():
        return None
    try:
        return json.loads(rp.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None


def discard_receipt(output_path) -> None:
    """Remove the receipt for an output path (tolerant of it being absent)."""
    receipt_path(output_path).unlink(missing_ok=True)


# COMPLETENESS CHECKS


def _parquet_schema_ok(path) -> bool:
    """True when the parquet footer is readable and registry dtypes match.

    Guards against truncated or corrupted writes from killed jobs.
    Schema-only: no data is read. Registry-known numeric columns must map
    to numeric arrow types.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        schema = pq.read_schema(path)
    except Exception:
        return False
    for field in schema:
        expected = get_data_type(field.name)
        if expected in ('float', 'int'):
            t = field.type
            if not (
                pa.types.is_integer(t)
                or pa.types.is_floating(t)
                or pa.types.is_decimal(t)
                or pa.types.is_null(t)
            ):
                return False
    return True


def _required_subadmin_ids(admin_id) -> set[str] | None:
    """Sub-admin (level 4) units of an admin unit, or None when unknown.

    Used to verify enrich evidence coverage footers. Returns None when the
    admin data needed to enumerate sub-units is unavailable (fail safe:
    the caller must treat partial coverage as incomplete).
    """
    try:
        from openplaces.io.readers import get_admin_ids

        return set(get_admin_ids(4, admin_id=admin_id))
    except Exception:
        return None


def is_output_complete(recipe, admin_id, required_partitions=None) -> bool:
    """True when a recipe output physically exists and passes its checks.

    - plain parquet: exists AND the footer is readable AND registry-known
      columns pass the (schema-only) dtype check
    - aggregated/partitioned parquet with a coverage footer: additionally,
      the recorded coverage is a superset of `required_partitions`
    - enrich evidence: coverage covers all sub-admin units, or the
      COVERAGE_ALL sentinel

    Parameters
    ----------
    recipe : str or dict
        Recipe ID or loaded recipe dictionary.
    admin_id : str or AdminId or None
        Admin unit of the output.
    required_partitions : iterable of str, optional
        Partition or sub-admin IDs that the output's coverage footer must
        include (ignored when the footer is absent or COVERAGE_ALL).
    """
    if isinstance(recipe, str):
        try:
            recipe = get_recipe_by_id(recipe)
        except Exception:
            return False
    try:
        out_path = get_output_path(recipe, admin_id=admin_id)
    except Exception:
        return False
    if not out_path.exists():
        return False
    if out_path.is_dir():
        return any(out_path.iterdir())
    if out_path.suffix != '.parquet':
        return True
    if not _parquet_schema_ok(out_path):
        return False
    coverage = read_partition_coverage(out_path)
    if not coverage or COVERAGE_ALL in coverage:
        # No coverage footer (plain output or legacy file) or full coverage
        return True
    if required_partitions is not None:
        return set(map(str, required_partitions)) <= coverage
    if recipe.get('stage') == 'enrich':
        required = _required_subadmin_ids(admin_id)
        if required is None:
            return False
        return required <= coverage
    return True


def output_conceptually_exists(recipe, admin_id) -> bool:
    """True when the output exists on disk OR a valid receipt stands in.

    The receipt cascade rule (design section 4.3): a consumer that was
    itself cleaned up still counts as existing, so receipts of its inputs
    stay valid and nothing is needlessly re-ingested.
    """
    if is_output_complete(recipe, admin_id):
        return True
    if isinstance(recipe, str):
        try:
            recipe = get_recipe_by_id(recipe)
        except Exception:
            return False
    try:
        out_path = get_output_path(recipe, admin_id=admin_id)
    except Exception:
        return False
    return read_receipt(out_path) is not None


def _path_conceptually_exists(path: Path) -> bool:
    return path.exists() or read_receipt(path) is not None


def receipt_justifies_skip(recipe, admin_id, orchestrated=None) -> bool:
    """True when a tombstone receipt justifies skipping regeneration.

    Requires (design section 4.3): retention.cleanup.honor_receipts
    enabled; not running under an orchestrator; a readable receipt; every
    recorded consumer output conceptually exists (physically, or via its
    own receipt); and the consumer set recomputed from the current recipe
    tree contains no consumer absent from the receipt (a recipe added
    after the deletion voids the skip).
    """
    if orchestrated is None:
        orchestrated = is_orchestrated()
    if orchestrated:
        return False
    if not _cleanup_config().get('honor_receipts', True):
        return False
    if isinstance(recipe, str):
        try:
            recipe = get_recipe_by_id(recipe)
        except Exception:
            return False
    try:
        out_path = get_output_path(recipe, admin_id=admin_id)
    except Exception:
        return False
    receipt = read_receipt(out_path)
    if receipt is None:
        return False

    consumers_verified = receipt.get('consumers_verified') or []
    if not consumers_verified:
        return False
    for consumer in consumers_verified:
        consumer_path = _resolve_relative(consumer.get('path', ''))
        if not _path_conceptually_exists(consumer_path):
            return False

    # Recompute the consumer set from the current recipe tree; consumers
    # not recorded in the receipt void the skip (fail safe)
    recorded = {c.get('recipe_id') for c in consumers_verified}
    index = _dependency_index()
    current, unresolved = index.consumers(get_recipe_id(recipe), admin_id)
    if unresolved:
        return False
    if any(consumer_id not in recorded for consumer_id in current):
        return False
    return True


# DEPENDENCY INDEX


@cache
def _all_recipe_ids() -> tuple[str, ...]:
    root = cfg.code_root.joinpath('src', 'openplaces', 'recipes')
    return tuple(sorted(p.stem for p in root.rglob('*.yaml')))


class _DependencyIndex:
    """Inverted recipe-dependency index: consumers per (recipe, admin).

    Literal edges are admin-independent and extracted once; auto-discovered
    edges are resolved per admin unit on demand (cached), the same way the
    pipeline resolves them at run time.
    """

    def __init__(self, recipe_ids=None):
        self.errors: list[tuple[str, Exception]] = []
        self.recipes: dict[str, dict] = {}
        for recipe_id in recipe_ids or _all_recipe_ids():
            try:
                self.recipes[recipe_id] = get_recipe_by_id(recipe_id)
            except Exception as error:
                self.errors.append((recipe_id, error))

        self._literal: dict[str, set[str]] = {}
        self._auto_consumers: list[str] = []
        for recipe_id, recipe in self.recipes.items():
            try:
                edges = get_recipe_dependencies(recipe)
            except Exception as error:
                self.errors.append((recipe_id, error))
                continue
            has_auto = False
            for edge in edges:
                if edge.kind == 'auto_discover':
                    has_auto = True
                elif edge.upstream_recipe_id:
                    self._literal.setdefault(edge.upstream_recipe_id, set()).add(
                        recipe_id
                    )
            if has_auto:
                self._auto_consumers.append(recipe_id)
        self._auto_cache: dict[tuple[str, str], tuple[set[str], bool]] = {}

    def _auto_upstreams(self, consumer_id, admin_str) -> tuple[set[str], bool]:
        key = (consumer_id, admin_str)
        if key not in self._auto_cache:
            upstream_ids: set[str] = set()
            unresolved = False
            try:
                edges = get_recipe_dependencies(
                    self.recipes[consumer_id], admin_id=admin_str
                )
            except Exception:
                edges = []
                unresolved = True
            for edge in edges:
                if edge.kind != 'auto_discover':
                    continue
                if edge.upstream_recipe_id:
                    upstream_ids.add(edge.upstream_recipe_id)
                elif not edge.resolved:
                    unresolved = True
            self._auto_cache[key] = (upstream_ids, unresolved)
        return self._auto_cache[key]

    def consumers(self, recipe_id, admin_id) -> tuple[set[str], bool]:
        """Return (consumer recipe IDs, any-unresolved flag) for one node.

        When the flag is True, some recipe in the tree has an
        auto-discovered reference that could not be resolved for this
        admin unit; the caller must assume it may consume this node.
        """
        found = set(self._literal.get(recipe_id, ()))
        admin_str = str(admin_id) if admin_id is not None else ''
        any_unresolved = False
        for consumer_id in self._auto_consumers:
            upstream_ids, unresolved = self._auto_upstreams(consumer_id, admin_str)
            if recipe_id in upstream_ids:
                found.add(consumer_id)
            if unresolved:
                any_unresolved = True
        found.discard(recipe_id)
        return found, any_unresolved


@cache
def _dependency_index() -> _DependencyIndex:
    return _DependencyIndex()


def _truncate_admin(admin_id, level: int) -> AdminId | None:
    if admin_id is None:
        return None
    if not isinstance(admin_id, AdminId):
        admin_id = AdminId(admin_id)
    if level <= 0:
        return None
    return AdminId(*admin_id.levels[:level])


def _consumers_complete(
    recipe_id: str,
    admin_id,
    index: _DependencyIndex,
    memo: dict | None = None,
) -> tuple[bool, list[str], list[dict]]:
    """Check completeness of every consumer of one (recipe, admin) node.

    Returns
    -------
    (deletable, blocked_by, consumers_verified)
        deletable is True iff there is at least one consumer and every
        consumer's output conceptually exists; blocked_by lists what
        prevents deletion; consumers_verified records the verified
        consumer outputs for the tombstone receipt.
    """
    memo_key = (recipe_id, str(admin_id) if admin_id is not None else None)
    if memo is not None and memo_key in memo:
        return memo[memo_key]
    consumer_ids, unresolved = index.consumers(recipe_id, admin_id)
    blocked_by: list[str] = []
    if unresolved:
        blocked_by.append('unresolved')
    if not consumer_ids:
        blocked_by.append('no consumers')
    verified: list[dict] = []
    node_level = AdminId(str(admin_id)).get_level() if admin_id is not None else 0
    for consumer_id in sorted(consumer_ids):
        consumer_recipe = index.recipes.get(consumer_id)
        if consumer_recipe is None:
            blocked_by.append(consumer_id)
            continue
        try:
            consumer_level = get_save_admin_level(consumer_recipe)
        except Exception:
            blocked_by.append(consumer_id)
            continue
        if consumer_level > node_level:
            # The consumer saves finer than this node; verifying all its
            # children is not supported yet (fail safe: keep the node)
            blocked_by.append(consumer_id)
            continue
        consumer_admin = _truncate_admin(admin_id, consumer_level)
        required = {str(admin_id)} if consumer_level < node_level else None
        complete = is_output_complete(
            consumer_recipe, consumer_admin, required_partitions=required
        ) or output_conceptually_exists(consumer_recipe, consumer_admin)
        if not complete:
            blocked_by.append(consumer_id)
            continue
        try:
            consumer_path = get_output_path(consumer_recipe, admin_id=consumer_admin)
        except Exception:
            blocked_by.append(consumer_id)
            continue
        verified.append(
            {
                'recipe_id': consumer_id,
                'admin_id': str(consumer_admin) if consumer_admin else None,
                'path': _relative_posix(consumer_path),
            }
        )
    result = (not blocked_by, blocked_by, verified)
    if memo is not None:
        memo[memo_key] = result
    return result


# LOCKING AND CLUSTER GUARDS


class DataLock:
    """Exclusive lock file under the data root for destructive operations.

    A county-scoped lock (`.openplaces.<admin_id>.lock`) lets cleanups of
    different counties run concurrently; the global lock
    (`.openplaces.lock`) serializes data-root-wide operations like
    compact(). Stale locks (older than `stale_after_s`) are taken over.
    """

    def __init__(self, admin_id=None, timeout_s=10.0, stale_after_s=3600.0):
        name = f'.openplaces.{admin_id}.lock' if admin_id else '.openplaces.lock'
        self.path = Path(cfg.data_root) / name
        self.timeout_s = timeout_s
        self.stale_after_s = stale_after_s
        self._fd = None

    def __enter__(self):
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                self._fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                payload = f'{os.getpid()}@{socket.gethostname()} {_utc_now_iso()}'
                os.write(self._fd, payload.encode())
                return self
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                    if age > self.stale_after_s:
                        self.path.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f'Could not acquire lock {self.path}. Another '
                        'cleanup may be running; remove the lock file if '
                        'it is stale.'
                    ) from None
                time.sleep(0.5)

    def __exit__(self, *exc):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        self.path.unlink(missing_ok=True)
        return False


def _cluster_busy() -> bool:
    """True when a cluster queue has pending or running jobs for the user."""
    if shutil.which('qstat') is None:
        return False
    try:
        result = subprocess.run(
            ['qstat', '-u', getpass.getuser()],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        return False
    return bool(result.stdout.strip())


# DELETION


def _tree_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob('*'):
        try:
            if child.is_file():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def _delete_output_with_receipt(
    out_path: Path,
    recipe_id: str,
    admin_id,
    consumers_verified: list[dict],
) -> tuple[str, int]:
    """Delete one output (file+sidecar or directory) and write its receipt.

    Returns (action, bytes_reclaimed). A locked file never aborts a batch:
    PermissionError is retried once, then reported as
    'locked_retry_later'.
    """
    geo_path = out_path.with_name(out_path.stem + '_geo' + out_path.suffix)
    size = _tree_size_bytes(out_path)
    if geo_path.exists():
        size += geo_path.stat().st_size
    try:
        stat = out_path.stat()
        source_size, source_mtime = stat.st_size, stat.st_mtime
    except OSError:
        source_size, source_mtime = None, None
    partitions = (
        sorted(read_partition_coverage(out_path))
        if out_path.is_file() and out_path.suffix == '.parquet'
        else []
    )

    for attempt in (0, 1):
        try:
            if out_path.is_dir():
                shutil.rmtree(out_path)
            else:
                delete_parquet(out_path)
            break
        except FileNotFoundError:
            break  # another process got there first
        except PermissionError:
            if attempt == 0:
                time.sleep(1.0)
                continue
            return 'locked_retry_later', 0

    write_receipt(
        out_path,
        {
            'recipe_id': recipe_id,
            'admin_id': str(admin_id) if admin_id is not None else None,
            'deleted_at': _utc_now_iso(),
            'partitions': partitions,
            'source_size_bytes': source_size,
            'source_mtime': source_mtime,
            'consumers_verified': consumers_verified,
            'size_bytes_reclaimed': size,
        },
    )
    return 'deleted', size


# CLEANUP (DAG-SCOPED)


def _walk_dag(root_recipe, admin_id, index: _DependencyIndex):
    """Yield (recipe_id, recipe, node_admin) for every node upstream of root.

    The root itself is not yielded. Each upstream node's admin unit is the
    walk admin truncated to that recipe's save level; recipes saving finer
    than the walk admin (e.g. per-town image caches under a county walk)
    keep the walk admin and are expanded by their handler.
    """
    root_id = get_recipe_id(root_recipe)
    admin_level = AdminId(str(admin_id)).get_level() if admin_id else 0
    visited = {root_id}
    if not isinstance(root_recipe, dict):
        root_recipe = get_recipe_by_id(root_id)
    pending = [root_recipe]
    while pending:
        recipe = pending.pop()
        try:
            edges = get_recipe_dependencies(recipe, admin_id=admin_id)
        except Exception:
            continue
        for edge in edges:
            upstream_id = edge.upstream_recipe_id
            if not upstream_id or upstream_id in visited:
                continue
            visited.add(upstream_id)
            try:
                upstream = get_recipe_by_id(upstream_id)
            except Exception:
                continue
            try:
                save_level = get_save_admin_level(upstream)
            except Exception:
                save_level = admin_level
            node_admin = (
                _truncate_admin(admin_id, min(save_level, admin_level))
                if admin_id
                else None
            )
            yield upstream_id, upstream, node_admin
            pending.append(upstream)


def _admin_ids_with_output(recipe) -> list[str]:
    """Admin IDs that have an output file for a recipe on disk."""
    from openplaces.recipe import _get_save_to

    data_dir, _ = _get_save_to(recipe)
    root = Path(cfg.get_dir(data_dir or 'cache'))
    entity = recipe.get('entity') or recipe.get('dataset')
    if entity is None or not root.is_dir():
        return []
    token = str(entity)
    found = set()
    for path in root.rglob(f'*_{token}.parquet'):
        prefix = path.stem[: -(len(token) + 1)]
        try:
            found.add(str(AdminId(prefix)))
        except ValueError:
            continue
    return sorted(found)


def cleanup(
    recipe,
    admin_ids=None,
    stages=None,
    include_images=False,
    aggressive=False,
    dry_run=True,
    verbose=True,
    force=False,
) -> pd.DataFrame:
    """Reclaim consumed intermediate outputs upstream of a terminal recipe.

    Walks the dependency DAG rooted at `recipe`, finds outputs with
    retention 'until_consumed' whose consumers are all complete
    (section 4.2 of the design), deletes them, and writes tombstone
    receipts. Every decision is recomputed from disk.

    Parameters
    ----------
    recipe : str or dict
        Terminal recipe the DAG is rooted at (e.g.
        'US_footprint-cheer-2026').
    admin_ids : str or list, optional
        Admin units in scope; None scans the terminal recipe's outputs on
        disk.
    stages : tuple of str, optional
        Only consider upstream recipes of these stages (e.g. ('ingest',)
        to only reclaim the cache).
    include_images : bool
        Image caches are only deleted with this explicit opt-in (or
        retention.cleanup.include_images in the config); they are always
        listed in the report.
    aggressive : bool
        Additionally treat 'core' outputs as until_consumed for this call
        (kept only until the curated outputs exist). Enrich evidence
        stays 'keep' regardless: its input images may be gone.
    dry_run : bool
        Default True: only report. Pass dry_run=False to delete.
    verbose : bool
        Print a short summary.
    force : bool
        Skip the cluster-queue guard.

    Returns
    -------
    pd.DataFrame
        Columns: path, bucket, class, action, size_mb, recipe_id,
        admin_id, blocked_by.
    """
    if isinstance(recipe, str):
        recipe = get_recipe_by_id(recipe)
    root_id = get_recipe_id(recipe)
    if admin_ids is None:
        admin_ids = _admin_ids_with_output(recipe)
    elif isinstance(admin_ids, str | AdminId):
        admin_ids = [admin_ids]

    if not dry_run and not force and _cluster_busy():
        raise RuntimeError(
            'Cluster queue has pending or running jobs; refusing '
            'destructive cleanup. Pass force=True to override.'
        )

    include_images = include_images or _cleanup_config().get('include_images', False)
    index = _dependency_index()
    rows: list[dict] = []

    for admin_id in admin_ids:
        lock = DataLock(admin_id) if not dry_run else nullcontext()
        with lock:
            for node_id, node_recipe, node_admin in _walk_dag(recipe, admin_id, index):
                if stages and node_recipe.get('stage') not in stages:
                    continue
                rows.extend(
                    _cleanup_node(
                        node_id,
                        node_recipe,
                        node_admin,
                        index,
                        include_images=include_images,
                        aggressive=aggressive,
                        dry_run=dry_run,
                    )
                )

    report = pd.DataFrame(rows, columns=_REPORT_COLUMNS)
    if verbose and not report.empty:
        reclaimable = report.loc[
            report['action'].isin(['deleted', 'would_delete']), 'size_mb'
        ].sum()
        verb = 'reclaimed' if not dry_run else 'reclaimable'
        print(f'cleanup({root_id}): {reclaimable:,.1f} MB {verb}.')
        if dry_run:
            print('Dry run: pass dry_run=False to delete.')
    return report


def _cleanup_node(
    node_id,
    node_recipe,
    node_admin,
    index,
    include_images,
    aggressive,
    dry_run,
) -> list[dict]:
    """Classify one DAG node and delete it when allowed."""
    entity = node_recipe.get('entity')
    entity_type = str(entity.entity_type) if entity is not None else None
    from openplaces.recipe import _get_save_to

    data_dir, _ = _get_save_to(node_recipe)
    retention = get_recipe_retention(node_recipe)
    if aggressive and data_dir == 'core' and node_recipe.get('stage') != 'enrich':
        retention = 'until_consumed'
    if data_dir in NEVER_DELETE:
        retention = 'keep'

    if entity_type == 'image':
        return _cleanup_image_node(
            node_id,
            node_recipe,
            node_admin,
            index,
            retention,
            include_images,
            dry_run,
        )

    try:
        out_path = get_output_path(node_recipe, admin_id=node_admin)
    except Exception:
        return []
    if not out_path.exists():
        return []
    size_mb = round(_tree_size_bytes(out_path) / 2**20, 1)
    row = {
        'path': _relative_posix(out_path),
        'bucket': data_dir,
        'class': retention,
        'size_mb': size_mb,
        'recipe_id': node_id,
        'admin_id': str(node_admin) if node_admin else None,
        'blocked_by': None,
        'action': 'kept',
    }
    if retention != 'until_consumed':
        return [row]

    deletable, blocked_by, verified = _consumers_complete(node_id, node_admin, index)
    if not deletable:
        row['blocked_by'] = ', '.join(blocked_by)
        row['action'] = 'blocked'
        return [row]
    if dry_run:
        row['action'] = 'would_delete'
        return [row]
    action, _ = _delete_output_with_receipt(out_path, node_id, node_admin, verified)
    row['action'] = action
    return [row]


def _cleanup_image_node(
    node_id,
    node_recipe,
    node_admin,
    index,
    retention,
    include_images,
    dry_run,
) -> list[dict]:
    """Handle image-cache nodes: per-cache refcounting via coverage footers."""
    from openplaces.diagnostics import list_image_caches

    entity = node_recipe['entity']
    source = str(entity.source.source_id) if entity.source else None
    version = str(entity.version) if entity.version else None
    caches = list_image_caches()
    if caches.empty:
        return []
    caches = caches[(caches['source'] == source) & (caches['version'] == version)]
    if node_admin is not None:
        selector = AdminId(str(node_admin))
        caches = caches[
            [
                str(selector) == cache_admin
                or selector.is_parent_of(AdminId(cache_admin))
                for cache_admin in caches['admin_id']
            ]
        ]

    rows = []
    for image_cache in caches.itertuples():
        cache_path = Path(image_cache.path)
        row = {
            'path': _relative_posix(cache_path),
            'bucket': 'external',
            'class': retention,
            'size_mb': image_cache.size_mb,
            'recipe_id': node_id,
            'admin_id': image_cache.admin_id,
            'blocked_by': None,
            'action': 'kept',
        }
        if retention != 'until_consumed':
            rows.append(row)
            continue
        if not include_images:
            row['action'] = 'kept'
            row['blocked_by'] = 'include_images not set (paid re-fetch)'
            rows.append(row)
            continue
        deletable, blocked_by, verified = _consumers_complete(
            node_id, image_cache.admin_id, index
        )
        if not deletable:
            row['action'] = 'blocked'
            row['blocked_by'] = ', '.join(blocked_by)
        elif dry_run:
            row['action'] = 'would_delete'
        else:
            action, _ = _delete_output_with_receipt(
                cache_path, node_id, image_cache.admin_id, verified
            )
            row['action'] = action
        rows.append(row)
    return rows


def cleanup_consumed_inputs(
    recipe,
    admin_id,
    include_images=False,
    verbose=False,
) -> pd.DataFrame:
    """Reclaim the consumed direct inputs of one recipe for one admin unit.

    Backs the stage entrypoints' cleanup='consumed' hook: after a stage
    finishes an admin unit, each of its direct inputs is deleted iff every
    consumer in the recipe tree is complete. Safe when called early —
    consumers with no output yet block deletion (e.g. the NSI parquet
    survives the footprint-spine hook until the parcel spine also exists).
    No-op when retention.cleanup.enabled is false.

    Parameters
    ----------
    recipe : str or dict
        The stage recipe whose inputs to consider.
    admin_id : str or AdminId
        The admin unit just finished.
    include_images : bool
        Opt-in for image-cache deletion (or
        retention.cleanup.include_images in the config).
    verbose : bool
        Print a one-line summary when something was reclaimed.
    """
    if not _cleanup_config().get('enabled', True):
        return pd.DataFrame(columns=_REPORT_COLUMNS)
    if isinstance(recipe, str):
        recipe = get_recipe_by_id(recipe)
    include_images = include_images or _cleanup_config().get('include_images', False)
    index = _dependency_index()
    admin_level = AdminId(str(admin_id)).get_level() if admin_id else 0

    rows: list[dict] = []
    with DataLock(admin_id):
        try:
            edges = get_recipe_dependencies(recipe, admin_id=admin_id)
        except Exception:
            edges = []
        seen: set[str] = set()
        for edge in edges:
            upstream_id = edge.upstream_recipe_id
            if not upstream_id or upstream_id in seen:
                continue
            seen.add(upstream_id)
            try:
                upstream = get_recipe_by_id(upstream_id)
                save_level = get_save_admin_level(upstream)
            except Exception:
                continue
            node_admin = (
                _truncate_admin(admin_id, min(save_level, admin_level))
                if admin_id
                else None
            )
            rows.extend(
                _cleanup_node(
                    upstream_id,
                    upstream,
                    node_admin,
                    index,
                    include_images=include_images,
                    aggressive=False,
                    dry_run=False,
                )
            )

    report = pd.DataFrame(rows, columns=_REPORT_COLUMNS)
    if verbose and not report.empty:
        reclaimed = report.loc[report['action'] == 'deleted', 'size_mb'].sum()
        if reclaimed:
            print(f'  cleanup: reclaimed {reclaimed:,.1f} MB of consumed inputs.')
    return report


# COMPACT (BUCKET GARBAGE COLLECTOR)


@cache
def _recipe_token_index() -> dict:
    """Map each recipe's entity/dataset token to its recipe IDs.

    A recipe ID reads {admin}_{token}[_{filename}]; output files read
    {file_admin}_{token}[_{filename}][_{suffix}].parquet, where file_admin
    may be deeper than the recipe's admin scope.
    """
    index: dict[str, list[str]] = {}
    for recipe_id in _all_recipe_ids():
        parts = recipe_id.split('_')
        try:
            AdminId(parts[0])
            token_pos = 1
        except ValueError:
            token_pos = 0
        if len(parts) > token_pos:
            index.setdefault(parts[token_pos], []).append(recipe_id)
    return index


def _match_recipe_for_file(stem: str) -> tuple[str | None, str | None]:
    """Match an output filename stem to (recipe_id, admin_id).

    Returns (None, admin) when the stem parses but matches no recipe in
    the current tree (an orphan candidate).
    """
    parts = stem.split('_')
    admin = None
    try:
        admin = AdminId(parts[0])
        rest = parts[1:]
    except ValueError:
        rest = parts
    if not rest:
        return None, None
    admin_str = str(admin) if admin else None
    for recipe_id in _recipe_token_index().get(rest[0], []):
        recipe_parts = recipe_id.split('_')
        try:
            recipe_admin = AdminId(recipe_parts[0])
            recipe_rest = recipe_parts[1:]
        except ValueError:
            recipe_admin = None
            recipe_rest = recipe_parts
        # The recipe's admin scope must cover the file's admin unit
        if recipe_admin is not None:
            if admin is None or not recipe_admin.is_parent_or_equal_of(admin):
                continue
        # The recipe's filename parts must prefix the file's remaining parts
        recipe_filename = recipe_rest[1:]
        if list(rest[1 : 1 + len(recipe_filename)]) != recipe_filename:
            continue
        return recipe_id, admin_str
    return None, admin_str


def _match_recipe_for_path(
    path: Path, bucket_root: Path
) -> tuple[str | None, str | None]:
    """Match an on-disk file to (recipe_id, admin_id).

    Tries the output filename convention first, then falls back to the
    directory layout: downloaded archives and image caches keep their
    source filenames, but live under recipe-derived directory paths
    ({admin levels...}/_all/{entity or dataset path...}/...).
    """
    from openplaces.core.constants import ESCAPE_DIR

    recipe_id, admin_str = _match_recipe_for_file(path.stem)
    if recipe_id is not None:
        return recipe_id, admin_str
    try:
        parts = path.parent.relative_to(bucket_root).parts
    except ValueError:
        return None, admin_str
    if ESCAPE_DIR in parts:
        cut = parts.index(ESCAPE_DIR)
        dir_admin = '-'.join(parts[:cut]) or None
        dataset_parts = parts[cut + 1 :]
    else:
        dir_admin = None
        dataset_parts = parts
    if dir_admin is not None:
        try:
            AdminId(dir_admin)
        except ValueError:
            dir_admin = None
    admin_str = admin_str or dir_admin
    if not dataset_parts:
        return None, admin_str
    index = _recipe_token_index()
    # The dataset path may extend past the recipe's token (nested cache
    # subdirectories), so try progressively shorter prefixes
    for length in range(len(dataset_parts), 1, -1):
        token = '-'.join(dataset_parts[:length])
        for recipe_id in index.get(token, []):
            recipe_parts = recipe_id.split('_')
            try:
                recipe_admin = AdminId(recipe_parts[0])
            except ValueError:
                recipe_admin = None
            if recipe_admin is not None:
                if dir_admin is None or not recipe_admin.is_parent_or_equal_of(
                    AdminId(dir_admin)
                ):
                    continue
            return recipe_id, admin_str
    return None, admin_str


def _bucket_roots(buckets) -> list[tuple[str, Path]]:
    """Configured roots of the requested buckets plus nested known buckets,
    deepest first, so each file is attributed to its most specific bucket."""
    roots = []
    for bucket in set(buckets) | set(STANDARD_DIRS) - {'data_root'}:
        try:
            root = Path(cfg.get_dir(bucket))
        except KeyError:
            continue
        roots.append((bucket, root.resolve()))
    roots.sort(key=lambda item: len(item[1].parts), reverse=True)
    return roots


def _bucket_of(path: Path, roots) -> str | None:
    for bucket, root in roots:
        if path.is_relative_to(root):
            return bucket
    return None


def compact(
    buckets=('cache', 'heap', 'external', 'core', 'out'),
    recipes=None,
    admin_ids=None,
    delete=(),
    dry_run=True,
    orphan_min_age_days=14,
    min_size_mb=0.0,
    include_shared=False,
    force=False,
) -> pd.DataFrame:
    """Garbage-collect the data buckets: report and optionally delete.

    Existent-first scan: every file on disk in the requested buckets is
    parsed for its recipe and admin unit and matched against the active
    recipe tree, then classified:

    - 'final': expected output with retention 'keep' (report only)
    - 'intermediate/needed': until_consumed with incomplete consumers
    - 'intermediate/consumed': until_consumed, all consumers complete;
      deleted when 'consumed' is in `delete`
    - 'heap': anything under the heap; deleted when 'heap' is in `delete`
    - 'orphan': matches no recipe in the current tree; deleted only when
      'orphans' is in `delete` (guards below)
    - 'receipt/stale': receipt whose recorded consumers all vanished;
      pruned automatically when any deletion is enabled

    Orphan-GC guards (all mandatory): compact aborts destructive runs when
    any recipe fails to parse; refuses orphan deletion when implausibly
    few recipes loaded; and never touches files matching
    retention.cleanup.exclude_patterns.

    Parameters
    ----------
    buckets : tuple of str
        Buckets to scan.
    recipes : list of str, optional
        Restrict the recipe tree to these recipe IDs (None = all).
    admin_ids : list, optional
        Restrict to files under these admin units.
    delete : tuple of str
        Subset of {'consumed', 'heap', 'orphans'}; empty = report only.
    dry_run : bool
        Must be False IN ADDITION to a non-empty `delete` for any
        deletion (two explicit acts).
    orphan_min_age_days : int
        A file younger than this is never classified as orphan.
    min_size_mb : float
        Drop report rows smaller than this.
    include_shared : bool
        Allow orphan deletion in shared buckets (external, share, raw).
    force : bool
        Skip the cluster-queue guard.

    Returns
    -------
    pd.DataFrame
        Columns: path, bucket, class, action, size_mb, recipe_id,
        admin_id, blocked_by.
    """
    invalid = set(delete) - {'consumed', 'heap', 'orphans'}
    if invalid:
        raise ValueError(f'Unknown delete selection(s): {sorted(invalid)}')
    destructive = bool(delete) and not dry_run

    if destructive and not force and _cluster_busy():
        raise RuntimeError(
            'Cluster queue has pending or running jobs; refusing '
            'destructive compact. Pass force=True to override.'
        )

    index = _dependency_index()
    if destructive and index.errors:
        failed = ', '.join(recipe_id for recipe_id, _ in index.errors[:5])
        raise RuntimeError(
            'Refusing destructive compact: recipe(s) failed to load '
            f'({failed}). A recipe with a transient error must not turn '
            'its outputs into orphans.'
        )
    orphan_gc_allowed = (
        'orphans' in delete and len(index.recipes) >= _MIN_RECIPES_FOR_ORPHAN_GC
    )
    recipe_scope = set(recipes) if recipes else None
    admin_scope = [AdminId(str(a)) for a in admin_ids] if admin_ids else None
    exclude_patterns = _cleanup_config().get('exclude_patterns') or []
    now = time.time()
    roots = _bucket_roots(buckets)
    shared_buckets = {
        name for name, info in STANDARD_DIRS.items() if info.get('shared')
    }

    lock = DataLock() if destructive else nullcontext()
    rows: list[dict] = []
    scanned_receipts: list[Path] = []
    memo: dict = {}

    with lock:
        seen_dirs: set[Path] = set()
        for bucket in buckets:
            try:
                bucket_root = Path(cfg.get_dir(bucket)).resolve()
            except KeyError:
                continue
            if not bucket_root.is_dir():
                continue
            for dirpath, dirnames, filenames in os.walk(bucket_root):
                current = Path(dirpath).resolve()
                if current in seen_dirs:
                    dirnames[:] = []
                    continue
                seen_dirs.add(current)
                owner = _bucket_of(current, roots) or bucket
                for filename in filenames:
                    path = current / filename
                    rel = _relative_posix(path)
                    if any(
                        fnmatch.fnmatch(rel, pattern) for pattern in exclude_patterns
                    ):
                        continue
                    if filename.endswith('.lock'):
                        continue
                    if filename.endswith(RECEIPT_SUFFIX):
                        scanned_receipts.append(path)
                        continue
                    row = _classify_file(
                        path,
                        owner,
                        bucket_root,
                        index,
                        recipe_scope,
                        admin_scope,
                        orphan_min_age_days,
                        now,
                        memo,
                    )
                    if row is None:
                        continue
                    rows.append(row)

        for row in rows:
            rows_action = _compact_action(
                row,
                delete,
                dry_run,
                orphan_gc_allowed,
                shared_buckets,
                include_shared,
                index,
            )
            row['action'] = rows_action

        rows.extend(_process_receipts(scanned_receipts, prune=destructive))

    report = pd.DataFrame(rows, columns=_REPORT_COLUMNS)
    if not report.empty and min_size_mb:
        report = report[
            (report['size_mb'] >= min_size_mb) | (report['class'] != 'final')
        ]
    return report.sort_values('size_mb', ascending=False, ignore_index=True)


def _classify_file(
    path: Path,
    bucket: str,
    bucket_root: Path,
    index: _DependencyIndex,
    recipe_scope,
    admin_scope,
    orphan_min_age_days,
    now,
    memo: dict,
) -> dict | None:
    try:
        size_mb = round(path.stat().st_size / 2**20, 3)
    except OSError:
        return None
    if bucket in ('heap', 'logs', 'models', 'reports') or bucket in NEVER_DELETE:
        recipe_id, admin_str = None, None
    else:
        recipe_id, admin_str = _match_recipe_for_path(path, bucket_root)
    if admin_scope is not None:
        if admin_str is None:
            return None
        admin = AdminId(admin_str)
        if not any(
            str(sel) == admin_str or sel.is_parent_of(admin) for sel in admin_scope
        ):
            return None
    if recipe_scope is not None and recipe_id not in recipe_scope:
        return None

    row = {
        'path': _relative_posix(path),
        'bucket': bucket,
        'size_mb': size_mb,
        'recipe_id': recipe_id,
        'admin_id': admin_str,
        'blocked_by': None,
        'action': 'report',
    }
    if bucket == 'heap':
        row['class'] = 'heap'
        return row
    if bucket in NEVER_DELETE or bucket in ('logs', 'models', 'reports'):
        # Protected or user-owned buckets are report-only
        row['class'] = 'final'
        return row
    if recipe_id is None:
        try:
            age_days = (now - path.stat().st_mtime) / 86400
        except OSError:
            age_days = 0
        row['class'] = 'orphan' if age_days >= orphan_min_age_days else 'recent'
        return row
    recipe = index.recipes.get(recipe_id)
    retention = 'keep'
    if recipe is not None:
        # Retention follows the bucket the file actually lives in: a
        # downloaded archive in 'external' is an input copy protected by
        # the bucket default, not by the recipe's (cache) output retention
        from openplaces.recipe import _get_save_to

        save_dir, _ = _get_save_to(recipe)
        recipe_retention = (
            (recipe.get('save_to') or {}).get('retention')
            if save_dir == bucket
            else None
        )
        try:
            retention = cfg.retention_for(
                bucket, recipe_id=recipe_id, recipe_retention=recipe_retention
            )
        except Exception:
            pass
    if retention != 'until_consumed':
        row['class'] = 'final'
        return row
    deletable, blocked_by, _ = _consumers_complete(
        recipe_id, admin_str, index, memo=memo
    )
    if deletable:
        row['class'] = 'intermediate/consumed'
    else:
        row['class'] = 'intermediate/needed'
        row['blocked_by'] = ', '.join(blocked_by)
    return row


def _compact_action(
    row,
    delete,
    dry_run,
    orphan_gc_allowed,
    shared_buckets,
    include_shared,
    index,
) -> str:
    cls = row['class']
    path = _resolve_relative(row['path'])
    if cls == 'heap' and 'heap' in delete:
        return _compact_delete(path, dry_run)
    if cls == 'intermediate/consumed' and 'consumed' in delete:
        recipe = index.recipes.get(row['recipe_id']) or {}
        entity = recipe.get('entity')
        if entity is not None and str(entity.entity_type) == 'image':
            # Image caches are deleted per cache with one receipt, not per
            # tile file; compact only reports them
            row['blocked_by'] = 'image cache: use cleanup(include_images=True)'
            return 'blocked'
        if dry_run:
            return 'would_delete'
        _, blocked_by, verified = _consumers_complete(
            row['recipe_id'], row['admin_id'], index
        )
        if blocked_by:
            row['blocked_by'] = ', '.join(blocked_by)
            return 'blocked'
        action, _ = _delete_output_with_receipt(
            path, row['recipe_id'], row['admin_id'], verified
        )
        return action
    if cls == 'orphan' and 'orphans' in delete:
        if row['bucket'] in shared_buckets and not include_shared:
            row['blocked_by'] = 'shared bucket (pass include_shared=True)'
            return 'blocked'
        if not orphan_gc_allowed:
            row['blocked_by'] = 'orphan GC guard'
            return 'blocked'
        return _compact_delete(path, dry_run)
    return 'report'


def _compact_delete(path: Path, dry_run: bool) -> str:
    if dry_run:
        return 'would_delete'
    for attempt in (0, 1):
        try:
            path.unlink()
            return 'deleted'
        except FileNotFoundError:
            return 'deleted'
        except PermissionError:
            if attempt == 0:
                time.sleep(1.0)
                continue
            return 'locked_retry_later'
    return 'locked_retry_later'


def _process_receipts(receipt_paths, prune: bool) -> list[dict]:
    """Classify receipts; stale ones (all consumers vanished) are pruned."""
    rows = []
    for path in receipt_paths:
        try:
            receipt = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            receipt = None
        consumers = (receipt or {}).get('consumers_verified') or []
        stale = (
            bool(receipt)
            and consumers != []
            and not any(
                _path_conceptually_exists(_resolve_relative(c.get('path', '')))
                for c in consumers
            )
        )
        if receipt is None:
            stale = True
        if not stale:
            continue
        action = 'report'
        if prune:
            path.unlink(missing_ok=True)
            action = 'deleted'
        rows.append(
            {
                'path': _relative_posix(path),
                'bucket': None,
                'class': 'receipt/stale',
                'action': action,
                'size_mb': 0.0,
                'recipe_id': (receipt or {}).get('recipe_id'),
                'admin_id': (receipt or {}).get('admin_id'),
                'blocked_by': None,
            }
        )
    return rows
