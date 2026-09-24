"""
The cleanup walk: delete consumed until_consumed outputs under a
root recipe and admin unit, node by node over the recipe DAG,
leaving a receipt for each, and the per-stage entry point the
harmonizer, enricher and curator call after a successful save.
discard_input_receipts lives here rather than with the receipts
because it walks the DAG to find them.
"""

import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import pandas as pd

from openplaces.config import cfg
from openplaces.core.constants import NEVER_DELETE
from openplaces.core.schema import AdminId
from openplaces.io import delete_parquet
from openplaces.io.aggregate import read_partition_coverage
from openplaces.io.cleanup.consumption import (
    _consumers_complete,
    _dependency_index,
    _truncate_admin,
)
from openplaces.io.cleanup.lock import DataLock, _cluster_busy
from openplaces.io.cleanup.receipts import (
    _cleanup_config,
    _recipe_retention_override,
    _relative_posix,
    _utc_now_iso,
    discard_receipt,
    receipt_path,
    write_receipt,
)
from openplaces.recipe import (
    get_output_path,
    get_recipe_by_id,
    get_recipe_dependencies,
    get_recipe_id,
    get_recipe_retention,
    get_save_admin_level,
)

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


def discard_input_receipts(recipe, admin_id=None) -> list[Path]:
    """Discard the tombstone receipts of one recipe's direct inputs.

    A stage discards its own receipt when reprocessing, but an input's
    receipt records *this* output as one of the consumers that justified
    deleting it. Left standing, it makes the next ingest skip
    regenerating an input the rerun is about to read, which then fails
    on the missing file. A deliberate rerun supersedes its inputs'
    receipts the same way it supersedes its own, so the stage
    entrypoints call this on reprocess.

    Parameters
    ----------
    recipe : str or dict
        The recipe being reprocessed.
    admin_id : str or AdminId, optional
        Admin unit being reprocessed; None for a global run.

    Returns
    -------
    list of pathlib.Path
        Output paths whose receipts were removed.
    """
    if isinstance(recipe, str):
        try:
            recipe = get_recipe_by_id(recipe)
        except Exception:
            return []
    try:
        edges = get_recipe_dependencies(recipe, admin_id=admin_id)
    except Exception:
        return []
    admin_level = AdminId(str(admin_id)).get_level() if admin_id else 0
    discarded: list[Path] = []
    seen: set[str] = set()
    for edge in edges:
        upstream_id = edge.upstream_recipe_id
        if not upstream_id or upstream_id in seen:
            continue
        seen.add(upstream_id)
        try:
            upstream = get_recipe_by_id(upstream_id)
        except Exception:
            continue
        for node_admin in _node_admins(upstream, admin_id, admin_level):
            try:
                out_path = get_output_path(upstream, admin_id=node_admin)
            except Exception:
                continue
            if receipt_path(out_path).exists():
                discard_receipt(out_path)
                discarded.append(out_path)
    return discarded


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
    partitions: list[str] = []
    if out_path.is_file() and out_path.suffix == '.parquet':
        try:
            partitions = sorted(read_partition_coverage(out_path))
        except Exception:
            # A truncated footer must not abort the batch: this runs
            # inside the stages' cleanup='consumed' hook, where raising
            # would crash the harmonizer or curator after the unit's own
            # output was already written. The file is being deleted
            # anyway, so record no coverage for it.
            partitions = []

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


def _node_admins(upstream, admin_id, admin_level: int, expand_finer=False) -> list:
    """Admin units at which one upstream recipe's output is reclaimable.

    Normally a single unit: the walk admin truncated to the recipe's save
    level. A recipe that saves finer than the walk admin (town-level
    inputs of a county spine) has no output at the walk admin at all, and
    leaving it there made `get_output_path` raise, which was swallowed:
    the node vanished from the report and was never reclaimable. With
    *expand_finer* its finer units are read off disk instead, so only
    units with something to reclaim are visited; that scan is why a
    caller not about to delete anything leaves the flag off. Image
    recipes stay at the walk admin, which is what their own handler
    expands.
    """
    if admin_id is None:
        return [None]
    try:
        save_level = get_save_admin_level(upstream)
    except Exception:
        save_level = admin_level
    if save_level <= admin_level or not expand_finer:
        return [_truncate_admin(admin_id, min(save_level, admin_level))]
    entity = upstream.get('entity')
    if entity is not None and str(entity.entity_type) == 'image':
        return [_truncate_admin(admin_id, admin_level)]
    walk_admin = AdminId(str(admin_id))
    finer = []
    for candidate in _admin_ids_with_output(upstream, under_admin=walk_admin):
        node_admin = AdminId(candidate)
        if node_admin.get_level() == save_level and walk_admin.is_parent_of(node_admin):
            finer.append(node_admin)
    # With nothing on disk there is nothing to reclaim, but the node
    # itself must still be yielded: `flow.dag` walks this graph to
    # enumerate jobs, and on a fresh install no output exists yet
    return finer or [_truncate_admin(admin_id, admin_level)]


def _walk_dag(root_recipe, admin_id, exclude_recipe_ids=None, expand_finer=False):
    """Yield (recipe_id, recipe, node_admin) for every node upstream of root.

    The root itself is not yielded. Each upstream node's admin unit is the
    walk admin truncated to that recipe's save level. With
    *expand_finer*, a recipe saving finer than the walk admin is yielded
    once per finer unit that has an output on disk (see `_node_admins`),
    except image recipes, which keep the walk admin and are expanded by
    their own handler. That expansion reads the disk, so callers that
    only enumerate recipes (flow.dag) leave it off.

    exclude_recipe_ids : set of str, optional
        Forwarded to `get_recipe_dependencies`. An excluded recipe's edges
        are never evaluated, so it (and anything only reachable through it)
        is pruned from the walk transitively -- see that function's
        docstring.
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
            edges = get_recipe_dependencies(
                recipe, admin_id=admin_id, exclude_recipe_ids=exclude_recipe_ids
            )
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
            for node_admin in _node_admins(
                upstream, admin_id, admin_level, expand_finer=expand_finer
            ):
                yield upstream_id, upstream, node_admin
            pending.append(upstream)


def _admin_ids_with_output(recipe, under_admin=None) -> list[str]:
    """Admin IDs that have an output file for a recipe on disk.

    Parameters
    ----------
    recipe : dict
        Loaded recipe whose outputs to look for.
    under_admin : str or AdminId, optional
        Only scan this unit's own subtree. The scan is a recursive glob
        of the output bucket, so narrowing it matters on a real data
        root; without it every finer-saving node in a walk would sweep
        the whole bucket.
    """
    from openplaces.recipe import _get_save_to

    data_dir, _ = _get_save_to(recipe)
    root = Path(cfg.get_dir(data_dir or 'cache'))
    if under_admin is not None:
        admin = AdminId(str(under_admin))
        root = root.joinpath(*admin.levels)
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
        'US_footprint-openplaces-2026').
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
            for node_id, node_recipe, node_admin in _walk_dag(
                recipe, admin_id, expand_finer=True
            ):
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
    # Aggressive mode demotes core outputs to until_consumed -- but only
    # those with no retention of their own. A recipe that declares
    # save_to: retention: keep (the geospine recipes: their outputs and
    # link sidecars are exactly what `--reprocess attributes` reuses, so
    # deleting them turns the next attribute-only rerun into a full
    # geometry rerun) keeps its declared class even under aggressive.
    # A per-recipe retention.recipes entry in the user's own config is
    # the documented protection lever and counts the same way: it is the
    # only way to protect a recipe whose YAML declares no retention, so
    # ignoring it here deleted exactly the outputs a user had pinned.
    explicit_retention = (node_recipe.get('save_to') or {}).get('retention')
    if explicit_retention is None:
        explicit_retention = _recipe_retention_override(node_id)
    if (
        aggressive
        and data_dir == 'core'
        and node_recipe.get('stage') != 'enrich'
        and explicit_retention is None
    ):
        retention = 'until_consumed'
    if data_dir in NEVER_DELETE:
        retention = 'keep'

    if entity_type == 'image':
        return _cleanup_image_node(
            node_id,
            node_recipe,
            node_admin,
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
    include_images,
    dry_run,
) -> list[dict]:
    """Reclaim leftover image caches, which nothing produces any more.

    Imagery has no ingest stage: Google's Static API policy prohibits
    storing or caching its content, so enrichment fetches pixels in memory
    per run and drops them. Any cache directory still on disk predates that
    and has no consumer to refcount against, so this neither reads the
    recipe's retention class (an image recipe declares no `save_to`) nor
    waits for a coverage footer -- both would only keep a stale artifact
    alive. Deleting is still opt-in through *include_images*, because
    removing files a user already has is their call, not a side effect.
    """
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
            'class': 'leftover',
            'size_mb': image_cache.size_mb,
            'recipe_id': node_id,
            'admin_id': image_cache.admin_id,
            'blocked_by': None,
            'action': 'kept',
        }
        if not include_images:
            row['action'] = 'kept'
            row['blocked_by'] = 'include_images not set (leftover image cache)'
            rows.append(row)
            continue
        if dry_run:
            row['action'] = 'would_delete'
        else:
            # No consumers to record: nothing reads an image cache now.
            action, _ = _delete_output_with_receipt(
                cache_path, node_id, image_cache.admin_id, []
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
    consumer in the recipe tree is complete. Safe when called early:
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
            except Exception:
                continue
            for node_admin in _node_admins(
                upstream, admin_id, admin_level, expand_finer=True
            ):
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
