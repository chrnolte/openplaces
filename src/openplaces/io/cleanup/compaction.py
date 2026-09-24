"""
compact(), the bucket garbage collector: classify every file in
a bucket against the recipe index (current, superseded, orphan,
stale receipt), decide an action per class, and delete or report.
"""

import fnmatch
import json
import os
import time
from contextlib import nullcontext
from functools import cache
from pathlib import Path

import pandas as pd

from openplaces.config import cfg
from openplaces.core.constants import NEVER_DELETE, STANDARD_DIRS
from openplaces.core.schema import AdminId, sanitize
from openplaces.io.cleanup.consumption import (
    _all_recipe_ids,
    _consumers_complete,
    _dependency_index,
    _DependencyIndex,
    _path_conceptually_exists,
)
from openplaces.io.cleanup.lock import DataLock, _cluster_busy, _touch_lock
from openplaces.io.cleanup.receipts import (
    RECEIPT_SUFFIX,
    _cleanup_config,
    _relative_posix,
    _resolve_relative,
)
from openplaces.io.cleanup.walk import _REPORT_COLUMNS, _delete_output_with_receipt
from openplaces.recipe import (
    get_recipe_by_id,
)

# Destructive orphan deletion is refused when fewer recipes than this
# loaded successfully (misconfigured recipe path, broken environment).
_MIN_RECIPES_FOR_ORPHAN_GC = 25


def _recipe_id_rest(recipe_id: str) -> list[str]:
    """Recipe ID parts after the leading admin ID, if it has one."""
    parts = recipe_id.split('_')
    try:
        AdminId(parts[0])
    except ValueError:
        return parts
    return parts[1:]


def _layer_output_names(recipe_id: str) -> list[tuple[str, tuple[str, ...]]]:
    """Return (token, filename parts) of each `additional_layers` output.

    A secondary entity writes its own token, which appears in no recipe
    ID at all, so it can only be read off the loaded recipe. Without
    these entries its output matches no recipe and is classed as an
    orphan while the recipe still produces it.
    """
    try:
        recipe = get_recipe_by_id(recipe_id)
    except Exception:
        return []
    names = []
    for layer_spec in recipe.get('additional_layers') or []:
        entity = layer_spec.get('entity')
        if entity is None:
            continue
        filename = (layer_spec.get('save_to') or {}).get('filename')
        names.append((str(entity), tuple(str(filename).split('_')) if filename else ()))
    return names


@cache
def _recipe_token_index() -> dict[str, list[tuple[str, tuple[str, ...]]]]:
    """Map each output token to its (recipe ID, filename parts) candidates.

    A recipe ID reads {admin}_{token}[_{filename}]; output files read
    {file_admin}_{token}[_{filename}][_{suffix}].parquet, where file_admin
    may be deeper than the recipe's admin scope. Entities declared in
    `additional_layers` are indexed under their own token as well.
    """
    index: dict[str, list[tuple[str, tuple[str, ...]]]] = {}
    for recipe_id in _all_recipe_ids():
        rest = _recipe_id_rest(recipe_id)
        if not rest:
            continue
        index.setdefault(rest[0], []).append((recipe_id, tuple(rest[1:])))
        for token, filename_parts in _layer_output_names(recipe_id):
            index.setdefault(token, []).append((recipe_id, filename_parts))
    return index


@cache
def _enrich_suffix_index() -> dict[str, list[str]]:
    """Map an enrich recipe's dataset suffix to its recipe IDs.

    An enrich output is named after the entity recipe it enriches plus
    its own dataset ({admin}_{entity token}_{dataset}), so the token in
    its filename is the spine's, never the enrich recipe's own. Matching
    the trailing dataset instead keeps the evidence file from being
    judged by the spine's retention and consumer set.
    """
    index: dict[str, list[str]] = {}
    for recipe_id in _all_recipe_ids():
        try:
            recipe = get_recipe_by_id(recipe_id)
        except Exception:
            continue
        if recipe.get('stage') != 'enrich':
            continue
        dataset = recipe.get('dataset')
        if dataset is None:
            continue
        index.setdefault(sanitize(str(dataset)), []).append(recipe_id)
    return index


def _recipe_admin_covers(recipe_id: str, admin) -> bool:
    """Whether a recipe's own admin scope covers a file's admin unit."""
    try:
        recipe_admin = AdminId(recipe_id.split('_')[0])
    except ValueError:
        return True
    return admin is not None and recipe_admin.is_parent_or_equal_of(admin)


def _match_recipe_for_file(stem: str) -> tuple[str | None, str | None]:
    """Match an output filename stem to (recipe_id, admin_id).

    The most specific candidate wins: an enrich evidence file is matched
    on its trailing dataset before the spine token it is named after, and
    among token candidates the one whose declared filename parts match
    the most of the stem. Taking the first candidate instead attributed
    every sibling `_suffix` recipe's output, and every enrich evidence
    file, to the primary recipe, which judged it by the wrong retention
    and the wrong consumer set.

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
    # A '_geo' sidecar belongs to the output it sits beside
    if len(rest) > 1 and rest[-1] == 'geo':
        rest = rest[:-1]
    if not rest:
        return None, None
    admin_str = str(admin) if admin else None

    suffix_index = _enrich_suffix_index()
    for start in range(1, len(rest)):
        for recipe_id in suffix_index.get('_'.join(rest[start:]), []):
            if _recipe_admin_covers(recipe_id, admin):
                return recipe_id, admin_str

    best_id, best_len = None, -1
    for recipe_id, filename_parts in _recipe_token_index().get(rest[0], []):
        if not _recipe_admin_covers(recipe_id, admin):
            continue
        # The recipe's filename parts must prefix the file's own
        if tuple(rest[1 : 1 + len(filename_parts)]) != filename_parts:
            continue
        if len(filename_parts) > best_len:
            best_id, best_len = recipe_id, len(filename_parts)
    return best_id, admin_str


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
        for recipe_id, _ in index.get(token, []):
            dir_admin_id = AdminId(dir_admin) if dir_admin else None
            if not _recipe_admin_covers(recipe_id, dir_admin_id):
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
                _touch_lock(lock)
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
                memo,
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
    if (
        bucket in NEVER_DELETE
        or bucket in ('logs', 'models', 'reports')
        or STANDARD_DIRS.get(bucket, {}).get('custom')
        or bucket not in STANDARD_DIRS
    ):
        # Protected or user-owned buckets are report-only. A directory
        # the user registered themselves (or one this build does not
        # know at all) holds files no recipe claims, and everything a
        # recipe does not claim is an orphan below, so without this the
        # aggressive path deleted the user's own data.
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
        # downloaded archive in 'external' is an input copy protected
        # by the bucket default, not by the recipe's output retention
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
    memo: dict | None = None,
) -> str:
    cls = row['class']
    path = _resolve_relative(row['path'])
    if cls == 'heap' and 'heap' in delete:
        return _compact_delete(path, dry_run)
    if cls == 'intermediate/consumed' and 'consumed' in delete:
        recipe = index.recipes.get(row['recipe_id']) or {}
        entity = recipe.get('entity')
        if entity is not None and str(entity.entity_type) == 'image':
            # Image caches are deleted per cache with one receipt,
            # not per tile file; compact only reports them
            row['blocked_by'] = 'image cache: use cleanup(include_images=True)'
            return 'blocked'
        if dry_run:
            return 'would_delete'
        _, blocked_by, verified = _consumers_complete(
            row['recipe_id'], row['admin_id'], index, memo=memo
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
