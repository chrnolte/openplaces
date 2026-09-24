"""
Whether an output is complete and whether it has been consumed.
Completeness is a readable parquet footer whose dtypes match the
registry, a geometry sidecar that agrees with it and every
required sub-admin partition present; the receipt-justified skip
lets a deliberate deletion stand. Consumption is the dependency
index over every recipe (literal and auto-discovered consumers of
each output) and the test that all of an output's consumers are
complete, which is what permits its deletion. One module because
the two halves call each other (a consumer is satisfied by a
complete output; a skip is justified by complete consumers).
"""

from functools import cache
from pathlib import Path

from openplaces.config import cfg
from openplaces.core.attribute_registry import get_data_type
from openplaces.core.schema import AdminId
from openplaces.io.aggregate import COVERAGE_ALL, read_partition_coverage
from openplaces.io.cleanup.receipts import (
    _cleanup_config,
    _relative_posix,
    _resolve_relative,
    is_orchestrated,
    read_receipt,
)
from openplaces.recipe import (
    get_output_path,
    get_recipe_by_id,
    get_recipe_dependencies,
    get_recipe_id,
    get_save_admin_level,
    resolve_attribute_name,
    saves_geometry,
)


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
        expected = get_data_type(resolve_attribute_name(field.name))
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


def _geometry_sidecar_ok(path: Path) -> bool:
    """True when an output's '_geo' sidecar is present and readable.

    save_parquet writes the attribute table first and the geometry
    sidecar second, so a job killed between the two leaves an attribute
    file that passes every check on its own while its geometry is gone.
    A '_join_id' column exists only to join to that sidecar, which makes
    it the one unambiguous on-disk signal that the sidecar is owed. An
    output keyed on 'geo_id' carries no such signal, so a missing
    sidecar there still goes unnoticed; a sidecar that is present is
    validated either way.
    """
    geo_path = path.with_name(path.stem + '_geo' + path.suffix)
    if geo_path.exists():
        return _parquet_schema_ok(geo_path)
    try:
        import pyarrow.parquet as pq

        return '_join_id' not in pq.read_schema(path).names
    except Exception:
        return False


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
    - split geometry layout: the '_geo' sidecar, which save_parquet writes
      second, is present and readable (see `_geometry_sidecar_ok`)
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
    if saves_geometry(recipe) and not _geometry_sidecar_ok(out_path):
        return False
    coverage = read_partition_coverage(out_path)
    if not coverage or COVERAGE_ALL in coverage:
        # No coverage footer (a plain or legacy file), or full
        # coverage
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


def _consumer_satisfies(recipe, admin_id, required_partitions=None) -> bool:
    """True when a consumer counts as complete for reclaiming an input.

    Physical completeness first; a consumer that was itself cleaned up
    still counts through its own tombstone receipt (the receipt cascade),
    but only when that receipt records coverage of the partitions the
    input is required to be in. Falling back to bare existence instead
    dropped the requirement altogether, so a county input a state
    aggregate had not consumed yet was deleted with a receipt that then
    made ingest skip regenerating it.

    Parameters
    ----------
    recipe : str or dict
        Consumer recipe ID or loaded recipe dictionary.
    admin_id : str or AdminId or None
        Admin unit of the consumer's output.
    required_partitions : iterable of str, optional
        Partition or sub-admin IDs the consumer must have consumed.
    """
    if is_output_complete(recipe, admin_id, required_partitions=required_partitions):
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
    receipt = read_receipt(out_path)
    if receipt is None:
        return False
    if required_partitions is None:
        return True
    recorded = set(map(str, receipt.get('partitions') or []))
    if COVERAGE_ALL in recorded:
        return True
    return set(map(str, required_partitions)) <= recorded


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
            consumer = self.recipes[consumer_id]
            # A global node has no admin unit of its own, and '' is not
            # one either: it raised in AdminId, which marked every
            # auto-discovering consumer unresolved, so a level-0 output
            # was neither deletable nor receipt-skippable and a deleted
            # global tile was re-ingested every run. Resolve such a node
            # against the consumer's own scope instead.
            resolve_admin = admin_str or consumer.get('admin_id')
            try:
                edges = get_recipe_dependencies(consumer, admin_id=resolve_admin)
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
    """AdminId.truncate_to_level, tolerating None and a plain string."""
    if admin_id is None:
        return None
    if not isinstance(admin_id, AdminId):
        admin_id = AdminId(admin_id)
    return admin_id.truncate_to_level(level)


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
        complete = _consumer_satisfies(
            consumer_recipe, consumer_admin, required_partitions=required
        )
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
