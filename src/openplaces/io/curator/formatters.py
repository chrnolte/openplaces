"""Curation steps for structural, non-semantic output formatting.

Formatters change dtypes or column layout only; they do not fill, infer, or
reconcile any values.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from openplaces.io.curator import CurateState, _register
from openplaces.io.curator.provenance import SOURCE_SUFFIX
from openplaces.recipe import provenance_suffixes as _provenance_suffixes
from openplaces.recipe import resolve_attribute_name
from openplaces.recipe import split_provenance_suffix as _split_source


@_register('declare_columns')
def declare_columns(state: CurateState, columns: list[str]) -> CurateState:
    """Ensure the listed columns exist, writing missing ones as all-null.

    A recipe that runs across regions with uneven reference coverage
    (e.g. a footprint source or point inventory that was never built for
    one state) inherits a spine missing that reference's evidence
    columns. Downstream steps and curated-reference readers treat a
    missing declared column as a recipe error, deliberately; this step
    is the per-recipe declaration that a listed column may legitimately
    be absent, mirroring the enricher's write-declared-columns-as-null
    convention for uncovered admin units. Existing columns are never
    touched.

    Parameters
    ----------
    columns : list of str
        Column names to declare. Each missing name is added as an
        all-null column; present names are left unchanged.
    """
    curated = state.curated
    added = [col for col in columns if col not in curated.columns]
    for col in added:
        # np.nan, not pd.NA: a float-NaN column survives arithmetic and
        # comparisons in later steps, where NAType raises on bool().
        curated[col] = np.nan
    if added and state.verbose:
        print(
            f'  declare_columns: {len(added)} absent column(s) '
            f'declared null: {", ".join(added)}'
        )
    return state


@_register('cast_categoricals')
def cast_categoricals(state: CurateState) -> CurateState:
    """Cast registry-defined categoricals and provenance sidecars to Categorical."""
    from openplaces.core.attribute_registry import get_categorical_attrs

    curated = state.curated
    cat_attrs = get_categorical_attrs()
    cat_sorted = sorted(cat_attrs, key=len, reverse=True)
    for col in curated.columns:
        if curated[col].dtype == 'category':
            continue
        base = next((a for a in cat_sorted if col.startswith(a)), None)
        if base is not None or col.endswith(SOURCE_SUFFIX):
            curated[col] = pd.Categorical(curated[col])
    state.curated = curated
    return state


@_register('cast_integers')
def cast_integers(state: CurateState, columns: list[str]) -> CurateState:
    """Cast numeric columns to a nullable integer dtype.

    Unlike ``fill_missing_numeric``, this does not fill missing values with a
    placeholder — a row with no evidence stays missing (``pd.NA``), not a
    misleading 0 (e.g. year 0). Values are rounded before casting, since a
    reconciled column can legitimately be non-integer (e.g. a fallback median
    across several buildings) even though the canonical attribute is a whole
    number.

    Parameters
    ----------
    columns : list of str
        Numeric columns to cast to pandas' nullable ``Int64``. Missing
        columns are skipped.
    """
    curated = state.curated
    for col in columns:
        if col in curated.columns:
            curated[col] = (
                pd.to_numeric(curated[col], errors='coerce').round().astype('Int64')
            )
    state.curated = curated
    return state


# Explicit display precedence for the source-variable block; auto-derived sources
# not listed here sort after the known ones (tolerant lookup via _source_rank).
_SOURCE_RANK = {'parcel': 0, 'overture': 1, 'nsi': 2, 'fema': 3, None: 9}


def _source_rank(source: str | None) -> int:
    return _SOURCE_RANK.get(source, 8)


# Canonical occupancy / enrichment outputs (block 0). Listed explicitly because
# they carry no provenance suffix to key on.
_FINAL_COLUMNS = (
    'occupancy_type',
    'roof_shape',
    'n_stories',
)
# Flag columns -> the flags/viz block (block 2), in this order.
_FLAG_COLUMNS = (
    'occupancy_type_conflict',
    'occupancy_type_review',
    'land_use_class_conflict',
    'land_use_review',
    'manufactured_home_community',
)
_MODIFIERS = ('_all', '_per_area', '_inferred')
_RELATIONAL_COUNT = re.compile(r'^n_.+_per_.+$')
_BARE_ID = re.compile(r'^(.+?)_id(_.+)?$')
_BIG = 10_000


def _attr_rank(name: str) -> int:
    from openplaces.core.attribute_registry import get_attribute_order

    rank = get_attribute_order(resolve_attribute_name(name))
    return _BIG if rank is None else rank


def _key_fields(col: str) -> tuple:
    """Return the (block, a, b, c, d) ranking fields for a non-sidecar column.

    Blocks: 0 = canonical/final (incl. m2), with ``priority_on_parcel`` last;
    1 = source variables (relational counts first, then suffixed evidence);
    2 = flags then visualization-only (``*_per_area``).

    A bare id column whose entity prefix is itself an entity-only provenance
    suffix (``parcel_id``, ``parcel_id_local`` — parcels map by ``_parcel``)
    leads that source's evidence group in block 1: the id naming which
    reference the group's columns describe belongs at its head, not stranded
    in block 0 without a registry rank.
    """
    if col == 'priority_on_parcel':
        return (0, _BIG + 1, 0, 0, 0)  # end of the canonical block
    if _RELATIONAL_COUNT.match(col):
        return (1, -1, 0, 0, 0)  # lead the source block
    if col in _FLAG_COLUMNS:
        return (2, _FLAG_COLUMNS.index(col), 0, 0, 0)

    modifier = ''
    base = col
    for mod in _MODIFIERS:
        if base.endswith(mod):
            modifier = mod
            base = base[: -len(mod)]
            break

    if modifier == '_per_area':
        attr, _source = _split_source(base)
        return (2, _BIG, _attr_rank(attr), 0, 0)  # viz, after flags
    if modifier == '_inferred':
        return (2, _BIG, _attr_rank(base.split('_')[0]), 1, 0)
    if col == 'm2':
        return (0, _attr_rank('m2'), 0, 0, 0)
    if col in _FINAL_COLUMNS:
        return (0, _attr_rank(col), 0, 0, 0)

    id_match = _BARE_ID.match(base)
    if id_match:
        source = dict(_provenance_suffixes()).get(f'_{id_match.group(1)}')
        if source is not None:
            # kind -1: lead the source's evidence group (existing kinds:
            # 0 counts, 1 suffixed ids, 2 other attributes)
            return (1, _source_rank(source), -1, _attr_rank(base), 0)

    attr, source = _split_source(base)
    if source is not None:
        kind = 0 if attr.startswith('n_') else (1 if '_id' in attr else 2)
        return (
            1,
            _source_rank(source),
            kind,
            _attr_rank(attr),
            1 if modifier == '_all' else 0,
        )

    return (0, _attr_rank(base), 0, 0, 0)


def _sort_key(col: str, index_of: dict[str, int]) -> tuple:
    """Deterministic sort key.

    Every ``{base}_source`` sidecar is grouped into one provenance band that
    sorts after all canonical columns (block 0) and before the source-variable
    block (block 1), ordered among themselves by their base column's rank.
    An ``{base}_all`` variant whose base column is present sorts immediately
    after that base (its key is the base's key plus a trailing marker), so
    e.g. ``parcel_id_all`` follows ``parcel_id`` even though the bare id
    columns tie on registry rank. A ``{base}_original`` variant (a raw,
    pre-reconciliation value preserved under its own name -- not evidence
    attributed from another entity, so it carries no provenance suffix to key
    on either) follows its base the same way, e.g. ``address_original`` after
    ``address``. Likewise a ``{base}_conflict`` column follows its
    ``{base}_source`` sidecar (``address_conflict`` after ``address_source``),
    unless it has an explicit ``_FLAG_COLUMNS`` slot.
    """
    if col.endswith(SOURCE_SUFFIX):
        # Band 0.5 groups every sidecar between canonical (0) and source (1),
        # ordered among themselves by their base column's rank.
        return (0.5, _key_fields(col[: -len(SOURCE_SUFFIX)]), index_of[col])
    if col.endswith('_all'):
        base = col[: -len('_all')]
        if base in index_of:
            return (*_sort_key(base, index_of), 1)
    if col.endswith('_original'):
        base = col[: -len('_original')]
        if base in index_of:
            return (*_sort_key(base, index_of), 1)
    if col.endswith('_conflict') and col not in _FLAG_COLUMNS:
        sidecar = col[: -len('_conflict')] + SOURCE_SUFFIX
        if sidecar in index_of:
            return (*_sort_key(sidecar, index_of), 1)
    fields = _key_fields(col)
    return (float(fields[0]), fields, index_of[col])


@_register('order_columns')
def order_columns(
    state: CurateState,
    overrides: list[str] | None = None,
    drop: list[str] | None = None,
) -> CurateState:
    """Order columns: canonical -> sources -> linked evidence -> flags/viz.

    Four bands: (0) canonical/final variables (incl. ``m2``), ordered by the
    attribute-registry ``sort`` rank, with ``priority_on_parcel`` ending the
    band; (0.5) every ``{col}_source`` provenance sidecar, grouped together and
    ordered by their base column's rank; (1) source variables inherited from
    other entities (relational counts first, then each source's evidence group
    led by its bare id columns — ``parcel_id``/``parcel_id_all``/
    ``parcel_id_local`` head the ``_parcel`` group); (2) flag and
    visualization-only columns (``occupancy_type_conflict``/``_review``,
    ``*_per_area``). An ``{col}_all`` or ``{col}_original`` variant always
    directly follows its base column. ``geometry`` is always kept last.
    Computed from each column's name and the registry, so the recipe needs no
    explicit column list.

    Parameters
    ----------
    overrides : list of str, optional
        Columns to force to the front, in the given order (escape hatch for any
        bespoke placement the rule does not capture). Defaults to none.
    drop : list of str, optional
        Transient columns to remove from the output (e.g. an intermediate used
        only during inference). Defaults to none.
    """
    curated = state.curated
    drop_set = set(drop or [])
    if drop_set:
        curated = curated.drop(columns=[c for c in drop_set if c in curated.columns])
    overrides = overrides or []
    cols = list(curated.columns)
    geom = [c for c in cols if c == 'geometry']
    others = [c for c in cols if c != 'geometry']

    lead = [c for c in overrides if c in others]
    rest = [c for c in others if c not in lead]
    index_of = {c: i for i, c in enumerate(others)}
    rest_sorted = sorted(rest, key=lambda c: _sort_key(c, index_of))

    state.curated = curated[lead + rest_sorted + geom]
    return state
