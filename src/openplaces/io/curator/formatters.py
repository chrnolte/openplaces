"""Curation steps for structural, non-semantic output formatting.

Formatters change dtypes or column layout only; they do not fill, infer, or
reconcile any values.
"""

from __future__ import annotations

import re
from functools import cache

import pandas as pd

from openplaces.io.curator import CurateState, _register
from openplaces.io.curator.provenance import SOURCE_SUFFIX


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


@cache
def _provenance_suffixes() -> tuple[tuple[str, str], ...]:
    """Provenance suffix -> source key, auto-generated from existing recipes.

    For every ``(entity_type, source)`` pair known to the recipes, generate the
    column suffixes the harmonizer can produce: ``_{entity}_{source}`` and the
    bare ``_{source}`` fallback (e.g. ``_building_nsi`` and ``_nsi``;
    ``_footprint_fema`` and ``_fema``). Parcels are interchangeable, so they map
    by the entity-only ``_parcel``. Returned longest-first so a specific suffix
    wins over its bare fallback. No hardcoded list — adding a source recipe
    extends this automatically.
    """
    from openplaces.recipe import iter_entity_sources

    suffixes: dict[str, str] = {}
    for entity, source in iter_entity_sources():
        if source is None:
            continue
        if entity == 'parcel':
            suffixes.setdefault('_parcel', 'parcel')
        else:
            suffixes.setdefault(f'_{entity}_{source}', source)
            suffixes.setdefault(f'_{source}', source)
    return tuple(sorted(suffixes.items(), key=lambda kv: len(kv[0]), reverse=True))


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
    'manufactured_home_community',
)
_MODIFIERS = ('_all', '_per_area', '_inferred')
_RELATIONAL_COUNT = re.compile(r'^n_.+_per_.+$')
_BIG = 10_000


def _split_source(name: str) -> tuple[str, str | None]:
    """Split a trailing provenance suffix off *name*; return (base, source)."""
    for suffix, source in _provenance_suffixes():
        if name.endswith(suffix):
            return name[: -len(suffix)], source
    return name, None


def _attr_rank(base: str) -> int:
    from openplaces.core.attribute_registry import get_attribute_order

    rank = get_attribute_order(base)
    return _BIG if rank is None else rank


def _key_fields(col: str) -> tuple:
    """Return the (block, a, b, c, d) ranking fields for a non-sidecar column.

    Blocks: 0 = canonical/final (incl. m2), with ``priority_on_parcel`` last;
    1 = source variables (relational counts first, then suffixed evidence);
    2 = flags then visualization-only (``*_per_area``).
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


def _sort_key(col: str, original_index: int) -> tuple:
    """Deterministic sort key.

    Every ``{base}_source`` sidecar is grouped into one provenance band that
    sorts after all canonical columns (block 0) and before the source-variable
    block (block 1), ordered among themselves by their base column's rank.
    """
    if col.endswith(SOURCE_SUFFIX):
        # Band 0.5 groups every sidecar between canonical (0) and source (1),
        # ordered among themselves by their base column's rank.
        return (0.5, _key_fields(col[: -len(SOURCE_SUFFIX)]), original_index)
    fields = _key_fields(col)
    return (float(fields[0]), fields, original_index)


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
    other entities (relational counts first, then the suffixed evidence); (2)
    flag and visualization-only columns (``occupancy_type_conflict``/``_review``,
    ``*_per_area``). ``geometry`` is always kept last. Computed from each
    column's name and the registry, so the recipe needs no explicit column
    list.

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
    rest_sorted = sorted(rest, key=lambda c: _sort_key(c, index_of[c]))

    state.curated = curated[lead + rest_sorted + geom]
    return state
