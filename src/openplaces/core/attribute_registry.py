"""
Global registry of well-known attributes: their data types, units, default
aggregation functions, output ``sort`` rank, owning ``entity_type``, and the
pipeline ``stage`` that first produces them.  It is the authoritative, codebase
list of canonical attribute names, spanning every entity type (parcel, building,
footprint, transaction, admin, ...) and every pipeline stage (ingest, harmonize,
enrich, curate).  Used to:
  - drive groupby aggregations without hardcoded column lists
  - convert derived columns to the correct pandas dtype (e.g. Categorical)
  - warn when ingested data does not match the expected type
  - order curated output columns deterministically (``sort`` rank)
  - list the canonical attributes for an entity type (:func:`get_attributes`)

``entity_type`` is a scoping tag: blank means the name is shared across entity
types; a value marks a name that is uniquely specific to one type.  Names are
unique, so if a name ever needed a genuinely different definition per entity
type the loader would switch to a composite ``(name, entity_type)`` lookup; that
is not needed today.

The registry is loaded from ``attribute_registry.csv`` (same directory) and
cached on first access.

Lookups are keyed on base attribute names only. A caller holding a possibly
provenance-suffixed column (e.g. ``improvement_value_parcel``) should resolve
it via :func:`openplaces.recipe.resolve_attribute_name` first. That helper
lives in the higher recipe layer — not here — because the provenance suffix
vocabulary is derived from the recipes directory, which layer 0 cannot import.

One suffix *is* handled here rather than in the recipe layer:
``{base}_source`` provenance sidecars (see :data:`PROVENANCE_SOURCE_SUFFIX`).
Unlike the recipe-derived suffixes above, this one is a fixed literal, not
tied to any particular ingest source, so :func:`get_agg_func` and
:func:`get_data_type` fall back to it generically for any ``{base}_source``
column whose base is itself registered -- no per-column CSV row needed.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import pandas as pd

_REGISTRY_PATH = Path(__file__).parent / 'attribute_registry.csv'

PROVENANCE_SOURCE_SUFFIX = '_source'


@cache
def load_registry() -> pd.DataFrame:
    """Return the attribute registry as a DataFrame indexed by attribute name."""
    return pd.read_csv(_REGISTRY_PATH).set_index('name')


def get_agg_func(attr: str) -> str | None:
    """Return the default aggregation function name for *attr*, or ``None``.

    Falls back to ``'first'`` for an unregistered ``{base}_source``
    provenance sidecar when *base* is itself registered.
    """
    reg = load_registry()
    if attr in reg.index:
        return reg.at[attr, 'aggregation']
    if attr.endswith(PROVENANCE_SOURCE_SUFFIX):
        base = attr[: -len(PROVENANCE_SOURCE_SUFFIX)]
        if base in reg.index:
            return 'first'
    return None


def get_data_type(attr: str) -> str | None:
    """Return the expected data type string for *attr*, or ``None``.

    Falls back to ``'categorical'`` for an unregistered ``{base}_source``
    provenance sidecar when *base* is itself registered.
    """
    reg = load_registry()
    if attr in reg.index:
        return reg.at[attr, 'data_type']
    if attr.endswith(PROVENANCE_SOURCE_SUFFIX):
        base = attr[: -len(PROVENANCE_SOURCE_SUFFIX)]
        if base in reg.index:
            return 'categorical'
    return None


def get_null_placeholder(attr: str) -> object | None:
    """Return the raw sentinel value that means "missing" for *attr*.

    Some sources use a placeholder (e.g. ``0`` for "year never recorded")
    instead of a true null. Declaring it here lets ingestion null it out
    generically for every recipe that maps a column to this attribute --
    current or future -- rather than requiring each recipe to declare its
    own transformation. Returns ``None`` when *attr* has no declared
    placeholder.
    """
    reg = load_registry()
    if attr not in reg.index or 'null_placeholder' not in reg.columns:
        return None
    value = reg.at[attr, 'null_placeholder']
    return None if pd.isna(value) else value


def get_categorical_attrs() -> frozenset[str]:
    """Return the set of attribute names whose ``data_type`` is ``'categorical'``."""
    reg = load_registry()
    return frozenset(reg.index[reg['data_type'] == 'categorical'])


def get_attributes(
    entity_type: str | None = None, stage: str | None = None
) -> pd.DataFrame:
    """Return registry rows for an entity type and/or pipeline stage.

    The canonical attributes that apply to a given entity type are the rows
    tagged with that ``entity_type`` plus the shared rows (blank ``entity_type``),
    which apply to every entity type. ``stage`` filters the same way, but a blank
    ``stage`` means ingest/general and is only returned when no stage is
    requested (it is not treated as matching every stage).

    Parameters
    ----------
    entity_type : str, optional
        Entity type to select (e.g. ``'parcel'``). When ``None``, no entity
        filtering is applied. When given, shared (untagged) attributes are
        always included.
    stage : str, optional
        Pipeline stage to select (``'harmonize'``, ``'enrich'``, ``'curate'``).
        When ``None``, no stage filtering is applied.

    Returns
    -------
    pandas.DataFrame
        The matching subset of the registry, indexed by attribute name.
    """
    reg = load_registry()
    result = reg
    if entity_type is not None and 'entity_type' in reg.columns:
        et = reg['entity_type']
        result = result[et.isna() | (et == entity_type)]
    if stage is not None and 'stage' in reg.columns:
        result = result[result['stage'] == stage]
    return result


def get_attribute_order(attr: str) -> int | None:
    """Return the registry ``sort`` rank for *attr*, or ``None`` if unset.

    The rank gives the intra-phase ordering of attributes used by the curation
    ``order_columns`` step. Lower ranks come first.
    """
    reg = load_registry()
    if attr not in reg.index or 'sort' not in reg.columns:
        return None
    value = reg.at[attr, 'sort']
    return None if pd.isna(value) else int(value)
