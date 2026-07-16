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
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import pandas as pd

_REGISTRY_PATH = Path(__file__).parent / 'attribute_registry.csv'


@cache
def load_registry() -> pd.DataFrame:
    """Return the attribute registry as a DataFrame indexed by attribute name."""
    return pd.read_csv(_REGISTRY_PATH).set_index('name')


def get_agg_func(attr: str) -> str | None:
    """Return the default aggregation function name for *attr*, or ``None``."""
    reg = load_registry()
    return reg.at[attr, 'aggregation'] if attr in reg.index else None


def get_data_type(attr: str) -> str | None:
    """Return the expected data type string for *attr*, or ``None``."""
    reg = load_registry()
    return reg.at[attr, 'data_type'] if attr in reg.index else None


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
