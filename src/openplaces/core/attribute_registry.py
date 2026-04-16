"""
Global registry of well-known attributes: their data types, units, and
default aggregation functions.  Used by the harmonizer to:
  - drive groupby aggregations without hardcoded column lists
  - convert derived columns to the correct pandas dtype (e.g. Categorical)
  - warn when ingested data does not match the expected type

The registry is loaded from ``attribute_registry.csv`` (same directory) and
cached on first access.
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
