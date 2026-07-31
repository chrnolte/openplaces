"""Registry mapping canonical dataset identity to a QGIS template layer.

Mirrors the caching/lookup pattern of `openplaces.core.attribute_registry`:
canonical (entity_type, source) identity maps to which prototype layer in
the standardized QGIS template to clone, and where to place it in the layer
tree. Extending the registry for a new source is a CSV edit, not a code
change.

The registry is loaded from ``qgis_map_style_registry.csv`` (same
directory) and cached on first access.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pandas as pd

_REGISTRY_PATH = Path(__file__).parent / 'qgis_map_style_registry.csv'

#: Reserved style_key for the generic unstyled fallback layer.
FALLBACK_STYLE_KEY = '_fallback'

_CARRY_THROUGH_ROLES = ('basemap', 'static')


@dataclass(frozen=True)
class LayerStyle:
    """One row of the QGIS map style registry.

    A base row (`variant_of` blank) is a full geo+attr prototype pair. A
    variant row (`variant_of` set to a base row's `style_key`) is a
    geometry-only prototype styled on a different column, already joined in
    the template to the *base* row's `_attr` layer — see
    :func:`get_style_variants`.

    `dynamic_categorize_attr`, when set, names the resolved-data column whose
    actual unique values (for this run) should replace the template's baked
    categorized-symbol category list — for columns whose value set is
    data-dependent rather than a fixed enum (e.g. an evidence-conflict
    summary column, or a per-source crosswalked classification).
    """

    style_key: str
    entity_type: str
    source: str | None
    role: str
    template_layer_name: str
    group_path: str
    default_visible: bool
    priority: int
    variant_of: str | None
    variant_label: str | None
    notes: str | None
    dynamic_categorize_attr: str | None


@cache
def load_registry() -> pd.DataFrame:
    """Return the QGIS map style registry as a DataFrame indexed by style_key."""
    df = pd.read_csv(_REGISTRY_PATH, dtype=str, keep_default_na=False)
    df['default_visible'] = df['default_visible'].str.lower().isin({'true', '1', 'yes'})
    df['priority'] = df['priority'].replace('', '0').astype(int)
    return df.set_index('style_key')


def _row_to_style(style_key: str, row: pd.Series) -> LayerStyle:
    return LayerStyle(
        style_key=style_key,
        entity_type=row['entity_type'],
        source=row['source'] or None,
        role=row['role'],
        template_layer_name=row['template_layer_name'],
        group_path=row['group_path'],
        default_visible=bool(row['default_visible']),
        priority=int(row['priority']),
        variant_of=row['variant_of'] or None,
        variant_label=row['variant_label'] or None,
        notes=row['notes'] or None,
        dynamic_categorize_attr=row.get('dynamic_categorize_attr') or None,
    )


def get_style(
    entity_type: str,
    source: str | None,
    role: str,
    *,
    registry: pd.DataFrame | None = None,
) -> LayerStyle | None:
    """Return the registered style for (*entity_type*, *source*, *role*), or None.

    Tries an exact ``(entity_type, source)`` match among rows with the given
    *role* first, then falls back to a wildcard row with a blank `source`
    for the same *entity_type* and *role*. *role* must be one of 'output',
    'input', or 'admin' — matching never considers 'basemap'/'static' rows,
    which are looked up separately via :func:`get_static_styles`. Variant
    rows (`variant_of` set) are never returned here — use
    :func:`get_style_variants` to fetch a base style's column variants.

    Parameters
    ----------
    entity_type : str
    source : str or None
    role : str
        One of 'output', 'input', 'admin' — the calling `LayerSpec`'s own
        role. Required so that, e.g., a footprint 'input' spec cannot pick
        up a footprint 'output' row's style just because both have a blank
        `source`.
    registry : pandas.DataFrame, optional
        Pre-loaded registry to look up in, e.g. for tests. Defaults to the
        cached production registry (:func:`load_registry`).
    """
    reg = registry if registry is not None else load_registry()
    candidates = reg[
        (reg['entity_type'] == entity_type)
        & (reg['role'] == role)
        & (reg['variant_of'] == '')
    ]
    source = source or ''
    exact = candidates[candidates['source'] == source]
    if len(exact):
        style_key = exact.index[0]
        return _row_to_style(style_key, exact.iloc[0])
    wildcard = candidates[candidates['source'] == '']
    if len(wildcard):
        style_key = wildcard.index[0]
        return _row_to_style(style_key, wildcard.iloc[0])
    return None


def get_style_variants(
    style_key: str, *, registry: pd.DataFrame | None = None
) -> list[LayerStyle]:
    """Return the column-styled variants registered under *style_key*.

    Each variant is a geometry-only template layer, styled on a different
    canonical attribute, already joined in the template to *style_key*'s own
    `_attr` layer. Returns an empty list for style keys with no variants.
    """
    reg = registry if registry is not None else load_registry()
    variants = reg[reg['variant_of'] == style_key]
    return [_row_to_style(key, row) for key, row in variants.iterrows()]


def get_fallback_style(*, registry: pd.DataFrame | None = None) -> LayerStyle:
    """Return the reserved generic fallback style (`style_key` `'_fallback'`)."""
    reg = registry if registry is not None else load_registry()
    if FALLBACK_STYLE_KEY not in reg.index:
        raise ValueError(
            f'No {FALLBACK_STYLE_KEY!r} row in the QGIS map style registry; '
            'the template must include a generic unstyled prototype layer.'
        )
    return _row_to_style(FALLBACK_STYLE_KEY, reg.loc[FALLBACK_STYLE_KEY])


def get_static_styles(*, registry: pd.DataFrame | None = None) -> list[LayerStyle]:
    """Return 'basemap'/'static' rows: template layers always kept, never cloned.

    Unlike matched output/input/admin rows, these survive pruning
    unconditionally and are never cloned — but their `default_visible` still
    drives whether the generator checks them on by default (see
    `generator.py`'s pruning step). Excludes the reserved `_fallback` row
    even though it is tagged ``role='static'`` — the fallback prototype must
    only survive pruning when a resolved layer actually falls back to it
    (see :func:`get_fallback_style`), not unconditionally like a basemap.
    """
    reg = registry if registry is not None else load_registry()
    static = reg[
        reg['role'].isin(_CARRY_THROUGH_ROLES) & (reg.index != FALLBACK_STYLE_KEY)
    ]
    return [_row_to_style(key, row) for key, row in static.iterrows()]
