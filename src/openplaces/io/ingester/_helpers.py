"""Module-level helpers of the ingester: matching an extracted file,
partition keys, registry type warnings and the scraper module cache."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from openplaces.core.attribute_registry import load_registry as _load_attr_registry


def _match_extracted_file(heap_dir: Path, expected_path: Path) -> Path | None:
    """Locate an unzipped file when its inner name varies slightly from expected.

    Some archives ship an inner filename that differs from the declared
    ``uncompressed_file_name`` by a minor variation (e.g. a doubled extension
    ``sales_2019_YTD.xlsx.xlsx``). This searches *heap_dir* recursively for the
    exact expected name first, then, failing that, for files sharing the expected
    stem; the stem fallback resolves only when it is unambiguous (exactly one
    match), so genuine multi-file archives are left strict.

    Parameters
    ----------
    heap_dir : Path
        Directory the archive was extracted into.
    expected_path : Path
        The declared (concrete, non-wildcard) path whose name was not found.

    Returns
    -------
    Path or None
        The resolved file, or ``None`` if no unambiguous match exists.
    """
    exact = next(heap_dir.rglob(expected_path.name), None)
    if exact is not None:
        return exact
    candidates = sorted(
        p for p in heap_dir.rglob(expected_path.stem + '*') if p.is_file()
    )
    return candidates[0] if len(candidates) == 1 else None


def _transform_partition_key(value, spec: dict) -> str:
    """Apply a scalar string transformation to a resolved partition key.

    A deliberately small vocabulary mirroring the string operations in
    io.transform, applied to one value rather than a column. Declared as
    ``download_by: partition_key_transformation: {placeholder: spec}``.

    Parameters
    ----------
    value : str
        The resolved partition key.
    spec : dict
        ``operation`` plus ``args``: ``substring`` ([start, stop]),
        ``zfill`` ([width]), ``add_prefix``/``add_suffix`` ([text]).
    """
    value = str(value)
    operation = spec.get('operation')
    args = spec.get('args') or []
    # Each operation's arity, checked before indexing: a short `args` list
    # otherwise raised a bare IndexError naming neither the recipe key nor
    # the operation that wanted more arguments.
    arity = {'substring': 2, 'zfill': 1, 'add_prefix': 1, 'add_suffix': 1}
    if operation in arity and len(args) < arity[operation]:
        raise ValueError(
            f"partition_key_transformation operation '{operation}' needs "
            f'{arity[operation]} argument(s), got {args!r}.'
        )
    if operation == 'substring':
        return value[args[0] : args[1]]
    if operation == 'zfill':
        return value.zfill(args[0])
    if operation == 'add_prefix':
        return f'{args[0]}{value}'
    if operation == 'add_suffix':
        return f'{value}{args[0]}'
    raise ValueError(
        f'Unknown partition_key_transformation operation {operation!r}; '
        "expected one of 'substring', 'zfill', 'add_prefix', 'add_suffix'."
    )


def _warn_registry_type_mismatches(gdf) -> None:
    """Warn when a column's dtype disagrees with the attribute registry."""
    reg = _load_attr_registry()
    for col in gdf.columns:
        if col not in reg.index:
            continue
        expected = reg.at[col, 'data_type']
        actual = gdf[col].dtype
        if expected == 'categorical' and str(actual) not in ('category', 'object'):
            warnings.warn(
                f"Column '{col}' expected dtype categorical but got {actual}.",
                stacklevel=2,
            )
        elif expected in ('float', 'int') and not pd.api.types.is_numeric_dtype(actual):
            warnings.warn(
                f"Column '{col}' expected numeric dtype but got {actual}.",
                stacklevel=2,
            )


# Scraper modules are loaded dynamically by file path (see
# `Ingester._load_scraper_fetch`) rather than via a normal `import`, since
# their filenames may contain hyphens (e.g. geography/recipe-specific
# scrapers). A normal `import` is cached by `sys.modules` for free; this
# cache restores the same "load and execute once per process" behavior for
# the dynamic path, so a scraper's own module-level state (e.g. caches it
# keeps for itself) survives across repeated calls within a run.
_SCRAPER_MODULE_CACHE: dict[str, object] = {}
