"""Pinned, git-tracked registry of CRS-to-CRS reprojection operations.

By default, `to_crs()` dynamically resolves operations via PROJ's
`Transformer.from_crs()`. For datum changes (e.g., NAD83-based state plane
to WGS84), this can silently select low-accuracy transforms depending on local
grid cache and network status, producing only a `UserWarning`. This caused a
reprojection mismatch between independently ingested Massachusetts parcel datasets in
`reproject()` in `geo/polygon.py`.

To ensure deterministic, offline-capable reprojections, this module pins transforms.
It stores a JSON file per `(source_crs, target_crs)` pair under `crs_transforms/`
and downloads the required grid files to `crs_grids/`. Both are committed to the
repository. Entries are registered either ahead of time via the CLI:

    python -m openplaces.geo.crs_transforms EPSG:26986

or lazily upon the first unregistered `reproject()` call.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from datetime import date
from pathlib import Path

import pyproj
import requests

REGISTRY_DIR = Path(__file__).parent / 'crs_transforms'
GRIDS_DIR = Path(__file__).parent / 'crs_grids'
GRID_URL_TEMPLATE = 'https://cdn.proj.org/{filename}'

# Let PROJ find grid files committed here, without touching the environment's
# own PROJ_DATA directory.
pyproj.datadir.append_data_dir(str(GRIDS_DIR))


def _crs_slug(crs) -> str:
    """Generate a filesystem-safe identifier for a CRS, preferring authority code."""
    crs = pyproj.CRS(crs)
    authority = crs.to_authority()
    if authority is None and crs.name != 'unknown':
        # A source CRS parsed from a real .prj/database definition can differ from
        # the canonical EPSG definition by sub-millimeter rounding in a projection
        # constant (e.g. NC State Plane ftUS false easting), which drops PROJ's
        # match confidence below to_authority()'s default threshold (70) to 0. At
        # PROJ's lowest confidence bucket (25) it usually still resolves -- but only
        # accept it if it's the sole candidate: ad hoc CRSs (e.g. the AEQD
        # projections in local_metric_crs()) report name 'unknown' and match
        # several unrelated ESRI codes at confidence 25, so an unguarded lookup
        # would collide unrelated local projections onto one registry file.
        candidates = crs.list_authority(min_confidence=25)
        if len(candidates) == 1:
            authority = (candidates[0].auth_name, candidates[0].code)
    if authority is not None:
        return f'{authority[0]}-{authority[1]}'
    return re.sub(r'[^A-Za-z0-9]+', '-', crs.name).strip('-')


def _registry_path(source_crs, target_crs) -> Path:
    return REGISTRY_DIR / f'{_crs_slug(source_crs)}_to_{_crs_slug(target_crs)}.json'


def load_crs_transform(source_crs, target_crs) -> dict | None:
    """Load the registered transform entry for a CRS pair.

    Parameters
    ----------
    source_crs : Any
        Source CRS in any form `pyproj.CRS` accepts.
    target_crs : Any
        Target CRS in any form `pyproj.CRS` accepts.

    Returns
    -------
    dict or None
        The registry entry, or `None` if the pair is not registered.
    """
    path = _registry_path(source_crs, target_crs)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def _grid_filenames(definition: str) -> list[str]:
    return re.findall(r'grids=(\S+)', definition)


def _pipeline_template(definition: str) -> str:
    """Return the PROJ pipeline definition with grid filenames blanked out.

    This template is used to identify and merge sibling operations that share the
    same structure but use different regional grid files. For example, regional
    HPGN/HARN corrections for `EPSG:5070 -> EPSG:4326` are identical except for
    their specific grid files. Blanking out filenames prevents pinning a multi-region
    CRS (like `EPSG:5070` Conus Albers) to a single region's grid, which would
    silently return `inf` for coordinates outside that region.
    """
    return re.sub(r'grids=\S+', 'grids=<GRID>', definition)


def _download_grid(filename: str) -> None:
    target = GRIDS_DIR / filename
    if target.exists():
        return
    GRIDS_DIR.mkdir(parents=True, exist_ok=True)
    response = requests.get(GRID_URL_TEMPLATE.format(filename=filename), timeout=30)
    response.raise_for_status()
    target.write_bytes(response.content)


def _write_registry_entry(path: Path, entry: dict) -> None:
    """Write a registry entry to a file atomically.

    Uses a write-then-rename pattern to ensure safety under concurrent callers.
    If multiple processes resolve the same unregistered pair, they generate
    identical content, making the final overwrite safe from corruption.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(entry, f, indent=2, sort_keys=True)
            f.write('\n')
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def resolve_crs_transform(source_crs, target_crs='EPSG:4326') -> dict:
    """Resolve, download, and register the best transform for a CRS pair.

    Picks the most accurate available operation from PROJ. If the operation uses
    regional grids, it merges sibling candidate grids into a single multi-grid
    definition to ensure coverage across the source domain, downloads required
    grid files to `crs_grids/`, and registers the transform in a JSON file.

    Parameters
    ----------
    source_crs : Any
        Source CRS in any form `pyproj.CRS` accepts.
    target_crs : Any, default 'EPSG:4326'
        Target CRS in any form `pyproj.CRS` accepts.

    Returns
    -------
    dict
        The resolved registry entry. The entry is persisted to disk unless the
        operation is exact (`accuracy_m == 0.0`) and requires no grid files.

    Raises
    ------
    ValueError
        If no usable reprojection operation is found.
    """
    pyproj.network.set_network_enabled(True)
    source_crs = pyproj.CRS(source_crs)
    target_crs = pyproj.CRS(target_crs)

    group = pyproj.transformer.TransformerGroup(source_crs, target_crs, always_xy=True)
    if not group.transformers:
        raise ValueError(
            f'No usable reprojection operation found for {source_crs.to_string()} '
            f'-> {target_crs.to_string()} ({len(group.unavailable_operations)} '
            'operation(s) known but unavailable -- check network access).'
        )

    # Sort available transformers by accuracy, breaking ties by description.
    def _sort_key(t):
        accuracy = t.accuracy if (t.accuracy or -1) >= 0 else float('inf')
        return (accuracy, t.description)

    best = min(group.transformers, key=_sort_key)
    template = _pipeline_template(best.definition)

    # Merge same-template sibling operations. This handles cases where a source
    # CRS covers multiple regions (e.g., EPSG:5070 Conus Albers). Using only the
    # single best-rated candidate might cover only a fraction of the domain
    # (e.g., a Canadian NRCan grid that doesn't cover the US), returning inf
    # for points outside it. Merging allows PROJ to try each grid in turn.
    siblings = [
        t for t in group.transformers if _pipeline_template(t.definition) == template
    ]

    grid_files = sorted({f for t in siblings for f in _grid_filenames(t.definition)})
    for filename in grid_files:
        _download_grid(filename)

    # Append @null as a final fallback for grids. A region's correction might
    # not share the winning template (e.g., Alaska in EPSG:4269 -> EPSG:4326,
    # which uses a grid-free proj=noop instead of hgridshift). Coordinates in
    # such regions would silently become inf, causing GEOSExceptions later.
    # Appending @null ensures points outside the regional grids fallback to
    # ballpark/identity accuracy instead of failing.
    if grid_files:
        definition = re.sub(
            r'grids=\S+', f'grids={",".join(grid_files)},@null', best.definition
        )
        if len(grid_files) > 1:
            operation_name = (
                f'{best.description} (merged with {len(grid_files) - 1} '
                'same-shaped region grid(s), with identity fallback outside '
                'their combined coverage)'
            )
        else:
            operation_name = (
                f'{best.description} (with identity fallback outside its coverage)'
            )
    else:
        definition = best.definition
        operation_name = best.description

    entry = {
        'source_crs': source_crs.to_string(),
        'target_crs': target_crs.to_string(),
        'operation_name': operation_name,
        'operation_definition': definition,
        'accuracy_m': best.accuracy,
        'grid_files': grid_files,
        'registered': date.today().isoformat(),
    }

    # Write to the registry unless the operation is exact and grid-free
    # (e.g., same-datum conversions like WGS84 to Web Mercator). These are pure
    # mathematical formulas with no environment-dependent files, so there is
    # nothing to pin or download.
    if entry['accuracy_m'] != 0.0 or entry['grid_files']:
        _write_registry_entry(_registry_path(source_crs, target_crs), entry)
    return entry


def get_or_resolve_crs_transform(source_crs, target_crs='EPSG:4326') -> dict:
    """Get the registered transform for a CRS pair, resolving it if not present.

    Parameters
    ----------
    source_crs : Any
        Source CRS in any form `pyproj.CRS` accepts.
    target_crs : Any, default 'EPSG:4326'
        Target CRS in any form `pyproj.CRS` accepts.

    Returns
    -------
    dict
        The pre-existing or newly resolved registry entry.
    """
    entry = load_crs_transform(source_crs, target_crs)
    if entry is not None:
        return entry
    return resolve_crs_transform(source_crs, target_crs)


def _main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        raise SystemExit(
            'Usage: python -m openplaces.geo.crs_transforms SOURCE_CRS [TARGET_CRS]'
        )
    source_crs, target_crs = args[0], args[1] if len(args) > 1 else 'EPSG:4326'
    entry = resolve_crs_transform(source_crs, target_crs)
    print(f'Registered {entry["source_crs"]} -> {entry["target_crs"]}:')
    print(f'  {entry["operation_name"]} (accuracy={entry["accuracy_m"]}m)')
    if entry['grid_files']:
        print(f'  grid file(s): {", ".join(entry["grid_files"])}')
    if entry['accuracy_m'] == 0.0 and not entry['grid_files']:
        print('  Exact, grid-free conversion -- not persisted (no pinning needed).')
    else:
        print(f'  -> {_registry_path(source_crs, target_crs)}')
        print('Review and commit the registry entry (and any new grid files) via a PR.')


if __name__ == '__main__':
    _main()
