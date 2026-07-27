"""
Pinned, git-tracked registry of CRS-to-CRS reprojection operations.

Plain `to_crs()` accepts whatever operation PROJ's `Transformer.from_crs()`
considers "best available" for a source/target CRS pair, resolved dynamically
at call time. For a datum change (e.g. NAD83-based state plane to WGS84) that
resolution can silently pick a much lower-accuracy operation depending on
network access and which grid files happen to be cached locally, with no
error -- only an easily-missed `UserWarning`. See `reproject()` in
`geo/polygon.py` for the mismatch this caused between two independently
ingested Massachusetts parcel datasets, traced to exactly this.

Registry: one small JSON file per `(source_crs, target_crs)` pair under
`crs_transforms/`, plus the actual grid files those operations reference
under `crs_grids/` -- both committed to the repo, so a `git clone` alone
gives deterministic, offline-capable reprojection. Entries are written
either ahead of time via the CLI (``python -m openplaces.geo.crs_transforms
EPSG:26986``) or lazily, the first time `reproject()` encounters an
unregistered pair.
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
    """Filesystem-safe identifier for a CRS, preferring its authority code."""
    crs = pyproj.CRS(crs)
    authority = crs.to_authority()
    if authority is not None:
        return f'{authority[0]}-{authority[1]}'
    return re.sub(r'[^A-Za-z0-9]+', '-', crs.name).strip('-')


def _registry_path(source_crs, target_crs) -> Path:
    return REGISTRY_DIR / f'{_crs_slug(source_crs)}_to_{_crs_slug(target_crs)}.json'


def load_crs_transform(source_crs, target_crs) -> dict | None:
    """Return the registered transform entry for *source_crs* -> *target_crs*.

    Parameters
    ----------
    source_crs, target_crs : Any
        CRS in any form `pyproj.CRS` accepts.

    Returns
    -------
    dict or None
        The registry entry, or `None` if this pair hasn't been registered.
    """
    path = _registry_path(source_crs, target_crs)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def _grid_filenames(definition: str) -> list[str]:
    return re.findall(r'grids=(\S+)', definition)


def _pipeline_template(definition: str) -> str:
    """Return *definition* with its grid filename(s) blanked out.

    Two candidate operations that share a template differ only in which
    single region-specific grid file they use -- e.g. every US state's own
    HPGN/HARN correction for `EPSG:5070 -> EPSG:4326` is byte-identical
    except for `grids=us_noaa_XXhpgn.tif`. Used to detect that case so those
    candidates can be merged (see `resolve_crs_transform`) instead of one
    single-region grid being picked and silently returning `inf` for every
    point outside its own coverage -- the mechanism behind a real incident
    where a source CRS spanning many states (`EPSG:5070`, Conus Albers) got
    pinned to *one* state's grid.
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
    """Write *entry* to *path* via write-temp-then-atomic-rename.

    Safe under concurrent callers: a race to resolve the same never-before-seen
    pair computes identical content, so whichever write lands last is a no-op
    in substance, not a corruption.
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

    Picks the most accurate operation `pyproj.transformer.TransformerGroup`
    reports as available (lowest `accuracy`, ties broken by operation name for
    determinism). If other available candidates share that operation's
    pipeline template (same steps, differing only in which single grid file
    they use -- the shape every per-state HPGN/HARN correction takes for a
    source CRS spanning many regions, e.g. `EPSG:5070`/Conus Albers), they are
    merged into one operation listing all of those grids
    (`+proj=hgridshift +grids=a.tif,b.tif,...`), which PROJ resolves per point
    by trying each grid in turn for the one whose area-of-use covers it.
    Picking only the single globally-best-rated candidate is not safe here:
    it may be the best-*rated* grid while covering only a fraction of the
    source CRS's actual domain, silently returning `inf` for any point
    outside it -- confirmed for `EPSG:5070 -> EPSG:4326`, whose single best
    candidate is a Canadian NRCan grid that doesn't cover any US state.

    Downloads any grid file(s) the resolved operation needs into
    `crs_grids/`, and writes the registry entry.

    Parameters
    ----------
    source_crs : Any
        Source CRS, in any form `pyproj.CRS` accepts.
    target_crs : Any, default 'EPSG:4326'
        Target CRS.

    Returns
    -------
    dict
        The registry entry that was written.

    Raises
    ------
    ValueError
        If no operation is available for this CRS pair.
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

    def _sort_key(t):
        accuracy = t.accuracy if (t.accuracy or -1) >= 0 else float('inf')
        return (accuracy, t.description)

    best = min(group.transformers, key=_sort_key)
    template = _pipeline_template(best.definition)
    siblings = [
        t for t in group.transformers if _pipeline_template(t.definition) == template
    ]

    grid_files = sorted({f for t in siblings for f in _grid_filenames(t.definition)})
    for filename in grid_files:
        _download_grid(filename)

    if len(grid_files) > 1:
        definition = re.sub(
            r'grids=\S+', f'grids={",".join(grid_files)}', best.definition
        )
        operation_name = (
            f'{best.description} (merged with {len(grid_files) - 1} '
            'same-shaped region grid(s))'
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
    _write_registry_entry(_registry_path(source_crs, target_crs), entry)
    return entry


def get_or_resolve_crs_transform(source_crs, target_crs='EPSG:4326') -> dict:
    """Return the registered transform for a CRS pair, resolving it if absent.

    Parameters
    ----------
    source_crs, target_crs : Any
        CRS in any form `pyproj.CRS` accepts.

    Returns
    -------
    dict
        The registry entry (pre-existing or newly resolved).
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
    print(f'  -> {_registry_path(source_crs, target_crs)}')
    print('Review and commit the registry entry (and any new grid files) via a PR.')


if __name__ == '__main__':
    _main()
