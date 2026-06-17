"""
System diagnostics: recipe availability, geographic coverage, disk usage, etc.
"""

import os
import shutil
import time
import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from openplaces.core.constants import ESCAPE_DIR, STRING_SEPARATOR_WITHIN_IDS
from openplaces.core.schema import AdminId


def find_recipes(
    entity_type: str,
    stage: str | None = None,
) -> pd.DataFrame:
    """Find all recipes for a given entity type.

    Scans the recipes directory and returns a table of available recipes.

    Parameters
    ----------
    entity_type : str
        Entity type to search for (e.g. ``'building'``, ``'parcel'``).
    stage : str or None
        If given, only return recipes whose ``stage`` field matches
        (e.g. ``'ingest'``, ``'harmonize'``).  Recipes without an explicit
        ``stage`` field are treated as ``'ingest'``.

    Returns
    -------
    pd.DataFrame
        Columns: ``admin_id``, ``stage``, ``entity_type``, ``source_id``,
        ``version``, ``n_companion_files``. Sorted by admin_id then source_id.
        Global recipes have an empty string for ``admin_id``.
    """
    recipes_root = Path(__file__).parent / 'recipes'

    rows = []
    for yaml_path in sorted(recipes_root.rglob('*.yaml')):
        # Fast pre-filter: entity_type must appear as a directory component
        if entity_type not in yaml_path.parts:
            continue

        with open(yaml_path, encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if 'entity' not in data:
            continue
        entity = data['entity']
        if entity.get('entity_type') != entity_type:
            continue

        recipe_stage = data.get('stage') or 'ingest'
        if stage is not None and recipe_stage != stage:
            continue

        admin_id = data.get('admin_id')
        admin_id_str = str(admin_id) if admin_id is not None else ''

        source = entity.get('source', {})
        source_id = source.get('source_id', '')
        version = str(entity.get('version', ''))

        n_companion = sum(1 for p in yaml_path.parent.iterdir() if p != yaml_path)

        rows.append(
            {
                'admin_id': admin_id_str,
                'stage': recipe_stage,
                'entity_type': entity_type,
                'source_id': source_id,
                'version': version,
                'n_companion_files': n_companion,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(['admin_id', 'source_id']).reset_index(drop=True)
    return df


def map_recipe_coverage(
    entity_type: str,
    stage: str | None = None,
    figsize: tuple = (14, 7),
    ax: plt.Axes | None = None,
    verbose: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Map geographic coverage of recipes for a given entity type.

    Plots admin geometries fetched via :func:`~openplaces.io.readers.get_admin`,
    layering smaller admin units on top of larger ones.  Requires admin
    boundary data (GADM) to be ingested.

    Colors encode both source (hue) and admin level (lightness): global
    recipes appear in a pastel shade, country-level recipes are slightly
    darker, state-level darker still, and so on.

    Parameters
    ----------
    entity_type : str
        Entity type to map (e.g. ``'building'``).
    stage : str or None
        If given, only map recipes whose ``stage`` field matches
        (e.g. ``'ingest'``, ``'harmonize'``).
    figsize : tuple
        Figure size (width, height) in inches.  Ignored when *ax* is given.
    ax : matplotlib.axes.Axes or None
        Axes to plot into.  A new figure is created when *None*.
    verbose : bool
        Print timing for each stage.

    Returns
    -------
    (fig, ax) : tuple[plt.Figure, plt.Axes]
    """

    from openplaces.io.readers import get_admin

    def _tick(label: str, t0: float) -> float:
        t1 = time.perf_counter()
        if verbose:
            print(f'  {label}: {t1 - t0:.2f}s')
        return t1

    # Single base hue (steel blue) blended with white at level-dependent weights.
    # weight=0 → white (pastel); weight=1 → full base colour.
    _BASE = (0.13, 0.47, 0.71, 1.0)  # tab10 blue
    _UNCOVERED = (0.91, 0.91, 0.91, 1.0)
    _LEVEL_WEIGHTS = {0: 0.22, 1: 0.48, 2: 0.72, 3: 0.92}
    # Edge grays: lighter for coarser levels, darker for finer
    _LEVEL_EDGECOLORS = {0: '#d8d8d8', 1: '#c0c0c0', 2: '#888888', 3: '#505050'}

    def _level_color(level: int) -> tuple:
        w = _LEVEL_WEIGHTS.get(level, min(0.22 + level * 0.24, 0.95))
        r, g, b, _ = _BASE
        return ((1 - w) + w * r, (1 - w) + w * g, (1 - w) + w * b, 1.0)

    def _level_edgecolor(level: int) -> str:
        return _LEVEL_EDGECOLORS.get(level, '#303030')

    df = find_recipes(entity_type, stage=stage)
    if df.empty:
        raise ValueError(f"No recipes found for entity_type='{entity_type}'.")

    def _admin_level(admin_id: str) -> int:
        return len(admin_id.split('-')) if admin_id else 0

    df = df.copy()
    df['_level'] = df['admin_id'].apply(_admin_level)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    _admin_geo_recipe = 'admin-openplaces-2026'

    # Load world once (level 1): background + levels 0 and 1 coverage
    world = None
    t = time.perf_counter()
    if verbose:
        print('Loading world (admin level 1)...')
    try:
        world = get_admin(
            level=1, geom='simplified', recipe=f'{_admin_geo_recipe}_admin1'
        )
        t = _tick(f'loaded {len(world):,} countries', t)
    except Exception:
        warnings.warn(
            'World background unavailable — admin level-1 data not ingested.',
            stacklevel=2,
        )

    if world is not None:
        # Build one color per country in a single pass:
        #   - start gray (uncovered)
        #   - paint global (level 0) coverage first (pastel)
        #   - then override with country-level (level 1) coverage (darker)
        # Result: world.plot() is called exactly once.
        color_map = {idx: _UNCOVERED for idx in world.index}

        if not df[df['_level'] == 0].empty:
            for idx in color_map:
                color_map[idx] = _level_color(level=0)

        for admin_id in df.loc[df['_level'] == 1, 'admin_id']:
            if admin_id in color_map:
                color_map[admin_id] = _level_color(level=1)

        t = time.perf_counter()
        world.plot(
            ax=ax,
            color=[color_map[idx] for idx in world.index],
            edgecolor=_level_edgecolor(1),
            linewidth=0.1,
        )
        t = _tick('Plot world (background + levels 0–1)', t)

    # Level 2+: one get_admin call per level, one plot call per level
    max_level = int(df['_level'].max())
    for level in range(2, max_level + 1):
        unique_ids = df.loc[df['_level'] == level, 'admin_id'].unique().tolist()
        if verbose:
            print(f'Level {level}: get_admin({unique_ids})...')
        t = time.perf_counter()
        try:
            gdf = get_admin(
                unique_ids,
                geom='simplified',
                recipe=f'{_admin_geo_recipe}_admin{level}',
            )
        except Exception as e:
            warnings.warn(
                f'Could not load geometries for level {level}: {e}',
                stacklevel=2,
            )
            continue
        t = _tick(
            f'get_admin level {level} ({len(unique_ids)} ids → {len(gdf)} rows)', t
        )

        t = time.perf_counter()
        gdf.plot(
            ax=ax,
            color=_level_color(level=level),
            edgecolor=_level_edgecolor(level),
            linewidth=0.1,
        )
        _tick(f'Plot level {level}', t)

    # Legend: one swatch per admin level present in the recipes
    level_labels = {0: 'global', 1: 'country', 2: 'state', 3: 'county'}
    handles = [
        mpatches.Patch(
            color=_level_color(level=level),
            label=level_labels.get(level, f'level {level}'),
        )
        for level in sorted(df['_level'].unique())
    ]
    ax.legend(handles=handles, loc='lower left', framealpha=0.9)

    ax.set_title(f'Recipe coverage: {entity_type}')
    ax.set_axis_off()

    return fig, ax


def profile_disk_usage(
    roots: dict[str, Path | str] | None = None,
    min_size_mb: float = 1.0,
) -> pd.DataFrame:
    """Profile disk usage of the openplaces data directories.

    Walks each root directory once and aggregates file sizes by admin unit
    and dataset, using the on-disk layout
    {root}/{admin levels...}/_all/{entity or dataset path...}/{files}.

    Parameters
    ----------
    roots : dict or None
        Mapping of label to directory to scan. Defaults to the configured
        core, external, heap, cache, and out directories that exist.
    min_size_mb : float
        Drop groups smaller than this size.

    Returns
    -------
    pd.DataFrame
        Columns: root, admin_id, dataset, n_files, size_mb.
        Sorted by size, descending. Files that do not follow the standard
        layout are aggregated with the path relative to the root as dataset.
    """
    from openplaces.config import cfg

    if roots is None:
        candidates = {
            'core': cfg.core_dir,
            'external': cfg.external_dir,
            'heap': cfg.heap_dir,
            'cache': cfg.cache_dir,
            'out': cfg.out_dir,
        }
        roots = {label: Path(d) for label, d in candidates.items() if Path(d).is_dir()}

    resolved_roots = {Path(d).resolve() for d in roots.values()}
    groups: dict[tuple[str, str, str], list[float]] = {}
    for label, root in roots.items():
        root = Path(root)
        for dirpath, dirnames, filenames in os.walk(root):
            # Avoid double counting roots nested in other roots (e.g. heap
            # inside cache).
            dirnames[:] = [
                d
                for d in dirnames
                if (Path(dirpath) / d).resolve() not in resolved_roots
                or (Path(dirpath) / d).resolve() == root.resolve()
            ]
            if not filenames:
                continue
            parts = Path(dirpath).relative_to(root).parts
            if ESCAPE_DIR in parts:
                cut = parts.index(ESCAPE_DIR)
                admin_id = STRING_SEPARATOR_WITHIN_IDS.join(parts[:cut])
                dataset = '/'.join(parts[cut + 1 :])
            else:
                admin_id = ''
                dataset = '/'.join(parts)
            size = 0
            n_files = 0
            for filename in filenames:
                try:
                    size += os.stat(os.path.join(dirpath, filename)).st_size
                    n_files += 1
                except OSError:
                    continue
            stats = groups.setdefault((label, admin_id, dataset), [0, 0])
            stats[0] += n_files
            stats[1] += size

    df = pd.DataFrame(
        [
            {
                'root': label,
                'admin_id': admin_id,
                'dataset': dataset,
                'n_files': stats[0],
                'size_mb': stats[1] / 2**20,
            }
            for (label, admin_id, dataset), stats in groups.items()
        ]
    )
    if df.empty:
        return df
    df = df[df['size_mb'] >= min_size_mb]
    df['size_mb'] = df['size_mb'].round(1)
    return df.sort_values('size_mb', ascending=False, ignore_index=True)


def list_image_caches() -> pd.DataFrame:
    """List downloaded image caches in the external data directory.

    Image caches are directories of the form
    {external}/{admin path}/_all/image/{source}/{version}, written by the
    image ingestion recipes (e.g. image-googlesatellite-z20).

    Returns
    -------
    pd.DataFrame
        One row per cache: admin_id, source, version, n_files, size_mb,
        path. Sorted by size, descending.
    """
    from openplaces.config import cfg

    df = profile_disk_usage({'external': cfg.external_dir}, min_size_mb=0.0)
    if df.empty:
        return df
    parts = df['dataset'].str.split('/')
    df = df[(parts.str[0] == 'image') & (parts.str.len() >= 3)].copy()
    if df.empty:
        return df

    parts = df['dataset'].str.split('/')
    df['source'] = parts.str[1]
    df['version'] = parts.str[2]
    caches = (
        df.groupby(['admin_id', 'source', 'version'], as_index=False)
        .agg(n_files=('n_files', 'sum'), size_mb=('size_mb', 'sum'))
        .sort_values('size_mb', ascending=False, ignore_index=True)
    )
    caches['path'] = [
        str(
            cfg.external_dir.joinpath(
                *AdminId(row.admin_id).to_path().parts,
                'image',
                row.source,
                row.version,
            )
        )
        for row in caches.itertuples()
    ]
    return caches


def delete_image_caches(
    admin_ids: str | list | None = None,
    source: str | None = None,
    version: str | None = None,
    dry_run: bool = True,
) -> pd.DataFrame:
    """Delete location-specific image caches from the external directory.

    Parameters
    ----------
    admin_ids : str, AdminId, list, or None
        Admin units whose caches to delete; a coarser unit (e.g. a county)
        matches all caches of its children. None matches all locations.
    source : str or None
        Restrict to one image source (e.g. 'googlesatellite').
    version : str or None
        Restrict to one recipe version (e.g. 'z20').
    dry_run : bool
        If True (default), only report what would be deleted. If False,
        remove each matched cache directory, including images and the
        image metadata parquet.

    Returns
    -------
    pd.DataFrame
        The matched caches: admin_id, source, version, n_files, size_mb,
        path.
    """
    caches = list_image_caches()
    if caches.empty:
        print('No image caches found.')
        return caches

    if admin_ids is not None:
        if isinstance(admin_ids, str | AdminId):
            admin_ids = [admin_ids]
        selectors = [AdminId(str(a)) for a in admin_ids]
        caches = caches[
            [
                any(
                    str(sel) == str(aid) or sel.is_parent_of(aid)
                    for sel in selectors
                    for aid in [AdminId(cache_admin_id)]
                )
                for cache_admin_id in caches['admin_id']
            ]
        ]
    if source is not None:
        caches = caches[caches['source'] == source]
    if version is not None:
        caches = caches[caches['version'] == str(version)]
    caches = caches.reset_index(drop=True)

    total_mb = caches['size_mb'].sum()
    if dry_run:
        print(
            f'Dry run: would delete {len(caches)} image cache(s), '
            f'{total_mb:,.1f} MB total. Pass dry_run=False to delete.'
        )
        return caches

    for cache_path in caches['path']:
        shutil.rmtree(cache_path, ignore_errors=True)
    print(f'Deleted {len(caches)} image cache(s), {total_mb:,.1f} MB total.')
    return caches
