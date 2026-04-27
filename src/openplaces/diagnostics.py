"""
System diagnostics: recipe availability, geographic coverage, etc.
"""

import time
import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import yaml


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
    _LEVEL_WEIGHTS = {0: 0.22, 1: 0.48, 2: 0.72, 3: 0.92}
    # Edge grays: lighter for coarser levels, darker for finer
    _LEVEL_EDGECOLORS = {0: '#d8d8d8', 1: '#c0c0c0', 2: '#888888', 3: '#505050'}

    def _level_color(level: int) -> tuple:
        w = _LEVEL_WEIGHTS.get(level, min(0.22 + level * 0.24, 0.95))
        r, g, b, _ = _BASE
        return ((1 - w) + w * r, (1 - w) + w * g, (1 - w) + w * b, 1.0)

    def _level_edgecolor(level: int) -> str:
        return _LEVEL_EDGECOLORS.get(level, '#303030')

    df = find_recipes(entity_type)
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
        color_map = {idx: '#e8e8e8' for idx in world.index}

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
