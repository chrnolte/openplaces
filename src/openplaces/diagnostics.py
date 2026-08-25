"""
System diagnostics: recipe availability, geographic coverage, disk usage, etc.
"""

import os
import time
import warnings
from functools import cache
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
        ``version``, ``n_companion_files``, ``exclude_from_auto_discover``,
        ``level``, ``recipe_id``, ``filename_suffix``. Sorted by admin_id
        then source_id. Global recipes have an empty string for
        ``admin_id``. ``level`` is the admin level the recipe's own output
        targets (parsed from its filename, e.g. `4` for ``..._admin4.yaml``)
        -- only meaningful for ``entity_type='admin'``; ``None`` otherwise,
        since only admin recipes are split one file per level. This is
        independent of how specific the recipe's ``admin_id`` scope is: two
        recipes scoped to the same country can each target a different
        level. ``recipe_id`` is the file's own stem (the recipe path
        convention guarantees this equals the true recipe id, filename
        suffix included, so it is read rather than reconstructed).
        ``filename_suffix`` is the trailing ``[_{filename}]`` part of
        ``recipe_id`` distinguishing a sibling recipe that otherwise shares
        the same admin_id/entity_type/source_id/version (e.g. two flat
        files bundled in the same source ZIP) -- empty string for a
        recipe with no such suffix.
    """
    index = _recipe_index()
    if index.empty:
        return index
    mask = index['entity_type'] == entity_type
    if stage is not None:
        mask &= index['stage'] == stage
    return index[mask].reset_index(drop=True)


@cache
def _recipe_index() -> pd.DataFrame:
    """Every recipe's summary row, parsed once per process.

    :func:`find_recipes` used to rglob and `yaml.safe_load` the whole
    recipe tree on every call, and auto-discovery calls it once per
    resolution. Measured on a 3-county CHEER graph: **8,173 YAML loads
    taking 86 s, 84% of the entire DAG construction** -- against 182
    distinct files that parse in 0.6 s all together. The tree is static
    for the life of a process, so the redundancy is pure waste.

    Cached like :func:`~openplaces.recipe.get_recipe_by_id` and the
    attribute registry, and with the same caveat: a session that edits a
    recipe on disk must call ``_recipe_index.cache_clear()`` (or restart)
    to see it.
    """
    recipes_root = Path(__file__).parent / 'recipes'

    rows = []
    for yaml_path in sorted(recipes_root.rglob('*.yaml')):
        try:
            with open(yaml_path, encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except Exception:  # noqa: BLE001 - an unparseable recipe is not
            continue  # discoverable; that is the pre-index behavior

        if not isinstance(data, dict) or 'entity' not in data:
            continue
        entity = data['entity']
        entity_type = entity.get('entity_type')
        # Preserved from the pre-index scan, which pre-filtered on the
        # path before parsing: a recipe whose declared entity_type is not
        # a directory component of its own path was never discoverable,
        # and making it so here would be a silent behavior change.
        if not entity_type or entity_type not in yaml_path.parts:
            continue

        recipe_stage = data.get('stage') or 'ingest'

        admin_id = data.get('admin_id')
        admin_id_str = str(admin_id) if admin_id is not None else ''

        source = entity.get('source', {})
        source_id = source.get('source_id', '')
        version = str(entity.get('version', ''))

        n_companion = sum(1 for p in yaml_path.parent.iterdir() if p != yaml_path)

        level = None
        if entity_type == 'admin':
            suffix = yaml_path.stem.rsplit('admin', 1)[-1]
            level = int(suffix) if suffix.isdigit() else None

        recipe_id = yaml_path.stem
        base_id = (
            f'{admin_id_str}_{entity_type}-{source_id}-{version}'
            if admin_id_str
            else f'{entity_type}-{source_id}-{version}'
        )
        filename_suffix = (
            recipe_id[len(base_id) :].lstrip('_')
            if recipe_id.startswith(base_id)
            else ''
        )

        rows.append(
            {
                'admin_id': admin_id_str,
                'stage': recipe_stage,
                'entity_type': entity_type,
                'source_id': source_id,
                'version': version,
                'n_companion_files': n_companion,
                'exclude_from_auto_discover': bool(
                    data.get('exclude_from_auto_discover', False)
                ),
                'level': level,
                'recipe_id': recipe_id,
                'filename_suffix': filename_suffix,
            }
        )

    df = pd.DataFrame(
        rows,
        columns=[
            'admin_id',
            'stage',
            'entity_type',
            'source_id',
            'version',
            'n_companion_files',
            'exclude_from_auto_discover',
            'level',
            'recipe_id',
            'filename_suffix',
        ],
    )
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


def map_admin_recipe_source_coverage(
    levels: tuple = (2, 3, 4),
    figsize: tuple = (8, 12),
    verbose: bool = False,
) -> tuple:
    """Map, per admin level, which countries have their own admin recipe.

    One world map per requested level, stacked vertically, colored by
    whether a country-specific admin recipe replaces GADM there (e.g.
    `GB_admin-ons-2024_admin4`) or GADM is still the fallback. Useful for
    spotting, at a glance, where GADM is likely incomplete or poorly
    named and nothing better has been added yet.

    Unlike :func:`map_recipe_coverage`, which colors by how specific a
    recipe's own `admin_id` scope is (global/country/state/...), this
    looks at each recipe's own output level (`level`, from
    :func:`find_recipes`) -- so a country whose admin4 recipe replaces
    GADM but whose admin2/3 don't (e.g. `DE`, `FR`) shows up correctly on
    each level's own map, not lumped together.

    Parameters
    ----------
    levels : tuple of int
        Admin levels to map, one subplot per level.
    figsize : tuple
        Figure size (width, height) in inches, for the whole figure.
    verbose : bool
        Print timing for each stage.

    Returns
    -------
    (fig, axes) : tuple[plt.Figure, list[plt.Axes]]
    """
    from openplaces.io.readers import get_admin

    df = find_recipes('admin', stage='ingest')
    if df.empty:
        raise ValueError('No admin recipes found.')

    # A country-specific recipe: `admin_id` scoped to exactly one
    # country (e.g. 'GB'), not '' (global, e.g. GADM) or 'US-MA' (a
    # state, out of scope for a country-level map).
    country_recipes = df[(df['admin_id'] != '') & (~df['admin_id'].str.contains('-'))]

    t0 = time.perf_counter()
    world = get_admin(level=1, geom='simplified', recipe='admin-openplaces-2026_admin1')
    if verbose:
        print(f'Loaded {len(world):,} countries: {time.perf_counter() - t0:.2f}s')

    fig, axes = plt.subplots(len(levels), 1, figsize=figsize)
    axes = [axes] if len(levels) == 1 else list(axes)

    _HAS_RECIPE = (0.13, 0.47, 0.71, 1.0)
    _GADM_FALLBACK = (0.91, 0.91, 0.91, 1.0)

    for ax, level in zip(axes, levels):
        covered = set(
            country_recipes.loc[country_recipes['level'] == level, 'admin_id']
        )
        colors = [
            _HAS_RECIPE if idx in covered else _GADM_FALLBACK for idx in world.index
        ]
        world.plot(ax=ax, color=colors, edgecolor='#888888', linewidth=0.1)
        ax.set_title(f'admin{level}: {len(covered)} countries')
        ax.set_axis_off()

    handles = [
        mpatches.Patch(color=_HAS_RECIPE, label='country-specific recipe'),
        mpatches.Patch(color=_GADM_FALLBACK, label='GADM fallback'),
    ]
    axes[-1].legend(handles=handles, loc='lower left', framealpha=0.9)
    fig.tight_layout()

    return fig, axes


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


def profile_columns(
    gdf: pd.DataFrame,
    entity_type: str | None = None,
    stage: str | None = None,
) -> pd.DataFrame:
    """Profile column completeness and registry schema conformance.

    For each column, reports how often it is populated -- both a strict
    non-null fraction and a "meaningfully populated" fraction that also
    excludes placeholder-looking values (``0`` or ``''``). The gap between
    the two flags data that looks complete but isn't: e.g. an ingest
    source's ``year_built`` field can be 100% non-null while being ``0``
    (a placeholder, not a real construction year) for every row in a given
    admin unit -- a plain ``notna()`` check never surfaces that, but the
    meaningfully-populated fraction drops straight to 0%.

    When *entity_type* is given, each column is resolved to its canonical
    registry name (:func:`openplaces.recipe.resolve_attribute_name`, which
    strips a provenance suffix such as ``_parcel``/``_building_nsi``) and
    checked against the registry's declared ``data_type`` for that
    entity_type/stage (see :func:`~openplaces.core.attribute_registry.
    get_attributes`) -- the same check
    :func:`openplaces.io.ingester._warn_registry_type_mismatches` makes at
    ingest time, but as a queryable report rather than a runtime warning.

    Parameters
    ----------
    gdf : DataFrame or GeoDataFrame
        Table to profile. A ``geometry`` column, if present, is skipped
        (see :func:`check_geometry` for geometry-specific diagnostics).
    entity_type : str, optional
        Entity type to check declared data types against (e.g.
        ``'parcel'``). When ``None``, ``dtype_mismatch`` is always
        ``False`` and ``expected_data_type`` is always ``None``.
    stage : str, optional
        Pipeline stage to scope the registry lookup to (see
        :func:`~openplaces.core.attribute_registry.get_attributes`).

    Returns
    -------
    pandas.DataFrame
        Indexed by column name. Columns: ``dtype`` (the column's pandas
        dtype, as a string), ``n_values``/``frac_values`` (non-null
        count/fraction), ``n_meaningful``/``frac_meaningful`` (non-null and
        not a placeholder-looking ``0``/``''``), ``expected_data_type``
        (registry value, or ``None`` when unregistered or *entity_type*
        wasn't given), ``dtype_mismatch`` (bool).
    """
    from openplaces.core.attribute_registry import get_attributes
    from openplaces.recipe import resolve_attribute_name

    registry = get_attributes(entity_type, stage) if entity_type else None

    n = len(gdf)
    rows = []
    for col in gdf.columns:
        if col == 'geometry':
            continue
        series = gdf[col]
        non_null = series.notna()
        n_values = int(non_null.sum())
        meaningful = non_null & ~series.isin([0, ''])
        n_meaningful = int(meaningful.sum())

        expected_data_type = None
        dtype_mismatch = False
        if registry is not None:
            canonical = resolve_attribute_name(col)
            if canonical in registry.index:
                expected_data_type = registry.at[canonical, 'data_type']
                actual = series.dtype
                if expected_data_type == 'categorical':
                    # A categorical column is often still plain string/object
                    # dtype until an explicit `columns_to_categorical` cast
                    # runs -- accept any string-like dtype (object, pandas
                    # 'string', pandas 3's default 'str', or an already-cast
                    # CategoricalDtype), not just a literal 'category'/
                    # 'object' name match.
                    dtype_mismatch = not (
                        pd.api.types.is_string_dtype(actual)
                        or isinstance(actual, pd.CategoricalDtype)
                    )
                elif expected_data_type in ('float', 'int'):
                    dtype_mismatch = not pd.api.types.is_numeric_dtype(actual)

        rows.append(
            {
                'column': col,
                'dtype': str(series.dtype),
                'n_values': n_values,
                'frac_values': n_values / n if n else 0.0,
                'n_meaningful': n_meaningful,
                'frac_meaningful': n_meaningful / n if n else 0.0,
                'expected_data_type': expected_data_type,
                'dtype_mismatch': dtype_mismatch,
            }
        )
    return pd.DataFrame(rows).set_index('column')


_NON_CATEGORICAL_SUFFIXES = ('_id', '_id_local', '_id_assessor', '_date')
_NON_CATEGORICAL_SUBSTRINGS = ('city', 'postal_code')


def _looks_non_categorical(column: str) -> bool:
    """True for an id/date/city/postal_code-like column name.

    These can have a high duplication ratio (many rows share the same
    city, for instance) without being meaningfully categorical for
    :func:`summarize_categoricals`'s purposes.
    """
    return column.endswith(_NON_CATEGORICAL_SUFFIXES) or any(
        token in column for token in _NON_CATEGORICAL_SUBSTRINGS
    )


def summarize_categoricals(
    gdf: pd.DataFrame,
    nmax: int = 100,
) -> dict[str, pd.Series]:
    """Value-count near-categorical columns.

    Auto-detects candidate categorical columns the same way
    ``notebooks/diagnostics/inspect_building_statistics.ipynb``'s own
    ``get_value_counts()`` did before this function existed: object/
    category dtype columns whose values repeat heavily
    (``duplicated(keep=False).mean() > 0.9``), excluding id-like/date-like/
    city/postal_code-suffixed columns (see :func:`_looks_non_categorical`)
    -- these are high-duplication but not meaningfully categorical. Any
    attribute already registered as categorical (see
    :func:`~openplaces.core.attribute_registry.get_categorical_attrs`,
    resolved through :func:`openplaces.recipe.resolve_attribute_name` to
    handle a provenance-suffixed column) is always included regardless of
    the duplication heuristic, since a sparse but genuinely categorical
    column (e.g. a rare occupancy class) would otherwise be missed.

    Parameters
    ----------
    gdf : DataFrame or GeoDataFrame
        Table to summarize.
    nmax : int
        Maximum number of distinct values to report per column.

    Returns
    -------
    dict[str, pandas.Series]
        Maps each detected column name to its ``value_counts().head(nmax)``.
    """
    from openplaces.core.attribute_registry import get_categorical_attrs
    from openplaces.recipe import resolve_attribute_name

    registered_categorical = get_categorical_attrs()

    result: dict[str, pd.Series] = {}
    for col in gdf.columns:
        if col == 'geometry':
            continue
        series = gdf[col]
        is_registered = resolve_attribute_name(col) in registered_categorical
        if is_registered:
            candidate = True
        elif series.dtype.name in ('object', 'category') and not (
            _looks_non_categorical(col)
        ):
            dup_ratio = series.duplicated(keep=False).mean() if len(series) else 0.0
            candidate = dup_ratio > 0.9
        else:
            candidate = False
        if candidate:
            result[col] = series.value_counts().head(nmax)
    return result


def check_geometry(gdf) -> dict:
    """Check a GeoDataFrame's geometry column for common defects.

    Flags exactly the two defect classes that (independently) crashed the
    harmonizer in production before being fixed: a degenerate non-polygon
    geometry (e.g. a 2-point ``LineString`` parcel boundary) mixed into an
    otherwise-polygon layer, and a null geometry -- both silently tolerated
    by a plain non-null check but fatal to ``geopandas.overlay`` (see
    :func:`openplaces.geo.polygon.overlay_polygons`, which now defensively
    drops both before any spatial join, rather than crashing the whole
    admin unit).

    Parameters
    ----------
    gdf : GeoDataFrame
        Table to check. Must have a ``geometry`` column.

    Returns
    -------
    dict
        ``n_total``, ``n_null`` (missing geometry), ``n_empty`` (present
        but empty, e.g. ``POLYGON EMPTY``), ``n_invalid`` (present,
        non-empty, but topologically invalid per ``geometry.is_valid``),
        and ``geom_type_counts`` (a :class:`pandas.Series` of
        ``geometry.geom_type`` value counts, including a count for null
        geometries).
    """
    geometry = gdf.geometry
    is_null = geometry.isna()
    is_present = ~is_null
    is_empty = is_present & geometry.is_empty
    is_valid = is_present & ~is_empty & geometry.is_valid
    n_invalid = int((is_present & ~is_empty & ~is_valid).sum())

    return {
        'n_total': len(gdf),
        'n_null': int(is_null.sum()),
        'n_empty': int(is_empty.sum()),
        'n_invalid': n_invalid,
        'geom_type_counts': geometry.geom_type.value_counts(dropna=False),
    }
