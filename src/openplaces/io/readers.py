"""
Low-level data-access functions for admin units and entities.
Imported by internal modules (io/*, geo/*). Also re-exported from api.py.
"""

import warnings
from collections.abc import Sequence

import geopandas as gpd
import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.io import read_parquet
from openplaces.recipe import (
    find_admin_recipe_id,
    get_output_path,
    get_recipe_by_id,
    get_save_admin_level,
    get_table_recipe,
)
from openplaces.utils import format_list

ADMIN_SOURCE_DEFAULT = 'admin-spine-2026'
REGION_SOURCE_DEFAULT = 'admin-regions-2026'
ADMIN_GEO_SOURCE_DEFAULT = 'admin-gadm-4~1'
ADMIN_PRIMARY_COLUMNS = {
    1: ['name', 'admin1_id_a3'],
    2: [
        'name',
        'type',
        'admin1_name',
        'admin2_id_admin1',
    ],
    3: [
        'name',
        'name_long',
        'type',
        'admin2_name',
        'admin1_name',
        'admin3_id_admin1',
    ],
    4: [
        'name',
        'name_long',
        'type',
        'admin3_name',
        'admin2_name',
        'admin1_name',
        'admin4_id_admin1',
    ],
}


def get_admin(
    admin_id=None,
    level=None,
    recipe=None,
    geom=False,
    columns=None,
    all_columns=False,
    silent=True,
):
    """Get admin units of any administrative level

    Parameters
    ----------
    admin_id : str, list, or openplaces.core.schema.AdminId
        Identifier(s) of admin units to return.
        Can include higher-level Admin IDs to select many lower levels
    level : int
        Admin level for which to return units.
        If none, use level of `admin_id` (deepest if a list is passed).
    recipe : str
        Use this recipe to import geometries and additional attributes.
    geom : bool
        If False or None, return DataFrame without geometries.
        If True, return GeoDataFrame with geometries.
    columns : list of str or None
        If a list of strings, will be used to select columns.
    all_columns : bool
        If True, returns not only the most important columns
    silent : True
        Silence warnings
    geom : bool or 'simplified'
        If False, return a DataFrame without geometries.
        If True, return a GeoDataFrame with full geometries.
        If ``'simplified'``, return a GeoDataFrame with simplified geometries
        from the ``_geo_simplified`` companion file written by
        ``AdminHarmonizer``.
    """

    if level is not None and level < 1:
        raise ValueError('The lowest level for admin units is 1. 0 is the planet.')

    # Cast scalar `admin_id` to list
    if isinstance(admin_id, str | AdminId):
        admin_id = [admin_id]
    elif admin_id is not None and not isinstance(admin_id, list):
        raise ValueError(f'AdminID not recognized: {type(admin_id)} {admin_id}.')

    # Cast all entries in the list to AdminId to get level for the recipe
    if admin_id is not None:
        admin_ids = []
        for _admin_id in admin_id:
            if isinstance(_admin_id, AdminId):
                admin_ids += [_admin_id]
            elif isinstance(_admin_id, str):
                admin_ids += [AdminId(_admin_id)]
            else:
                raise ValueError(
                    f'AdminID not recognized: {type(_admin_id)} {_admin_id}.'
                )

    # Cast recipe to a dict if a str (id) is provided
    if isinstance(recipe, str):
        recipe = get_recipe_by_id(recipe)
    if isinstance(recipe, dict) and admin_id is None:
        admin_id = recipe['admin_id']
        admin_ids = [admin_id]

    # Try to infer level from recipe if not explicitly specified
    if not isinstance(level, int) and isinstance(recipe, dict):
        filename = recipe.get('save_to', {}).get('filename', '')
        if filename.startswith('admin') and filename[5:].isdigit():
            level = int(filename[5:])
        elif 'recipe_id' in recipe:
            parts = recipe['recipe_id'].split('_')
            for part in parts:
                if part.startswith('admin') and part[5:].isdigit():
                    level = int(part[5:])
                    break

    # If level is still not specified, use deep level from admin_ids
    if not isinstance(level, int) and admin_id is not None:
        admin_id_level = max(_admin_id.get_level() for _admin_id in admin_ids)
        level = admin_id_level

    # Pick default recipe for geometry attributes if None is provided
    if geom and recipe is None:
        if level is None:
            # Default to countries
            level = 1
        # Prefer an in-country recipe when all requested admin IDs share the
        # same country; fall back to the global GADM default otherwise.
        in_country_recipe_id = None
        if admin_id is not None:
            country_ids = list(
                dict.fromkeys(str(AdminId(_aid.levels[0])) for _aid in admin_ids)
            )
            if len(country_ids) == 1:
                in_country_recipe_id = find_admin_recipe_id(
                    country_ids[0], level, silent=True
                )
        recipe = in_country_recipe_id or f'{ADMIN_GEO_SOURCE_DEFAULT}_admin{level}'
        if isinstance(recipe, str):
            recipe = get_recipe_by_id(recipe)

    if admin_id is None and level is None:
        # Default to countries
        level = 1

    if admin_id is not None:
        # Recompute level (as it could have been set by recipe admin ID)
        admin_id_level = max(_admin_id.get_level() for _admin_id in admin_ids)

        if level < admin_id_level:
            # Clip admin_ids to upper level (= load parent admin units)
            admin_ids = [AdminId(*_admin_id.levels[:level]) for _admin_id in admin_ids]
            admin_ids = list(dict.fromkeys(admin_ids))
            if not silent:
                print('Inferred Admin IDs: ' + admin_ids)

    if isinstance(recipe, dict):
        # Set recipe_parquet_path: will be used twice
        recipe_parquet_path = get_output_path(recipe, recipe['admin_id'])

    try:
        # Load admin spine from default source
        # `keep_default_na` is `True` to keep `NA` (Namibia)
        admin = get_recipe_by_id(
            f'{ADMIN_SOURCE_DEFAULT}_admin{level}', dtype=str, keep_default_na=False
        )
        if f'admin{level}_id' not in admin:
            raise ValueError(f"'admin{level}_id' does not exist:\n\n" + str(admin))

        admin = admin.set_index(f'admin{level}_id')

    except OSError:
        # Should only happen before the spine exists.
        warnings.warn(
            f'\n\nAdmin spine not found: {ADMIN_SOURCE_DEFAULT}_admin{level}.\n\n'
            'Falling back to `admin_recipe`.\n'
        )
        if isinstance(recipe, dict):
            # Load spine from recipe
            admin = pd.read_parquet(recipe_parquet_path)[[]]
        else:
            raise OSError(
                f'\n\nAdmin spine not found: {ADMIN_SOURCE_DEFAULT}_admin{level}.\n'
            )

    if isinstance(recipe, dict):
        # If a recipe was provided, check whether it contains new IDs

        # Get path of recipe data in filesystem
        admin_ids_in_spine = list(admin.index)

        # Read only the ID column
        admin_ids_from_recipe = read_parquet(recipe_parquet_path, columns=[]).index
        admin_ids_to_add_to_spine = sorted(
            set(admin_ids_from_recipe) - set(admin_ids_in_spine)
        )
        if admin_ids_to_add_to_spine:
            if not silent:
                txt_warnings = (
                    f'\n\n{len(admin_ids_to_add_to_spine):,d} admin IDs from recipe `'
                    + (
                        f'{recipe["admin_id"]}_'
                        if recipe['admin_id'].get_level() > 1
                        else ''
                    )
                    + f'{recipe.get("entity") or recipe.get("dataset")}_admin{level}`'
                    ' not found in reference Admin IDs:\n\n- '
                    + '\n- '.join(admin_ids_to_add_to_spine[:5])
                    + ('\n- ...' if len(admin_ids_to_add_to_spine) > 5 else '')
                    + '\n\nOptions to silence this warning:\n'
                    '- Add Admin IDs to reference list in '
                    f'`{ADMIN_SOURCE_DEFAULT}_admin{level}.csv`\n'
                    '- Call get_admin() with silent=True.\n'
                )
                warnings.warn(txt_warnings)

            admin_to_add = pd.DataFrame(
                index=pd.Index(admin_ids_to_add_to_spine, name=f'admin{level}_id'),
                columns=admin.columns,
            )
            admin = pd.concat([admin, admin_to_add]).sort_index()

    # Select admin units
    if admin_id is not None:
        mask_select = pd.Series(False, index=admin.index)
        for _admin_id_to_get in admin_ids:
            mask_select |= admin.index.str.startswith(str(_admin_id_to_get))
        if not mask_select.any():
            raise ValueError(
                'No admin IDs from reference spine found. Perhaps they have not been '
                f'defined at level {level} for `admin_id`: {format_list(admin_ids)}?'
            )
        admin = admin[mask_select].copy()

    if isinstance(recipe, dict):
        if admin_id is None or not mask_select.any():
            filters = None
        else:
            filters = [(f'admin{level}_id', 'in', sorted(set(admin.index)))]

        # Read attribute data from filesystem
        admin_from_recipe = read_parquet(
            recipe_parquet_path, geom=geom, filters=filters
        )

        # Get column order (retain admin, add recipe)
        column_order = list(dict.fromkeys(list(admin) + list(admin_from_recipe)))

        # Join recipe data to spine, overwriting columns from spine
        shared_columns = list(set(admin) & set(admin_from_recipe))
        # Fill empty recipe data with available data from spine
        admin_from_recipe[shared_columns] = admin_from_recipe[shared_columns].fillna(
            admin[shared_columns]
        )
        admin = admin.drop(columns=shared_columns).join(admin_from_recipe, how='outer')[
            column_order
        ]
        # Cast to GeoDataFrame if geometries are included
        if isinstance(admin_from_recipe, gpd.GeoDataFrame):
            admin = gpd.GeoDataFrame(admin, crs=admin_from_recipe.crs)

    # Filter columns
    if not all_columns or columns:
        if columns:
            if isinstance(columns, str):
                columns = [columns]
            elif not isinstance(columns, list):
                raise ValueError(f'`columns` must be a string or list: {columns}')
        columns_to_retain = columns if columns else ADMIN_PRIMARY_COLUMNS[level]
        admin = admin[
            [x for x in columns_to_retain + ['geometry'] if x in admin]
        ].copy()

    # Return without empty columns
    column_is_empty = admin.eq('').all() | admin.isnull().all()
    non_empty_columns = list(column_is_empty[~column_is_empty].index)
    return admin[non_empty_columns]


def get_admin_ids(admin_level, admin_id=None, admin_recipe=None):
    """Get list of administrative unit IDs"""
    admin_ids = get_admin(
        admin_id,
        admin_level,
        columns=[],
        recipe=get_recipe_by_id(admin_recipe) if admin_recipe is not None else None,
    ).index.tolist()
    return sorted(admin_ids)


def get_regions(region_id=None):
    """Get the named-region registry: a 1:n mapping of region to admin unit.

    A region is any named group of admin units that the admin hierarchy
    cannot express -- a study area, a delivery footprint, a funder's
    geography. The CHEER regions are the motivating case: 45 of North
    Carolina's 100 counties and 42 of Texas's 254, neither of them a
    complete state.

    Kept in one registry rather than beside whichever recipe first needed
    it, because the same grouping is wanted by delivery, by mapping, and by
    ad-hoc analysis, and three copies of a county list drift.

    Parameters
    ----------
    region_id : str, optional
        Return only this region's rows. Raises `KeyError` when it is not
        registered, listing what is.

    Returns
    -------
    pandas.DataFrame
        Columns `region_id`, `name`, `region_admin_id` (the admin unit the
        region rolls up to, which may be blank) and `admin_id`, one row per
        member unit.
    """
    regions = get_recipe_by_id(REGION_SOURCE_DEFAULT, dtype=str, keep_default_na=False)
    if region_id is None:
        return regions
    rows = regions[regions['region_id'] == str(region_id)]
    if rows.empty:
        known = format_list(sorted(regions['region_id'].unique()))
        raise KeyError(f'Unknown region {region_id!r}; registered: {known}.')
    return rows


def get_region_admin_ids(region_id):
    """Get the admin unit IDs a named region groups, in registry order."""
    return list(dict.fromkeys(get_regions(region_id)['admin_id'].dropna()))


def _as_admin_id(value):
    return value if isinstance(value, AdminId) else AdminId(value)


def _get_output_admin_ids(recipe, admin_id):
    """Resolve requested admin IDs to the recipe's output granularity.

    Returns ``(output_admin_ids, finer_requested_ids)``: the deduped save-level
    AdminIds whose files must be read, and the subset of originally requested
    AdminIds strictly finer than the recipe's save level -- each such saved
    file may cover admin units the caller didn't ask for (see get_entities'
    post-read filtering).
    """
    save_level = get_save_admin_level(recipe)
    if save_level == 0:
        return [None], []

    recipe_admin_id = _as_admin_id(recipe['admin_id'])
    requested = recipe_admin_id if admin_id is None else admin_id
    if isinstance(requested, str | AdminId):
        requested = [requested]

    output_admin_ids = []
    finer_requested_ids = []
    for value in requested:
        requested_admin_id = _as_admin_id(value)
        if requested_admin_id.is_parent_or_equal_of(recipe_admin_id):
            requested_admin_id = recipe_admin_id
        elif not recipe_admin_id.is_parent_or_equal_of(requested_admin_id):
            raise ValueError(
                f'{requested_admin_id} is outside recipe scope {recipe_admin_id}'
            )

        if requested_admin_id.get_level() >= save_level:
            output_admin_ids.append(AdminId(*requested_admin_id.levels[:save_level]))
            if requested_admin_id.get_level() > save_level:
                finer_requested_ids.append(requested_admin_id)
            continue

        child_admin_ids = get_admin(
            requested_admin_id,
            save_level,
            columns=[],
        ).index
        output_admin_ids.extend(
            _as_admin_id(child_admin_id)
            for child_admin_id in child_admin_ids
            if recipe_admin_id.is_parent_or_equal_of(_as_admin_id(child_admin_id))
        )

    return list(dict.fromkeys(output_admin_ids)), finer_requested_ids


def get_entities(
    recipe,
    admin_id=None,
    geom=False,
    layer=None,
    partition_id=None,
    columns: Sequence[str] | None = None,
    missing='raise',
    bbox: tuple[float, float, float, float] | None = None,
):
    """Load and combine processed Parquet tables for entities.

    Parent administrative IDs are expanded to the recipe's save level.
    Recipes aggregated to a single file read their ``_all`` output by
    default.

    Parameters
    ----------
    recipe : str or dict
        Recipe that defines the entity.
    admin_id : str, AdminId, or sequence
        Administrative unit(s) to load. Defaults to the recipe admin ID.
    geom : bool or 'simplified'
        If True, include geometries and return a GeoDataFrame. If
        ``'simplified'``, join simplified geometries from the
        ``_geo_simplified`` sidecar where one exists (see
        :func:`openplaces.io.read_parquet`).
    layer : str, optional
        Secondary layer defined in ``additional_layers``.
    partition_id : str, optional
        Explicit partition value to read.
    columns : sequence of str, optional
        Columns to read.
    missing : {'raise', 'warn', 'ignore'}
        How to handle missing output files.
    bbox : tuple of (minx, miny, maxx, maxy), optional
        Spatial bounding box filter in EPSG:4326, forwarded to
        :func:`openplaces.io.read_parquet` for each resolved output file.
        Exploits per-file covering-bbox predicate pushdown, so files whose
        extent doesn't overlap contribute no rows to the combined result —
        this bounds memory use when loading a recipe across many
        administrative units without loading every file in full.
    """
    if missing not in {'raise', 'warn', 'ignore'}:
        raise ValueError("missing must be 'raise', 'warn', or 'ignore'")
    if isinstance(recipe, str):
        recipe = get_recipe_by_id(recipe)

    if layer is not None:
        recipe = get_table_recipe(recipe, layer)

    if partition_id is None and (recipe.get('aggregate_by') or {}).get('single_file'):
        partition_id = 'all'

    save_level = get_save_admin_level(recipe)
    admin_col = f'admin{save_level}_id' if save_level > 0 else None
    read_columns = columns
    if admin_col is not None and columns is not None:
        if isinstance(columns, str):
            columns = [columns]
        if admin_col in columns:
            read_columns = [col for col in columns if col != admin_col]
        else:
            admin_col = None

    output_admin_ids, finer_requested_ids = _get_output_admin_ids(recipe, admin_id)
    finer_by_level: dict[int, set[str]] = {}
    for finer_admin_id in finer_requested_ids:
        finer_by_level.setdefault(finer_admin_id.get_level(), set()).add(
            str(finer_admin_id)
        )

    frames = []
    output_paths = []
    missing_paths = []
    for output_admin_id in output_admin_ids:
        path = get_output_path(
            recipe,
            output_admin_id,
            partition_id=partition_id,
        )
        if not path.exists():
            missing_paths.append(path)
            continue
        df = read_parquet(path, geom=geom, columns=read_columns, bbox=bbox)
        if admin_col is not None and output_admin_id is not None:
            if admin_col not in df.columns:
                df[admin_col] = str(output_admin_id)
        frames.append(df)
        output_paths.append(path)

    if missing_paths:
        message = (
            f'{len(missing_paths)} recipe output file(s) do not exist; '
            f'first missing path: {missing_paths[0]}'
        )
        if missing == 'raise':
            raise FileNotFoundError(message)
        if missing == 'warn':
            warnings.warn(message, stacklevel=2)

    if not frames:
        data = gpd.GeoDataFrame() if geom else pd.DataFrame()
    elif len(frames) == 1:
        data = frames[0]
    else:
        ignore_index = all(isinstance(frame.index, pd.RangeIndex) for frame in frames)
        data = pd.concat(frames, ignore_index=ignore_index)
        if geom:
            data = gpd.GeoDataFrame(data, crs=frames[0].crs)

    if admin_col is not None and admin_col in data.columns:
        data[admin_col] = data[admin_col].astype('category')

    # A requested admin_id finer than the recipe's save level (e.g. a town
    # when the recipe saves one file per county) gets truncated to its
    # save-level ancestor above, so a single read file can cover admin units
    # the caller didn't ask for. Narrow back down to exactly what was
    # requested, using a stored per-row id column when the recipe happens to
    # carry one, or a spatial fallback (join to just the requested units'
    # boundary polygons) otherwise.
    for level, ids in finer_by_level.items():
        if data.empty:
            break
        col = f'admin{level}_id'
        if col in data.columns:
            data = data[data[col].astype(str).isin(ids)]
        elif geom:
            from openplaces.geo.overlay import overlay_admin_ids

            boundaries = get_admin(sorted(ids), level, geom=True)['geometry']
            data = overlay_admin_ids(data, admin_geometries=boundaries)
            data = data[data[col].astype(str).isin(ids)]
        else:
            warnings.warn(
                f'Cannot restrict output to the requested admin{level} IDs: '
                f"this recipe's saved data has no {col!r} column and "
                'geom=False rules out a spatial fallback; returning '
                f'unfiltered admin{save_level}-level data instead.',
                stacklevel=2,
            )

    partition_ids = set()
    if partition_id == 'all':
        from openplaces.io.aggregate import read_partition_coverage

        for path in output_paths:
            partition_ids.update(read_partition_coverage(path))

    data.attrs['openplaces_output_paths'] = [str(path) for path in output_paths]
    data.attrs['openplaces_missing_paths'] = [str(path) for path in missing_paths]
    data.attrs['openplaces_partition_ids'] = sorted(partition_ids)
    return data


def get_dataset(recipe, admin_id=None, partition_id=None, geom=False):
    """Load a processed dataset by recipe.

    Handles both raster and tabular dataset recipes. For raster datasets
    (Cloud Optimized GeoTIFFs written by `fetch_rasters_by_admin`), returns
    the path to the .tif file so the caller controls resource management
    (e.g. with rasterio or xarray). For tabular datasets, returns a
    DataFrame or GeoDataFrame exactly as `get_entities` does.

    Parameters
    ----------
    recipe : str or dict
        Recipe that defines the dataset. Can be a loaded recipe (dict) or
        a string recipe ID.
    admin_id : str or AdminId, optional
        Administrative unit for which to load the data. If None, uses
        the admin_id from the recipe.
    partition_id : str, optional
        Partition value to locate a specific partition file, e.g. '2020'
        for a year-partitioned recipe. Pass None (default) for recipes
        without partitioning.
    geom : bool
        If True, include geometries and return a GeoDataFrame.
        Ignored for raster datasets.

    Returns
    -------
    Path
        Path to the .tif file, for raster datasets.
    pandas.DataFrame or geopandas.GeoDataFrame
        Loaded tabular data, for non-raster datasets.

    Raises
    ------
    ValueError
        If the recipe does not have a 'dataset' key. Use `get_entities`
        for entity recipes.
    """

    if isinstance(recipe, str):
        recipe = get_recipe_by_id(recipe)

    if 'dataset' not in recipe:
        raise ValueError(
            "Recipe does not have a 'dataset' key. "
            'Use `get_entities` for entity recipes.'
        )

    if admin_id is None:
        admin_id = recipe['admin_id']

    output_path = get_output_path(recipe, admin_id, partition_id=partition_id)

    if recipe['dataset'].is_raster:
        return output_path

    return read_parquet(output_path, geom=geom)
