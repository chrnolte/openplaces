"""
src/openplaces/api.py

Public API for openplaces.
"""

import warnings

import pandas as pd
import geopandas as gpd

from openplaces.core.schema import AdminId
from openplaces.io import read_parquet
from openplaces.path import cache_path
from openplaces.recipe import get_recipe, get_recipe_by_id

ADMIN_SOURCE_DEFAULT = 'admin-openplaces-2026'
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
    ],
}

# Deprecated - delete after transition
ADMIN_SOURCE = 'admin-gadm-4~1'
ADMIN1_PRIMARY_COLUMNS = ['name', 'admin1_id_a3', 'lat', 'long', 'ha']
ADMIN2_PRIMARY_COLUMNS = [
    'name',
    'type',
    'admin1_name',
    'admin2_id_admin1',
]
ADMIN3_PRIMARY_COLUMNS = [
    'name',
    'name_long',
    'type',
    'admin2_name',
    'admin1_name',
    'admin3_id_admin1',
]


def get_admin(
    admin_id=None,
    level=None,
    recipe=None,
    geom=False,
    columns=None,
    all_columns=False,
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
    """

    if level is None or level < 1:
        raise ValueError('The lowest level for admin units is 1. 0 is the planet.')

    # Pick default recipe for geometry attributes if None is provided
    if geom is True and recipe is None:
        recipe = f'{ADMIN_GEO_SOURCE_DEFAULT}_admin{level}'

    # Cast recipe to a dict if a str (id) is provided
    if isinstance(recipe, str):
        recipe = get_recipe_by_id(recipe)
    if isinstance(recipe, dict) and admin_id is None:
        admin_id = recipe['admin_id']

    # Default to countries
    if admin_id is None and level is None:
        level = 0

    # Cast scalar `admin_id` to list
    if isinstance(admin_id, (str, AdminId)):
        admin_id = [admin_id]
    elif admin_id is not None and not isinstance(admin_id, list):
        raise ValueError(f'AdminID not recognized: {type(admin_id)} {admin_id}.')

    # Cast all entries in the list to AdminId
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

        # Get lowest level of requested Admin IDs
        admin_id_level = max(_admin_id.get_level() for _admin_id in admin_ids)
        if not isinstance(level, int):
            level = admin_id_level
        if level < admin_id_level:
            # Clip admin_ids to upper level (= load parent admin units)
            admin_ids = [AdminId(*_admin_id.levels[:level]) for _admin_id in admin_ids]
            admin_ids = list(dict.fromkeys(admin_ids))
            print('Inferred Admin IDs: ' + admin_ids)

    if isinstance(recipe, dict):
        # Set recipe_parquet_path: will be used twice
        recipe_parquet_path = cache_path(
            recipe['admin_id'],
            recipe['entity'],
            filename=recipe['cache_filename'] if 'cache_filename' in recipe else None,
        )

    try:
        # Load admin spine from default source
        # `keep_default_na` is `True` to keep `NA` (Namibia)
        admin = get_recipe_by_id(
            f'{ADMIN_SOURCE_DEFAULT}_admin{level}', dtype=str, keep_default_na=False
        ).set_index(f'admin{level}_id')
    except IOError:
        # Should only happen before the spine exists.
        warnings.warn(
            f'Admin spine not found: {ADMIN_SOURCE_DEFAULT}_admin{level}. '
            'Falling back to `admin_recipe`.'
        )
        if isinstance(recipe, dict):
            # Load spine from recipe
            admin = pd.read_parquet(recipe_parquet_path)[[]]
        else:
            raise IOError(
                f'Admin spine not found: {ADMIN_SOURCE_DEFAULT}_admin{level}.'
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
            txt_warnings = (
                '\n\nAdmin IDs from recipe `'
                + (
                    f'{recipe["admin_id"]}_'
                    if recipe["admin_id"].get_level() > 1
                    else ''
                )
                + f'{recipe["entity"]}_admin{level}`'
                ' not found in reference Admin IDs:\n\n'
                + ', '.join(admin_ids_to_add_to_spine)
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
        admin = admin[mask_select].copy()

    if isinstance(recipe, dict):
        if admin_id is None:
            filters = None
        else:
            filters = [(f'admin{level}_id', 'in', sorted(set(admin.index)))]

        # Read attribute data from filesystem
        admin_from_recipe = read_parquet(
            recipe_parquet_path, geom=geom, filters=filters
        )

        # Join recipe data to spine, overwriting columns from spine
        admin = admin.drop(columns=set(admin) & set(admin_from_recipe)).join(
            admin_from_recipe, how='outer'
        )
        # Cast to GeoDataFrame if geometries are included
        if isinstance(admin_from_recipe, gpd.GeoDataFrame):
            admin = gpd.GeoDataFrame(admin, crs=admin_from_recipe.crs)

    # Filter columns
    if not all_columns or columns:
        if columns:
            if isinstance(columns, str):
                columns = [columns]
            elif not isinstance(columns, list):
                raise ValueError(f"`columns` must be a string or list: {columns}")
        columns_to_retain = columns if columns else ADMIN_PRIMARY_COLUMNS[level]
        admin = admin[
            [x for x in columns_to_retain + ['geometry'] if x in admin]
        ].copy()

    return admin


def get_admin_ids(admin_level, admin_id=None, admin_recipe=None):
    """Get list of administrative unit IDs"""
    admin_ids = get_admin(
        admin_id,
        admin_level,
        columns=[],
        recipe=get_recipe_by_id(admin_recipe) if admin_recipe is not None else None,
    ).index.tolist()
    return sorted(admin_ids)


def read_entities(recipe, admin_id=None, geom=False):
    """Generic function to load a processed Parquet table for entities

    Entities are administrative units (`admin`), parcels, buildings,
    transactions, etc., as defined by the `recipe`, and carry

    Parameters
    ----------
    recipe : str or dict
        Recipe that defines the entity. Can be a loaded recipe (dict) or
        a string of the `recipe_id` (which includes admin_id)
    admin_id : str or AdminId
        Administrative unit for which to load the data. If None,
        choose admin_id of recipe.
    geom : bool
        If True, include geometries and return a GeoDataFrame
    """

    if isinstance(recipe, str):
        recipe = get_recipe_by_id(recipe)

    if admin_id is None:
        admin_id = recipe['admin_id']

    filename = recipe['cache_filename'] if 'cache_filename' in recipe else None
    parquet_path = cache_path(
        admin_id,
        recipe['entity'],
        filename=filename,
    )
    return read_parquet(parquet_path, geom=geom)
