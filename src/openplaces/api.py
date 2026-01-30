"""
src/openplaces/api.py

Public API for openplaces.
"""


from openplaces.core.schema import AdminId
from openplaces.io import read_parquet
from openplaces.path import cache_path
from openplaces.recipe import get_recipe, get_recipe_by_id

ADMIN_SOURCE = 'admin-gadm-4~1'
ADMIN0_PRIMARY_COLUMNS = ['name', 'admin0_id_a3', 'lat', 'long', 'ha']
ADMIN1_PRIMARY_COLUMNS = [
    'name',
    'type',
    'admin0_name',
    'admin1_id_admin0',
    'lat',
    'long',
    'ha',
]
ADMIN2_PRIMARY_COLUMNS = [
    'name',
    'type',
    'admin1_name',
    'admin0_name',
    'lat',
    'long',
    'ha',
]


def get_admin0(admin_id=None, geom=False, all_columns=False):
    """Get units of admin level 0 (countries).

    Parameters
    ----------
    admin_id : str, list, or openplaces.core.schema.AdminId
        Admin unit identifier.
        Set to pick one or more countries ('CO' for Colombia)
    geom : bool
        If False or None, return DataFrame without geometries.
        If True, return GeoDataFrame with default polygon geometries.
    all_columns : bool
        If True, returns not only the most important columns
    """

    recipe = get_recipe(AdminId(), ADMIN_SOURCE, filename='admin0')

    parquet_path = cache_path(
        recipe['admin_id'],
        recipe['entity'],
        filename=recipe['cache_filename'] if 'cache_filename' in recipe else None,
    )

    admin0 = read_parquet(parquet_path, geom=geom)

    # Filter to AdminId
    if isinstance(admin_id, str):
        admin0 = admin0.loc[[admin_id]]
    elif isinstance(admin_id, AdminId):
        admin0 = admin0.loc[[str(admin_id)]]
    elif isinstance(admin_id, list):
        admin0 = admin0.loc[admin_id]
    elif admin_id is not None:
        raise ValueError(
            f'Type of `admin_id` not yet supported: {admin_id} type: {type(admin_id)}'
        )

    # Filter columns
    if not all_columns:
        admin0 = admin0[
            [x for x in ADMIN0_PRIMARY_COLUMNS + ['geometry'] if x in admin0]
        ]

    return admin0


def get_admin1(admin_id=None, geom=False, recipe=None, columns=None, all_columns=False):
    """Get units of admin level 1 (states / departments).

    Parameters
    ----------
    admin_id : str
        Admin unit identifier.
        Set to pick a single administrative unit ('US-TX' for Texas)
    geom : bool
        If False or None, return DataFrame without geometries.
        If True, return GeoDataFrame with default polygon geometries.
    recipe : str
        If a valid recipe, will use outcomes of that recipe.
    columns : list of str or None
        If a list of strings, will be used to select columns.
    all_columns : bool
        If True, returns not only the most important columns
    """

    if recipe is None:
        recipe = get_recipe(AdminId(), ADMIN_SOURCE, filename='admin1')

    parquet_path = cache_path(
        recipe['admin_id'],
        recipe['entity'],
        filename=recipe['cache_filename'] if 'cache_filename' in recipe else None,
    )
    admin1 = read_parquet(parquet_path, geom=geom)

    # Filter to AdminId
    if isinstance(admin_id, str):
        admin_id = AdminId(admin_id)
    if isinstance(admin_id, AdminId):
        if len(admin_id.levels) < 2:
            # Filter level-1 IDs for Admin0 Id
            admin1 = admin1.loc[
                [x for x in admin1.index if x.startswith(str(admin_id))]
            ]
        elif len(admin_id.levels) == 2:
            # Select admin1 level directly
            admin1 = admin1.loc[[str(admin_id)]]
        else:
            raise ValueError(f'`admin_id` has too many levels: {admin_id}.')
    elif isinstance(admin_id, list):
        admin1 = admin1.loc[admin_id]
    elif admin_id is not None:
        raise ValueError(
            f'Type of `admin_id` not yet supported: {admin_id} type: {type(admin_id)}'
        )

    # Filter columns
    if not all_columns or columns:
        if columns:
            if isinstance(columns, str):
                columns = [columns]
            elif not isinstance(columns, list):
                raise ValueError(f"`columns` must be a string or list: {columns}")
        columns_to_retain = columns if columns else ADMIN1_PRIMARY_COLUMNS
        admin1 = admin1[[x for x in columns_to_retain + ['geometry'] if x in admin1]]

    return admin1


def get_admin2(admin_id=None, geom=False, recipe=None, columns=None, all_columns=False):
    """Get units of admin level 2 (counties / municipalities).

    Parameters
    ----------
    admin_id : str
        Admin unit identifier.
        Set to pick a single administrative unit
        ('US-MA-MI' for Middlesex county, Massachusetts, US)
    geom : bool
        If False or None, return DataFrame without geometries.
        If True, return GeoDataFrame with default polygon geometries.
    recipe : str
        If a valid recipe, will use outcomes of that recipe.
    columns : list of str or None
        If a list of strings, will be used to select columns.
    all_columns : bool
        If True, returns all columns (not only primary ones)
    """

    if recipe is None:
        recipe = get_recipe(AdminId(), ADMIN_SOURCE, filename='admin2')

    parquet_path = cache_path(
        recipe['admin_id'],
        recipe['entity'],
        filename=recipe['cache_filename'] if 'cache_filename' in recipe else None,
    )

    admin2 = read_parquet(parquet_path, geom=geom)

    if isinstance(admin_id, str):
        admin_id = AdminId(admin_id)
    if isinstance(admin_id, AdminId):
        if len(admin_id.levels) < 3:
            # Filter level-2 IDs for AdminId
            admin2 = admin2.loc[
                [x for x in admin2.index if x.startswith(str(admin_id))]
            ]
        elif len(admin_id.levels) == 3:
            # Select admin1 level directly
            admin2 = admin2.loc[[str(admin_id)]]
        else:
            raise ValueError(f'`admin_id` has too many levels: {admin_id}.')
    elif isinstance(admin_id, list):
        admin2 = admin2.loc[admin_id]
    elif admin_id is not None:
        raise ValueError(
            f'Type of `admin_id` not yet supported: {admin_id} type: {type(admin_id)}'
        )

    # Filter columns
    if not all_columns or columns:
        if columns:
            if isinstance(columns, str):
                columns = [columns]
            elif not isinstance(columns, list):
                raise ValueError(f"`columns` must be a string or list: {columns}")
        columns_to_retain = columns if columns else ADMIN2_PRIMARY_COLUMNS
        admin2 = admin2[[x for x in columns_to_retain + ['geometry'] if x in admin2]]

    return admin2


def get_admin3(admin_id=None, geom=False, recipe=None, columns=None, all_columns=False):
    """Get units of admin level 2 (counties / municipalities).

    Parameters
    ----------
    admin_id : str
        Admin unit identifier.
        Set to pick a single administrative unit
        ('US-MA-MI' for Middlesex county, Massachusetts, US)
    geom : bool
        If False or None, return DataFrame without geometries.
        If True, return GeoDataFrame with default polygon geometries.
    recipe : str
        If a valid recipe, will use outcomes of that recipe.
    columns : str, list of str, or None
        If a list of strings, will be used to select columns.
    all_columns : bool
        If True, returns all columns (not only primary ones)
    """

    if recipe is None:
        raise NotImplementedError('No default recipe for `get_admin3()` defined yet.')

    parquet_path = cache_path(
        recipe['admin_id'],
        recipe['entity'],
        filename=recipe['cache_filename'] if 'cache_filename' in recipe else None,
    )

    admin3 = read_parquet(parquet_path, geom=geom)

    if isinstance(admin_id, str):
        admin_id = AdminId(admin_id)
    if isinstance(admin_id, AdminId):
        if len(admin_id.levels) < 4:
            # Filter level-3 IDs for AdminId
            admin3 = admin3.loc[
                [x for x in admin3.index if x.startswith(str(admin_id))]
            ]
        elif len(admin_id.levels) == 4:
            # Select admin1 level directly
            admin3 = admin3.loc[[str(admin_id)]]
        else:
            raise ValueError(f'`admin_id` has too many levels: {admin_id}.')
    elif isinstance(admin_id, list):
        admin3 = admin3.loc[admin_id]
    elif admin_id is not None:
        raise ValueError(
            f'Type of `admin_id` not yet supported: {admin_id} type: {type(admin_id)}'
        )

    # Filter columns
    if not all_columns or columns:
        if columns:
            if isinstance(columns, str):
                columns = [columns]
            elif not isinstance(columns, list):
                raise ValueError(f"`columns` must be a string or list: {columns}")
        columns_to_retain = columns if columns else ADMIN2_PRIMARY_COLUMNS
        admin3 = admin3[[x for x in columns_to_retain + ['geometry'] if x in admin3]]

    return admin3


def get_admin_by_level(level, *args, **kwargs):
    """Get units of selected administrative units by level

    Parameters
    ----------
    level : int
        Admin level (0: countries, 1: states/departments, 2: counties)
    args : list
        Positional arguments (passed on to get_admin)
    kwargs : list
        Keyword arguments (passed on to get_admin)
    """
    level = int(level)
    if level == 0:
        return get_admin0(*args, **kwargs)
    elif level == 1:
        return get_admin1(*args, **kwargs)
    elif level == 2:
        return get_admin2(*args, **kwargs)
    elif level == 3:
        return get_admin3(*args, **kwargs)


def get_admin_ids(admin_level, admin_id=None, admin_recipe=None):
    """Get list of administrative unit IDs"""
    admin_ids = get_admin_by_level(
        admin_level,
        admin_id,
        columns=[],
        recipe=get_recipe_by_id(admin_recipe) if admin_recipe is not None else None,
    ).index.tolist()
    return sorted(admin_ids)


def read_entities(admin_id, recipe, geom=False):
    """Generic function to load a processed Parquet table for entities

    Entities are administrative units (`admin`), parcels, buildings,
    transactions, etc., as defined by the `recipe`, and carry

    Parameters
    ----------
    admin_id : str or AdminId
        Administrative unit for which to load the data
    recipe : str or dict
        Recipe that defines the entity. Can be a loaded recipe (dict) or
        a string of the `recipe_id` (which includes admin_id)
    geom : bool
        If True, include geometries and return a GeoDataFrame
    """

    if isinstance(recipe, str):
        recipe = get_recipe_by_id(recipe)
    parquet_path = cache_path(admin_id, recipe['entity'])
    return read_parquet(parquet_path, geom=geom)
