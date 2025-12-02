"""
src/openplaces/api.py

Public API for openplaces.
"""

import geopandas as gpd
import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.path import cache_path
from openplaces.recipe import get_recipe

ADMIN0_SOURCE = 'admin0-gadm-4~1'
ADMIN1_SOURCE = 'admin1-gadm-4~1'
ADMIN2_SOURCE = 'admin2-gadm-4~1'

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

    recipe = get_recipe(AdminId(), 'admin', source=ADMIN0_SOURCE)

    parquet_path = cache_path(
        recipe['admin_id'],
        recipe['entity'],
        filename=recipe['cache_filename'] if 'cache_filename' in recipe else None,
    )

    admin0 = pd.read_parquet(parquet_path)

    if geom:
        geo_parquet_path = cache_path(
            recipe['admin_id'],
            recipe['entity'],
            filename=(recipe['cache_filename'] if 'cache_filename' in recipe else '')
            + '_geo',
        )

        # Join polygons to table (to keep CRS), then rearrange columns
        admin0 = gpd.read_parquet(geo_parquet_path).join(admin0)[
            list(admin0.columns) + ['geometry']
        ]

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
        recipe = get_recipe(AdminId(), 'admin', source=ADMIN1_SOURCE)

    parquet_path = cache_path(
        recipe['admin_id'],
        recipe['entity'],
        filename=recipe['cache_filename'] if 'cache_filename' in recipe else None,
    )

    admin1 = pd.read_parquet(parquet_path)

    if geom:
        parquet_geo_path = cache_path(
            recipe['admin_id'],
            recipe['entity'],
            filename=(recipe['cache_filename'] if 'cache_filename' in recipe else '')
            + '_geo',
        )

        # Join polygons to table (keeping CRS), move geometry to end
        admin1 = gpd.read_parquet(parquet_geo_path).join(admin1)[
            list(admin1.columns) + ['geometry']
        ]

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
            if not isinstance(columns, list):
                raise ValueError(f"`columns` must be a list: {columns}")
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
        recipe = get_recipe(AdminId(), 'admin', source=ADMIN2_SOURCE)

    parquet_path = cache_path(
        recipe['admin_id'],
        recipe['entity'],
        filename=recipe['cache_filename'] if 'cache_filename' in recipe else None,
    )

    admin2 = pd.read_parquet(parquet_path)

    if geom:
        parquet_geo_path = cache_path(
            recipe['admin_id'],
            recipe['entity'],
            filename=(recipe['cache_filename'] if 'cache_filename' in recipe else '')
            + '_geo',
        )

        # Join polygons to table (keeping CRS), move geometry to end
        admin2 = gpd.read_parquet(parquet_geo_path).join(admin2)[
            list(admin2.columns) + ['geometry']
        ]

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
            if not isinstance(columns, list):
                raise ValueError(f"`columns` must be a list: {columns}")
        columns_to_retain = columns if columns else ADMIN2_PRIMARY_COLUMNS
        admin2 = admin2[[x for x in columns_to_retain + ['geometry'] if x in admin2]]

    return admin2
