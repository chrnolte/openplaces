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


def get_admin0(admin_id=None, geom=False):
    """Get units of admin level 0 (countries).

    Parameters
    ----------
    admin_id : str
        Admin unit identifier.
        Set to pick a single country ('CO' for Colombia)
    geom : bool
        If False or None, return DataFrame without geometries.
        If True, return GeoDataFrame with default polygon geometries.
    """

    recipe = get_recipe(AdminId(), 'admin', source=ADMIN0_SOURCE)

    parquet_path = cache_path(
        recipe['admin_id'],
        recipe['entity'],
        filename=recipe['cache_filename'] if 'cache_filename' in recipe else None,
    )

    admin0 = pd.read_parquet(parquet_path)

    if geom:
        parquet_geo_path = cache_path(
            recipe['admin_id'],
            recipe['entity'],
            filename=(recipe['cache_filename'] if 'cache_filename' in recipe else '')
            + '_geo',
        )

        # Join polygons to table (keeping CRS), move geometry to end
        admin0 = gpd.read_parquet(parquet_geo_path).join(admin0)[
            list(admin0.columns) + ['geometry']
        ]

    if isinstance(admin_id, str):
        admin0 = admin0.loc[[admin_id]]
    elif isinstance(admin_id, AdminId):
        admin0 = admin0.loc[[str(AdminId(admin_id))]]

    return admin0


def get_admin1(admin_id=None, geom=False):
    """Get units of admin level 0 (countries).

    Parameters
    ----------
    admin_id : str
        Admin unit identifier.
        Set to pick a single administrative unit ('US-TX' for Texas)
    geom : bool
        If False or None, return DataFrame without geometries.
        If True, return GeoDataFrame with default polygon geometries.
    """

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

    if isinstance(admin_id, str):
        admin1 = admin1.loc[[admin_id]]
    elif isinstance(admin_id, AdminId):
        admin1 = admin1.loc[[str(admin_id)]]

    return admin1


def get_admin2(admin_id=None, geom=False):
    """Get units of admin level 0 (countries).

    Parameters
    ----------
    admin_id : str
        Admin unit identifier.
        Set to pick a single administrative unit
        ('US-MA-MI' for Middlesex county, Massachusetts, US)
    geom : bool
        If False or None, return DataFrame without geometries.
        If True, return GeoDataFrame with default polygon geometries.
    """

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
        admin2 = admin2.loc[[admin_id]]
    elif isinstance(admin_id, AdminId):
        admin2 = admin2.loc[[str(admin_id)]]

    return admin2
