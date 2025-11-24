"""
Administration

Worldwide administrative referencing and mapping

- Manage global admin files
- Manage globally unique identifiers (admin_ids)
"""
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path

from openplaces.recipe import get_recipe

# # Regex patterns of ISO and HASC codes
# AA_AA = '^[A-Z]{2}\\-[A-Z]{2}$'
# AA_AA_EXTRACT = '^([A-Z]{2})\\-([A-Z]{2})$'
# HASC1 = '^[A-Z]{2}\\.[A-Z]{2}$'
# HASC2 = '^[A-Z]{2}\\.[A-Z]{2}\\.[A-Z]{2}$'


def get_admin0_iso():
    """Get dataframe with country ISO alpha codes and names"""

    ADMIN0_ISO_RENAME_COLUMNS = {
        'Country or Area': 'name',
        'ISO-alpha2 Code': 'admin0_id',
        'ISO-alpha3 Code': 'admin0_id_a3',
    }

    admin0_iso = (
        get_recipe(entity='admin', filename='admin0_iso_alpha_2', keep_default_na=False)
        .rename(columns=ADMIN0_ISO_RENAME_COLUMNS)
        .query('admin0_id != ""')
        .set_index('admin0_id')
        .sort_index()[['name', 'admin0_id_a3']]
    )

    admin0_iso_additions = get_recipe(
        None, 'admin', filename='admin0_iso_additions'
    ).set_index('admin0_id')
    admin0_iso = pd.concat(
        [
            admin0_iso,
            admin0_iso_additions[['name', 'admin0_id_a3']],
        ]
    )

    # Manual addition
    if admin0_iso.index.duplicated().any():
        raise Exception('admin0_iso index duplicated.')

    # Joining regional groupings
    admin0_iso_regions = (
        get_recipe(
            entity='admin',
            filename='admin0_iso3166-country-regions',
            keep_default_na=False,
        )
        .rename(columns={'alpha-2': 'admin0_id'})
        .set_index('admin0_id')[['region', 'sub-region', 'intermediate-region']]
    )

    for admin0_id_to, admin0_id_from in admin0_iso_additions[
        'admin0_id_copy_region'
    ].items():
        admin0_iso_regions.loc[admin0_id_to] = admin0_iso_regions.loc[admin0_id_from]

    # Ensure no duplicates were introduced
    if admin0_iso_regions.index.duplicated().any():
        raise Exception(
            'admin0_iso index duplicated: '
            + ', '.join(
                admin0_iso_regions.index[
                    admin0_iso_regions.index.duplicated(keep=False)
                ]
            )
        )

    admin0_iso = admin0_iso.join(admin0_iso_regions).sort_index()

    return admin0_iso


def index_from_admin0_id_a3(gdf):
    """Give dataframe `gdf` an `admin0_id` index from `admin0_id_a3`

    Single-use function to create linkage between GADM and ISO
    """

    admin0_id_a3_to_admin0_id = (
        get_admin0_iso().reset_index().set_index('admin0_id_a3')['admin0_id']
    )

    try:
        gdf_indexed = gdf.join(admin0_id_a3_to_admin0_id, on='admin0_id_a3').set_index(
            'admin0_id'
        )
    except:
        print(gdf)
        raise

    if gdf_indexed.index.isnull().any():
        print('Missing indices')
        print(gdf_indexed[gdf_indexed.index.isnull()])
        raise ValueError('Missing indices')
    return gdf_indexed


"""Get dataframe with state/province ISO3116-2 codes and names"""


def get_admin1_iso():
    """Get dataframe with state/province ISO3116-2 codes and names"""

    ADMIN1_ISO_RENAME_COLUMNS = {
        'country_code': 'admin0_id',
        'subdivision_name': 'name',
        'code': 'admin1_id_iso3166',
    }

    admin1_iso = (
        get_recipe(
            entity='admin',
            filename='admin1_iso3166-2_2021-03-01',
            keep_default_na=False,
        )
        .rename(columns=ADMIN1_ISO_RENAME_COLUMNS)
        .join(get_admin0_iso()['name'].rename('admin0_name'), on='admin0_id')
        .query('`admin1_id_iso3166` != "-"')
    )
    if admin1_iso['admin1_id_iso3166'].duplicated().any():
        raise Exception(
            'Unique ISO3116-2 code required in `openplaces.io.admin.get_admin1_iso()`.'
        )
    admin1_iso = admin1_iso.set_index('admin1_id_iso3166')[
        ['name', 'admin0_id', 'admin0_name']
    ].sort_index()
    return admin1_iso
