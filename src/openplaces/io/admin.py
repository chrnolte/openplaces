"""
Administration

Worldwide administrative referencing and mapping

- Manage global admin files
- Manage globally unique identifiers (admin_ids)
"""

import glob
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from openplaces.api import get_admin1, get_admin2
from openplaces.core.constants import (
    ADMIN0_ID_HASC1_A2,
    REGEX_ADMIN1_IDS_AA_AA,
    REGEX_ADMIN1_IDS_AA_AA_EXTRACT,
    REGEX_ADMIN1_IDS_HASC,
    REGEX_ADMIN2_IDS_HASC,
    STRING_SEPARATOR_WITHIN_IDS,
)
from openplaces.path import recipe_path
from openplaces.recipe import get_recipe, get_recipe_by_id
from openplaces.utils import create_comparable_name_link, standardize_names


# Admin 0: Countries
def get_admin0_iso():
    """Get dataframe with country ISO alpha codes and names"""

    ADMIN0_ISO_RENAME_COLUMNS = {
        'Country or Area': 'name',
        'ISO-alpha2 Code': 'admin0_id',
        'ISO-alpha3 Code': 'admin0_id_a3',
    }

    admin0_iso = (
        get_recipe(None, 'admin-iso', filename='admin0-alpha-2', keep_default_na=False)
        .rename(columns=ADMIN0_ISO_RENAME_COLUMNS)
        .query('admin0_id != ""')
        .set_index('admin0_id')
        .sort_index()[['name', 'admin0_id_a3']]
    )

    admin0_iso_additions = get_recipe(
        None, 'admin-iso', filename='admin0-additions'
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
            None,
            'admin-iso',
            filename='admin0-regions-iso3166',
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


def admin0_id_index_from_admin0_id_a3(gdf):
    """Give dataframe `gdf` an `admin0_id` index from `admin0_id_a3`

    Single-use function to create linkage between GADM and ISO
    """

    admin0_id_a3_to_admin0_id = (
        get_admin0_iso().reset_index().set_index('admin0_id_a3')['admin0_id']
    )

    gdf_indexed = gdf.join(admin0_id_a3_to_admin0_id, on='admin0_id_a3').set_index(
        'admin0_id'
    )

    if gdf_indexed.index.isnull().any():
        print('Missing indices')
        print(gdf_indexed[gdf_indexed.index.isnull()])
        raise ValueError('Missing indices')
    return gdf_indexed


# Admin 1: States / provinces


def get_admin1_iso():
    """Get dataframe with state/province ISO3116-2 codes and names"""

    ADMIN1_ISO_RENAME_COLUMNS = {
        'country_code': 'admin0_id',
        'subdivision_name': 'name',
        'code': 'admin1_id_iso3166',
    }

    admin1_iso = (
        get_recipe(
            None,
            'admin-iso-20210301',
            filename='admin1-iso3166-2',
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


def admin1_id_index_from_admin1_gadm(admin1):
    """Give dataframe `admin` an `admin1_id` index based on GADM data"""

    # Join with level-2 administrative units
    admin1 = admin1.join(
        get_admin0_iso().reset_index().set_index('admin0_id_a3')['admin0_id'],
        on='admin0_id_a3',
        how='inner',
    )
    if admin1['admin0_id'].isnull().any():
        raise ValueError('Empty `\'admin0_id\'` found in `admin1`.')

    admin1['_name'] = admin1['name'].apply(standardize_names).fillna('NA')

    # Read ISO
    admin1_iso = get_admin1_iso()
    admin1_iso['_name'] = admin1_iso['name'].apply(standardize_names)
    admin1_iso_join = admin1_iso.reset_index().set_index(['admin0_id', '_name'])
    admin1 = admin1.join(
        admin1_iso_join['admin1_id_iso3166'], on=['admin0_id', '_name']
    )

    # Initiate empty admin ID
    admin1['admin1_id'] = pd.Series(None, dtype='object')
    admin1['admin1_id_source'] = pd.Series(None, dtype='object')

    # First priority: official ISO3166-2 codes
    i = admin1['admin1_id_iso3166'].str.contains(REGEX_ADMIN1_IDS_AA_AA, na=False)
    hasc_from_iso = admin1[i]['admin1_id_iso3166'].str.extract(
        REGEX_ADMIN1_IDS_AA_AA_EXTRACT
    )
    admin1.loc[i, 'admin1_id'] = hasc_from_iso.apply(
        STRING_SEPARATOR_WITHIN_IDS.join, 1
    )
    admin1.loc[i, 'admin1_id_source'] = 'iso'

    # Second priority: existing HASC codes that are unique,
    # not already used by ISO3166-2, and use correct country-level code
    admin1_id_from_hasc1 = admin1['admin1_id_hasc'].str.replace(
        '.', STRING_SEPARATOR_WITHIN_IDS, regex=False
    )
    i = (
        admin1['admin1_id'].isnull()
        & admin1['admin1_id_hasc'].str.contains(REGEX_ADMIN1_IDS_HASC)
        & admin1['admin1_id_hasc'].str.slice(0, 2).eq(admin1['admin0_id'])
        & ~admin1['admin1_id_hasc'].duplicated(False)
        & ~admin1_id_from_hasc1.isin(admin1['admin1_id'])
    )
    admin1.loc[i, 'admin1_id'] = admin1_id_from_hasc1[i]
    admin1.loc[i, 'admin1_id_source'] = 'hasc1'

    # Third priority: capitalized letters
    admin1 = admin1.sort_values(['admin0_id', '_name'])
    i_fill = admin1['admin1_id'].isnull()
    admin1_id_caps = admin1[i_fill]['_name'].str.extract('^([A-Z]).*?([A-Z])')
    admin1_id_caps = admin1_id_caps[admin1_id_caps.notnull().mean(1).eq(1)].apply(
        ''.join, 1
    )
    admin1_id_caps = (
        admin1.loc[admin1_id_caps.index]['admin0_id']
        + STRING_SEPARATOR_WITHIN_IDS
        + admin1_id_caps
    )
    admin1_id_caps = admin1_id_caps[
        ~admin1_id_caps.isin(admin1['admin1_id']) & ~admin1_id_caps.duplicated()
    ]
    admin1.loc[admin1_id_caps.index, 'admin1_id'] = admin1_id_caps
    admin1.loc[admin1_id_caps.index, 'admin1_id_source'] = 'own.caps'

    # Fourth priority: first two letters
    i_fill = admin1['admin1_id'].isnull()
    admin1_id_two = (
        admin1[i_fill]['admin0_id']
        + STRING_SEPARATOR_WITHIN_IDS
        + admin1[i_fill]['_name'].str.upper().str.slice(0, 2)
    )
    admin1_id_two = admin1_id_two[
        ~admin1_id_two.isin(admin1['admin1_id']) & ~admin1_id_two.duplicated()
    ]
    admin1.loc[admin1_id_two.index, 'admin1_id'] = admin1_id_two
    admin1.loc[admin1_id_two.index, 'admin1_id_source'] = 'own.first'

    # Fifth priority: any two letters from the name
    i_fill = admin1['admin1_id'].isnull()
    for ix in admin1[i_fill].index:
        admin0_id = admin1.loc[ix, 'admin0_id']
        name = admin1.loc[ix, '_name'].replace(' ', '').replace('-', '')
        for x1, x2 in combinations(name.upper(), 2):
            admin1_id = admin0_id + STRING_SEPARATOR_WITHIN_IDS + x1 + x2
            if admin1_id not in set(admin1['admin1_id']):
                admin1.loc[ix, 'admin1_id'] = admin1_id
                admin1.loc[ix, 'admin1_id_source'] = 'own.any'
                break

    if admin1['admin1_id'].isnull().any() or admin1['admin1_id'].duplicated().any():
        raise Exception('Unable to resolve all admin1_ids')

    if admin1['admin1_id'].isnull().any() or admin1['admin1_id'].duplicated().any():
        raise Exception('Unable to resolve all `admin1_ids`.')

    return admin1.set_index('admin1_id').drop(columns='_name')


# Admin 2: Counties / municipalities


def admin2_id_index_from_admin2_gadm(admin2):
    admin1 = get_admin1(columns=['admin1_id_gadm'])

    # Join admin1
    admin2 = admin2.join(
        admin1.reset_index().set_index('admin1_id_gadm')['admin1_id'],
        on='admin1_id_gadm',
        how='inner',
    )
    admin2['admin0_id'] = admin2['admin1_id'].str.slice(0, 2)

    # Initiate empty AID
    admin2['admin2_id'] = pd.Series(None, dtype='object')
    admin2['admin2_id_source'] = pd.Series(None, dtype='object')

    # Standardize names and sort
    admin2['_name'] = admin2['name'].fillna('').apply(standardize_names)
    admin2 = admin2.sort_values(['admin1_id', '_name'])

    # First priority: unique existing HASC 2 codes, corrected for admin1_id
    HASC2_REGEX_EXTRACT = r"([A-Z0-9]{2})\.([A-Z0-9]{2})\.([A-Z0-9]{2})"
    i_has_hasc = (
        admin2['admin2_id_hasc'].str.match(HASC2_REGEX_EXTRACT)
        & ~admin2['admin2_id_hasc'].duplicated(keep=False)
        & ~admin2['admin0_id'].isin(ADMIN0_ID_HASC1_A2)
    )
    admin2_hasc_parts = admin2[i_has_hasc]['admin2_id_hasc'].str.extract(
        HASC2_REGEX_EXTRACT, expand=True
    )
    admin2_id_from_hasc2_harmonized = (
        admin2[i_has_hasc]['admin1_id']
        + STRING_SEPARATOR_WITHIN_IDS
        + admin2_hasc_parts[2]
    )

    # Remove duplicates introduced through harmonization (Admin1)
    mask_is_unique = ~admin2_id_from_hasc2_harmonized.duplicated(keep=False)
    i = admin2.index.isin(admin2_id_from_hasc2_harmonized[mask_is_unique].index)

    admin2.loc[i, 'admin2_id'] = admin2_id_from_hasc2_harmonized
    admin2.loc[i, 'admin2_id_source'] = 'hasc'

    # Second priority: unique existing HASC 1 codes, corrected for admin1_id
    # Countries using HASC1 code for level-2 administrative units
    i = (
        admin2['admin0_id'].isin(ADMIN0_ID_HASC1_A2)
        & admin2['admin2_id_hasc'].str.contains(REGEX_ADMIN2_IDS_HASC).fillna(False)
        & ~admin2['admin2_id_hasc'].duplicated(False)
    )
    admin2.loc[i, 'admin2_id'] = (
        admin2[i]['admin1_id']
        + STRING_SEPARATOR_WITHIN_IDS
        + admin2[i]['admin2_id_hasc'].str.slice(3, 5)
    )
    admin2.loc[i, 'admin2_id_source'] = 'hasc1'

    # Exception: Brazil has too many subdivisions, gets three-letter codes
    # (Minas Gerais has 854 subdivisions, São Paulo 644, 10 others > 200)
    # Brazil, first try: initials
    i_br = admin2['admin0_id'].eq('BR')
    admin2.loc[i_br, 'admin2_id'], admin2.loc[i_br, 'admin2_id_source'] = np.nan, np.nan
    regexes = [
        '^([A-Z]).*? ([A-Z]).*? ([A-Z])',
        '^([A-Z][a-z]).*?([A-Z])',
        '^([A-Z]).*?([A-Z][a-z])',
        '^([A-Z][a-z]{2})',
    ]
    for regex in regexes:
        i_fill = (
            admin2['admin2_id'].isnull()
            & admin2['_name'].notnull()
            & admin2['admin0_id'].eq('BR')
        )
        aids = admin2[i_fill]['_name'].str.extract(regex)
        aids = aids[aids.notnull().mean(1).eq(1)].apply(''.join, 1)
        aids = (
            admin2.loc[aids.index]['admin1_id']
            + STRING_SEPARATOR_WITHIN_IDS
            + aids.str.upper()
        )
        aids = aids[~aids.isin(admin2['admin2_id']) & ~aids.duplicated()]
        admin2.loc[aids.index, 'admin2_id'] = aids
        admin2_id_source = 'own.br.init' if regex == regexes[0] else 'own.br.first'
        admin2.loc[aids.index, 'admin2_id_source'] = admin2_id_source

    # Brazil, second try: any three
    i_fill = (
        admin2['admin2_id'].isnull()
        & admin2['_name'].notnull()
        & admin2['admin0_id'].eq('BR')
    )
    ixs = admin2[i_fill].index
    aids = set(admin2['admin2_id'])
    for ix in ixs:
        admin1_id = admin2.loc[ix, 'admin1_id']
        name = admin2.loc[ix, '_name'].upper().replace(' ', '').replace('-', '')
        for x1, x2, x3 in combinations(name, 3):
            admin2_id = admin1_id + STRING_SEPARATOR_WITHIN_IDS + x1 + x2 + x3
            if admin2_id not in aids:
                admin2.loc[ix, 'admin2_id'] = admin2_id
                admin2.loc[ix, 'admin2_id_source'] = 'own.br.any'
                aids.add(admin2_id)
                break

    # Exception: Uruguay has no names, gets generic codes (X01, X02, etc.)
    i_uy = admin2['admin0_id'].eq('UY')
    numbers = pd.Series(
        admin2[i_uy]
        .groupby('admin1_id')
        .apply(lambda x: pd.Series(range(1, len(x) + 1)), include_groups=False)
    )
    numbers.index = admin2[i_uy].index
    admin2.loc[i_uy, 'admin2_id'] = (
        admin2[i_uy]['admin1_id']
        + STRING_SEPARATOR_WITHIN_IDS
        + 'X'
        + numbers.astype(str).str.zfill(2)
    )
    admin2.loc[i_uy, 'admin2_id_source'] = 'own.uy'

    # Exception: Unnamed units with generic digits
    # Usually zones in cities, found in Vietnam, Praha (Prague), Guatemala
    i = admin2['admin2_id'].isnull() & admin2['name'].str.contains(
        ' [0-9]{1,2}$'
    ).fillna(False)
    initials = admin2['_name'].str.slice(0, 1)
    n_digits = i.groupby([admin2[i]['admin1_id'], initials[i]]).size()
    N_DIGITS_PER_AID1_MIN = 3
    for admin1_id, initial in n_digits[n_digits.ge(N_DIGITS_PER_AID1_MIN)].index:
        i_fill = i & admin2['admin1_id'].eq(admin1_id) & initials.eq(initial)
        digits = admin2[i_fill]['name'].str.extract(' ([0-9]{1,2})$')[0]
        # If digits are not unique, overwrite with unique digits
        if not len(set(digits)) == len(digits):
            digits = pd.Series(range(1, len(digits) + 1), index=digits.index).astype(
                str
            )
        n_zfill = int(np.ceil(np.log(i_fill.sum()) / np.log(10)))
        aids = (
            admin1_id
            + STRING_SEPARATOR_WITHIN_IDS
            + initial
            + digits.str.zfill(n_zfill)
        )
        admin2.loc[i_fill, 'admin2_id'] = aids
        admin2.loc[i_fill, 'admin2_id_source'] = 'own.a00'

    # Third priority: initials of first two words
    i_fill = admin2['admin2_id'].isnull() & admin2['_name'].notnull()
    aid_caps = admin2[i_fill]['_name'].str.extract('^([A-Z]).*?([A-Z])')
    aid_caps = aid_caps[aid_caps.notnull().mean(1).eq(1)].apply(''.join, 1)
    aid_caps = (
        admin2.loc[aid_caps.index]['admin1_id'] + STRING_SEPARATOR_WITHIN_IDS + aid_caps
    )
    aid_caps = aid_caps[~aid_caps.isin(admin2['admin2_id']) & ~aid_caps.duplicated()]
    admin2.loc[aid_caps.index, 'admin2_id'] = aid_caps
    admin2.loc[aid_caps.index, 'admin2_id_source'] = 'own.init'

    # Fourth priority: first two letters
    i_fill = (
        admin2['admin2_id'].isnull()
        & admin2['_name'].notnull()
        & admin2['_name'].ne('')
    )
    aid_two = (
        admin2[i_fill]['admin1_id']
        + STRING_SEPARATOR_WITHIN_IDS
        + admin2[i_fill]['_name'].str.upper().str.slice(0, 2)
    )
    aid_two = aid_two[
        ~aid_two.isin(admin2['admin2_id'])
        & ~aid_two.duplicated()
        & aid_two.str.len().ge(6)
    ]
    admin2.loc[aid_two.index, 'admin2_id'] = aid_two
    admin2.loc[aid_two.index, 'admin2_id_source'] = 'own.first'

    # Fifth priority: any two letters from the name
    i_fill = admin2['admin2_id'].isnull() & admin2['_name'].notnull()
    ixs = admin2[i_fill].index
    aids = set(admin2['admin2_id'])
    for ix in ixs:
        admin1_id = admin2.loc[ix, 'admin1_id']
        name = admin2.loc[ix, '_name'].upper().replace(' ', '').replace('-', '')
        for x1, x2 in combinations(name, 2):
            admin2_id = admin1_id + STRING_SEPARATOR_WITHIN_IDS + x1 + x2
            if admin2_id not in aids:
                admin2.loc[ix, 'admin2_id'] = admin2_id
                admin2.loc[ix, 'admin2_id_source'] = 'own.any'
                aids.add(admin2_id)
                break

    # Sixth priority: rename existing aids to make space for others
    i_fill = admin2['admin2_id'].isnull() & admin2['_name'].notnull()
    ixs = admin2[i_fill].index
    aids = set(admin2['admin2_id'])
    for ix in ixs:
        admin1_id = admin2.loc[ix, 'admin1_id']
        name = admin2.loc[ix, '_name'].upper().replace(' ', '').replace('-', '')
        for x1, x2 in combinations(name, 2):
            admin2_id = admin1_id + STRING_SEPARATOR_WITHIN_IDS + x1 + x2

            rep = admin2[admin2['admin2_id'].eq(admin2_id)]
            if len(rep) == 0:
                print('How did I miss this? ' + admin2_id)
                continue

            ix2 = rep.iloc[0].name
            name_rep = rep.iloc[0]['_name'].upper().replace(' ', '').replace('-', '')

            replacement_found = False
            for y1, y2 in combinations(name_rep, 2):
                admin2_id_rep = admin1_id + STRING_SEPARATOR_WITHIN_IDS + y1 + y2
                if admin2_id_rep not in aids:
                    replacement_found = True
                    admin2.loc[ix2, 'admin2_id'] = admin2_id_rep
                    admin2.loc[ix2, 'admin2_id_source'] = 'own.rep'
                    aids.add(admin2_id_rep)
                    break

            if replacement_found:
                admin2.loc[ix, 'admin2_id'] = admin2_id
                admin2.loc[ix, 'admin2_id_source'] = 'own.any'
                break

    # Last resort: filling in NAs
    i_fill = admin2['admin2_id'].isnull()
    numbers = pd.Series(
        admin2[i_fill]
        .groupby('admin1_id')
        .apply(lambda x: pd.Series(range(1, len(x) + 1)), include_groups=False)
    )
    numbers.index = admin2[i_fill].index
    admin2.loc[i_fill, 'admin2_id'] = (
        admin2[i_fill]['admin1_id']
        + STRING_SEPARATOR_WITHIN_IDS
        + 'X'
        + numbers.astype(str)
    )
    admin2.loc[i_fill, 'admin2_id_source'] = 'own.na'

    # Catch issues with nulls and duplicates
    admin2_id_isnull = admin2['admin2_id'].isnull()
    admin2_id_duplicated = admin2['admin2_id'].duplicated(keep=False)
    if admin2_id_isnull.any() or admin2_id_duplicated.any():
        message = 'Unable to resolve all AdminIds from GADM Level-2.\n\n'
        if admin2_id_isnull.any():
            message += 'Nulls:\n\n' + str(admin2[admin2_id_isnull])
        if admin2_id_duplicated.any():
            message += 'Duplicates:\n\n' + str(
                admin2[admin2_id_duplicated].sort_values('admin2_id')[
                    ['admin2_id_hasc', 'admin2_id', 'name']
                ]
            )
        raise Exception(message)

    return admin2.set_index('admin2_id').drop(columns='_name')


def admin2_id_index_from_admin2_US_nhgis(admin2_local):
    # Join states
    admin1_recipe = get_recipe('US', 'admin-nhgis-2020', filename='admin1')
    admin1_crosswalk = (
        get_admin1(recipe=admin1_recipe, columns=['admin1_id_admin0'])
        .reset_index()
        .set_index('admin1_id_admin0')
    )
    admin2_local = admin2_local.join(admin1_crosswalk, on='admin1_id_admin0')

    # Create name-based identifier
    admin2_local['name_link'] = admin2_local['name'].apply(create_comparable_name_link)

    # Add ' city' to the name_link for duplicate name + state
    # (e.g. Baltimore county vs. city)
    i_city_duplicates = admin2_local[['admin1_id', 'name']].duplicated(
        keep=False
    ) & admin2_local['name_long'].eq(admin2_local['name'] + ' city')
    admin2_local.loc[i_city_duplicates, 'name_link'] += ' city'

    # Load global reference layer (GADM)
    admin2 = get_admin2('US')
    admin2['admin1_id'] = admin2.index.str.slice(0, 5)

    # Correct (replace) names from global reference layer to official
    admin2_name_crosswalk = get_recipe(
        'US', 'admin-nhgis-2020', filename='admin2-names-from-gadm'
    )
    for _, row in admin2_name_crosswalk.iterrows():
        admin2.loc[
            admin2['admin1_id'].eq(row['admin1_id'])
            & admin2['name'].eq(row['admin2_name_gadm']),
            'name',
        ] = row['admin2_name_official']

    admin2['name_link'] = admin2['name'].str.lower().apply(create_comparable_name_link)

    # Join global admin-2 data (with identifier) to local admin-2 data
    admin2_local = admin2_local.join(
        admin2.reset_index().set_index(['admin1_id', 'name_link'])['admin2_id'],
        on=['admin1_id', 'name_link'],
    )

    # Set new admin2_ids for units that don't exist in the global layer
    new_admin2_ids = get_recipe(
        'US',
        'admin-nhgis-2020',
        filename='admin2-ids',
        dtype={'admin2_id_admin0': str},
    ).set_index('admin2_id_admin0')
    for admin2_id_admin0, admin2_id in new_admin2_ids['admin2_id'].items():
        admin2_local.loc[
            admin2_local['admin2_id_admin0'].eq(admin2_id_admin0), 'admin2_id'
        ] = admin2_id

    # Ensure the IDs are complete and unique
    i_null = admin2_local['admin2_id'].isnull()
    if i_null.any():
        raise ValueError('Empty `admin2_id`:\n' + str(admin2_local[i_null]))

    i_dupl = admin2_local['admin2_id'].duplicated(keep=False)
    if i_dupl.any():
        raise ValueError('Duplicate `admin2_id`:\n' + str(admin2_local[i_dupl]))

    return admin2_local.set_index('admin2_id')


# def get_admin_id_crosswalk(admin_id, admin_level, admin_id_col, admin_recipe_id):
#     """Get a crosswalk Series (source admin ID > openplaces admin ID)

#     Parameters
#     ----------
#     admin_id : str
#         Administrative unit ID of the dataset
#     admin_level : int
#         Level of the administrative units that need to be crosswalked
#     admin_id_col : str
#         Name of the column in the recipe. Becomes index of crosswalk
#     admin_recipe_id : str
#         ID of the admin recipe that contains the crosswalk.
#     """

#     admin_id_crosswalk = get_admin_by_level(
#         admin_level,
#         admin_id,
#         columns=[admin_id_col],
#         recipe=get_recipe_by_id(admin_recipe_id),
#     )
#     return admin_id_crosswalk.reset_index().set_index(admin_id_col)[
#         f'admin{admin_level}_id'
#     ]


def find_admin_recipe(admin_id, admin_level):
    """Find an administrative data ingestion recipe

    Parameters
    ----------
    admin_id : str
        Administrative unit identifier
    admin_level : int
        Administrative level for which a recipe is sought.
    """
    glob_recipe_path = recipe_path(
        admin_id, 'admin-*-*', filename=f'admin{admin_level}'
    )
    recipe_paths_found = glob.glob(str(glob_recipe_path))
    if len(recipe_paths_found) > 1:
        raise NotImplementedError(
            f'Multiple admin recipes found for {admin_id} at level {admin_level}:\n\n'
            + '\n'.join(recipe_paths_found)
            + '\n\nRewrite `find_local_admin_recipe` to make your selection.'
        )

    recipe_id = Path(recipe_paths_found[0]).name
    return get_recipe_by_id(recipe_id)


def generate_admin_ids(
    df,
    new_admin_id_col='admin2_id',
    parent_admin_id_col='admin1_id',
    name_col='name',
    name_long_col=None,
    id_separator=STRING_SEPARATOR_WITHIN_IDS,
    verbose=False,
):
    """
    Generate unique two-letter admin unit codes within parent units.

    Generate unique admin ID codes for administrative units

    Level-agnostic design: works for any parent-child relationship:
    admin1->admin2 (state->county), admin2->admin3 (county->town)

    Strategy:
    1. Initials from multi-word/hyphenated names (e.g., "Los Angeles" → "LA")
    2. First two letters of name (for single-word names)
    3. Any two letters from name
    4. Use 'C' or 'T' suffix for cities/townships when helpful
    5. Swap existing codes to free up better matches
    6. Three-letter codes if two-letter space exhausted
    7. Sequential numbering as last resort

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with administrative unit data
    new_admin_id_col : str
        Name for the new administrative ID column (default 'admin3_id')
    parent_admin_id_col : str
        Column name containing parent admin ID (e.g., 'admin2_id')
    name_col : str
        Column name containing subdivision name
    name_long_col : str, optional
        Column name containing long-form name.
        Example: in the US, this might include 'city' and 'township'
        suffixes that resolve ambiguities between entity names
        If None or column doesn't exist, city/township detection is skipped
    id_separator : str
        Separator to use in IDs (default '_')
    verbose : bool
        If True, prints statistics and other outputs

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by new_admin_id_col with diagnostics column

    Raises
    ------
    ValueError
        If unable to generate unique IDs for all rows
    """

    # Work on a copy
    admin = df.copy()

    # Auto-generate source column name
    id_source_col = new_admin_id_col + '_source'

    # Initialize columns
    admin[new_admin_id_col] = None
    admin[id_source_col] = None
    # Ensure _name_clean never has None/NaN - use empty string as fallback
    admin['_name_clean'] = (
        admin[name_col]
        .fillna('')
        .str.upper()
        .str.replace(' ', '', regex=False)
        .str.replace('-', '', regex=False)
    )

    # Detect city/township/borough types from name_long (if available)
    use_name_long = name_long_col is not None and name_long_col in admin.columns
    if use_name_long:
        admin['_is_city'] = admin[name_long_col].str.contains(
            ' city$', case=False, na=False
        )
        admin['_is_township'] = admin[name_long_col].str.contains(
            ' township$', case=False, na=False
        )
        admin['_is_borough'] = admin[name_long_col].str.contains(
            ' borough$', case=False, na=False
        )
    else:
        admin['_is_city'] = False
        admin['_is_township'] = False
        admin['_is_borough'] = False

    # Sort for consistent processing
    admin = admin.sort_values([parent_admin_id_col, name_col]).copy()

    # Track used IDs globally
    used_ids = set()

    # Priority 1: First letter + first letter of second word (hyphenated/multi-word)
    if verbose:
        print("Priority 1: Initials from multi-word/hyphenated names...")
    mask = admin[new_admin_id_col].isna()
    if mask.any():
        # Extract initials from multi-word names (treat hyphens as word separators)
        names = admin.loc[mask, name_col].str.upper().str.replace('-', ' ')
        has_multiple_words = names.str.contains(' ', na=False)
        if has_multiple_words.any():
            words_split = names[has_multiple_words].str.split(' ', n=1)
            codes = words_split.str[0].str[0] + words_split.str[1].str[0]
            candidate_ids = (
                admin.loc[mask & has_multiple_words, parent_admin_id_col]
                + id_separator
                + codes
            )
            is_unique = ~candidate_ids.duplicated(keep=False) & ~candidate_ids.isin(
                used_ids
            )
            idx_to_update = mask & has_multiple_words
            admin.loc[idx_to_update & is_unique, new_admin_id_col] = candidate_ids[
                is_unique
            ]
            admin.loc[idx_to_update & is_unique, id_source_col] = 'initials'
            used_ids.update(candidate_ids[is_unique])

    if verbose:
        print(f"  Assigned: {admin[new_admin_id_col].notna().sum()}/{len(admin)}")

    # Priority 2: First two letters (for single-word names)
    if verbose:
        print("Priority 2: First two letters...")
    mask = admin[new_admin_id_col].isna() & (admin['_name_clean'].str.len() >= 2)
    if mask.any():
        codes = admin.loc[mask, '_name_clean'].str[:2]
        candidate_ids = admin.loc[mask, parent_admin_id_col] + id_separator + codes
        # Only assign IDs that are unique within this batch and not already used
        is_unique = ~candidate_ids.duplicated(keep=False) & ~candidate_ids.isin(
            used_ids
        )
        admin.loc[mask & is_unique, new_admin_id_col] = candidate_ids[is_unique]
        admin.loc[mask & is_unique, id_source_col] = 'first2'
        used_ids.update(candidate_ids[is_unique])

    if verbose:
        print(f"  Assigned: {admin[new_admin_id_col].notna().sum()}/{len(admin)}")

    # Priority 3: Any two letters from name
    if verbose:
        print("Priority 3: Any two letters from name...")
    mask = admin[new_admin_id_col].isna()
    unassigned = admin[mask].copy()

    if len(unassigned) > 0:
        # Pre-compute all needed data
        indices = unassigned.index.tolist()
        names_clean = unassigned['_name_clean'].tolist()
        parent_ids = unassigned[parent_admin_id_col].tolist()

        # Process in batch
        for i, (idx, name_clean, parent_id) in enumerate(
            zip(indices, names_clean, parent_ids)
        ):
            if len(name_clean) < 2:
                continue

            # Try combinations
            for c1, c2 in combinations(name_clean, 2):
                code = c1 + c2
                new_id = parent_id + id_separator + code
                if new_id not in used_ids:
                    admin.loc[idx, new_admin_id_col] = new_id
                    admin.loc[idx, id_source_col] = 'any2'
                    used_ids.add(new_id)
                    break

    if verbose:
        print(f"  Assigned: {admin[new_admin_id_col].notna().sum()}/{len(admin)}")

    # Priority 3b: Letter + number combinations (for names with few letters)
    if verbose:
        print("Priority 3b: Letter + number combinations...")
    mask = admin[new_admin_id_col].isna()
    unassigned = admin[mask].copy()

    if len(unassigned) > 0:
        # Pre-extract all needed data in batch
        indices = unassigned.index.tolist()
        parent_ids = unassigned[parent_admin_id_col].tolist()

        # Extract letters and numbers from name and name_long columns
        names_upper = unassigned[name_col].fillna('').str.upper().tolist()
        if use_name_long:
            names_long_upper = unassigned[name_long_col].fillna('').str.upper().tolist()
        else:
            names_long_upper = [''] * len(names_upper)

        # Process in batch
        for idx, parent_id, name_upper, name_long_upper in zip(
            indices, parent_ids, names_upper, names_long_upper
        ):
            combined_name = name_upper + ' ' + name_long_upper
            letters = [c for c in combined_name if c.isalpha()]
            numbers = [c for c in combined_name if c.isdigit()]

            # Try letter + number combinations
            found = False
            if letters and numbers:
                for letter in letters:
                    for number in numbers:
                        code = letter + number
                        new_id = parent_id + id_separator + code
                        if new_id not in used_ids:
                            admin.loc[idx, new_admin_id_col] = new_id
                            admin.loc[idx, id_source_col] = 'letter_num'
                            used_ids.add(new_id)
                            found = True
                            break
                    if found:
                        break

            # If no letters at all, use X + number
            if not found and not letters and numbers:
                for number in numbers:
                    code = 'X' + number
                    new_id = parent_id + id_separator + code
                    if new_id not in used_ids:
                        admin.loc[idx, new_admin_id_col] = new_id
                        admin.loc[idx, id_source_col] = 'x_num'
                        used_ids.add(new_id)
                        break

    if verbose:
        print(f"  Assigned: {admin[new_admin_id_col].notna().sum()}/{len(admin)}")

    # Priority 4: Try C/T suffix for cities/townships
    if verbose:
        print("Priority 4: Using C/T suffix for cities/townships...")
    mask = admin[new_admin_id_col].isna() & (admin['_is_city'] | admin['_is_township'])
    if mask.any():
        first_letter = admin.loc[mask, '_name_clean'].str[0]
        suffix = admin.loc[mask, '_is_city'].map({True: 'C', False: 'T'})
        codes = first_letter + suffix
        candidate_ids = admin.loc[mask, parent_admin_id_col] + id_separator + codes
        is_unique = ~candidate_ids.duplicated(keep=False) & ~candidate_ids.isin(
            used_ids
        )
        admin.loc[mask & is_unique, new_admin_id_col] = candidate_ids[is_unique]
        admin.loc[mask & is_unique, id_source_col] = 'suffix_CT'
        used_ids.update(candidate_ids[is_unique])

    if verbose:
        print(f"  Assigned: {admin[new_admin_id_col].notna().sum()}/{len(admin)}")

    # Priority 5: Swap existing codes to free up better matches
    if verbose:
        print("Priority 5: Swapping existing codes...")
    mask = admin[new_admin_id_col].isna()
    swaps_made = 0
    unassigned = admin[mask].copy()

    if len(unassigned) > 0:
        indices = unassigned.index.tolist()
        names_clean = unassigned['_name_clean'].tolist()
        parent_ids = unassigned[parent_admin_id_col].tolist()

        for idx, name_clean, parent_id in zip(indices, names_clean, parent_ids):
            if len(name_clean) < 2:
                continue

            found = False
            for c1, c2 in combinations(name_clean, 2):
                code = c1 + c2
                new_id = parent_id + id_separator + code

                # Check if this ID is already taken
                if new_id in used_ids:
                    # Find who has it
                    existing_mask = admin[new_admin_id_col] == new_id
                    if not existing_mask.any():
                        continue
                    existing_idx = existing_mask.idxmax()
                    existing_name_clean = admin.at[existing_idx, '_name_clean']
                    existing_parent_id = admin.at[existing_idx, parent_admin_id_col]

                    # Skip if existing name is too short
                    if len(existing_name_clean) < 2:
                        continue

                    # Only swap if they're in the same parent unit
                    if existing_parent_id == parent_id:
                        # Try to find alternative for existing holder
                        swap_found = False
                        for d1, d2 in combinations(existing_name_clean, 2):
                            alt_code = d1 + d2
                            alt_new_id = existing_parent_id + id_separator + alt_code
                            if alt_new_id not in used_ids and alt_code != code:
                                # Perform swap
                                used_ids.remove(new_id)
                                admin.loc[existing_idx, new_admin_id_col] = alt_new_id
                                admin.loc[existing_idx, id_source_col] = 'swapped'
                                used_ids.add(alt_new_id)

                                admin.loc[idx, new_admin_id_col] = new_id
                                admin.loc[idx, id_source_col] = 'any2_after_swap'
                                used_ids.add(new_id)

                                swap_found = True
                                swaps_made += 1
                                break

                        if swap_found:
                            found = True
                            break

            if found:
                break

    if verbose:
        print(f"  Swaps made: {swaps_made}")
        print(f"  Assigned: {admin[new_admin_id_col].notna().sum()}/{len(admin)}")

    # Priority 6: Three-letter codes for remaining
    if verbose:
        print("Priority 6: Three-letter codes...")
    mask = admin[new_admin_id_col].isna()

    # First try: first three letters (vectorized)
    if mask.any():
        has_three = admin.loc[mask, '_name_clean'].str.len() >= 3
        if has_three.any():
            codes = admin.loc[mask & has_three, '_name_clean'].str[:3]
            candidate_ids = (
                admin.loc[mask & has_three, parent_admin_id_col] + id_separator + codes
            )
            is_unique = ~candidate_ids.duplicated(keep=False) & ~candidate_ids.isin(
                used_ids
            )
            admin.loc[mask & has_three & is_unique, new_admin_id_col] = candidate_ids[
                is_unique
            ]
            admin.loc[mask & has_three & is_unique, id_source_col] = 'first3'
            used_ids.update(candidate_ids[is_unique])

    # Second try: any three letters (needs iterative approach)
    mask = admin[new_admin_id_col].isna()
    unassigned = admin[mask].copy()

    if len(unassigned) > 0:
        indices = unassigned.index.tolist()
        names_clean = unassigned['_name_clean'].tolist()
        parent_ids = unassigned[parent_admin_id_col].tolist()

        for idx, name_clean, parent_id in zip(indices, names_clean, parent_ids):
            if len(name_clean) < 3:
                continue

            for c1, c2, c3 in combinations(name_clean, 3):
                code = c1 + c2 + c3
                new_id = parent_id + id_separator + code
                if new_id not in used_ids:
                    admin.loc[idx, new_admin_id_col] = new_id
                    admin.loc[idx, id_source_col] = 'any3'
                    used_ids.add(new_id)
                    break

    if verbose:
        print(f"  Assigned: {admin[new_admin_id_col].notna().sum()}/{len(admin)}")

    # Last resort: Sequential numbering
    if verbose:
        print("Priority 7: Sequential numbering...")
    mask = admin[new_admin_id_col].isna()
    if mask.any():
        # Group by parent_admin_id_col and assign sequential numbers
        remaining = admin[mask].groupby(parent_admin_id_col)
        for parent_id, group in remaining:
            for i, idx in enumerate(group.index, start=1):
                code = f'X{i:02d}'
                new_id = parent_id + id_separator + code
                # Ensure uniqueness
                counter = 1
                while new_id in used_ids:
                    code = f'X{i:02d}{chr(64+counter)}'
                    new_id = parent_id + id_separator + code
                    counter += 1

                admin.loc[idx, new_admin_id_col] = new_id
                admin.loc[idx, id_source_col] = 'sequential'
                used_ids.add(new_id)

    if verbose:
        print(f"  Final assigned: {admin[new_admin_id_col].notna().sum()}/{len(admin)}")

    # Verify uniqueness
    if admin[new_admin_id_col].isna().any():
        n_missing = admin[new_admin_id_col].isna().sum()
        raise ValueError(f"Failed to assign IDs to {n_missing} rows")

    if admin[new_admin_id_col].duplicated().any():
        n_dupes = admin[new_admin_id_col].duplicated().sum()
        dupes = admin[admin[new_admin_id_col].duplicated(keep=False)][
            [new_admin_id_col, name_col, parent_admin_id_col]
        ]
        raise ValueError(f"Found {n_dupes} duplicate IDs:\n{dupes}")

    if verbose:
        print("\n✓ All IDs assigned and verified unique!")

        # Print summary statistics
        print("\nID Generation Summary:")
        print(admin[id_source_col].value_counts().to_string())

    # Clean up temporary columns and set index
    admin = admin.drop(
        columns=['_name_clean', '_is_city', '_is_township', '_is_borough']
    )
    admin = admin.set_index(new_admin_id_col)

    return admin
