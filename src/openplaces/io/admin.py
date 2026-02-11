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

from openplaces.api import get_admin
from openplaces.core.constants import (
    ADMIN1_IDS_USING_HASC1_FOR_ADMIN2,
    REGEX_ADMIN2_IDS_AA_AA,
    REGEX_ADMIN2_IDS_AA_AA_EXTRACT,
    REGEX_ADMIN2_IDS_HASC,
    REGEX_ADMIN3_IDS_HASC,
    STRING_SEPARATOR_WITHIN_IDS,
)
from openplaces.path import recipe_path
from openplaces.recipe import get_recipe, get_recipe_by_id
from openplaces.utils import create_comparable_name_link, standardize_names


# Admin 1: Countries
def get_admin1_iso():
    """Get dataframe with country ISO alpha codes and names"""

    ADMIN1_ISO_RENAME_COLUMNS = {
        'Country or Area': 'name',
        'ISO-alpha2 Code': 'admin1_id',
        'ISO-alpha3 Code': 'admin1_id_a3',
    }

    admin1_iso = (
        get_recipe(None, 'admin-iso', filename='admin1-alpha-2', keep_default_na=False)
        .rename(columns=ADMIN1_ISO_RENAME_COLUMNS)
        .query('admin1_id != ""')
        .set_index('admin1_id')
        .sort_index()[['name', 'admin1_id_a3']]
    )

    admin1_iso_additions = get_recipe(
        None, 'admin-iso', filename='admin1-additions'
    ).set_index('admin1_id')
    admin1_iso = pd.concat(
        [
            admin1_iso,
            admin1_iso_additions[['name', 'admin1_id_a3']],
        ]
    )

    # Manual addition
    if admin1_iso.index.duplicated().any():
        raise Exception('admin1_iso index duplicated.')

    # Joining regional groupings
    admin1_iso_regions = (
        get_recipe(
            None,
            'admin-iso',
            filename='admin1-regions-iso3166',
            keep_default_na=False,
        )
        .rename(columns={'alpha-2': 'admin1_id'})
        .set_index('admin1_id')[['region', 'sub-region', 'intermediate-region']]
    )

    for admin1_id_to, admin1_id_from in admin1_iso_additions[
        'admin1_id_copy_region'
    ].items():
        admin1_iso_regions.loc[admin1_id_to] = admin1_iso_regions.loc[admin1_id_from]

    # Ensure no duplicates were introduced
    if admin1_iso_regions.index.duplicated().any():
        raise Exception(
            'admin1_iso index duplicated: '
            + ', '.join(
                admin1_iso_regions.index[
                    admin1_iso_regions.index.duplicated(keep=False)
                ]
            )
        )

    admin1_iso = admin1_iso.join(admin1_iso_regions).sort_index()

    return admin1_iso


def admin1_id_index_from_admin1_id_a3(gdf):
    """Give dataframe `gdf` an `admin1_id` index from `admin1_id_a3`

    Single-use function to create linkage between GADM and ISO
    """

    admin1_id_a3_to_admin1_id = (
        get_admin1_iso().reset_index().set_index('admin1_id_a3')['admin1_id']
    )

    gdf_indexed = gdf.join(admin1_id_a3_to_admin1_id, on='admin1_id_a3').set_index(
        'admin1_id'
    )

    if gdf_indexed.index.isnull().any():
        print('Missing indices')
        print(gdf_indexed[gdf_indexed.index.isnull()])
        raise ValueError('Missing indices')
    return gdf_indexed


# Admin 2: States / provinces


def get_admin2_iso():
    """Get dataframe with state/province ISO3116-2 codes and names"""

    ADMIN2_ISO_RENAME_COLUMNS = {
        'country_code': 'admin1_id',
        'subdivision_name': 'name',
        'code': 'admin2_id_iso3166',
    }

    admin2_iso = (
        get_recipe(
            None,
            'admin-iso-20210301',
            filename='admin2-iso3166-2',
            keep_default_na=False,
        )
        .rename(columns=ADMIN2_ISO_RENAME_COLUMNS)
        .join(get_admin1_iso()['name'].rename('admin1_name'), on='admin1_id')
        .query('`admin2_id_iso3166` != "-"')
    )
    if admin2_iso['admin2_id_iso3166'].duplicated().any():
        raise Exception(
            'Unique ISO3116-2 code required in `openplaces.io.admin.get_admin2_iso()`.'
        )
    admin2_iso = admin2_iso.set_index('admin2_id_iso3166')[
        ['name', 'admin1_id', 'admin1_name']
    ].sort_index()
    return admin2_iso


def admin2_id_index_from_admin2_gadm(admin2):
    """Give dataframe `admin` an `admin2_id` index based on GADM data"""

    # Join with level-2 administrative units
    admin2 = admin2.join(
        get_admin1_iso().reset_index().set_index('admin1_id_a3')['admin1_id'],
        on='admin1_id_a3',
        how='inner',
    )
    if admin2['admin1_id'].isnull().any():
        raise ValueError('Empty `\'admin1_id\'` found in `admin2`.')

    admin2['_name'] = admin2['name'].apply(standardize_names).fillna('NA')

    # Read ISO
    admin2_iso = get_admin2_iso()
    admin2_iso['_name'] = admin2_iso['name'].apply(standardize_names)
    admin2_iso_join = admin2_iso.reset_index().set_index(['admin1_id', '_name'])
    admin2 = admin2.join(
        admin2_iso_join['admin2_id_iso3166'], on=['admin1_id', '_name']
    )

    # Initiate empty admin ID
    admin2['admin2_id'] = pd.Series(None, dtype='object')
    admin2['admin2_id_source'] = pd.Series(None, dtype='object')

    # First priority: official ISO3166-2 codes
    i = admin2['admin2_id_iso3166'].str.contains(REGEX_ADMIN2_IDS_AA_AA, na=False)
    hasc_from_iso = admin2[i]['admin2_id_iso3166'].str.extract(
        REGEX_ADMIN2_IDS_AA_AA_EXTRACT
    )
    admin2.loc[i, 'admin2_id'] = hasc_from_iso.apply(
        STRING_SEPARATOR_WITHIN_IDS.join, 1
    )
    admin2.loc[i, 'admin2_id_source'] = 'iso'

    # Second priority: existing HASC codes that are unique,
    # not already used by ISO3166-2, and use correct country-level code
    admin2_id_from_hasc1 = admin2['admin2_id_hasc'].str.replace(
        '.', STRING_SEPARATOR_WITHIN_IDS, regex=False
    )
    i = (
        admin2['admin2_id'].isnull()
        & admin2['admin2_id_hasc'].str.contains(REGEX_ADMIN2_IDS_HASC)
        & admin2['admin2_id_hasc'].str.slice(0, 2).eq(admin2['admin1_id'])
        & ~admin2['admin2_id_hasc'].duplicated(False)
        & ~admin2_id_from_hasc1.isin(admin2['admin2_id'])
    )
    admin2.loc[i, 'admin2_id'] = admin2_id_from_hasc1[i]
    admin2.loc[i, 'admin2_id_source'] = 'hasc'

    # Third priority: capitalized letters
    admin2 = admin2.sort_values(['admin1_id', '_name'])
    i_fill = admin2['admin2_id'].isnull()
    admin2_id_caps = admin2[i_fill]['_name'].str.extract('^([A-Z]).*?([A-Z])')
    admin2_id_caps = admin2_id_caps[admin2_id_caps.notnull().mean(1).eq(1)].apply(
        ''.join, 1
    )
    admin2_id_caps = (
        admin2.loc[admin2_id_caps.index]['admin1_id']
        + STRING_SEPARATOR_WITHIN_IDS
        + admin2_id_caps
    )
    admin2_id_caps = admin2_id_caps[
        ~admin2_id_caps.isin(admin2['admin2_id']) & ~admin2_id_caps.duplicated()
    ]
    admin2.loc[admin2_id_caps.index, 'admin2_id'] = admin2_id_caps
    admin2.loc[admin2_id_caps.index, 'admin2_id_source'] = 'capitalized'

    # Fourth priority: first two letters
    i_fill = admin2['admin2_id'].isnull()
    admin2_id_two = (
        admin2[i_fill]['admin1_id']
        + STRING_SEPARATOR_WITHIN_IDS
        + admin2[i_fill]['_name'].str.upper().str.slice(0, 2)
    )
    admin2_id_two = admin2_id_two[
        ~admin2_id_two.isin(admin2['admin2_id']) & ~admin2_id_two.duplicated()
    ]
    admin2.loc[admin2_id_two.index, 'admin2_id'] = admin2_id_two
    admin2.loc[admin2_id_two.index, 'admin2_id_source'] = 'first2'

    # Fifth priority: any two letters from the name
    i_fill = admin2['admin2_id'].isnull()
    for ix in admin2[i_fill].index:
        admin1_id = admin2.loc[ix, 'admin1_id']
        name = admin2.loc[ix, '_name'].replace(' ', '').replace('-', '')
        for x1, x2 in combinations(name.upper(), 2):
            admin2_id = admin1_id + STRING_SEPARATOR_WITHIN_IDS + x1 + x2
            if admin2_id not in set(admin2['admin2_id']):
                admin2.loc[ix, 'admin2_id'] = admin2_id
                admin2.loc[ix, 'admin2_id_source'] = 'any2'
                break

    if admin2['admin2_id'].isnull().any() or admin2['admin2_id'].duplicated().any():
        raise Exception('Unable to resolve all admin2_ids')

    if admin2['admin2_id'].isnull().any() or admin2['admin2_id'].duplicated().any():
        raise Exception('Unable to resolve all `admin2_ids`.')

    return admin2.set_index('admin2_id').drop(columns='_name')


# Admin 3: Counties / municipalities


def admin3_id_index_from_admin3_gadm(admin3):
    admin2 = get_admin(level=2, columns=['admin2_id_gadm'])

    # Join admin2
    admin3 = admin3.join(
        admin2.reset_index().set_index('admin2_id_gadm')['admin2_id'],
        on='admin2_id_gadm',
        how='inner',
    )
    admin3['admin1_id'] = admin3['admin2_id'].str.slice(0, 2)

    # Initiate empty AID
    admin3['admin3_id'] = pd.Series(None, dtype='object')
    admin3['admin3_id_source'] = pd.Series(None, dtype='object')

    # Standardize names and sort
    admin3['_name'] = admin3['name'].fillna('').apply(standardize_names)
    admin3 = admin3.sort_values(['admin2_id', '_name'])

    # First priority: unique existing HASC 2 codes, corrected for admin2_id
    HASC2_REGEX_EXTRACT = r"([A-Z0-9]{2})\.([A-Z0-9]{2})\.([A-Z0-9]{2})"
    i_has_hasc = (
        admin3['admin3_id_hasc'].str.match(HASC2_REGEX_EXTRACT)
        & ~admin3['admin3_id_hasc'].duplicated(keep=False)
        & ~admin3['admin1_id'].isin(ADMIN1_IDS_USING_HASC1_FOR_ADMIN2)
    )
    admin3_hasc_parts = admin3[i_has_hasc]['admin3_id_hasc'].str.extract(
        HASC2_REGEX_EXTRACT, expand=True
    )
    admin3_id_from_hasc2_harmonized = (
        admin3[i_has_hasc]['admin2_id']
        + STRING_SEPARATOR_WITHIN_IDS
        + admin3_hasc_parts[2]
    )

    # Remove duplicates introduced through harmonization (Admin1)
    mask_is_unique = ~admin3_id_from_hasc2_harmonized.duplicated(keep=False)
    i = admin3.index.isin(admin3_id_from_hasc2_harmonized[mask_is_unique].index)

    admin3.loc[i, 'admin3_id'] = admin3_id_from_hasc2_harmonized
    admin3.loc[i, 'admin3_id_source'] = 'hasc'

    # Second priority: unique existing HASC 1 codes, corrected for admin2_id
    # Countries using HASC1 code for level-2 administrative units
    i = (
        admin3['admin1_id'].isin(ADMIN1_IDS_USING_HASC1_FOR_ADMIN2)
        & admin3['admin3_id_hasc'].str.contains(REGEX_ADMIN3_IDS_HASC).fillna(False)
        & ~admin3['admin3_id_hasc'].duplicated(False)
    )
    admin3.loc[i, 'admin3_id'] = (
        admin3[i]['admin2_id']
        + STRING_SEPARATOR_WITHIN_IDS
        + admin3[i]['admin3_id_hasc'].str.slice(3, 5)
    )
    admin3.loc[i, 'admin3_id_source'] = 'hasc'

    # Exception: Brazil has too many subdivisions, gets three-letter codes
    # (Minas Gerais has 854 subdivisions, São Paulo 644, 10 others > 200)
    # Brazil, first try: initials
    i_br = admin3['admin1_id'].eq('BR')
    admin3.loc[i_br, 'admin3_id'], admin3.loc[i_br, 'admin3_id_source'] = np.nan, np.nan
    regexes = [
        '^([A-Z]).*? ([A-Z]).*? ([A-Z])',
        '^([A-Z][a-z]).*?([A-Z])',
        '^([A-Z]).*?([A-Z][a-z])',
        '^([A-Z][a-z]{2})',
    ]
    for regex in regexes:
        i_fill = (
            admin3['admin3_id'].isnull()
            & admin3['_name'].notnull()
            & admin3['admin1_id'].eq('BR')
        )
        aids = admin3[i_fill]['_name'].str.extract(regex)
        aids = aids[aids.notnull().mean(1).eq(1)].apply(''.join, 1)
        aids = (
            admin3.loc[aids.index]['admin2_id']
            + STRING_SEPARATOR_WITHIN_IDS
            + aids.str.upper()
        )
        aids = aids[~aids.isin(admin3['admin3_id']) & ~aids.duplicated()]
        admin3.loc[aids.index, 'admin3_id'] = aids
        admin3_id_source = 'br.initials' if regex == regexes[0] else 'br.first3'
        admin3.loc[aids.index, 'admin3_id_source'] = admin3_id_source

    # Brazil, second try: any three
    i_fill = (
        admin3['admin3_id'].isnull()
        & admin3['_name'].notnull()
        & admin3['admin1_id'].eq('BR')
    )
    ixs = admin3[i_fill].index
    aids = set(admin3['admin3_id'])
    for ix in ixs:
        admin2_id = admin3.loc[ix, 'admin2_id']
        name = admin3.loc[ix, '_name'].upper().replace(' ', '').replace('-', '')
        for x1, x2, x3 in combinations(name, 3):
            admin3_id = admin2_id + STRING_SEPARATOR_WITHIN_IDS + x1 + x2 + x3
            if admin3_id not in aids:
                admin3.loc[ix, 'admin3_id'] = admin3_id
                admin3.loc[ix, 'admin3_id_source'] = 'br.any3'
                aids.add(admin3_id)
                break

    # Exception: Uruguay has no names, gets generic codes (X01, X02, etc.)
    i_uy = admin3['admin1_id'].eq('UY')
    numbers = pd.Series(
        admin3[i_uy]
        .groupby('admin2_id')
        .apply(lambda x: pd.Series(range(1, len(x) + 1)), include_groups=False)
    )
    numbers.index = admin3[i_uy].index
    admin3.loc[i_uy, 'admin3_id'] = (
        admin3[i_uy]['admin2_id']
        + STRING_SEPARATOR_WITHIN_IDS
        + 'X'
        + numbers.astype(str).str.zfill(2)
    )
    admin3.loc[i_uy, 'admin3_id_source'] = 'uy'

    # Exception: Unnamed units with generic digits
    # Usually zones in cities, found in Vietnam, Praha (Prague), Guatemala
    i = admin3['admin3_id'].isnull() & admin3['name'].str.contains(
        ' [0-9]{1,2}$'
    ).fillna(False)
    initials = admin3['_name'].str.slice(0, 1)
    n_digits = i.groupby([admin3[i]['admin2_id'], initials[i]]).size()
    N_DIGITS_PER_AID1_MIN = 3
    for admin2_id, initial in n_digits[n_digits.ge(N_DIGITS_PER_AID1_MIN)].index:
        i_fill = i & admin3['admin2_id'].eq(admin2_id) & initials.eq(initial)
        digits = admin3[i_fill]['name'].str.extract(' ([0-9]{1,2})$')[0]
        # If digits are not unique, overwrite with unique digits
        if not len(set(digits)) == len(digits):
            digits = pd.Series(range(1, len(digits) + 1), index=digits.index).astype(
                str
            )
        n_zfill = int(np.ceil(np.log(i_fill.sum()) / np.log(10)))
        aids = (
            admin2_id
            + STRING_SEPARATOR_WITHIN_IDS
            + initial
            + digits.str.zfill(n_zfill)
        )
        admin3.loc[i_fill, 'admin3_id'] = aids
        admin3.loc[i_fill, 'admin3_id_source'] = 'a00'

    # Third priority: initials of first two words
    i_fill = admin3['admin3_id'].isnull() & admin3['_name'].notnull()
    aid_caps = admin3[i_fill]['_name'].str.extract('^([A-Z]).*?([A-Z])')
    aid_caps = aid_caps[aid_caps.notnull().mean(1).eq(1)].apply(''.join, 1)
    aid_caps = (
        admin3.loc[aid_caps.index]['admin2_id'] + STRING_SEPARATOR_WITHIN_IDS + aid_caps
    )
    aid_caps = aid_caps[~aid_caps.isin(admin3['admin3_id']) & ~aid_caps.duplicated()]
    admin3.loc[aid_caps.index, 'admin3_id'] = aid_caps
    admin3.loc[aid_caps.index, 'admin3_id_source'] = 'initials'

    # Fourth priority: first two letters
    i_fill = (
        admin3['admin3_id'].isnull()
        & admin3['_name'].notnull()
        & admin3['_name'].ne('')
    )
    aid_two = (
        admin3[i_fill]['admin2_id']
        + STRING_SEPARATOR_WITHIN_IDS
        + admin3[i_fill]['_name'].str.upper().str.slice(0, 2)
    )
    aid_two = aid_two[
        ~aid_two.isin(admin3['admin3_id'])
        & ~aid_two.duplicated()
        & aid_two.str.len().ge(6)
    ]
    admin3.loc[aid_two.index, 'admin3_id'] = aid_two
    admin3.loc[aid_two.index, 'admin3_id_source'] = 'first2'

    # Fifth priority: any two letters from the name
    i_fill = admin3['admin3_id'].isnull() & admin3['_name'].notnull()
    ixs = admin3[i_fill].index
    aids = set(admin3['admin3_id'])
    for ix in ixs:
        admin2_id = admin3.loc[ix, 'admin2_id']
        name = admin3.loc[ix, '_name'].upper().replace(' ', '').replace('-', '')
        for x1, x2 in combinations(name, 2):
            admin3_id = admin2_id + STRING_SEPARATOR_WITHIN_IDS + x1 + x2
            if admin3_id not in aids:
                admin3.loc[ix, 'admin3_id'] = admin3_id
                admin3.loc[ix, 'admin3_id_source'] = 'any2'
                aids.add(admin3_id)
                break

    # Sixth priority: rename existing aids to make space for others
    i_fill = admin3['admin3_id'].isnull() & admin3['_name'].notnull()
    ixs = admin3[i_fill].index
    aids = set(admin3['admin3_id'])
    for ix in ixs:
        admin2_id = admin3.loc[ix, 'admin2_id']
        name = admin3.loc[ix, '_name'].upper().replace(' ', '').replace('-', '')
        for x1, x2 in combinations(name, 2):
            admin3_id = admin2_id + STRING_SEPARATOR_WITHIN_IDS + x1 + x2

            rep = admin3[admin3['admin3_id'].eq(admin3_id)]
            if len(rep) == 0:
                print('How did I miss this? ' + admin3_id)
                continue

            ix2 = rep.iloc[0].name
            name_rep = rep.iloc[0]['_name'].upper().replace(' ', '').replace('-', '')

            replacement_found = False
            for y1, y2 in combinations(name_rep, 2):
                admin3_id_rep = admin2_id + STRING_SEPARATOR_WITHIN_IDS + y1 + y2
                if admin3_id_rep not in aids:
                    replacement_found = True
                    admin3.loc[ix2, 'admin3_id'] = admin3_id_rep
                    admin3.loc[ix2, 'admin3_id_source'] = 'replaced'
                    aids.add(admin3_id_rep)
                    break

            if replacement_found:
                admin3.loc[ix, 'admin3_id'] = admin3_id
                admin3.loc[ix, 'admin3_id_source'] = 'any2'
                break

    # Last resort: filling in NAs
    i_fill = admin3['admin3_id'].isnull()
    numbers = pd.Series(
        admin3[i_fill]
        .groupby('admin2_id')
        .apply(lambda x: pd.Series(range(1, len(x) + 1)), include_groups=False)
    )
    numbers.index = admin3[i_fill].index
    admin3.loc[i_fill, 'admin3_id'] = (
        admin3[i_fill]['admin2_id']
        + STRING_SEPARATOR_WITHIN_IDS
        + 'X'
        + numbers.astype(str)
    )
    admin3.loc[i_fill, 'admin3_id_source'] = 'filled'

    # Catch issues with nulls and duplicates
    admin3_id_isnull = admin3['admin3_id'].isnull()
    admin3_id_duplicated = admin3['admin3_id'].duplicated(keep=False)
    if admin3_id_isnull.any() or admin3_id_duplicated.any():
        message = 'Unable to resolve all AdminIds from GADM Level-2.\n\n'
        if admin3_id_isnull.any():
            message += 'Nulls:\n\n' + str(admin3[admin3_id_isnull])
        if admin3_id_duplicated.any():
            message += 'Duplicates:\n\n' + str(
                admin3[admin3_id_duplicated].sort_values('admin3_id')[
                    ['admin3_id_hasc', 'admin3_id', 'name']
                ]
            )
        raise Exception(message)

    return admin3.set_index('admin3_id').drop(columns='_name')


def admin3_id_index_from_admin3_US_nhgis(admin3_local):
    # Join states
    admin2_recipe = get_recipe('US', 'admin-nhgis-2020', filename='admin2')
    admin2_crosswalk = (
        get_admin(level=2, recipe=admin2_recipe, columns=['admin2_id_admin1'])
        .reset_index()
        .set_index('admin2_id_admin1')
    )
    admin3_local = admin3_local.join(admin2_crosswalk, on='admin2_id_admin1')

    # Create name-based identifier
    admin3_local['name_link'] = admin3_local['name'].apply(create_comparable_name_link)

    # Add ' city' to the name_link for duplicate name + state
    # (e.g. Baltimore county vs. city)
    i_city_duplicates = admin3_local[['admin2_id', 'name']].duplicated(
        keep=False
    ) & admin3_local['name_long'].eq(admin3_local['name'] + ' city')
    admin3_local.loc[i_city_duplicates, 'name_link'] += ' city'

    # Load global reference layer (GADM)
    admin3 = get_admin('US', level=3)
    admin3['admin2_id'] = admin3.index.str.slice(0, 5)

    # Correct (replace) names from global reference layer to official
    admin3_name_crosswalk = get_recipe(
        'US', 'admin-nhgis-2020', filename='admin3-names-from-gadm'
    )
    for _, row in admin3_name_crosswalk.iterrows():
        admin3.loc[
            admin3['admin2_id'].eq(row['admin2_id'])
            & admin3['name'].eq(row['admin3_name_gadm']),
            'name',
        ] = row['admin3_name_official']

    admin3['name_link'] = admin3['name'].str.lower().apply(create_comparable_name_link)

    # Join global admin-2 data (with identifier) to local admin-2 data
    admin3_local = admin3_local.join(
        admin3.reset_index().set_index(['admin2_id', 'name_link'])['admin3_id'],
        on=['admin2_id', 'name_link'],
    )

    # Set new admin3_ids for units that don't exist in the global layer
    new_admin3_ids = get_recipe(
        'US',
        'admin-nhgis-2020',
        filename='admin3-ids',
        dtype={'admin3_id_admin1': str},
    ).set_index('admin3_id_admin1')

    for admin3_id_admin1, admin3_id in new_admin3_ids['admin3_id'].items():
        mask_replace = admin3_local['admin3_id_admin1'].eq(admin3_id_admin1)
        admin3_local.loc[mask_replace, 'admin3_id'] = admin3_id

    # Ensure the IDs are complete and unique
    i_null = admin3_local['admin3_id'].isnull()
    if i_null.any():
        raise ValueError('Empty `admin3_id`:\n' + str(admin3_local[i_null]))

    i_dupl = admin3_local['admin3_id'].duplicated(keep=False)
    if i_dupl.any():
        raise ValueError(
            'Duplicate `admin3_id`:\n'
            + str(
                admin3_local[i_dupl][
                    ['admin3_id_admin1', 'name', 'name_long', 'admin3_id']
                ]
            )
        )

    return admin3_local.set_index('admin3_id')


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
    new_admin_id_col='admin3_id',
    parent_admin_id_col='admin2_id',
    name_col='name',
    name_long_col=None,
    id_separator=STRING_SEPARATOR_WITHIN_IDS,
    verbose=False,
):
    """
    Generate unique two-letter admin unit codes within parent units.

    Generate unique admin ID codes for administrative units

    Level-agnostic design: works for any parent-child relationship:
    admin2->admin3 (state->county), admin3->admin4 (county->town)

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
        Name for the new administrative ID column (default 'admin4_id')
    parent_admin_id_col : str
        Column name containing parent admin ID (e.g., 'admin3_id')
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
