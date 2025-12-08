"""
Administration

Worldwide administrative referencing and mapping

- Manage global admin files
- Manage globally unique identifiers (admin_ids)
"""

from dataclasses import dataclass, field
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
from openplaces.recipe import get_recipe
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
        STRING_SEPARATOR_WITHIN_IDS.join, 1
    )
    admin1_id_caps = admin1.loc[admin1_id_caps.index]['admin0_id'] + admin1_id_caps
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
    aid_from_hasc2 = admin2['admin2_id_hasc'].str.replace(
        '.', STRING_SEPARATOR_WITHIN_IDS, regex=False
    )
    aid_from_hasc2_c = (
        admin2['admin1_id']
        + STRING_SEPARATOR_WITHIN_IDS
        + aid_from_hasc2.str.slice(4, 6)
    )
    i = (
        aid_from_hasc2_c.notnull()
        & aid_from_hasc2_c.str.len().ge(6)
        & ~aid_from_hasc2_c.duplicated(False)
    )
    admin2.loc[i, 'admin2_id'] = aid_from_hasc2_c[i]
    admin2.loc[i, 'admin2_id_source'] = np.where(
        aid_from_hasc2_c[i] == aid_from_hasc2[i], 'hasc2', 'hasc2.mod'
    )

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
    i_fill = admin2['admin2_id'].isnull() & admin2['_name'].notnull()
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

    if admin2['admin2_id'].isnull().any() or admin2['admin2_id'].duplicated().any():
        raise Exception('Unable to resolve all AdminIds from GADM Level-2.')

    return admin2.set_index('admin2_id').drop(columns='_name')


def admin2_id_index_from_admin2_US_nhgis(admin2_local):
    # Join states
    admin1_recipe = get_recipe('US', 'admin', source='admin1-nhgis-2020')
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
        'US', 'admin', filename='admin2-gadm-4~1_names_to_official'
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
        'admin',
        filename='admin2-nhgis-admin2_ids',
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
