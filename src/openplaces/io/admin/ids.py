"""
Build the id index of one admin level from the codes its units
already carry: ISO 3166 for countries and regions, GADM and HASC
for districts, ONS and GISCO LAU for level-4 units, and the NUTS
crosswalk. Each builder returns the frame indexed by the id it
minted.
"""

import warnings
from itertools import combinations

import numpy as np
import pandas as pd

from openplaces.core.constants import (
    ADMIN1_IDS_USING_HASC1_FOR_ADMIN2,
    REGEX_ADMIN2_IDS_AA_AA,
    REGEX_ADMIN2_IDS_AA_AA_EXTRACT,
    REGEX_ADMIN2_IDS_HASC,
    REGEX_ADMIN3_IDS_HASC,
    STRING_SEPARATOR_WITHIN_IDS,
)
from openplaces.io.admin_codes import assign_admin_ids
from openplaces.io.readers import get_admin, get_dataset
from openplaces.recipe import (
    get_output_path,
    get_recipe,
    get_recipe_by_id,
)
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
        get_recipe(
            None,
            'admin-iso-20210319',
            filename='admin1-alpha-2',
            keep_default_na=False,
        )
        .rename(columns=ADMIN1_ISO_RENAME_COLUMNS)
        .query('admin1_id != ""')
        .set_index('admin1_id')
        .sort_index()[['name', 'admin1_id_a3']]
    )

    admin1_iso_additions = get_recipe(
        None, 'admin-openplaces-2026', filename='admin1-additions'
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
            'admin-iso-20240619',
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
    """Give dataframe `admin` an `admin2_id` index based on GADM data

    See Also
    --------
    openplaces.io.admin_codes.assign_admin_ids : the successor, now wired
        into `admin-gadm-4~1_admin2`.

    Notes
    -----
    Superseded, and no longer called by any recipe. It mints an identifier
    from GADM's own ISO and HASC codes, which is what the spine did before
    the 2026 re-mint; the successor instead reproduces the code the
    committed spine already records for a unit and assigns only units the
    spine does not name. Because GADM's sibling groups are not the spine's
    (45,966 level-3 units against 48,695), minting here could not reproduce
    the spine even under the current rules. Kept as the record of how
    pre-2026 identifiers were derived; do not wire it into new recipes.
    """

    # Join with level-2 administrative units
    admin2 = admin2.join(
        get_admin1_iso().reset_index().set_index('admin1_id_a3')['admin1_id'],
        on='admin1_id_a3',
        how='inner',
    )
    if admin2['admin1_id'].isnull().any():
        raise ValueError("Empty `'admin1_id'` found in `admin2`.")

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
    """Give dataframe `admin3` an `admin3_id` index based on GADM data

    See Also
    --------
    openplaces.io.admin_codes.assign_admin_ids : the successor, now wired
        into `admin-gadm-4~1_admin3`.

    Notes
    -----
    Superseded, and no longer called by any recipe, for the reasons given
    on :func:`admin2_id_index_from_admin2_gadm`. The per-country exceptions
    below (Brazil's three-letter codes, Uruguay's unnamed units, the
    numbered city zones) are the shape of problem the successor's code
    derivation now handles generally. Kept as the record of how pre-2026
    identifiers were derived; do not wire it into new recipes.
    """
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
    HASC2_REGEX_EXTRACT = r'([A-Z0-9]{2})\.([A-Z0-9]{2})\.([A-Z0-9]{2})'
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


def admin3_id_index_from_local(admin3_local, admin2_recipe_id):
    """Index a country's local admin-3 source against the global spine.

    Joins the local admin-3 units to the global (GADM-derived) admin-3 layer by
    parent admin-2 and cleaned name, applies the country's name corrections and
    explicit ID overrides (CSV assets stored beside its admin recipe), and
    returns the frame indexed by ``admin3_id``.

    Wired from a recipe via ``create_index.function`` with
    ``args: {admin2_recipe_id}`` (``index_function:`` cannot pass args). The
    explicit recipe reference also declares the admin-2 data dependency to
    the flow DAG (recipe-ID keys are edge sources).

    Parameters
    ----------
    admin3_local : GeoDataFrame
        Local admin-3 source with at least ``name`` and ``admin2_id_admin1``
        (and usually ``name_long`` and ``admin3_id_admin1``).
    admin2_recipe_id : str
        Recipe ID of the admin-2 ingest of the same source (e.g.
        ``'US_admin-census-2021_admin2'``). Its country and admin entity
        also locate the crosswalk assets stored beside the recipe.
    """
    # Load the parent admin-2 recipe by ID and extract its components
    admin2_recipe = get_recipe_by_id(admin2_recipe_id)
    country_id = str(admin2_recipe['admin_id'])
    admin_entity = str(admin2_recipe['entity'])

    # Join states (admin-2)
    admin2_crosswalk = (
        get_admin(level=2, recipe=admin2_recipe, columns=['admin2_id_admin1'])
        .reset_index()
        .set_index('admin2_id_admin1')
    )
    admin3_local = admin3_local.join(admin2_crosswalk, on='admin2_id_admin1')

    # Create name-based identifier
    admin3_local['name_link'] = admin3_local['name'].apply(create_comparable_name_link)

    # Disambiguate duplicate names within a parent using the long name's type
    # suffix (e.g. Baltimore county vs. Baltimore city in the US). Only fires
    # where a ``name_long`` of "<name> city" is present, so it is a no-op for
    # countries without that convention.
    if 'name_long' in admin3_local:
        i_city_duplicates = admin3_local[['admin2_id', 'name']].duplicated(
            keep=False
        ) & admin3_local['name_long'].eq(admin3_local['name'] + ' city')
        admin3_local.loc[i_city_duplicates, 'name_link'] += ' city'

    # The spine, which is openplaces' own layer.
    admin3 = get_admin(country_id, level=3)
    admin3['admin2_id'] = admin3.index.str.slice(0, 5)

    # A sidecar of GADM-spelling corrections used to run here, because
    # the spine's names came from GADM and a local source spells them
    # officially. The spine no longer carries GADM names: every one of
    # the 51 US corrections matched nothing, and Colombia's last three
    # were applied to the spine directly. Nothing is corrected on the
    # way past any more, and the tables that held GADM's strings are
    # gone with it.
    admin3['name_link'] = admin3['name'].str.lower().apply(create_comparable_name_link)

    # Join global admin-2 data (with identifier) to local admin-2 data
    admin3_local = admin3_local.join(
        admin3.reset_index().set_index(['admin2_id', 'name_link'])['admin3_id'],
        on=['admin2_id', 'name_link'],
    )

    # Set new admin3_ids for units that don't exist in the global layer
    new_admin3_ids = get_recipe(
        country_id,
        admin_entity,
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
        report_cols = [
            c
            for c in ['admin3_id_admin1', 'name', 'name_long', 'admin3_id']
            if c in admin3_local
        ]
        raise ValueError(
            'Duplicate `admin3_id`:\n' + str(admin3_local[i_dupl][report_cols])
        )

    return admin3_local.set_index('admin3_id')


def admin3_id_index_from_admin3_code(gdf, country_id, code_column='admin3_id_admin1'):
    """Index a local admin-3 layer by joining its national code to the spine.

    The New England states are why this exists. Their towns govern and
    their counties do not, so the spine carries towns at level 3 -- and a
    Census county subdivision *is* the town, meaning its 10-digit GEOID
    already identifies a spine unit exactly. There is nothing to infer:
    no name matching, no prefix truncation, no parent lookup.

    That makes this the plainest of the index functions, and deliberately
    so. `admin3_id_index_from_local` joins on parent plus cleaned name
    because its sources carry no shared code; using it here would
    reintroduce name ambiguity (Massachusetts has both a Bridgewater and
    a West Bridgewater, and eight of its town names recur in other New
    England states) to resolve a code that is already unique.

    Parameters
    ----------
    gdf : GeoDataFrame
        Local admin-3 source carrying *code_column*.
    country_id : str
        Admin id whose level-3 spine to resolve against, e.g. `'US-MA'`.
        Scoping to the state keeps the join small and makes an unmatched
        row a real error rather than a cross-state near-miss.
    code_column : str, optional
        Column holding the national code (default `'admin3_id_admin1'`).

    Raises
    ------
    ValueError
        If *code_column* is absent, if the spine offers no codes to join
        against, if any row fails to match, or if two rows resolve to the
        same spine unit. An admin layer is what every other dataset is
        keyed on, so a partial or ambiguous index is worse than no index
        at all.
    """
    if code_column not in gdf:
        raise ValueError(
            f"Cannot index by '{code_column}': column not in the source "
            f'({sorted(gdf.columns)}).'
        )

    # 'name' as well as the code: the fallback below needs it, and a
    # second get_admin call would re-read the same file.
    spine = get_admin(country_id, level=3, columns=[code_column, 'name'])
    codes = spine.reset_index().dropna(subset=[code_column])
    if codes.empty:
        raise ValueError(
            f"The level-3 spine for '{country_id}' carries no "
            f"'{code_column}' values to join against."
        )
    if codes[code_column].duplicated().any():
        duplicated = sorted(codes.loc[codes[code_column].duplicated(), code_column])
        raise ValueError(
            f"Non-unique '{code_column}' in the level-3 spine for "
            f"'{country_id}': {duplicated[:10]}"
        )

    gdf = gdf.copy()
    gdf['admin3_id'] = (
        gdf[code_column].astype(str).map(codes.set_index(code_column)['admin3_id'])
    )

    # Fall back to the name for anything the code join missed. A town
    # that becomes a city gets a new Census subdivision code while the
    # spine still holds the old one -- measured in Massachusetts for
    # Watertown (spine 2501773440, TIGER 2501773405), Easthampton,
    # Amesbury and Methuen, all four of which converted. The spine calls
    # them "Watertown Town" and TIGER calls them "Watertown", so
    # stripping the type word matches them without a hand-maintained
    # code crosswalk that would need editing again at the next
    # conversion.
    unmatched = gdf['admin3_id'].isnull()
    if unmatched.any() and 'name' in gdf:
        from openplaces.utils import create_comparable_name_link

        def _link(value):
            link = create_comparable_name_link(value)
            if not isinstance(link, str):
                return link
            for suffix in (' town', ' city', ' township', ' plantation'):
                if link.endswith(suffix):
                    link = link[: -len(suffix)]
                    break
            return link.strip()

        by_name = spine.reset_index()
        by_name['_link'] = by_name['name'].map(_link)
        by_name = by_name.drop_duplicates('_link').set_index('_link')['admin3_id']
        recovered = gdf.loc[unmatched, 'name'].map(_link).map(by_name)
        gdf.loc[unmatched, 'admin3_id'] = recovered
        # The recovered subset, not `recovered.index`, which is every
        # unmatched row: naming rows that did not match points the
        # reader at units the next block is about to raise on.
        matched_by_name = recovered[recovered.notna()]
        if len(matched_by_name):
            warnings.warn(
                f'{len(matched_by_name):,d} admin-3 unit(s) in '
                f"'{country_id}' matched by name after their "
                f"'{code_column}' changed: "
                + ', '.join(sorted(gdf.loc[matched_by_name.index, 'name'])[:8]),
                stacklevel=2,
            )

    unmatched = gdf['admin3_id'].isnull()
    if unmatched.any():
        report = [c for c in ('name', 'name_long', code_column) if c in gdf]
        raise ValueError(
            f'{unmatched.sum():,d} of {len(gdf):,d} admin-3 unit(s) in '
            f"'{country_id}' had no spine match on '{code_column}':\n"
            + str(gdf.loc[unmatched, report].head(20))
        )

    # The spine side of the join is unique by construction, but the
    # source side is not, and neither is the name fallback: two rows
    # whose names strip to the same link, or a name-recovered row
    # landing on a unit a code match already claimed, both hand the
    # same id to two rows. Indexing on it would silently double a unit
    # everything downstream is keyed on.
    duplicated = gdf['admin3_id'].duplicated(keep=False)
    if duplicated.any():
        report = [c for c in ('name', 'name_long', code_column) if c in gdf]
        raise ValueError(
            f'{duplicated.sum():,d} admin-3 unit(s) in {country_id!r} share '
            f'an `admin3_id` after matching on {code_column!r} and name:\n'
            + str(gdf.loc[duplicated, [*report, 'admin3_id']].head(20))
        )

    return gdf.set_index('admin3_id')


def admin4_id_index_from_gb_ons(gdf, admin3_recipe_id, lookup_recipe_id):
    """Give dataframe `gdf` an `admin4_id` index for GB LADs.

    Resolves each LAD's admin3 (County/UA) parent via the ONS
    LAD-to-County/UA lookup (`lookup_recipe_id`), falling back to the
    LAD's own code where the lookup has no row for it -- Scotland's
    council areas and Northern Ireland's local government districts are
    already single-tier, so their own LAD code IS their County/UA code,
    and the lookup (England/Wales only) has no row for them. The
    resolved County/UA code is then matched against `admin3_recipe_id`'s
    own `admin3_id_admin1` attribute (the same ONS code, set when
    GB_admin-ons-2024_admin3 was ingested) to find each LAD's admin3_id,
    and `assign_admin_ids` mints the admin4_id under that parent.

    Wired from a recipe via `create_index.function` with
    `args: {admin3_recipe_id, lookup_recipe_id}` (the explicit recipe
    references also declare these data dependencies to the flow DAG).

    Parameters
    ----------
    gdf : GeoDataFrame
        ONS LAD boundary layer with an `admin4_id_admin1` column (ONS
        `LAD24CD`-style GSS code).
    admin3_recipe_id : str
        Recipe ID of the already-ingested GB admin3 (County/UA) layer,
        e.g. `'GB_admin-ons-2024_admin3'`.
    lookup_recipe_id : str
        Recipe ID of the already-ingested LAD-to-County/UA lookup
        dataset, e.g. `'GB_reference-ons-2024_lad-to-ctyua'`.
    """
    gdf = gdf.copy()

    lookup = get_dataset(lookup_recipe_id, admin_id='GB')['admin3_id_admin1']
    gdf['admin3_id_admin1'] = gdf['admin4_id_admin1'].map(lookup)
    gdf['admin3_id_admin1'] = gdf['admin3_id_admin1'].fillna(gdf['admin4_id_admin1'])

    # Read the admin3 recipe's own freshly-ingested output directly, not
    # through `get_admin`'s default-spine merge: at this point in the
    # pipeline the persisted spine CSV hasn't been updated with this
    # recipe's new admin3_ids yet (that happens later, via
    # `update_admin_spine`), and merging in the stale GADM-era admin3
    # rows -- most of which have no `admin3_id_admin1` at all -- would
    # collide as duplicated nulls below.
    admin3_recipe = get_recipe_by_id(admin3_recipe_id)
    admin3 = pd.read_parquet(
        get_output_path(admin3_recipe, admin_id=admin3_recipe['admin_id']),
        columns=['admin3_id_admin1'],
    )
    admin3_id_by_code = admin3.reset_index().set_index('admin3_id_admin1')['admin3_id']
    if admin3_id_by_code.index.duplicated().any():
        raise ValueError(
            'Non-unique `admin3_id_admin1` in admin3 layer '
            f"'{admin3_recipe_id}'; cannot resolve LAD parentage."
        )

    gdf['admin3_id'] = gdf['admin3_id_admin1'].map(admin3_id_by_code)
    unmatched = gdf['admin3_id'].isnull()
    if unmatched.any():
        raise ValueError(
            'Unmatched admin3 (County/UA) parent for LAD(s):\n'
            + str(gdf.loc[unmatched, ['admin4_id_admin1', 'admin3_id_admin1']])
        )

    return assign_admin_ids(
        gdf, new_admin_id_col='admin4_id', parent_admin_id_col='admin3_id'
    )


def admin4_id_index_from_gisco_lau(gdf, admin3_country_id, admin3_code_lengths):
    """Give dataframe `gdf` an `admin4_id` index for a GISCO LAU country.

    Resolves each LAU's admin3 parent by prefix-truncating its national
    LAU code (`admin4_id_admin1`), rather than through a separate lookup
    table or name-matching: many EU countries build municipal codes by
    appending digits to their county/department code -- confirmed against
    real data for both countries this is first used for (German AGS:
    Barnim's municipalities' 8-digit codes, e.g. `12060020`, all share
    Barnim's own 5-digit Kreis code `12060` as their prefix; French
    INSEE: Paris's commune code `75056` shares Paris department's own
    code `75` as its 2-digit prefix). `admin3_code_lengths` lists the
    candidate prefix lengths to try, longest first, since a country's own
    admin3 codes aren't always uniform length (e.g. France: 2 digits for
    metropolitan departments, 3 for overseas ones).

    Unlike `admin4_id_index_from_gb_ons`, this does not replace admin3 --
    it matches against the *current, unmodified* default admin3 spine via
    `get_admin`, so the ordinary (recipe-less) `get_admin` path is safe:
    there are no freshly-minted, not-yet-in-spine admin3_ids to collide
    against.

    Wired from a recipe via `create_index.function` with
    `args: {admin3_country_id, admin3_code_lengths}`.

    Parameters
    ----------
    gdf : GeoDataFrame
        GISCO LAU boundary layer already filtered to one country, with an
        `admin4_id_admin1` column (the national LAU code).
    admin3_country_id : str
        Country admin_id whose admin3 layer to resolve parentage against,
        e.g. `'DE'`.
    admin3_code_lengths : sequence of int
        Candidate prefix lengths of `admin4_id_admin1` to try, longest
        first, until one matches an existing `admin3_id_admin1`.
    """
    admin3 = get_admin(admin3_country_id, level=3, columns=['admin3_id_admin1'])
    valid_codes = set(admin3['admin3_id_admin1'].dropna())

    gdf = gdf.copy()
    code = gdf['admin4_id_admin1'].astype(str)
    admin3_id_admin1 = pd.Series(pd.NA, index=gdf.index, dtype=object)
    for length in sorted(set(admin3_code_lengths), reverse=True):
        candidate = code.str.slice(0, length)
        fill = admin3_id_admin1.isna() & candidate.isin(valid_codes)
        admin3_id_admin1[fill] = candidate[fill]
    gdf['admin3_id_admin1'] = admin3_id_admin1

    # An imperfect crosswalk (dropped, not raised) mirrors the top-level
    # `admin_id_crosswalk` join's own handling elsewhere in the ingester:
    # a boundary reform can retire/renumber a Kreis/department between
    # GADM's admin3 vintage and GISCO's current LAU vintage faster than
    # GADM's own re-ingest catches up (observed for Germany: ~40 of
    # ~11,000 municipalities under a since-renumbered Kreis code, e.g. the
    # pre-"Region Hannover" county reform), so this is expected drift, not
    # a systemic mismatch -- but it is loud (a printed warning) rather
    # than silent, since a large unmatched count would signal a real bug.
    unmatched = gdf['admin3_id_admin1'].isnull()
    if unmatched.any():
        warnings.warn(
            f'\n\n{unmatched.sum():,d} LAU(s) had no admin3 match (no prefix of '
            f'{sorted(set(admin3_code_lengths), reverse=True)} characters '
            'matched an existing admin3_id_admin1) and will be dropped:\n\n'
            + str(gdf.loc[unmatched, ['name', 'admin4_id_admin1']])
            + '\n',
            stacklevel=2,
        )
        gdf = gdf.loc[~unmatched].copy()

    admin3_id_by_code = admin3.reset_index().set_index('admin3_id_admin1')['admin3_id']
    if admin3_id_by_code.index.duplicated().any():
        raise ValueError(
            f"Non-unique `admin3_id_admin1` in admin3 for '{admin3_country_id}'; "
            'cannot resolve LAU parentage.'
        )
    gdf['admin3_id'] = gdf['admin3_id_admin1'].map(admin3_id_by_code)

    return assign_admin_ids(
        gdf, new_admin_id_col='admin4_id', parent_admin_id_col='admin3_id'
    )


def nuts_crosswalk_index_from_admin4(df, admin4_recipe_id):
    """Index a NUTS crosswalk table by `admin4_id`.

    Shared by every country's NUTS-crosswalk recipe (GB's ONS-sourced
    LAD-to-ITL lookup and the GISCO-LAU-sourced ones used for the rest of
    NUTS-Europe): translates each row's national admin4 code
    (`admin4_id_admin1`) into the matching `admin4_id` from the
    already-ingested admin4 layer, by reading that recipe's own output
    directly rather than through `get_admin`'s default-spine merge (which
    would collide freshly-minted, not-yet-in-spine admin_ids against
    unrelated stale rows when the admin4 source itself was replaced, as
    it was for GB and for GISCO-LAU countries -- see
    `admin4_id_index_from_gb_ons` / `admin4_id_index_from_gisco_lau`).
    Any NUTS-code normalization (e.g. GB's `TL`->`UK` prefix swap) must
    already be done by the recipe's own `transformations` before this
    function runs; it only resolves the join key.

    Some source tables carry more than one row per admin4 unit (e.g. a
    handful of large, sparsely-populated Scottish council areas split
    across two ITL3/NUTS3 sub-regions despite being a single LAD). Since
    the crosswalk is keyed one-row-per-`admin4_id`, such a unit keeps
    only its first matching row (arbitrary tie-break) -- coarser NUTS
    levels are unaffected, being truncations of NUTS3.

    Wired from a recipe via `create_index.function` with
    `args: {admin4_recipe_id}`.

    Parameters
    ----------
    df : DataFrame
        A NUTS/ITL lookup with an `admin4_id_admin1` column (the national
        admin4 code used by both this table and the admin4 recipe below).
    admin4_recipe_id : str
        Recipe ID of the already-ingested admin4 layer for the same
        country, e.g. `'GB_admin-ons-2024_admin4'` or
        `'DE_admin-gisco-2024_admin4'`.
    """
    admin4_recipe = get_recipe_by_id(admin4_recipe_id)
    admin4 = pd.read_parquet(
        get_output_path(admin4_recipe, admin_id=admin4_recipe['admin_id']),
        columns=['admin4_id_admin1'],
    )
    admin4_id_by_code = admin4.reset_index().set_index('admin4_id_admin1')['admin4_id']
    if admin4_id_by_code.index.duplicated().any():
        raise ValueError(
            'Non-unique `admin4_id_admin1` in admin4 layer '
            f"'{admin4_recipe_id}'; cannot resolve NUTS crosswalk keys."
        )

    df = df.copy()
    df['admin4_id'] = df['admin4_id_admin1'].map(admin4_id_by_code)
    # Dropped (with a warning), not raised, for the same reason as the
    # admin3-parentage join in `admin4_id_index_from_gisco_lau`: some rows
    # here can't match the admin4 layer at all -- either they were
    # themselves dropped there (a Kreis/department renumbered between
    # vintages), or they're a statistical aggregate/placeholder code
    # (observed for Germany: codes ending `999`, e.g. `10042999`) that
    # was never a real LAU boundary feature to begin with.
    unmatched = df['admin4_id'].isnull()
    if unmatched.any():
        warnings.warn(
            f'\n\n{unmatched.sum():,d} crosswalk row(s) had no matching '
            f"admin4_id in '{admin4_recipe_id}' and will be dropped:\n\n"
            + str(df.loc[unmatched, ['admin4_id_admin1']])
            + '\n',
            stacklevel=2,
        )
        df = df.loc[~unmatched]

    df = df.drop_duplicates(subset='admin4_id', keep='first')
    return df.set_index('admin4_id')


# Letters without a Unicode decomposition: NFKD alone cannot fold these to ASCII
