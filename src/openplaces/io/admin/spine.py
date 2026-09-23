"""
Update the committed admin spine from a level's recipe output,
keeping the columns a restricted source does not permit out of the
published copy and registering where each unit's codes came from.
"""

import re

import numpy as np
import pandas as pd

from openplaces.core.constants import (
    REGEX_ADMIN_TYPE_EXTRACT,
    STRING_SEPARATOR_WITHIN_IDS,
)
from openplaces.io.readers import get_admin
from openplaces.path import recipe_path
from openplaces.recipe import (  # noqa: F401
    find_admin_recipe_id,
    get_output_path,
    get_recipe,
    get_recipe_by_id,
)

_RESTRICTED_SPINE_COLUMNS = ('name_original', 'name_alternatives')


def _redistribution_restricted(recipe) -> bool:
    """Return True when a recipe's source may not be redistributed.

    Parameters
    ----------
    recipe : dict
        A loaded recipe. One carrying no source counts as open.

    Returns
    -------
    bool
        The source's `redistribution_restricted` flag.
    """
    entity = recipe.get('entity') or recipe.get('dataset') or {}
    source = getattr(entity, 'source', None)
    if source is None and isinstance(entity, dict):
        source = entity.get('source')
    if isinstance(source, dict):
        return bool(source.get('redistribution_restricted'))
    return bool(getattr(source, 'redistribution_restricted', False))


def _publishable_spine_columns(frame, level, restricted):
    """Drop the columns a restricted source may not put into the spine.

    Parameters
    ----------
    frame : pandas.DataFrame
        An admin recipe's output, indexed by admin id.
    level : int
        Admin level of the spine being updated.
    restricted : bool
        Whether the recipe's source forbids redistribution.

    Returns
    -------
    pandas.DataFrame
        `frame` itself for an open source; otherwise a copy without the
        source's own codes and its alternative and native-script names.
    """
    if not restricted:
        return frame
    drop = [
        c
        for c in frame.columns
        if c in _RESTRICTED_SPINE_COLUMNS
        or re.fullmatch(rf'admin{level}_id_admin[0-9]', c)
    ]
    return frame.drop(columns=drop)


def _register_code_source(level, recipe_id, admin_ids):
    """Record which recipe supplies the national codes of each country.

    `admin-spine-2026_code-sources.csv` names, per country and level,
    the recipe the spine's `admin{level}_id_admin1` codes come from, and
    `tests/io/test_admin_spine_provenance.py` refuses a code with no row
    there. Registering in the step that copies the codes keeps a new
    country's provenance in the same change as its codes.

    Parameters
    ----------
    level : int
        Admin level whose codes were copied.
    recipe_id : str
        Recipe the codes came from.
    admin_ids : iterable of str
        Units that received a code.
    """
    path = recipe_path(None, 'admin-spine-2026', filename='code-sources.csv')
    table = (
        pd.read_csv(path, dtype=str, keep_default_na=False)
        if path.exists()
        else pd.DataFrame(columns=['admin1_id', 'level', 'recipe_id', 'scheme'])
    )
    countries = {str(a).split(STRING_SEPARATOR_WITHIN_IDS, 1)[0] for a in admin_ids}
    for country in sorted(countries):
        row = (table['admin1_id'] == country) & (table['level'] == str(level))
        if row.any():
            table.loc[row, 'recipe_id'] = recipe_id
        else:
            table.loc[len(table)] = [country, str(level), recipe_id, '']
    table.to_csv(path, index=False, encoding='utf-8', lineterminator='\n')


def update_admin_spine(
    level, admin_recipe_id, test, silent=False, replace_countries=False
):
    """Update the `openplaces` admin spine with admin recipe info

    Parameters
    ----------
    level : int
        Administrative level of the spine to update
    admin_recipe : str
        ID of admin recipe to update the spine with
        (This function assumes the recipe is already ingested.)
    test : bool
        If True, writes to '{file}_test.csv' instead of the original
    silent : bool
        If True, silences printouts when new admin IDs are added.
    replace_countries : bool
        For a global recipe, replace the whole slice of every country it
        covers (and no national recipe claims at this level) rather than
        only adding units the spine lacks. This is how a country's units
        move from one global source to another: rows the new source does
        not name are dropped, not carried.
    """

    admin_recipe = get_recipe_by_id(admin_recipe_id)
    admin_id_prefix = str(admin_recipe['admin_id'])

    # Load admin spine
    admin_spine = get_admin(level=level, all_columns=True)
    # Load the recipe's own output directly, not through `get_admin`:
    # `get_admin` always outer-joins onto the (possibly stale) spine, so
    # for a country/region-scoped recipe it returns the *whole world's*
    # spine at this level with only the matching rows enriched -- never
    # just the recipe's own rows -- which would make every check below
    # against a stale, superseded admin_id a false negative.
    admin_local = pd.read_parquet(
        get_output_path(admin_recipe, admin_id=admin_recipe['admin_id'])
    )
    # A source that may not be redistributed contributes identity only:
    # which units exist and what they are called. Its own codes and its
    # alternative and native-script spellings stay out of the spine,
    # which ships with the package. GADM is the case in point.
    admin_local = _publishable_spine_columns(
        admin_local, level, _redistribution_restricted(admin_recipe)
    )

    if admin_id_prefix:
        # A country/region-scoped recipe (e.g. GB_admin-ons-2024_adminN)
        # is authoritative for every admin_id under its own prefix --
        # replace the whole slice rather than only adding rows to it, so
        # a source it supersedes (e.g. GADM) can't leave stale,
        # geometry-less rows behind in the spine.
        # The slice ends at a level boundary. `admin_id_prefix` is a
        # unit id, never a free-form prefix, so a bare string test let
        # a recipe scoped to the pre-2026 'US-NC-WA' (Wake) delete
        # 'US-NC-WAR' (Warren) and every unit under it from the spine.
        owned = (
            admin_spine.index == admin_id_prefix
        ) | admin_spine.index.str.startswith(
            admin_id_prefix + STRING_SEPARATOR_WITHIN_IDS
        )
        admin_spine = admin_spine[~owned]
    else:
        # A global recipe (e.g. GADM) must not (re-)populate admin_ids
        # under an admin1 unit that already has its own, more specific
        # admin recipe registered for this level -- that recipe is
        # authoritative there, whether or not it has been spine-updated
        # yet, so a later GADM refresh can't resurrect what it replaced.
        admin1_ids = {admin_id.split('-', 1)[0] for admin_id in admin_local.index}
        superseded_admin1_ids = {
            admin1_id
            for admin1_id in admin1_ids
            if find_admin_recipe_id(admin1_id, level, silent=True)
            not in (None, admin_recipe_id)
        }
        if superseded_admin1_ids:
            admin1_of = admin_local.index.to_series().str.split('-', n=1).str[0]
            admin_local = admin_local[~admin1_of.isin(superseded_admin1_ids)]
        if replace_countries:
            covered = admin1_ids - superseded_admin1_ids
            spine_admin1 = admin_spine.index.to_series().str.split('-', n=1).str[0]
            admin_spine = admin_spine[~spine_admin1.isin(covered)]

    # Initiate new admin spine
    new_admin_spine = admin_spine.copy()

    # Create new entries
    new_admin_ids = sorted(set(admin_local.index) - set(admin_spine.index))
    if new_admin_ids:
        if not silent:
            print(
                'Adding: '
                + ', '.join(new_admin_ids[:5])
                + (
                    f', and {len(new_admin_ids) - 5:,d} more.'
                    if len(new_admin_ids) > 5
                    else ''
                )
            )

        new_admin_entries = admin_local.loc[new_admin_ids].copy()

        if 'name_long' in new_admin_entries:
            # Extract the trailing admin-type word from the long name (e.g.
            # 'Census Area', 'Borough', 'City', 'Municipio'). Language-agnostic:
            # the term list lives in `REGEX_ADMIN_TYPE_EXTRACT`.
            new_admin_entries['type'] = (
                new_admin_entries['name_long']
                .str.title()
                .str.extract(REGEX_ADMIN_TYPE_EXTRACT)
            )
            # Disambiguate units that share a name with another unit under the
            # same parent (e.g. an independent city vs. its county) by appending
            # the lowercased type word to the name. Generic: fires only on a
            # genuine name collision, so it is a no-op where names are unique.
            sep = STRING_SEPARATOR_WITHIN_IDS

            def _parents(index):
                return index.to_series().str.rsplit(sep, n=1).str[0].to_numpy()

            spine_parents = _parents(admin_spine.index)
            new_parents = _parents(new_admin_entries.index)
            spine_names = admin_spine['name'].to_numpy()
            new_names = new_admin_entries['name'].to_numpy()
            combined = pd.DataFrame(
                {
                    'parent': np.concatenate([spine_parents, new_parents]),
                    'name': np.concatenate([spine_names, new_names]),
                }
            )
            dup = combined.duplicated(['parent', 'name'], keep=False)
            dup_pairs = set(map(tuple, combined[dup][['parent', 'name']].to_numpy()))
            i_collision = (
                np.array([(p, n) in dup_pairs for p, n in zip(new_parents, new_names)])
                & new_admin_entries['type'].notna().to_numpy()
            )
            new_admin_entries.loc[i_collision, 'name'] = (
                new_admin_entries.loc[i_collision, 'name']
                + ' '
                + new_admin_entries.loc[i_collision, 'type'].str.lower()
            )

        # A source's own stable key (`admin{N}_id_wikidata`) and native
        # name are worth carrying; the spine gains the column on first
        # sight rather than dropping the value.
        for column in (f'admin{level}_id_wikidata', 'name_original'):
            if column in new_admin_entries and column not in admin_spine:
                admin_spine[column] = ''
                new_admin_spine[column] = ''

        # Align columns
        new_admin_entries = new_admin_entries[
            [v for v in new_admin_entries if v in admin_spine]
        ]

        new_admin_spine = pd.concat([new_admin_spine, new_admin_entries]).sort_index()

    # Save official IDs worth keeping (e.g. FIPS codes)
    admin_id_columns = sorted(
        c for c in admin_local.columns if re.match(rf'admin{level}_id_admin[0-9]$', c)
    )
    if admin_id_columns:
        # Keep the first one of the sorted columns (should be highest
        # official ID, one used by the country, over one used by state)
        codes = admin_local[admin_id_columns[0]]
        new_admin_spine.loc[admin_local.index, admin_id_columns[0]] = codes
        if not test:
            coded = codes.notna() & (codes.astype(str).str.strip() != '')
            _register_code_source(
                level, admin_recipe_id, admin_local.index[coded.to_numpy()]
            )

    # Write in the same byte-exact form as `build.remint_spine`: no BOM,
    # and LF on every platform, so either writer reproduces the file.
    admin_recipe_path = recipe_path(
        None,
        'admin-spine-2026',
        filename=f'admin{level}' + ('_test' if test else '') + '.csv',
    )
    new_admin_spine.to_csv(admin_recipe_path, encoding='utf-8', lineterminator='\n')
