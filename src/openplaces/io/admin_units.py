"""The flat unit registry: one row per administrative unit, any level.

The spine is stored as one CSV per level, which makes depth the storage key
rather than an attribute. That works only while every branch of the world
has the same depth. It stops working as soon as one does not: if
Connecticut's towns sit at level 3 because the state has had no county
government since 1960, while California's sit at level 4, then "all towns in
the United States" spans two files and no query returns it.

This module builds a single table that answers those questions
independently of depth:

    admin_id, parent_admin_id, depth, type, name, in_path, valid_from,
    valid_to

`parent_admin_id` is explicit rather than implied by the identifier string.
That matters for units that exist but do not appear in the path -- an
abolished county kept for aggregation, say -- because a string can only
encode the hierarchy it is part of. Those units carry `in_path` false and
are reached through the region registry (`openplaces.io.readers.get_regions`)
rather than as a path segment.

The per-level CSVs stay authoritative until the migration completes. This
builder derives the flat table from them and is checked by regenerating the
per-level views and comparing them back, so the new storage is proven
faithful before anything depends on it.
"""

import pandas as pd

from openplaces.core.schema import STRING_SEPARATOR_WITHIN_IDS
from openplaces.recipe import get_recipe_by_id

SPINE_RECIPE = 'admin-spine-2026'
MAX_LEVEL = 4

UNIT_COLUMNS = [
    'admin_id',
    'parent_admin_id',
    'depth',
    'type',
    'name',
    'in_path',
    'valid_from',
    'valid_to',
]


def parent_of(admin_id: str) -> str:
    """Return the parent identifier, or empty for a top-level unit.

    Works at any depth, which is why it survives the migration: the parent
    is the identifier minus its last segment whether the unit sits at level
    two or level four.

    Parameters
    ----------
    admin_id : str
        Administrative identifier, e.g. 'US-MA-MI'.

    Returns
    -------
    str
        Parent identifier, or '' when there is none.

    Examples
    --------
    >>> parent_of('US-MA-MI')
    'US-MA'
    >>> parent_of('US')
    ''
    """
    parts = str(admin_id).split(STRING_SEPARATOR_WITHIN_IDS)
    return STRING_SEPARATOR_WITHIN_IDS.join(parts[:-1]) if len(parts) > 1 else ''


def load_level(level: int) -> pd.DataFrame:
    """Return one level's committed spine CSV.

    Parameters
    ----------
    level : int
        Administrative level, 1 through 4.

    Returns
    -------
    pandas.DataFrame
        The level's rows, all columns as strings.
    """
    return get_recipe_by_id(
        f'{SPINE_RECIPE}_admin{level}', dtype=str, keep_default_na=False
    )


def build_units(max_level: int = MAX_LEVEL) -> pd.DataFrame:
    """Build the flat unit registry from the per-level spine CSVs.

    Changes no identifier: this is a re-shaping of what is already
    committed, so that the two representations can be compared before
    either is trusted over the other.

    Parameters
    ----------
    max_level : int, optional
        Highest level to include, default 4.

    Returns
    -------
    pandas.DataFrame
        One row per unit, columns as in UNIT_COLUMNS, sorted by admin_id.
    """
    frames = []
    for level in range(1, max_level + 1):
        level_rows = load_level(level)
        id_column = f'admin{level}_id'
        units = pd.DataFrame(
            {
                'admin_id': level_rows[id_column],
                'parent_admin_id': level_rows[id_column].map(parent_of),
                'depth': level,
                'type': level_rows['type'] if 'type' in level_rows else '',
                'name': level_rows['name'] if 'name' in level_rows else '',
                # Every unit is on the path today. Level omission has not
                # happened yet, so writing False anywhere here would be an
                # assertion the current data does not support.
                'in_path': True,
                'valid_from': '',
                'valid_to': '',
            }
        )
        frames.append(units[units['admin_id'].str.strip() != ''])

    units = pd.concat(frames, ignore_index=True)
    return units.sort_values('admin_id').reset_index(drop=True)[UNIT_COLUMNS]


def to_level_view(units: pd.DataFrame, level: int) -> pd.DataFrame:
    """Regenerate one per-level view from the flat registry.

    The inverse of :func:`build_units` for a single level, used to prove
    the flat table loses nothing. Only the columns the flat table carries
    are reconstructed; the per-level CSVs hold further columns
    (name_original, the source foreign keys) that the registry does not
    duplicate.

    Parameters
    ----------
    units : pandas.DataFrame
        The flat registry.
    level : int
        Level to extract.

    Returns
    -------
    pandas.DataFrame
        Columns admin{level}_id, name and type, sorted by identifier.
    """
    rows = units[units['depth'].astype(int) == level].copy()
    view = pd.DataFrame(
        {
            f'admin{level}_id': rows['admin_id'],
            'name': rows['name'],
            'type': rows['type'],
        }
    )
    return view.sort_values(f'admin{level}_id').reset_index(drop=True)


def orphans(units: pd.DataFrame) -> pd.DataFrame:
    """Return units whose declared parent is not itself a registered unit.

    A non-empty result is a real defect: it means a unit hangs off a parent
    the registry cannot resolve, which is what silently drops rows in a
    join. Reported rather than repaired, because the fix depends on which
    side is wrong.

    Parameters
    ----------
    units : pandas.DataFrame
        The flat registry.

    Returns
    -------
    pandas.DataFrame
        The offending rows, empty when the hierarchy is closed.
    """
    known = set(units['admin_id'])
    has_parent = units['parent_admin_id'].str.strip() != ''
    missing = has_parent & ~units['parent_admin_id'].isin(known)
    return units[missing]
