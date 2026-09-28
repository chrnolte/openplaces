"""Track which countries have moved off GADM, one country at a time.

The spine's rows for about 190 countries came from GADM, which may not
be redistributed, and are being replaced from Wikidata (CC0). Doing that
in one pass means re-harvesting and re-minting the world before anything
can be checked, and a run that stops halfway leaves nothing behind.

It does not have to be one pass. `update_admin_spine(...,
replace_countries=True, admin_ids=[...])` replaces exactly the countries
it is given, and codes are minted among siblings, so one country's
identifiers never depend on another's. Every intermediate state is a
valid spine in which some countries are off GADM and the rest are not
yet.

What that needs is a record of where the work stands, kept in the
repository rather than in a session's notes, since the note a previous
session left behind was in a scratchpad and is gone. This module owns
that record: one row per country and level, with the counts each source
offers, the class the selection rule chose, and a status. A session
reads it, takes as many rows as its budget allows, and writes it back.

The measurements are advisory. `units_geoboundaries` and `agreement`
compare the harvest against an independent source, which is a cheap
signal that a country deserves a look before it is migrated, not a
verdict: geoBoundaries is missing for many countries, holds the wrong
tier for some, and is share-alike for others.
"""

import pandas as pd

from openplaces.io.admin_codes.anchors import normalize_name
from openplaces.io.readers import get_admin
from openplaces.path import recipe_path
from openplaces.recipe import find_admin_recipe_id, get_output_path, get_recipe_by_id

#: The recipe the ledger sits beside, and the file name it takes.
LEDGER_RECIPE = 'admin-wikidata-2026'
LEDGER_FILENAME = 'migration.csv'

#: The layer each country's rows are being moved to, and the independent
#: source the harvest is scored against.
UNIT_RECIPE = 'admin-wikidata-2026_admin{level}'
REFERENCE_RECIPE = 'admin-geoboundaries-6~0~0_admin{level}'

LEVELS = (2, 3, 4)

#: Where a country and level stands.
#:
#: pending    nothing done since the rule last changed
#: harvested  re-harvested under the current rule, not yet looked at
#: reviewed   a person read the counts and the class and accepted them
#: migrated   the spine's rows for it come from Wikidata
#: held       deliberately not migrated yet, with the reason in `note`
#: deferred   out of scope for now, so the rows were removed
#: unsourced  no open source names these units, so the spine has none
#:
#: The last three read alike and are not, and a reader who sees a
#: country missing deserves to know which it is.
#:
#: `held` is a to-do with rows still standing: a rule fix, a sidecar
#: override or a harvest at another level would rescue it, and until
#: then the spine carries the old source's units.
#:
#: `deferred` is a to-do with the rows gone. A source could name these
#: units and none has been wired up; the old rows could not stay while
#: that waits, because they are the source this project may not
#: publish. Re-running the country's recipe at that level is all it
#: takes to move it to `migrated`.
#:
#: `unsourced` is a finding rather than a to-do: the units exist, no
#: source this project may publish names them, and the shipped spine
#: therefore says nothing about them.
STATUSES = (
    'pending',
    'harvested',
    'reviewed',
    'migrated',
    'held',
    'deferred',
    'unsourced',
)

COLUMNS = (
    'admin1_id',
    'level',
    'status',
    'units_spine',
    'units_wikidata',
    'units_geoboundaries',
    'agreement',
    'unit_type',
    'harvested',
    'note',
)


def ledger_path():
    """Return the committed ledger's path, beside the Wikidata recipes."""
    return recipe_path(None, LEDGER_RECIPE, filename=LEDGER_FILENAME)


def load_ledger():
    """Return the ledger, or an empty one with the right columns.

    Returns
    -------
    pandas.DataFrame
        One row per country and level. Every column is read as text, so
        a status is never turned into a float by an empty column.
    """
    path = ledger_path()
    if not path.exists():
        return pd.DataFrame(columns=list(COLUMNS))
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in COLUMNS:
        if column not in table:
            table[column] = ''
    return table[list(COLUMNS)]


def save_ledger(table):
    """Write the ledger in the byte-exact form the spine files use.

    Parameters
    ----------
    table : pandas.DataFrame
        The ledger to write. Sorted by country and level, so a diff
        shows what changed rather than what moved.

    Returns
    -------
    pathlib.Path
        Where it was written.
    """
    path = ledger_path()
    filled = table.copy()
    for column in COLUMNS:
        if column not in filled:
            filled[column] = ''
    ordered = filled[list(COLUMNS)].fillna('')
    ordered = ordered.sort_values(['admin1_id', 'level'])
    ordered.to_csv(path, index=False, encoding='utf-8', lineterminator='\n')
    return path


def migrated(level=None):
    """Return the countries whose spine rows already come from Wikidata.

    Parameters
    ----------
    level : int, optional
        Restrict to one level. Unset returns every migrated pair.

    Returns
    -------
    set of str or set of tuple
        Country codes for one level, or (country, level) pairs.
    """
    table = load_ledger()
    done = table[table['status'] == 'migrated']
    if level is None:
        return {(row.admin1_id, int(row.level)) for row in done.itertuples()}
    return set(done.loc[done['level'] == str(level), 'admin1_id'])


def deferred(level=None):
    """Return the countries whose rows at a level were removed, for now.

    Like `unsourced`, a country here has no rows at this level; unlike
    it, that is a scheduling decision rather than a finding about what
    sources exist. See `STATUSES`.

    Parameters
    ----------
    level : int, optional
        Restrict to one level. Unset returns every pair.

    Returns
    -------
    set of str or set of tuple
        Country codes for one level, or (country, level) pairs.
    """
    table = load_ledger()
    rows = table[table['status'] == 'deferred']
    if level is None:
        return {(row.admin1_id, int(row.level)) for row in rows.itertuples()}
    return set(rows.loc[rows['level'] == str(level), 'admin1_id'])


def unsourced(level=None):
    """Return the countries the spine deliberately says nothing about.

    The counterpart of `migrated`: a country here has no rows at this
    level, and that is a decision rather than an omission.

    Parameters
    ----------
    level : int, optional
        Restrict to one level. Unset returns every pair.

    Returns
    -------
    set of str or set of tuple
        Country codes for one level, or (country, level) pairs.
    """
    table = load_ledger()
    rows = table[table['status'] == 'unsourced']
    if level is None:
        return {(row.admin1_id, int(row.level)) for row in rows.itertuples()}
    return set(rows.loc[rows['level'] == str(level), 'admin1_id'])


def in_scope(level):
    """Return the countries whose level a global layer may supply.

    A country whose own admin recipe covers the level keeps it: those
    are the national sources, and a global harvest would be discarded.
    """
    spine = get_admin(level=level, all_columns=True)
    countries = set(spine.index.to_series().str.split('-', n=1).str[0])
    scoped = set()
    for country in countries:
        found = find_admin_recipe_id(country, level, silent=True)
        if not (bool(found) and not str(found).startswith('admin-')):
            scoped.add(country)
    return scoped


def _units(recipe_id, alpha2):
    """Read one country's rows from an ingested layer, or an empty frame."""
    try:
        path = get_output_path(get_recipe_by_id(recipe_id), admin_id=alpha2)
    except (FileNotFoundError, KeyError, ValueError):
        return pd.DataFrame()
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _names(frame):
    if 'name' not in frame or not len(frame):
        return set()
    return {normalize_name(n) for n in frame['name'].dropna() if str(n).strip()}


def measure(alpha2, level, spine=None):
    """Measure one country and level against the spine and the reference.

    Parameters
    ----------
    alpha2 : str
        The country's level-1 id.
    level : int
        Admin level.
    spine : pandas.DataFrame, optional
        The level's spine, already loaded. Measuring the whole world
        otherwise re-reads it once per country.

    Returns
    -------
    dict
        A ledger row, without a status.
    """
    if spine is None:
        spine = get_admin(level=level, all_columns=True)
    country_of = spine.index.to_series().str.split('-', n=1).str[0]
    mine = spine[country_of.to_numpy() == alpha2]
    harvest = _units(UNIT_RECIPE.format(level=level), alpha2)
    reference = _units(REFERENCE_RECIPE.format(level=level), alpha2)
    harvest_names, reference_names = _names(harvest), _names(reference)
    shared = len(harvest_names & reference_names)
    types = (
        harvest['type'].value_counts() if 'type' in harvest else pd.Series(dtype=int)
    )
    return {
        'admin1_id': alpha2,
        'level': str(level),
        'units_spine': str(len(mine)),
        'units_wikidata': str(len(harvest)),
        'units_geoboundaries': str(len(reference)),
        # Blank rather than zero where there is nothing to compare
        # against, so an absent reference does not read as disagreement.
        'agreement': (
            f'{shared / len(reference_names):.2f}' if reference_names else ''
        ),
        # The class the rule chose, by its label. A Q-number here is a
        # class Wikidata has no English label for, which is worth
        # seeing rather than hiding.
        'unit_type': str(types.index[0]) if len(types) else '',
    }


def refresh(admin_ids=None, levels=LEVELS, status=None, note=None):
    """Re-measure countries and write the ledger back.

    Rows the ledger does not have are added as `pending`; rows it has
    keep their status unless one is given. A country and level the
    spine no longer has in scope is left alone rather than removed, so
    a `held` decision is never lost by a scope change.

    Parameters
    ----------
    admin_ids : iterable of str, optional
        Countries to measure. Unset measures every country in scope.
    levels : iterable of int, optional
        Levels to measure.
    status : str, optional
        Status to set on the measured rows.
    note : str, optional
        Note to set on the measured rows.

    Returns
    -------
    pandas.DataFrame
        The ledger as written.
    """
    if status is not None and status not in STATUSES:
        raise ValueError(f'status must be one of {STATUSES}, not {status!r}.')
    table = load_ledger()
    indexed = {
        (row.admin1_id, row.level): dict(row._asdict())
        for row in table.itertuples(index=False)
    }
    stamp = pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d')
    for level in levels:
        countries = sorted(in_scope(level) if admin_ids is None else admin_ids)
        spine = get_admin(level=level, all_columns=True)
        for alpha2 in countries:
            key = (alpha2, str(level))
            row = indexed.get(key, {'status': 'pending', 'note': '', 'harvested': ''})
            row.update(measure(alpha2, level, spine=spine))
            if status is not None:
                row['status'] = status
                if status in ('harvested', 'migrated'):
                    row['harvested'] = stamp
            if note is not None:
                row['note'] = note
            row.setdefault('status', 'pending')
            indexed[key] = row
    written = pd.DataFrame(list(indexed.values()))
    for column in COLUMNS:
        if column not in written:
            written[column] = ''
    save_ledger(written.fillna(''))
    return load_ledger()


def summary():
    """Return how many country-levels stand at each status, per level."""
    table = load_ledger()
    if table.empty:
        return pd.DataFrame()
    return (
        table.pivot_table(
            index='level', columns='status', values='admin1_id', aggfunc='count'
        )
        .fillna(0)
        .astype(int)
    )
