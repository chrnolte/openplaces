"""The committed spine carries nothing its sources forbid publishing.

GADM allows academic use but not redistribution, so no spine column may
hold a value only GADM supplied. Codes are checked against
admin-spine-2026_code-sources.csv, which names the license-clean recipe
behind each country's codes at each level: a code from any other
country can only have come from GADM. Alternative and native-script
names came from GADM alone, so none may be published at all.

Only the present spine is kept. Snapshots of earlier vintages carried
the same GADM values and resolved nothing a present-only spine needs.

A country whose rows have been moved to Wikidata (CC0) says so with its
`admin{N}_id_wikidata`, and the migration ledger records which countries
those are; a native name (`name_original`) is Wikidata's too, and may
stand only on a row carrying that id. The move runs one country at a
time, so the check is against the ledger rather than against the whole
world at once.
"""

import pandas as pd
import pytest

from openplaces.path import spine_path
from openplaces.recipe import find_admin_recipe_id, get_recipe_by_id

LEVELS = (2, 3, 4)


def _national(country, level):
    """True when the country's own admin recipe covers this level."""
    found = find_admin_recipe_id(country, level, silent=True)
    return bool(found) and not str(found).startswith('admin-')


def _sources():
    path = spine_path(2).parent / 'admin-spine-2026_code-sources.csv'
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _spine(level):
    return pd.read_csv(spine_path(level), dtype=str, keep_default_na=False)


@pytest.mark.parametrize('level', LEVELS)
def test_every_code_has_a_registered_source(level):
    column = f'admin{level}_id_admin1'
    sources = _sources()
    allowed = set(sources.loc[sources['level'] == str(level), 'admin1_id'])
    frame = _spine(level)
    country = frame[f'admin{level}_id'].str.split('-').str[0]
    stray = sorted(set(country[(frame[column] != '') & ~country.isin(allowed)]))
    assert not stray, f'level {level}: {column} with no registered source: {stray}'


@pytest.mark.parametrize('level', LEVELS)
def test_no_alternative_names_are_published(level):
    frame = _spine(level)
    filled = int((frame.get('name_alternatives', pd.Series(dtype=str)) != '').sum())
    assert filled == 0, f'level {level}: {filled} values in name_alternatives'


@pytest.mark.parametrize('level', LEVELS)
def test_a_native_name_stands_only_on_a_wikidata_row(level):
    frame = _spine(level)
    native = frame.get('name_original', pd.Series('', index=frame.index)) != ''
    sourced = frame.get(f'admin{level}_id_wikidata', pd.Series('', index=frame.index))
    stray = int((native & (sourced == '')).sum())
    assert stray == 0, f'level {level}: {stray} native names without a Wikidata id'


# The world's rows move off GADM one country at a time, so this cannot
# be a single assertion about every country at once. It is an assertion
# about the countries the migration ledger says have moved, which is a
# real invariant from the first country onward and grows with the work.
#
# "Migrated" means off GADM, not specifically onto Wikidata. Germany
# went to its own federal mapping agency because no global source could
# name its Kreise, and that is a better outcome than Wikidata, not a
# failure of one. So a migrated country satisfies this by carrying
# Wikidata identifiers *or* by having a national recipe at that level.
@pytest.mark.parametrize('level', LEVELS)
def test_every_migrated_country_is_off_gadm(level):
    from openplaces.io.admin_migration import migrated

    done = migrated(level)
    if not done:
        pytest.skip(f'level {level}: no country has been migrated yet')
    frame = _spine(level)
    column = f'admin{level}_id_wikidata'
    assert column in frame, f'level {level}: no {column} column'
    country = frame[f'admin{level}_id'].str.split('-').str[0]
    from_wikidata = frame[column] != ''
    from_national = country.map(lambda c: _national(c, level))
    stray = sorted(
        set(
            frame.loc[
                country.isin(done) & ~from_wikidata & ~from_national,
                f'admin{level}_id',
            ]
        )
    )
    assert not stray, (
        f'level {level}: the ledger calls these countries migrated, but '
        f'{len(stray)} of their rows come from neither Wikidata nor a '
        f'national recipe: {stray[:10]}'
    )


@pytest.mark.parametrize('level', LEVELS)
def test_an_unsourced_country_has_no_rows_at_all(level):
    """What no open source names, the shipped spine does not carry.

    This is also the guard a contributor needs. A user who holds GADM
    may fill these countries in locally, which their own licence allows
    and this project's does not: doing so edits the spine inside the
    installed package, and without this test that edit would reach a
    pull request unnoticed.
    """
    from openplaces.io.admin_migration import unsourced

    empty = unsourced(level)
    if not empty:
        pytest.skip(f'level {level}: no country is recorded as unsourced')
    frame = _spine(level)
    country = frame[f'admin{level}_id'].str.split('-').str[0]
    stray = sorted(set(frame.loc[country.isin(empty), f'admin{level}_id']))
    assert not stray, (
        f'level {level}: the ledger says no open source names these units, '
        f'but the spine carries {len(stray)} of them: {stray[:10]}'
    )


@pytest.mark.parametrize('level', LEVELS)
def test_a_deferred_country_has_no_rows_at_that_level(level):
    """A level put out of scope is emptied, not left on the old source.

    The reason `deferred` exists rather than a note on `held`: the rows
    that were there came from GADM, and "we will get to this level
    later" is not a licence to keep publishing them meanwhile. The same
    contributor guard as the unsourced test above applies, and here it
    matters more, because refilling one of these is a plausible
    good-faith mistake: a source does name these units.
    """
    from openplaces.io.admin_migration import deferred

    empty = deferred(level)
    if not empty:
        pytest.skip(f'level {level}: no country is recorded as deferred')
    frame = _spine(level)
    country = frame[f'admin{level}_id'].str.split('-').str[0]
    stray = sorted(set(frame.loc[country.isin(empty), f'admin{level}_id']))
    assert not stray, (
        f'level {level}: the ledger defers these countries, so their rows '
        f'were removed, but the spine carries {len(stray)} again: {stray[:10]}'
    )


@pytest.mark.parametrize('level', LEVELS)
def test_a_country_not_yet_migrated_is_not_claimed_to_be(level):
    """The ledger may not run ahead of the spine, only behind it."""
    from openplaces.io.admin_migration import in_scope, migrated

    claimed = migrated(level)
    assert claimed <= in_scope(level) | {c for c in claimed if _national(c, level)}, (
        f'level {level}: the ledger lists countries the spine has no rows for'
    )


def test_every_registered_source_is_a_recipe():
    for recipe_id in _sources()['recipe_id']:
        assert get_recipe_by_id(recipe_id), recipe_id


def test_only_the_present_spine_is_kept():
    stale = sorted(p.name for p in spine_path(2).parent.glob('*superseded*'))
    assert not stale, f'snapshots of earlier spines are committed: {stale}'
