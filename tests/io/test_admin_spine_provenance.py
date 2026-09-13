"""The committed spine carries nothing its sources forbid publishing.

GADM allows academic use but not redistribution, so no spine column may
hold a value only GADM supplied. Codes are checked against
admin-spine-2026_code-sources.csv, which names the license-clean recipe
behind each country's codes at each level: a code from any other
country can only have come from GADM. Alternative and native-script
names came from GADM alone, so none may be published at all.

Only the present spine is kept. Snapshots of earlier vintages carried
the same GADM values and resolved nothing a present-only spine needs.

Every row of a country without a national admin recipe comes from
Wikidata (CC0) and says so with its `admin{N}_id_wikidata`; a native
name (`name_original`) is Wikidata's too, and may stand only on such a
row.
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


# The world rows are not yet replaced from the Wikidata layer: the
# harvest and the update are built and tested, the replacement itself
# (a re-mint of every non-national country) is postponed until the
# project works in those countries. Strict, so the day it lands this
# marker has to go.
@pytest.mark.xfail(strict=True, reason='world rows from Wikidata postponed')
@pytest.mark.parametrize('level', LEVELS)
def test_every_row_without_a_national_source_comes_from_wikidata(level):
    frame = _spine(level)
    column = f'admin{level}_id_wikidata'
    assert column in frame, f'level {level}: no {column} column'
    country = frame[f'admin{level}_id'].str.split('-').str[0]
    national = {c for c in set(country) if _national(c, level)}
    # Z1 to Z9 are the disputed-area placeholders GADM invented; no
    # source but GADM knows them, and their removal from level 1 is a
    # decision still open (see the phase-3 plan).
    placeholder = country.str.fullmatch(r'Z\d')
    unsourced = frame[~country.isin(national) & ~placeholder & (frame[column] == '')]
    stray = sorted(set(unsourced[f'admin{level}_id'].str.split('-').str[0]))
    assert not stray, f'level {level}: rows with no Wikidata id in {stray}'


def test_every_registered_source_is_a_recipe():
    for recipe_id in _sources()['recipe_id']:
        assert get_recipe_by_id(recipe_id), recipe_id


def test_only_the_present_spine_is_kept():
    stale = sorted(p.name for p in spine_path(2).parent.glob('*superseded*'))
    assert not stale, f'snapshots of earlier spines are committed: {stale}'
