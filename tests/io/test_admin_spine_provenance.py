"""The committed spine carries nothing its sources forbid publishing.

GADM allows academic use but not redistribution, so no spine column may
hold a value only GADM supplied. Codes are checked against
admin-spine-2026_code-sources.csv, which names the license-clean recipe
behind each country's codes at each level: a code from any other
country can only have come from GADM. Alternative and native-script
names came from GADM alone, so none may be published at all.

Only the present spine is kept. Snapshots of earlier vintages carried
the same GADM values and resolved nothing a present-only spine needs.
"""

import pandas as pd
import pytest

from openplaces.path import spine_path
from openplaces.recipe import get_recipe_by_id

LEVELS = (2, 3, 4)


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
def test_no_alternative_or_native_names_are_published(level):
    frame = _spine(level)
    for column in ('name_alternatives', 'name_original'):
        filled = int((frame.get(column, pd.Series(dtype=str)) != '').sum())
        assert filled == 0, f'level {level}: {filled} values in {column}'


def test_every_registered_source_is_a_recipe():
    for recipe_id in _sources()['recipe_id']:
        assert get_recipe_by_id(recipe_id), recipe_id


def test_only_the_present_spine_is_kept():
    stale = sorted(p.name for p in spine_path(2).parent.glob('*superseded*'))
    assert not stale, f'snapshots of earlier spines are committed: {stale}'
