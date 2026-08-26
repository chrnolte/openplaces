"""Tests for the code-keyed bundled `parcel_id_links.csv`.

The table used to be keyed on `admin_id`, which openplaces re-mints. Three
retargets by identifier string later, 163 ids were duplicated, 54 named no
live unit, and twenty rows carried a neighboring county's conversion rule.
These tests hold the key to something openplaces does not issue, and would
have caught every one of those failures.
"""

import numpy as np
import pandas as pd
import pytest

from openplaces.geo import ids
from openplaces.geo.build_parcel_id_links import measure_parcel_id_fit
from openplaces.geo.ids import (
    _PARCEL_ID_DIR,
    _PARCEL_ID_KINDS,
    _PARCEL_ID_RULE_COLUMNS,
    PARCEL_ID_RELINK_THRESHOLD,
    _parcel_id_link_table,
    _parcel_id_link_units,
    _parcel_id_links,
    _resolve_instruction,
    parcel_id_link_library,
)

IMPORT_COMMAND = 'python -m openplaces.geo.import_parcel_id_links'
KEY = ['country_id', 'admin_id_admin1', 'kind']


@pytest.fixture(scope='module')
def committed():
    """The table exactly as committed, before any spine resolution."""
    return pd.read_csv(
        _PARCEL_ID_DIR / 'parcel_id_links.csv', dtype=str, keep_default_na=False
    )


def test_key_columns_are_present_and_populated(committed):
    for column in KEY:
        assert column in committed.columns
        assert (committed[column].str.strip() != '').all(), (
            f'{column} is part of the key and may not be blank'
        )


def test_key_is_unique(committed):
    """Two rows claiming one rank for one unit and side means at least one
    describes a unit it does not name, which is how a neighboring county's
    rule gets applied."""
    duplicated = committed[committed.duplicated(KEY, keep=False)]
    assert duplicated.empty, (
        f'{len(duplicated)} rows share a (country, code, kind) key, '
        f'starting with {duplicated.iloc[0][KEY].to_dict()}. Rebuild with '
        f'`{IMPORT_COMMAND}`.'
    )


def test_every_key_resolves_to_a_live_unit(committed):
    """A key that resolves to nothing is a row that has stopped applying."""
    units = _parcel_id_link_units()
    unresolved = {
        (country, code)
        for country, code in zip(committed['country_id'], committed['admin_id_admin1'])
        if (country, code) not in units
    }
    assert not unresolved, (
        f'{len(unresolved)} keys name no unit in the current spine, starting '
        f'with {sorted(unresolved)[:5]}. Rebuild with `{IMPORT_COMMAND}`.'
    )


def test_admin_id_column_matches_what_the_key_resolves_to(committed):
    """Hygiene, not correctness: `admin_id` and `name` are read by people and
    joined on by nothing, so a re-mint leaves them stale without changing any
    behavior. Failing here is the signal to refresh them."""
    units = _parcel_id_link_units()
    resolved = [
        units.get((country, code))
        for country, code in zip(committed['country_id'], committed['admin_id_admin1'])
    ]
    stale = {
        (was, now) for was, now in zip(committed['admin_id'], resolved) if was != now
    }
    assert not stale, (
        f'{len(stale)} units carry an out-of-date `admin_id` comment, starting '
        f'with {sorted(stale)[:5]}. Refresh them with `{IMPORT_COMMAND}`.'
    )


def test_every_unit_has_a_rule_for_both_sides():
    """A unit present for one side only would silently fall through to
    `simple` on the other."""
    table = _parcel_id_link_table()
    counts = table.groupby('admin_id')['kind'].nunique()
    incomplete = counts[counts != len(_PARCEL_ID_KINDS)]
    assert incomplete.empty, (
        f'{len(incomplete)} units carry a rule for only one side, '
        f'starting with {list(incomplete.index[:5])}.'
    )


def test_one_rule_per_unit_and_side():
    """The table stores a decision, not a menu: the alternatives the search
    measured are collapsed into the smallest shared conversion library."""
    table = _parcel_id_link_table()
    assert not table.duplicated(['admin_id', 'kind']).any()


def test_the_conversion_library_is_far_smaller_than_the_table():
    """The point of the covering set. If a rebuild ever stopped sharing
    conversions between units this would climb toward one per unit."""
    library = parcel_id_link_library()
    distinct = library[_PARCEL_ID_RULE_COLUMNS].drop_duplicates()
    table = _parcel_id_link_table()
    assert len(distinct) < len(table) / 10
    assert library['n_units'].sum() == len(table)
    assert list(library['n_units']) == sorted(library['n_units'], reverse=True)


def test_loaded_index_is_unique_and_covers_every_unit(committed):
    links = _parcel_id_links()
    assert links.index.is_unique
    assert len(links) == committed['admin_id'].nunique()


def test_resolve_instruction_returns_scalars_for_a_linked_unit():
    """The duplicate-key failure surfaced here: `links.loc[aid]` returns a
    DataFrame for a duplicated index, and the `or 'simple'` fallback then
    evaluates a Series in a boolean context."""
    links = _parcel_id_links()
    sample = [links.index[0], links.index[len(links) // 2], links.index[-1]]
    for admin_id in sample:
        for kind in _PARCEL_ID_KINDS:
            pattern, conv, tolerance = _resolve_instruction(admin_id, None, kind)
            assert not isinstance(pattern, pd.Series)
            assert not isinstance(conv, pd.Series)
            assert isinstance(conv, str) or pd.isna(conv)
            assert tolerance is None


def test_both_admin_levels_are_represented():
    """The search resolved 123 counties by their subdivisions, and New England
    towns are the units that matter there. A rebuild that collapsed level 4
    onto counties would still look healthy without this."""
    links = _parcel_id_links()
    depths = links.index.str.count('-') + 1
    assert set(depths) == {3, 4}
    assert (depths == 4).sum() > 500


def test_an_unlinked_unit_falls_through_to_simple():
    pattern, conv, tolerance = _resolve_instruction('ZZ-ZZ-ZZ', None, 'parcel')
    assert (pattern, conv, tolerance) == (None, 'simple', None)


def test_an_absent_conversion_stays_absent(committed):
    """An empty `conv` means "apply this row's pattern and join on '|'",
    which is not what the `or 'simple'` fallback would make of an empty
    string. Rows like that exist, so the distinction has to survive loading."""
    blank = committed[
        (committed['conv'].str.strip() == '') & (committed['kind'] == 'parcel')
    ]
    if blank.empty:
        pytest.skip('no unit currently ships a blank parcel conversion')
    _, conv, _ = _resolve_instruction(blank.iloc[0]['admin_id'], None, 'parcel')
    assert (
        conv is None or conv is pd.NA or (isinstance(conv, float) and np.isnan(conv))
    ), f'a blank conversion resolved to {conv!r}, not to a missing value'


def test_a_duplicated_key_raises_rather_than_resolving_one_of_them(
    monkeypatch, tmp_path, committed
):
    """Failing loudly is the point: a silently wrong conversion produces a
    plausible key that joins to nothing, which nothing downstream notices."""
    doubled = pd.concat([committed, committed.head(1)], ignore_index=True)
    doubled.to_csv(tmp_path / 'parcel_id_links.csv', index=False)
    monkeypatch.setattr(ids, '_PARCEL_ID_DIR', tmp_path)
    _parcel_id_link_table.cache_clear()
    _parcel_id_links.cache_clear()
    try:
        with pytest.raises(ValueError, match='shared admin ids'):
            _parcel_id_links()
    finally:
        _parcel_id_link_table.cache_clear()
        _parcel_id_links.cache_clear()


def test_the_library_is_what_a_failed_link_should_try():
    """A newly ingested source that does not fit its unit's rule needs
    somewhere to go, and the library is deliberately small enough to try."""
    library = parcel_id_link_library()
    assert len(library)
    assert set(library['kind']) == set(_PARCEL_ID_KINDS)
    for column in _PARCEL_ID_RULE_COLUMNS:
        assert column in library.columns
    top = library.iloc[0]
    assert top['n_units'] > 100


def test_a_units_rule_is_a_member_of_the_library():
    links = _parcel_id_links()
    library = parcel_id_link_library()
    known = set(
        zip(
            library['pattern'],
            library['conv'],
            library['source_column'],
            library['kind'],
        )
    )
    admin_id = links.index[len(links) // 2]
    row = links.loc[admin_id]
    for kind in _PARCEL_ID_KINDS:
        rule = tuple(
            '' if pd.isna(row[f'{part}_{kind}']) else row[f'{part}_{kind}']
            for part in ('pattern', 'conv', 'source_column')
        )
        assert (*rule, kind) in known


def test_the_relink_threshold_is_a_fraction():
    """`link_by_id` compares an achieved match share against it, and
    `recheck_parcel_id_links` re-runs the search below it."""
    assert 0 < PARCEL_ID_RELINK_THRESHOLD <= 1


def test_fit_reports_a_rule_that_matches_nothing():
    """The single-source check. A rule written for another source's id
    format converts nothing, which is visible without a second dataset --
    this is how Maine's 557,000 ingested parcels showed the bundled rules
    did not fit them."""
    links = _parcel_id_links()
    admin_id = next(
        a
        for a in links.index
        if isinstance(links.loc[a, 'pattern_parcel'], str)
        and links.loc[a, 'pattern_parcel'] not in ('Ignored', 'Unrecognized')
    )
    fit = measure_parcel_id_fit(pd.Series(['@@@ not an id @@@'] * 20), admin_id)
    assert fit['n'] == 20
    assert fit['converted'] == 0.0
    assert fit['converted_simple'] > 0
    assert fit['excess_loss'] > 0


def test_fit_is_empty_for_an_empty_column():
    fit = measure_parcel_id_fit(pd.Series([], dtype='string'), 'US-AL-AU')
    assert fit == {
        'n': 0,
        'converted': 0.0,
        'converted_simple': 0.0,
        'excess_loss': 0.0,
        'uniqueness': 0.0,
    }


def test_fit_reports_uniqueness_separately_from_coverage():
    """The opposite failure: a conversion that matches everything and
    collapses distinct ids onto one key."""
    ids = pd.Series(['1-1', '1-2', '1-3', '1-4'])
    fit = measure_parcel_id_fit(ids, 'ZZ-ZZ-ZZ')
    assert fit['converted'] == 1.0
    assert fit['uniqueness'] == 1.0
