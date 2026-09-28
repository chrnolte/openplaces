"""Units a person has decided the spine will not carry.

The spine's standing position, set by the maintainer on 2026-09-23, is
to carry no territorial claim it does not already carry. Migrating the
four largest remaining countries would otherwise have reversed that in
four disputes at once: Wikidata types Taiwan as a province of China,
Zaporozhye as an oblast of Russia, and puts Kashmir on both the Indian
and the Pakistani side of that dispute.

These tests pin the mechanism and the file, not the politics: that the
exclusions are read by Q-number, that every row carries a reason, and
that the four decisions are present.
"""

import pandas as pd
import pytest

from openplaces.io.scrapers import wikidata_admin_scraper as scraper


def test_every_exclusion_carries_a_reason():
    """The file is a record of decisions, not a list of deletions."""
    table = scraper.load_exclusions()
    if table.empty:
        pytest.skip('no exclusions recorded')
    blank = table[table['reason'].astype(str).str.strip() == '']
    assert blank.empty, (
        f'{len(blank)} exclusion(s) give no reason: {list(blank["name"])}'
    )


def test_every_exclusion_names_a_q_number():
    """Keyed by Q-number so a relabelling upstream cannot re-admit one."""
    table = scraper.load_exclusions()
    if table.empty:
        pytest.skip('no exclusions recorded')
    ids = table['wikidata_id'].astype(str)
    assert ids.str.fullmatch(r'Q\d+').all(), list(ids[~ids.str.fullmatch(r'Q\d+')])


def test_every_exclusion_names_a_level():
    table = scraper.load_exclusions()
    if table.empty:
        pytest.skip('no exclusions recorded')
    levels = set(table['level'].astype(str))
    assert levels <= {'2', '3', '4'}, levels


@pytest.mark.parametrize(
    ('alpha2', 'qid'),
    [
        ('CN', 'Q57251'),  # Taiwan
        ('IN', 'Q66278313'),  # Jammu and Kashmir
        ('IN', 'Q200667'),  # Ladakh
        ('PK', 'Q200130'),  # Azad Kashmir
        ('PK', 'Q200697'),  # Gilgit-Baltistan
        ('RU', 'Q114333615'),  # Zaporozhye
    ],
)
def test_the_recorded_decisions_are_in_force(alpha2, qid):
    """Each of the four disputes the 2026-09-23 decision covered."""
    assert qid in scraper._excluded_ids(alpha2, 2)


def test_an_exclusion_applies_only_to_its_own_country_and_level():
    """Taiwan is excluded from China's level 2, not from everything."""
    assert 'Q57251' not in scraper._excluded_ids('CN', 3)
    assert 'Q57251' not in scraper._excluded_ids('JP', 2)


def test_a_country_with_no_exclusions_gets_an_empty_set():
    assert scraper._excluded_ids('FR', 2) == set()


def test_a_missing_file_is_not_an_error(monkeypatch, tmp_path):
    """A checkout without the sidecar still harvests, excluding nothing."""
    monkeypatch.setattr(scraper, 'recipe_path', lambda *a, **k: tmp_path / 'absent.csv')
    table = scraper.load_exclusions()
    assert isinstance(table, pd.DataFrame)
    assert table.empty
    assert list(table.columns) == list(scraper.EXCLUSION_COLUMNS)
