"""The ledger that records which countries have moved off GADM.

The move runs one country at a time, so the record of where it stands
has to live in the repository: the note the previous session left was in
a scratchpad and did not survive it.
"""

import pandas as pd
import pytest

from openplaces.io import admin_migration as mig


def _stub_measure(monkeypatch):
    """Measure nothing: these tests are about the ledger, not the counts."""
    monkeypatch.setattr(
        mig,
        'measure',
        lambda alpha2, level, spine=None: {
            'admin1_id': alpha2,
            'level': str(level),
        },
    )
    monkeypatch.setattr(mig, 'get_admin', lambda *a, **k: pd.DataFrame())


@pytest.fixture
def ledger(monkeypatch, tmp_path):
    """Point the ledger at a temporary file."""
    path = tmp_path / 'migration.csv'
    monkeypatch.setattr(mig, 'ledger_path', lambda: path)
    return path


def test_an_absent_ledger_reads_as_empty_with_its_columns(ledger):
    table = mig.load_ledger()
    assert table.empty
    assert list(table.columns) == list(mig.COLUMNS)


def test_a_round_trip_keeps_every_value_as_text(ledger):
    mig.save_ledger(
        pd.DataFrame(
            [
                {
                    'admin1_id': 'KE',
                    'level': '2',
                    'status': 'migrated',
                    'units_spine': '47',
                    'units_wikidata': '47',
                    'units_geoboundaries': '47',
                    'agreement': '0.98',
                    'unit_type': 'county of Kenya',
                    'harvested': '2026-09-20',
                    'note': '',
                }
            ]
        )
    )
    table = mig.load_ledger()
    assert table.loc[0, 'status'] == 'migrated'
    # A count read back as 47.0 would not compare equal to what a
    # measurement writes, and the ledger is diffed in git.
    assert table.loc[0, 'units_spine'] == '47'


def test_it_is_written_without_a_byte_order_mark_and_with_lf(ledger):
    mig.save_ledger(pd.DataFrame([{'admin1_id': 'KE', 'level': '2'}]))
    raw = ledger.read_bytes()
    assert not raw.startswith(b'\xef\xbb\xbf')
    assert b'\r\n' not in raw


def test_migrated_reads_only_the_rows_that_say_so(ledger):
    mig.save_ledger(
        pd.DataFrame(
            [
                {'admin1_id': 'KE', 'level': '2', 'status': 'migrated'},
                {'admin1_id': 'UG', 'level': '2', 'status': 'harvested'},
                {'admin1_id': 'RW', 'level': '3', 'status': 'migrated'},
            ]
        )
    )
    assert mig.migrated(2) == {'KE'}
    assert mig.migrated(3) == {'RW'}
    assert mig.migrated() == {('KE', 2), ('RW', 3)}


def test_a_status_outside_the_vocabulary_is_refused(ledger):
    with pytest.raises(ValueError, match='status must be one of'):
        mig.refresh(admin_ids=['KE'], levels=(2,), status='done')


def test_refresh_keeps_a_held_decision_and_its_note(ledger, monkeypatch):
    mig.save_ledger(
        pd.DataFrame(
            [
                {
                    'admin1_id': 'DE',
                    'level': '3',
                    'status': 'held',
                    'note': 'no Wikidata class for its Kreise',
                }
            ]
        )
    )
    _stub_measure(monkeypatch)
    table = mig.refresh(admin_ids=['DE'], levels=(3,))
    row = table[table['admin1_id'] == 'DE'].iloc[0]
    assert row['status'] == 'held'
    assert row['note'] == 'no Wikidata class for its Kreise'


def test_refresh_adds_an_unknown_country_as_pending(ledger, monkeypatch):
    _stub_measure(monkeypatch)
    table = mig.refresh(admin_ids=['KE'], levels=(2,))
    assert table.loc[0, 'status'] == 'pending'


def test_refresh_stamps_the_date_when_it_records_a_harvest(ledger, monkeypatch):
    _stub_measure(monkeypatch)
    table = mig.refresh(admin_ids=['KE'], levels=(2,), status='harvested')
    assert table.loc[0, 'status'] == 'harvested'
    assert table.loc[0, 'harvested']


def test_a_country_with_its_own_recipe_is_not_in_scope():
    # The US Census covers every US level, so a global layer never
    # supplies one; Kenya has no national recipe at all.
    scope = mig.in_scope(2)
    assert 'US' not in scope
    assert 'KE' in scope
