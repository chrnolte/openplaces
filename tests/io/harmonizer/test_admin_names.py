"""Assigning a town id from the name a record states.

Every name here is fabricated.
"""

import pandas as pd

from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer import admin_names as an

UNITS = pd.DataFrame(
    {
        'name': ['Quillby city', 'Quillby', 'St. Orrin', 'Marlowe Falls'],
        'type': ['City', 'Town', 'Town', 'Village'],
    },
    index=pd.Index(['XX-YY-ZZ-QC', 'XX-YY-ZZ-QT', 'XX-YY-ZZ-SO', 'XX-YY-ZZ-MF']),
)
PATTERN = r'^(?P<name>.*?),?\s+(?P<type>town|village|city)\b(?:\s+of?)?$'


def test_a_town_and_the_city_of_the_same_name_are_told_apart_by_type():
    names = pd.Series(['Quillby', 'QUILLBY', 'Saint Orrin', 'marlowe falls'])
    types = pd.Series(['City', 'town', 'TOWN', 'Village'])
    out = an.match_admin_names(names, types, UNITS)
    assert out.tolist() == [
        'XX-YY-ZZ-QC',
        'XX-YY-ZZ-QT',
        'XX-YY-ZZ-SO',
        'XX-YY-ZZ-MF',
    ]


def test_nothing_close_is_accepted_and_nothing_ambiguous_is_guessed():
    # A misspelling, a unit of another type, and (without types) a name
    # two units share: all stay missing.
    out = an.match_admin_names(
        pd.Series(['Quilby', 'Marlowe Falls']), pd.Series(['Town', 'Town']), UNITS
    )
    assert out.isna().all()
    untyped = an.match_admin_names(pd.Series(['Quillby', 'St Orrin']), None, UNITS)
    assert pd.isna(untyped[0])
    assert untyped[1] == 'XX-YY-ZZ-SO'


def _state(spine):
    state = HarmonizeState.__new__(HarmonizeState)
    state.spine = spine
    state.admin_id = 'XX-YY-ZZ'
    state.metadata = {}
    state.timer = None
    state.verbose = False
    return state


def test_step_reads_the_sources_spelling_and_keeps_an_existing_id(monkeypatch):
    monkeypatch.setattr(an, 'get_admin', lambda admin_id, level: UNITS)
    spine = pd.DataFrame(
        {
            'city': [
                'Quillby, Town of',
                'QUILLBY, CITY OF',
                'Marlowe Falls, Village O',
                'Saint Orrin Town of',
                'Nowhere, Town of',
                None,
            ],
            'admin4_id': [None, None, None, None, None, 'XX-YY-ZZ-QT'],
        }
    )
    out = an.assign_admin_id_from_name(
        _state(spine), name_column='city', level=4, name_pattern=PATTERN
    ).spine
    assert out['admin4_id'].tolist()[:4] == [
        'XX-YY-ZZ-QT',
        'XX-YY-ZZ-QC',
        'XX-YY-ZZ-MF',
        'XX-YY-ZZ-SO',
    ]
    assert pd.isna(out['admin4_id'][4])
    assert out['admin4_id'][5] == 'XX-YY-ZZ-QT'


def test_step_is_a_no_op_where_the_level_does_not_exist(monkeypatch):
    def no_level(admin_id, level):
        raise ValueError('No admin IDs from reference spine found.')

    monkeypatch.setattr(an, 'get_admin', no_level)
    spine = pd.DataFrame({'city': ['Quillby, Town of']})
    out = an.assign_admin_id_from_name(
        _state(spine), name_column='city', level=4, name_pattern=PATTERN
    ).spine
    assert 'admin4_id' not in out.columns
