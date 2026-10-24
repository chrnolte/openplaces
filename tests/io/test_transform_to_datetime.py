"""The `to_datetime` unary transform with a format and a century rule.

Florida DEP writes closing dates as DD-MON-YY over closings since the
1930s; strptime reads '15-MAR-52' as 2052. `latest_year` moves a date
parsed after it back a century.
"""

import pandas as pd

from openplaces.io.transform import apply_transformations


def _recipe(args=None):
    config = {
        'type': 'unary',
        'operation': 'to_datetime',
        'input': 'x',
        'output': 'y',
    }
    if args is not None:
        config['args'] = args
    return {'transformations': [config]}


def test_without_args_it_parses_and_blanks_what_does_not_parse():
    df = pd.DataFrame({'x': ['2001-02-03', 'not a date']})
    out = apply_transformations(df, _recipe())
    assert out['y'].iloc[0] == pd.Timestamp('2001-02-03')
    assert pd.isna(out['y'].iloc[1])


def test_a_format_reads_upper_case_month_abbreviations():
    df = pd.DataFrame({'x': ['03-FEB-01']})
    out = apply_transformations(df, _recipe({'format': '%d-%b-%y'}))
    assert out['y'].iloc[0] == pd.Timestamp('2001-02-03')


def test_a_year_after_latest_year_moves_back_a_century():
    df = pd.DataFrame({'x': ['15-MAR-52', '15-MAR-26', '15-MAR-27', '15-MAR-99']})
    out = apply_transformations(
        df, _recipe({'format': '%d-%b-%y', 'latest_year': 2026})
    )
    assert out['y'].dt.year.tolist() == [1952, 2026, 1927, 1999]
    assert out['y'].dt.day.tolist() == [15, 15, 15, 15]


def test_a_list_of_dates_stays_missing():
    df = pd.DataFrame({'x': ['15-MAR-52;16-APR-53', None]})
    out = apply_transformations(
        df, _recipe({'format': '%d-%b-%y', 'latest_year': 2026})
    )
    assert out['y'].isna().all()
