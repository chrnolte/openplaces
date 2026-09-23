"""Two parcel sources' split units: the more specific source wins.

Every id and value below is fabricated.
"""

import pandas as pd

from openplaces.io.harmonizer.spine import _most_specific_units


class _State:
    verbose = False


def _part(specificity, label, rows):
    frame = pd.DataFrame({'parcel_id_local': [f'u{i}' for i in rows]})
    return specificity, label, frame


def test_one_source_is_kept_whole():
    parts = [_part(2, 'statewide:units', range(3))]
    kept = _most_specific_units(parts, _State())
    assert len(kept) == 1 and len(kept[0]) == 3


def test_the_county_layers_units_win_over_the_statewide_layers(capsys):
    parts = [
        _part(2, 'statewide:units', range(728)),
        _part(3, 'county:units', range(638)),
    ]
    kept = _most_specific_units(parts, _State())
    assert [len(df) for df in kept] == [638]
    printed = capsys.readouterr().out
    assert 'statewide:units (728 rows)' in printed
    assert 'county:units' in printed


def test_two_sources_at_one_level_are_both_kept():
    parts = [_part(3, 'a:units', range(2)), _part(3, 'b:units', range(5))]
    assert sorted(len(df) for df in _most_specific_units(parts, _State())) == [2, 5]


def test_no_units_at_all_is_no_rows_and_no_message(capsys):
    assert _most_specific_units([], _State()) == []
    assert capsys.readouterr().out == ''
